use crate::application::migrations::{StoreKind, migrate_store, prepare_store};
use crate::domain::ports::EmbeddingIdentity;
use crate::infrastructure::config::ensure_private_dir;
use anyhow::Context;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior};
use std::path::Path;
use std::sync::Once;

/// How long a connection waits on a locked database before failing. WAL gives
/// concurrent readers + one writer across processes (e.g. a transient MCP
/// session sharing the stores with a running daemon); writers queue up to this.
const BUSY_TIMEOUT_MS: u32 = 5000;

/// Register sqlite-vec as an auto-extension exactly once per process (the
/// registration is process-global; repeating it on every open is redundant).
fn register_sqlite_vec() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        #[allow(clippy::missing_transmute_annotations)]
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

pub fn init_db(path: Option<&impl AsRef<Path>>) -> rusqlite::Result<Connection> {
    register_sqlite_vec();

    let mut conn = match path {
        Some(p) => Connection::open(p)?,
        None => Connection::open_in_memory()?,
    };

    // busy_timeout MUST be set before any statement that takes a lock —
    // including `journal_mode=WAL`, which needs a write lock to convert a fresh
    // file. Set after, a second process opening the store while another holds
    // the lock failed immediately with "database is locked" (QA-7).
    conn.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS.into()))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    // Write transactions take the write lock up front (BEGIN IMMEDIATE) so
    // busy_timeout applies to them. A DEFERRED transaction that reads first and
    // upgrades to a writer later fails with SQLITE_BUSY without waiting when a
    // concurrent writer committed in between (QA-7: lost writes under
    // contention). Every `unchecked_transaction()` in the repositories is a write.
    conn.set_transaction_behavior(TransactionBehavior::Immediate);

    Ok(conn)
}

/// On-disk size of a SQLite database in bytes (`page_count * page_size`). For
/// an in-memory DB this is the in-memory footprint. Used by the metrics CT scan.
pub fn db_size_bytes(conn: &Connection) -> rusqlite::Result<i64> {
    let page_count: i64 = conn.query_row("PRAGMA page_count", [], |r| r.get(0))?;
    let page_size: i64 = conn.query_row("PRAGMA page_size", [], |r| r.get(0))?;
    Ok(page_count * page_size)
}

/// The two independent SQLite memory stores that make up NeuroLithe V2.
///
/// `stm` is today's decaying fact engine; `ltm` is the permanent knowledge
/// tree. They live in separate files so each can be reset and evolve
/// independently — STM is wiped by soft/hard reset, LTM only by hard reset, and
/// the decay path never touches LTM. One process owns both connections and
/// serializes access (as the repository does today).
pub struct MemoryStores {
    pub stm: Connection,
    pub ltm: Connection,
    /// Shared advisory lock on the workspace, held for as long as the stores
    /// are open (keep it alive alongside them). While any process holds one,
    /// `workspace delete` and `reembed` refuse the workspace (P2R-3).
    pub lease: WorkspaceLease,
}

/// Name of the advisory lock file inside a workspace directory.
pub const LOCK_FILE: &str = ".lock";

/// A **shared** advisory lock on `<workspace>/.lock`: "this workspace is open".
/// Any number of processes (a daemon plus transient MCP sessions) may hold one;
/// it blocks only an exclusive lock. Released on drop.
#[derive(Debug)]
pub struct WorkspaceLease {
    _file: std::fs::File,
}

/// An **exclusive** advisory lock on `<workspace>/.lock`: taken by destructive
/// maintenance (delete, reembed) and refused while any lease exists.
#[derive(Debug)]
pub struct ExclusiveWorkspaceLock {
    _file: std::fs::File,
}

/// Open (creating, 0600) the lock file of the workspace in `dir`.
fn open_lock_file(dir: &Path) -> anyhow::Result<std::fs::File> {
    let mut opts = std::fs::OpenOptions::new();
    opts.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let path = dir.join(LOCK_FILE);
    opts.open(&path)
        .with_context(|| format!("opening workspace lock {}", path.display()))
}

/// Take a shared lease on the workspace in `dir` (which must exist). Fails
/// only while another process holds the exclusive lock (a delete/reembed in
/// progress).
pub fn lease_workspace(dir: &Path) -> anyhow::Result<WorkspaceLease> {
    let file = open_lock_file(dir)?;
    match file.try_lock_shared() {
        Ok(()) => Ok(WorkspaceLease { _file: file }),
        Err(std::fs::TryLockError::WouldBlock) => anyhow::bail!(
            "workspace {} is being deleted or re-embedded by another process",
            dir.display()
        ),
        Err(std::fs::TryLockError::Error(e)) => {
            Err(anyhow::Error::new(e).context(format!("locking workspace {}", dir.display())))
        }
    }
}

/// Take the exclusive lock on the workspace `name` in `dir`, refusing while it
/// is open anywhere — in another process, or in this one (e.g. a daemon's own
/// workspace while its MCP session switched away).
pub fn lock_workspace_exclusive(dir: &Path, name: &str) -> anyhow::Result<ExclusiveWorkspaceLock> {
    let file = open_lock_file(dir)?;
    match file.try_lock() {
        Ok(()) => Ok(ExclusiveWorkspaceLock { _file: file }),
        Err(std::fs::TryLockError::WouldBlock) => anyhow::bail!(
            "workspace {name:?} is in use (open in a running neurolithe process); \
             stop that process or switch it to another workspace first"
        ),
        Err(std::fs::TryLockError::Error(e)) => {
            Err(anyhow::Error::new(e).context(format!("locking workspace {}", dir.display())))
        }
    }
}

/// Delete the workspace `name` in `dir`, refusing while it is open anywhere.
pub fn delete_workspace_dir(dir: &Path, name: &str) -> anyhow::Result<()> {
    let lock = lock_workspace_exclusive(dir, name)?;
    // Windows can't remove a directory holding an open file; elsewhere keep
    // the lock until the tree is gone so nobody opens it mid-delete.
    #[cfg(windows)]
    drop(lock);
    std::fs::remove_dir_all(dir)
        .with_context(|| format!("deleting workspace {}", dir.display()))?;
    #[cfg(not(windows))]
    drop(lock);
    Ok(())
}

/// File name of the STM store inside a workspace directory.
pub const STM_FILE: &str = "stm.sqlite";
/// File name of the LTM store inside a workspace directory.
pub const LTM_FILE: &str = "ltm.sqlite";

/// Busy wait for the shutdown checkpoint: short, so exit never stalls for
/// long behind another process (e.g. a daemon) reading the same store.
const CHECKPOINT_BUSY_MS: u64 = 1000;

/// Graceful-shutdown step: fold each existing store's WAL back into the main
/// file and truncate it (`PRAGMA wal_checkpoint(TRUNCATE)`), so a stopped
/// workspace is a self-contained pair of `.sqlite` files. Uses its own short-
/// lived connection; the process's store connections must be idle. Best-effort:
/// a store still in use elsewhere is reported (`busy`) rather than awaited.
/// Missing stores are skipped. Logs each outcome; returns whether every
/// existing store was fully checkpointed.
pub fn checkpoint_workspace(dir: &Path) -> bool {
    let mut all_ok = true;
    for file in [STM_FILE, LTM_FILE] {
        let path = dir.join(file);
        if !path.is_file() {
            continue;
        }
        let result = (|| -> rusqlite::Result<(i64, i64, i64)> {
            let conn = Connection::open(&path)?;
            conn.busy_timeout(std::time::Duration::from_millis(CHECKPOINT_BUSY_MS))?;
            conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
        })();
        match result {
            Ok((0, _, _)) => tracing::debug!("checkpointed {}", path.display()),
            Ok((_, log, done)) => {
                all_ok = false;
                tracing::warn!(
                    "checkpoint of {} incomplete (store busy; {done}/{log} WAL frames)",
                    path.display()
                );
            }
            Err(e) => {
                all_ok = false;
                tracing::warn!("checkpoint of {} failed: {e}", path.display());
            }
        }
    }
    all_ok
}

/// Open a workspace's two stores (`<dir>/stm.sqlite`, `<dir>/ltm.sqlite`),
/// creating the directory (0700) and store files (0600) on first use, and
/// prepare each: version guard, automatic pre-migration backup, migrations,
/// `meta`, and the embedding-identity check (a store embedded by another
/// model/dimension is refused with a pointer to `neurolithe reembed`).
///
/// Returns the stores plus human-readable notes (backups taken, migration
/// warnings) for the caller to log. Errors name the store and file (QA-15).
pub fn open_workspace_stores(
    dir: &Path,
    identity: &EmbeddingIdentity,
) -> anyhow::Result<(MemoryStores, Vec<String>)> {
    open_workspace_with(dir, |conn, kind, path| {
        prepare_store(conn, kind, Some(path), identity)
    })
}

/// Like [`open_workspace_stores`] but only migrates (no embedding check): used
/// by `workspace import`, whose legacy stores may predate the current embedder
/// and are fixed afterwards with `neurolithe reembed`. `dim` only shapes vector
/// tables that don't exist yet.
pub fn migrate_workspace_stores(
    dir: &Path,
    dim: usize,
) -> anyhow::Result<(MemoryStores, Vec<String>)> {
    open_workspace_with(dir, |conn, kind, path| {
        migrate_store(conn, kind, Some(path), dim)
    })
}

fn open_workspace_with(
    dir: &Path,
    prepare: impl Fn(
        &Connection,
        StoreKind,
        &Path,
    ) -> anyhow::Result<crate::application::migrations::PrepareReport>,
) -> anyhow::Result<(MemoryStores, Vec<String>)> {
    ensure_private_dir(dir).with_context(|| format!("creating workspace dir {}", dir.display()))?;
    let lease = lease_workspace(dir)?;
    let mut notes = Vec::new();
    let mut open = |label: &str, file: &str, kind: StoreKind| -> anyhow::Result<Connection> {
        let path = dir.join(file);
        create_private_file(&path)?;
        open_store(label, Some(&path), |conn| {
            let report = prepare(conn, kind, &path)?;
            if let Some(backup) = &report.backup {
                notes.push(format!(
                    "{label} store migrated v{} → v{} (backup: {})",
                    report.from_version,
                    report.to_version,
                    backup.display()
                ));
            }
            notes.extend(report.warnings.iter().map(|w| format!("{label}: {w}")));
            Ok(())
        })
    };
    let stm = open("STM", STM_FILE, StoreKind::Stm)?;
    let ltm = open("LTM", LTM_FILE, StoreKind::Ltm)?;
    Ok((MemoryStores { stm, ltm, lease }, notes))
}

/// Create an empty file with mode 0600 if it doesn't exist yet. SQLite gives
/// its `-wal`/`-shm` side files the main file's permissions, so the whole store
/// stays owner-only (2.1). An empty file is a valid empty database.
fn create_private_file(path: &Path) -> anyhow::Result<()> {
    if path.exists() {
        return Ok(());
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    match opts.open(path) {
        Ok(_) => Ok(()),
        // Another process won the race — fine, it's the same empty store.
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(anyhow::anyhow!("creating {}: {e}", path.display())),
    }
}

/// The on-disk footprint of one store: the main file plus its WAL (committed
/// data can sit in the WAL until a checkpoint). 0 if absent.
pub fn store_bytes(path: &Path) -> u64 {
    let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let mut wal = path.as_os_str().to_owned();
    wal.push("-wal");
    size(path) + size(Path::new(&wal))
}

/// The workspace directories under `root` (`<home>/workspaces`) whose names
/// pass `is_valid_name`, sorted. A missing root means "no workspaces".
pub fn list_workspace_dirs(
    root: &Path,
    is_valid_name: impl Fn(&str) -> bool,
) -> anyhow::Result<Vec<(String, std::path::PathBuf)>> {
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root).with_context(|| format!("listing {}", root.display()))? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        if let Some(name) = entry.file_name().to_str()
            && is_valid_name(name)
        {
            out.push((name.to_string(), entry.path()));
        }
    }
    out.sort();
    Ok(out)
}

/// Open an existing store file read-only (no schema changes, no WAL
/// conversion) — for export and backup of workspaces that may not be active.
fn open_read_only(path: &Path) -> anyhow::Result<Connection> {
    register_sqlite_vec();
    let conn = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("opening {} read-only", path.display()))?;
    conn.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS.into()))?;
    Ok(conn)
}

/// Whether `conn` has a table named `name`.
fn has_table(conn: &Connection, name: &str) -> rusqlite::Result<bool> {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
        [name],
        |r| r.get(0),
    )
}

/// JSON dump of a workspace: every STM fact (payload + cognitive attributes)
/// and every LTM document leaf (dataId, title, summary, provenance). Opens the
/// files read-only, so it works for any workspace, active or not.
pub fn export_workspace(dir: &Path, name: &str) -> anyhow::Result<serde_json::Value> {
    let mut facts = Vec::new();
    let stm_path = dir.join(STM_FILE);
    if stm_path.is_file() && std::fs::metadata(&stm_path)?.len() > 0 {
        let conn = open_read_only(&stm_path)?;
        if has_table(&conn, "nodes")? {
            let mut stmt = conn.prepare(
                "SELECT payload, ccl, status, relevance_score, support_count, created_at
                 FROM nodes ORDER BY id",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, Option<f64>>(3)?,
                    r.get::<_, Option<i64>>(4)?,
                    r.get::<_, Option<String>>(5)?,
                ))
            })?;
            for row in rows {
                let (payload, ccl, status, relevance, support, created_at) = row?;
                let payload: serde_json::Value =
                    serde_json::from_str(&payload).unwrap_or(serde_json::Value::String(payload));
                facts.push(serde_json::json!({
                    "payload": payload,
                    "ccl": ccl,
                    "status": status,
                    "relevance_score": relevance,
                    "support_count": support,
                    "created_at": created_at,
                }));
            }
        }
    }

    let mut leaves = Vec::new();
    let ltm_path = dir.join(LTM_FILE);
    if ltm_path.is_file() && std::fs::metadata(&ltm_path)?.len() > 0 {
        let conn = open_read_only(&ltm_path)?;
        if has_table(&conn, "leaves")? && has_table(&conn, "tree_nodes")? {
            let mut stmt = conn.prepare(
                "SELECT l.data_id, n.name, n.summary, l.provenance
                 FROM leaves l JOIN tree_nodes n ON n.id = l.tree_node_id
                 ORDER BY n.id",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })?;
            for row in rows {
                let (data_id, title, summary, provenance) = row?;
                let provenance: serde_json::Value = serde_json::from_str(&provenance)
                    .unwrap_or(serde_json::Value::String(provenance));
                leaves.push(serde_json::json!({
                    "data_id": data_id,
                    "title": title,
                    "summary": summary,
                    "provenance": provenance,
                }));
            }
        }
    }

    Ok(serde_json::json!({
        "workspace": name,
        "stm_facts": facts,
        "ltm_leaves": leaves,
    }))
}

/// `VACUUM INTO` a consistent, compacted copy of `src` at `dest` (which must
/// not exist). Works on a live store (it reads a snapshot) and folds the WAL in.
fn vacuum_into(src: &Path, dest: &Path) -> anyhow::Result<()> {
    if dest.exists() {
        anyhow::bail!("refusing to overwrite {}", dest.display());
    }
    let conn = open_read_only(src)?;
    let dest_str = dest
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF-8 path {}", dest.display()))?;
    // Create the destination empty with 0600 first (VACUUM INTO accepts an
    // empty existing file), so the copy is never readable under the umask,
    // not even briefly (P2R-9).
    create_private_file(dest)?;
    let result = conn
        .execute("VACUUM INTO ?1", [dest_str])
        .with_context(|| format!("copying {} to {}", src.display(), dest.display()));
    if result.is_err() {
        let _ = std::fs::remove_file(dest);
    }
    result.map(|_| ())
}

/// Back up a workspace's two stores into `out_dir` as timestamped files
/// (`<name>-<UTC stamp>-stm.sqlite` / `-ltm.sqlite`) via `VACUUM INTO`.
/// Returns the paths written.
pub fn backup_workspace(
    dir: &Path,
    name: &str,
    out_dir: &Path,
) -> anyhow::Result<Vec<std::path::PathBuf>> {
    ensure_private_dir(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let stamp: String = Connection::open_in_memory()?.query_row(
        "SELECT strftime('%Y%m%dT%H%M%SZ', 'now')",
        [],
        |r| r.get(0),
    )?;
    let mut written = Vec::new();
    for (file, kind) in [(STM_FILE, "stm"), (LTM_FILE, "ltm")] {
        let src = dir.join(file);
        if !src.is_file() || std::fs::metadata(&src)?.len() == 0 {
            continue;
        }
        let dest = out_dir.join(format!("{name}-{stamp}-{kind}.sqlite"));
        vacuum_into(&src, &dest)?;
        written.push(dest);
    }
    if written.is_empty() {
        anyhow::bail!("workspace {name:?} has no store files to back up");
    }
    Ok(written)
}

/// Import legacy store files into a **new** workspace directory. Each source
/// is copied byte-for-byte — the main file plus its `-wal` if present (a
/// WAL-mode store's latest commits may live only there) — into
/// `<dest>/{stm,ltm}.sqlite[-wal]` with mode 0600; the caller then opens the
/// copies, which runs the migrations. The sources are only read: SQLite never
/// opens them, so no `-wal`/`-shm` appears next to them and a read-only
/// source directory works (P2R-1).
pub fn import_store_files(stm_src: &Path, ltm_src: &Path, dest_dir: &Path) -> anyhow::Result<()> {
    import_store_files_with(stm_src, ltm_src, dest_dir, &mut |_| {})
}

/// [`import_store_files`] with a hook called after each file is copied (tests
/// use it to change a source mid-copy).
fn import_store_files_with(
    stm_src: &Path,
    ltm_src: &Path,
    dest_dir: &Path,
    after_copy: &mut dyn FnMut(&Path),
) -> anyhow::Result<()> {
    for src in [stm_src, ltm_src] {
        if !src.is_file() {
            anyhow::bail!("no such store file: {}", src.display());
        }
    }
    if dest_dir.exists() {
        anyhow::bail!("{} already exists", dest_dir.display());
    }
    ensure_private_dir(dest_dir)?;
    let result = copy_store(stm_src, &dest_dir.join(STM_FILE), after_copy)
        .and_then(|()| copy_store(ltm_src, &dest_dir.join(LTM_FILE), after_copy));
    if result.is_err() {
        // Don't leave a half-imported workspace behind.
        let _ = std::fs::remove_dir_all(dest_dir);
    }
    result
}

/// Size + mtime of a file (`None` if absent) — a cheap "did it change?" stamp.
fn file_stamp(path: &Path) -> Option<(u64, std::time::SystemTime)> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.len(), meta.modified().ok()?))
}

/// PRAGMA quick_check on a store file; `Err` unless it reports "ok".
pub fn quick_check(path: &Path) -> anyhow::Result<()> {
    let conn = init_db(Some(&path))
        .with_context(|| format!("opening {} for an integrity check", path.display()))?;
    let result: String = conn
        .query_row("PRAGMA quick_check", [], |r| r.get(0))
        .with_context(|| format!("integrity check of {}", path.display()))?;
    if result != "ok" {
        anyhow::bail!("{} failed its integrity check: {result}", path.display());
    }
    Ok(())
}

/// The embedding model recorded in a store's `meta` (`None` for a store that
/// predates embedding metadata). Read-only.
pub fn stored_embedding_model(path: &Path) -> anyhow::Result<Option<String>> {
    let conn = open_read_only(path)?;
    if !has_table(&conn, "meta")? {
        return Ok(None);
    }
    Ok(conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'embedding_model'",
            [],
            |r| r.get::<_, String>(0),
        )
        .optional()?)
}

/// Append a suffix to a path's file name (`x.sqlite` → `x.sqlite-wal`).
fn with_suffix(path: &Path, suffix: &str) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    std::path::PathBuf::from(name)
}

/// Byte-copy one store (+ its `-wal`, if any) into a new 0600 file. A byte
/// copy is only a consistent snapshot if nothing writes the source meanwhile,
/// so the source's size + mtime (main file and WAL) are compared before and
/// after; any change refuses the import (N2).
fn copy_store(src: &Path, dest: &Path, after_copy: &mut dyn FnMut(&Path)) -> anyhow::Result<()> {
    let pairs = [
        (src.to_path_buf(), dest.to_path_buf()),
        (with_suffix(src, "-wal"), with_suffix(dest, "-wal")),
    ];
    let before: Vec<_> = pairs.iter().map(|(from, _)| file_stamp(from)).collect();
    for (from, to) in &pairs {
        if !from.is_file() {
            continue; // no WAL: a rollback-journal or cleanly closed store
        }
        let mut reader =
            std::fs::File::open(from).with_context(|| format!("reading {}", from.display()))?;
        create_private_file(to)?;
        let mut writer = std::fs::OpenOptions::new()
            .write(true)
            .open(to)
            .with_context(|| format!("writing {}", to.display()))?;
        std::io::copy(&mut reader, &mut writer)
            .with_context(|| format!("copying {} to {}", from.display(), to.display()))?;
        writer.sync_all()?;
        after_copy(from);
    }
    let after: Vec<_> = pairs.iter().map(|(from, _)| file_stamp(from)).collect();
    if before != after {
        anyhow::bail!(
            "source store {} is in use (it changed during the copy); stop the process using it \
             and retry",
            src.display()
        );
    }
    Ok(())
}

/// How long [`open_store`] keeps retrying a store that is busy at startup.
const OPEN_RETRY_WINDOW: std::time::Duration = std::time::Duration::from_secs(10);

/// `SQLITE_BUSY` / `SQLITE_LOCKED` (any extended code) anywhere in the chain.
fn is_busy(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<rusqlite::Error>()
            .and_then(rusqlite::Error::sqlite_error_code)
            .is_some_and(|code| {
                matches!(
                    code,
                    rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
                )
            })
    })
}

/// Open one store and prepare it (schema/migrations), attaching the store name
/// + path to any failure.
///
/// Retries the whole open while the file is busy. `busy_timeout` alone is not
/// enough at startup: when several processes open a *fresh* file at once, the
/// rollback→WAL conversion (and first-time DDL) can deadlock on lock upgrades,
/// and SQLite then returns SQLITE_BUSY immediately *without* invoking the busy
/// handler (QA-7). Preparing is idempotent, so dropping the connection and
/// trying again with jittered backoff is safe.
fn open_store(
    label: &str,
    path: Option<&Path>,
    mut prepare: impl FnMut(&Connection) -> anyhow::Result<()>,
) -> anyhow::Result<Connection> {
    let shown = path.map_or_else(|| ":memory:".to_string(), |p| p.display().to_string());
    let deadline = std::time::Instant::now() + OPEN_RETRY_WINDOW;
    let mut attempt: u64 = 0;
    loop {
        let result = match init_db(path.as_ref()) {
            Err(e) => Err((anyhow::Error::new(e), "opening")),
            Ok(conn) => match prepare(&conn) {
                Ok(()) => Ok(conn),
                Err(e) => Err((e, "preparing")),
            },
        };
        match result {
            Ok(conn) => return Ok(conn),
            Err((e, _)) if is_busy(&e) && std::time::Instant::now() < deadline => {
                attempt += 1;
                // 10–60 ms, varied per process + attempt so contenders desync.
                let jitter = (u64::from(std::process::id()) + attempt * 17) % 50;
                std::thread::sleep(std::time::Duration::from_millis(10 + jitter));
            }
            Err((e, what)) => {
                return Err(e.context(format!("{what} {label} store at '{shown}'")));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_init_db_in_memory() {
        let _conn = init_db(None as Option<&String>).expect("Failed to init in-memory db");
    }

    const DIM: usize = 8;

    fn ident() -> EmbeddingIdentity {
        EmbeddingIdentity {
            provider: "test".into(),
            model: "test-model".into(),
            dim: DIM,
        }
    }

    /// `<home>/workspaces` for a throwaway home.
    fn workspaces(home: &Path) -> PathBuf {
        home.join("workspaces")
    }

    fn open(dir: &Path) -> MemoryStores {
        open_workspace_stores(dir, &ident()).expect("open").0
    }

    /// Graceful shutdown folds the WAL back into the store and truncates it,
    /// even while the process still holds its (idle) store connections.
    #[test]
    fn test_checkpoint_workspace_truncates_wal() {
        let home = tempfile::tempdir().unwrap();
        let dir = workspaces(home.path()).join("default");
        let stores = open(&dir);
        stores
            .stm
            .execute_batch("CREATE TABLE t(x); INSERT INTO t VALUES (1), (2), (3);")
            .unwrap();
        let wal = dir.join(format!("{STM_FILE}-wal"));
        assert!(
            std::fs::metadata(&wal).unwrap().len() > 0,
            "writes sit in the WAL"
        );

        assert!(checkpoint_workspace(&dir), "idle stores checkpoint fully");
        assert_eq!(std::fs::metadata(&wal).unwrap().len(), 0, "WAL truncated");
        // The data is in the main file: a fresh read sees it.
        drop(stores);
        let conn = Connection::open(dir.join(STM_FILE)).unwrap();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM t", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 3);
        // A missing workspace is a no-op, not an error.
        assert!(checkpoint_workspace(&home.path().join("nope")));
    }

    /// Both stores init into their own files independently, and their vector
    /// dimensions don't interfere: STM's `vec_nodes` is built at the STM
    /// dimension while LTM carries a different one.
    #[test]
    fn test_init_stores_independent() {
        let home = tempfile::tempdir().expect("tempdir");
        let dir = workspaces(home.path()).join("default");
        let stm_path = dir.join(STM_FILE);
        let ltm_path = dir.join(LTM_FILE);

        let stores = open(&dir);

        // Both files were created on disk — the stores are distinct.
        assert!(stm_path.exists(), "STM file should exist");
        assert!(ltm_path.exists(), "LTM file should exist");

        // STM ran its schema: the sqlite-vec virtual table is present and was
        // built at the STM dimension (a node-shaped embedding fits).
        let vec_table_exists: bool = stores
            .stm
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_nodes'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);
        assert!(vec_table_exists, "STM vec_nodes should be created");

        // The STM vec table accepts a vector of the STM dimension — proof the
        // dimension applied to STM and was not crossed with LTM's.
        let embedding = vec![0.0_f32; DIM];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                embedding.as_ptr() as *const u8,
                std::mem::size_of_val(embedding.as_slice()),
            )
        };
        stores
            .stm
            .execute(
                "INSERT INTO vec_nodes(node_id, embedding) VALUES (1, ?1)",
                rusqlite::params![bytes],
            )
            .expect("STM vec insert at STM dimension should succeed");
    }

    /// QA-7: opening a store while another connection holds the write lock must
    /// wait (busy_timeout) instead of failing with "database is locked". Before
    /// the fix `journal_mode=WAL` ran before `busy_timeout`, so the conversion
    /// of a fresh file hit SQLITE_BUSY immediately.
    #[test]
    fn test_open_waits_for_lock_instead_of_failing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("locked.sqlite");

        // A fresh (rollback-journal) file held under an EXCLUSIVE lock.
        let holder = Connection::open(&path).unwrap();
        holder
            .execute_batch("CREATE TABLE t(x); BEGIN EXCLUSIVE; INSERT INTO t VALUES (1);")
            .unwrap();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(300));
            holder.execute_batch("COMMIT").unwrap();
        });

        let opened = init_db(Some(&path));
        releaser.join().unwrap();
        let conn = opened.expect("open should wait for the lock, not fail");
        let mode: String = conn
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "wal");
    }

    /// QA-7: write transactions begin IMMEDIATE, so they queue on the lock
    /// (busy_timeout) rather than failing on a deferred read→write upgrade.
    #[test]
    fn test_write_transactions_are_immediate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("imm.sqlite");
        let a = init_db(Some(&path)).unwrap();
        a.execute_batch("CREATE TABLE t(x)").unwrap();
        let b = init_db(Some(&path)).unwrap();

        // An open IMMEDIATE transaction on `a` holds the write lock, so a second
        // IMMEDIATE begin on `b` must see the lock (with a tiny timeout, BUSY).
        let tx = a.unchecked_transaction().unwrap();
        b.busy_timeout(std::time::Duration::from_millis(50))
            .unwrap();
        let err = b.unchecked_transaction().err();
        assert!(
            err.is_some(),
            "a DEFERRED begin would succeed here; IMMEDIATE must contend for the lock"
        );
        tx.commit().unwrap();
        assert!(b.unchecked_transaction().is_ok());
    }

    /// QA-15: a store that cannot be opened names the store and the file.
    #[test]
    fn test_init_stores_error_names_the_file() {
        let home = tempfile::tempdir().expect("tempdir");
        let dir = workspaces(home.path()).join("broken");
        std::fs::create_dir_all(&dir).unwrap();
        let stm_path = dir.join(STM_FILE);
        // Not a database: SQLite fails on first use.
        std::fs::write(
            &stm_path,
            b"this is definitely not a sqlite file, just junk bytes",
        )
        .unwrap();
        let err = match open_workspace_stores(&dir, &ident()) {
            Ok(_) => panic!("corrupt store must fail"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("STM"), "error should name the store: {err}");
        assert!(
            err.contains(&*stm_path.to_string_lossy()),
            "error should name the file: {err}"
        );
    }

    /// 2.1: a new workspace dir is 0700 and its store files 0600.
    #[cfg(unix)]
    #[test]
    fn test_workspace_files_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let home = tempfile::tempdir().unwrap();
        let dir = workspaces(home.path()).join("default");
        let stores = open(&dir);
        stores
            .stm
            .execute_batch("CREATE TABLE t(x); INSERT INTO t VALUES (1);")
            .unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&dir), 0o700);
        assert_eq!(mode(&dir.join(STM_FILE)), 0o600);
        assert_eq!(mode(&dir.join(LTM_FILE)), 0o600);
        let wal = dir.join(format!("{STM_FILE}-wal"));
        if wal.exists() {
            assert_eq!(mode(&wal), 0o600, "WAL inherits the store's mode");
        }
    }

    /// 2.2: two workspaces share nothing — a row written in one is invisible
    /// in the other (separate files).
    #[test]
    fn test_workspaces_are_physically_separate() {
        let home = tempfile::tempdir().unwrap();
        let a = open(&workspaces(home.path()).join("a"));
        let b = open(&workspaces(home.path()).join("b"));
        a.stm
            .execute(
                "INSERT INTO nodes (tenant_id, payload) VALUES ('default', '{\"fact\":\"only in a\"}')",
                [],
            )
            .unwrap();
        let count = |c: &Connection| -> i64 {
            c.query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))
                .unwrap()
        };
        assert_eq!(count(&a.stm), 1);
        assert_eq!(count(&b.stm), 0);
        let names: Vec<String> = list_workspace_dirs(&workspaces(home.path()), |_| true)
            .unwrap()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    /// 2.2: export dumps STM facts + LTM leaves; backup writes timestamped
    /// VACUUM INTO copies; import copies legacy files into a new workspace.
    #[test]
    fn test_export_backup_and_import() {
        let home = tempfile::tempdir().unwrap();
        let dir = workspaces(home.path()).join("src");
        {
            let stores = open(&dir);
            stores
                .stm
                .execute(
                    "INSERT INTO nodes (tenant_id, payload) VALUES ('default', '{\"fact\":\"exported fact\"}')",
                    [],
                )
                .unwrap();
            stores
                .ltm
                .execute(
                    "INSERT INTO tree_nodes (name, summary, kind) VALUES ('Doc', 'sum', 'leaf')",
                    [],
                )
                .unwrap();
            let id = stores.ltm.last_insert_rowid();
            stores
                .ltm
                .execute(
                    "INSERT INTO leaves (tree_node_id, data_id, provenance) VALUES (?1, 'doc_1', '{}')",
                    [id],
                )
                .unwrap();
        }

        let dump = export_workspace(&dir, "src").unwrap();
        assert_eq!(dump["workspace"], "src");
        assert_eq!(dump["stm_facts"][0]["payload"]["fact"], "exported fact");
        assert_eq!(dump["ltm_leaves"][0]["data_id"], "doc_1");
        assert_eq!(dump["ltm_leaves"][0]["title"], "Doc");

        let out = home.path().join("backups");
        let written = backup_workspace(&dir, "src", &out).unwrap();
        assert_eq!(written.len(), 2);
        for p in &written {
            let name = p.file_name().unwrap().to_string_lossy();
            assert!(
                name.starts_with("src-") && name.ends_with(".sqlite"),
                "{name}"
            );
        }

        // Import the backups as a new workspace: same data, fresh dir.
        let dest = workspaces(home.path()).join("imported");
        import_store_files(&written[0], &written[1], &dest).unwrap();
        let stores = migrate_workspace_stores(&dest, DIM).unwrap().0;
        let facts: i64 = stores
            .stm
            .query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(facts, 1);
        // Importing over an existing workspace is refused.
        assert!(import_store_files(&written[0], &written[1], &dest).is_err());
    }

    /// 2.4 via the workspace opener: a workspace embedded by one model is
    /// refused under another, with a pointer to `neurolithe reembed`.
    #[test]
    fn test_workspace_open_refuses_other_embedder() {
        let home = tempfile::tempdir().unwrap();
        let dir = workspaces(home.path()).join("default");
        drop(open(&dir));
        let other = EmbeddingIdentity {
            provider: "test".into(),
            model: "another-model".into(),
            dim: DIM,
        };
        let err = match open_workspace_stores(&dir, &other) {
            Ok(_) => panic!("mismatched embedder must be refused"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("reembed"), "{err}");
    }

    fn listing(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// P2R-1: importing a WAL-mode legacy store (its latest commits only in
    /// `-wal`, no `-shm` — e.g. after a crash) copies the files byte-for-byte:
    /// the source directory is left exactly as it was (SQLite never opens it,
    /// so no `-shm`/`-wal` appears), a read-only source dir works, and the
    /// WAL-only rows arrive in the new workspace.
    #[test]
    fn test_import_wal_source_is_copied_not_opened() {
        let home = tempfile::tempdir().unwrap();
        let live = home.path().join("live");
        let legacy = home.path().join("legacy");
        std::fs::create_dir_all(&legacy).unwrap();
        {
            let stores = open(&live);
            stores
                .stm
                .execute(
                    "INSERT INTO nodes (tenant_id, payload) VALUES ('default', '{\"fact\":\"in the wal\"}')",
                    [],
                )
                .unwrap();
            // Snapshot the files while the writer is open: the row is only in
            // the WAL. Leave out -shm, as a crashed process would.
            for f in [STM_FILE, LTM_FILE] {
                for suffix in ["", "-wal"] {
                    let from = live.join(format!("{f}{suffix}"));
                    if from.exists() {
                        std::fs::copy(&from, legacy.join(format!("{f}{suffix}"))).unwrap();
                    }
                }
            }
        }
        assert!(
            legacy.join(format!("{STM_FILE}-wal")).exists(),
            "fixture has a WAL"
        );
        let before = listing(&legacy);
        #[cfg(unix)]
        let set_mode = |mode: u32| {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&legacy, std::fs::Permissions::from_mode(mode)).unwrap();
        };
        #[cfg(unix)]
        set_mode(0o555); // read-only source dir

        let dest = workspaces(home.path()).join("imported");
        let result = import_store_files(&legacy.join(STM_FILE), &legacy.join(LTM_FILE), &dest);
        #[cfg(unix)]
        set_mode(0o755);
        result.unwrap();
        assert_eq!(listing(&legacy), before, "source dir must be untouched");

        let stores = migrate_workspace_stores(&dest, DIM).unwrap().0;
        let fact: String = stores
            .stm
            .query_row(
                "SELECT json_extract(payload, '$.fact') FROM nodes",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(fact, "in the wal");
    }

    /// P2R-3: an open workspace holds a shared lease, so delete (exclusive)
    /// is refused until every opener has closed it; and while an exclusive
    /// lock is held, the workspace can't be opened.
    #[test]
    fn test_workspace_lock_blocks_delete_while_open() {
        let home = tempfile::tempdir().unwrap();
        let dir = workspaces(home.path()).join("busy");
        let stores = open(&dir);
        let second = open(&dir); // leases are shared: two openers are fine
        let err = delete_workspace_dir(&dir, "busy").unwrap_err().to_string();
        assert!(err.contains("in use"), "{err}");
        assert!(dir.exists());
        drop(stores);
        assert!(
            delete_workspace_dir(&dir, "busy").is_err(),
            "still open once"
        );
        drop(second);

        let lock = lock_workspace_exclusive(&dir, "busy").unwrap();
        let err = match open_workspace_stores(&dir, &ident()) {
            Ok(_) => panic!("must not open under an exclusive lock"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("being deleted or re-embedded"), "{err}");
        drop(lock);

        delete_workspace_dir(&dir, "busy").unwrap();
        assert!(!dir.exists());
    }

    /// P2R-9: backup copies are owner-only, and VACUUM INTO refuses to
    /// clobber an existing non-empty file.
    #[cfg(unix)]
    #[test]
    fn test_backup_files_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let home = tempfile::tempdir().unwrap();
        let dir = workspaces(home.path()).join("b");
        drop(open(&dir));
        let out = home.path().join("backups");
        for path in backup_workspace(&dir, "b", &out).unwrap() {
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "{}", path.display());
            assert!(vacuum_into(&dir.join(STM_FILE), &path).is_err());
        }
    }

    /// N2: a source written to during the copy is refused (the copy would not
    /// be a consistent snapshot) and nothing is left behind.
    #[test]
    fn test_import_refuses_source_changed_mid_copy() {
        let home = tempfile::tempdir().unwrap();
        let live = home.path().join("live");
        let stores = open(&live); // a "running process" holding the source
        let dest = workspaces(home.path()).join("imported");
        let mut first = true;
        let err = import_store_files_with(
            &live.join(STM_FILE),
            &live.join(LTM_FILE),
            &dest,
            &mut |_| {
                if std::mem::take(&mut first) {
                    stores
                        .stm
                        .execute(
                            "INSERT INTO nodes (tenant_id, payload) VALUES ('default', '{\"fact\":\"late write\"}')",
                            [],
                        )
                        .unwrap();
                }
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("is in use"), "{err}");
        assert!(!dest.exists(), "partial import removed");
    }

    /// N2: a corrupt store fails `quick_check`.
    #[test]
    fn test_quick_check_detects_corruption() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join("c.sqlite");
        {
            let conn = init_db(Some(&path)).unwrap();
            conn.execute_batch(
                "PRAGMA journal_mode=DELETE; CREATE TABLE t(x TEXT);
                 WITH RECURSIVE c(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM c WHERE i < 2000)
                 INSERT INTO t SELECT printf('%0100d', i) FROM c;",
            )
            .unwrap();
        }
        quick_check(&path).unwrap();
        // Scribble over a b-tree page in the middle of the table.
        let mut bytes = std::fs::read(&path).unwrap();
        let page = 4096 * 10;
        for b in &mut bytes[page..page + 4096] {
            *b = 0xAB;
        }
        std::fs::write(&path, bytes).unwrap();
        assert!(quick_check(&path).is_err());
    }
}
