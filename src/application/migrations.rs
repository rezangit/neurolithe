//! Store metadata and schema migrations (Phase 2 §4).
//!
//! Every store (STM and LTM, per workspace) carries:
//! - `PRAGMA user_version`: the schema version, advanced by an ordered list of
//!   migrations (0 → current). A legacy store with no `meta` table is v0.
//! - A `meta(key, value)` table: `schema_version`, `store_kind` (`stm`|`ltm`),
//!   `created_at`, and the embedding identity (`embedding_provider`,
//!   `embedding_model`, `embedding_dim`) that produced its vectors.
//!
//! [`prepare_store`] is the single entry point used whenever a store is opened.
//! It refuses stores that are newer than this binary, of the wrong kind, or
//! embedded by a different model/dimension (pointing at `neurolithe reembed`).
//! Before migrating an existing store it takes an automatic `VACUUM INTO` backup.

use crate::domain::models::WORKSPACE_TENANT;
use crate::domain::ports::EmbeddingIdentity;
use crate::infrastructure::schema::{
    LTM_VEC_TABLES, STM_VEC_TABLES, create_ltm_vec_tables, create_stm_vec_table, drop_vec_tables,
    init_ltm_schema, init_schema, vec_table_dim,
};
use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::path::{Path, PathBuf};

/// Which of a workspace's two stores a database file is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreKind {
    Stm,
    Ltm,
}

impl StoreKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            StoreKind::Stm => "stm",
            StoreKind::Ltm => "ltm",
        }
    }

    /// The schema version this binary brings the store to.
    pub fn latest_version(&self) -> i64 {
        self.migrations().len() as i64
    }

    fn migrations(&self) -> &'static [Migration] {
        match self {
            StoreKind::Stm => STM_MIGRATIONS,
            StoreKind::Ltm => LTM_MIGRATIONS,
        }
    }

    pub(crate) fn vec_tables(&self) -> &'static [&'static str] {
        match self {
            StoreKind::Stm => STM_VEC_TABLES,
            StoreKind::Ltm => LTM_VEC_TABLES,
        }
    }

    /// (a table only this kind has, a table only the other kind has) — to
    /// recognise a legacy store that predates `meta.store_kind`.
    fn signature_tables(&self) -> (&'static str, &'static str) {
        match self {
            StoreKind::Stm => ("nodes", "tree_nodes"),
            StoreKind::Ltm => ("tree_nodes", "nodes"),
        }
    }

    pub(crate) fn create_vec_tables(&self, conn: &Connection, dim: usize) -> rusqlite::Result<()> {
        match self {
            StoreKind::Stm => create_stm_vec_table(conn, dim),
            StoreKind::Ltm => create_ltm_vec_tables(conn, dim),
        }
    }
}

/// STM schema version: 2 (v1 base schema + meta, v2 single tenant).
pub const STM_SCHEMA_VERSION: i64 = 2;
/// LTM schema version: 1 (base schema + meta).
pub const LTM_SCHEMA_VERSION: i64 = 1;

/// State threaded through one migration run.
pub struct MigrationCtx {
    /// Vector dimension for tables a migration creates on a fresh store.
    pub dim: usize,
    /// Human-readable notes for the caller to log.
    pub warnings: Vec<String>,
}

/// One schema step, `user_version` N-1 → N. Runs inside the migration
/// transaction; must be idempotent against partially-shaped legacy stores.
pub type Migration = fn(&Transaction, &mut MigrationCtx) -> Result<()>;

const STM_MIGRATIONS: &[Migration] = &[stm_v1_base, stm_v2_single_tenant];
const LTM_MIGRATIONS: &[Migration] = &[ltm_v1_base];

fn create_meta_table(tx: &Transaction) -> Result<()> {
    tx.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        [],
    )?;
    Ok(())
}

/// v1: the base STM schema (idempotent `CREATE … IF NOT EXISTS`, so a legacy
/// v0 store keeps its tables and data) plus the `meta` table.
fn stm_v1_base(tx: &Transaction, ctx: &mut MigrationCtx) -> Result<()> {
    init_schema(tx, ctx.dim)?;
    create_meta_table(tx)
}

/// v2: workspaces replace tenants. Every STM row moves to
/// [`WORKSPACE_TENANT`]; if the store held several tenants, their data is
/// merged into this workspace and a warning lists them.
fn stm_v2_single_tenant(tx: &Transaction, ctx: &mut MigrationCtx) -> Result<()> {
    let mut stmt = tx.prepare(
        "SELECT tenant_id FROM nodes UNION SELECT tenant_id FROM episodes
         UNION SELECT tenant_id FROM ccl_registry ORDER BY 1",
    )?;
    let tenants: Vec<String> = stmt
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);

    if tenants.len() > 1 {
        ctx.warnings.push(format!(
            "store held {} tenants ({}); all merged into this workspace as '{}'. \
             Use separate workspaces to keep data apart.",
            tenants.len(),
            tenants.join(", "),
            WORKSPACE_TENANT
        ));
    }

    tx.execute("UPDATE nodes SET tenant_id = ?1", params![WORKSPACE_TENANT])?;
    tx.execute(
        "UPDATE episodes SET tenant_id = ?1",
        params![WORKSPACE_TENANT],
    )?;
    // `ccl_registry` is UNIQUE(tenant_id, name): the first definition of each
    // layer name wins, later duplicates from other tenants are dropped.
    tx.execute(
        "UPDATE OR IGNORE ccl_registry SET tenant_id = ?1",
        params![WORKSPACE_TENANT],
    )?;
    tx.execute(
        "DELETE FROM ccl_registry WHERE tenant_id != ?1",
        params![WORKSPACE_TENANT],
    )?;
    Ok(())
}

/// v1: the base LTM schema plus the `meta` table.
fn ltm_v1_base(tx: &Transaction, ctx: &mut MigrationCtx) -> Result<()> {
    init_ltm_schema(tx, ctx.dim)?;
    create_meta_table(tx)
}

/// What [`prepare_store`] did, for the caller to log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareReport {
    pub from_version: i64,
    pub to_version: i64,
    /// The automatic pre-migration backup, if one was taken.
    pub backup: Option<PathBuf>,
    pub warnings: Vec<String>,
}

/// Open-time preparation of a store: version guard, kind guard, automatic
/// backup + migrations, `meta` bookkeeping, and the embedding identity check.
///
/// `db_path` is the store's file (used for backups and messages); `None` for an
/// in-memory store. Errors are actionable: a newer store asks for a newer
/// binary; an embedding mismatch points at `neurolithe reembed`.
pub fn prepare_store(
    conn: &Connection,
    kind: StoreKind,
    db_path: Option<&Path>,
    embedding: &EmbeddingIdentity,
) -> Result<PrepareReport> {
    let mut report = migrate_store(conn, kind, db_path, embedding.dim)?;
    check_embedding(conn, kind, db_path, embedding, &mut report.warnings)?;
    Ok(report)
}

/// Version/kind guards, backup and migrations — without the embedding check
/// (used by `reembed`, which is how a mismatched store gets fixed).
pub fn migrate_store(
    conn: &Connection,
    kind: StoreKind,
    db_path: Option<&Path>,
    dim: usize,
) -> Result<PrepareReport> {
    let label = store_label(kind, db_path);
    let from_version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    let latest = kind.latest_version();

    if from_version > latest {
        bail!(
            "{label} has schema version {from_version}, but this neurolithe build supports \
             up to {latest}. It was written by a newer version — upgrade neurolithe to open it."
        );
    }
    check_kind(conn, kind, &label)?;

    let mut report = PrepareReport {
        from_version,
        to_version: from_version,
        backup: None,
        warnings: Vec::new(),
    };
    if from_version == latest {
        return Ok(report);
    }

    // An existing store (any user table) is backed up before it is touched.
    if let Some(path) = db_path
        && has_user_tables(conn)?
    {
        let backup = backup_store(conn, path, &format!("v{from_version}"))
            .with_context(|| format!("backing up {label} before migrating it"))?;
        report.backup = Some(backup);
    }

    let mut ctx = MigrationCtx {
        dim,
        warnings: Vec::new(),
    };
    let tx = conn.unchecked_transaction()?;
    for (i, migration) in kind
        .migrations()
        .iter()
        .enumerate()
        .skip(from_version as usize)
    {
        migration(&tx, &mut ctx)
            .with_context(|| format!("migrating {label} to schema v{}", i + 1))?;
    }
    set_meta(&tx, "schema_version", &latest.to_string())?;
    set_meta(&tx, "store_kind", kind.as_str())?;
    tx.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('created_at', datetime('now'))",
        [],
    )?;
    tx.pragma_update(None, "user_version", latest)?;
    tx.commit()?;

    report.to_version = latest;
    report.warnings = ctx.warnings;
    Ok(report)
}

/// Refuse to migrate/open a file that is the *other* kind of store.
fn check_kind(conn: &Connection, kind: StoreKind, label: &str) -> Result<()> {
    if table_exists(conn, "meta")? {
        if let Some(stored) = get_meta(conn, "store_kind")?
            && stored != kind.as_str()
        {
            bail!("{label} is a '{stored}' store, not '{}'", kind.as_str());
        }
        return Ok(());
    }
    let (own, other) = kind.signature_tables();
    if table_exists(conn, other)? && !table_exists(conn, own)? {
        bail!(
            "{label} looks like a legacy '{}' store (it has `{other}`, not `{own}`)",
            if kind == StoreKind::Stm { "ltm" } else { "stm" }
        );
    }
    Ok(())
}

/// Compare the store's recorded embedding identity with the configured one.
/// A store without a recorded identity (fresh or legacy) adopts the configured
/// one — rebuilding its (empty) vector tables if their dimension differs — or,
/// if it already holds vectors of another dimension, is refused.
fn check_embedding(
    conn: &Connection,
    kind: StoreKind,
    db_path: Option<&Path>,
    embedding: &EmbeddingIdentity,
    warnings: &mut Vec<String>,
) -> Result<()> {
    let label = store_label(kind, db_path);
    let stored_model = get_meta(conn, "embedding_model")?;
    let stored_dim = get_meta(conn, "embedding_dim")?.and_then(|d| d.parse::<usize>().ok());

    if let (Some(model), Some(dim)) = (&stored_model, stored_dim) {
        if *model != embedding.model || dim != embedding.dim {
            bail!(mismatch_message(&label, model, dim, embedding));
        }
        return Ok(());
    }

    // No identity recorded yet.
    let vectors = count_vectors(conn, kind)?;
    let table_dim = vec_table_dim(conn, kind.vec_tables()[0])?;
    if let Some(table_dim) = table_dim
        && table_dim != embedding.dim
    {
        if vectors > 0 {
            bail!(mismatch_message(
                &label,
                "an unrecorded model",
                table_dim,
                embedding
            ));
        }
        // Empty index at the wrong size: rebuild it at the embedder's size.
        drop_vec_tables(conn, kind.vec_tables())?;
        kind.create_vec_tables(conn, embedding.dim)?;
    }
    if vectors > 0 {
        warnings.push(format!(
            "{label} predates embedding metadata; assuming its {vectors} vector(s) came from \
             '{}' ({}-dim). If they did not, run `neurolithe reembed`.",
            embedding.model, embedding.dim
        ));
    }
    record_embedding(conn, embedding)?;
    Ok(())
}

fn mismatch_message(
    label: &str,
    stored_model: &str,
    stored_dim: usize,
    configured: &EmbeddingIdentity,
) -> String {
    format!(
        "{label} was embedded with {stored_model} ({stored_dim}-dim), but the configured \
         embedder is '{}' ({}-dim). Its vectors cannot be searched with the new model. \
         Run `neurolithe reembed` (with `--workspace <name>` for a non-default workspace) \
         to re-embed it, or switch the embedding config back.",
        configured.model, configured.dim
    )
}

/// Write the embedding identity into `meta`.
pub(crate) fn record_embedding(conn: &Connection, embedding: &EmbeddingIdentity) -> Result<()> {
    set_meta(conn, "embedding_provider", &embedding.provider)?;
    set_meta(conn, "embedding_model", &embedding.model)?;
    set_meta(conn, "embedding_dim", &embedding.dim.to_string())?;
    Ok(())
}

fn count_vectors(conn: &Connection, kind: StoreKind) -> Result<i64> {
    let mut total = 0;
    for table in kind.vec_tables() {
        if table_exists(conn, table)? {
            total += conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| {
                r.get::<_, i64>(0)
            })?;
        }
    }
    Ok(total)
}

/// Read one `meta` value (`None` if the table or key is absent).
pub fn get_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    if !table_exists(conn, "meta")? {
        return Ok(None);
    }
    Ok(conn
        .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
        .optional()?)
}

fn set_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

fn table_exists(conn: &Connection, name: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
        [name],
        |r| r.get(0),
    )?)
}

fn has_user_tables(conn: &Connection) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%')",
        [],
        |r| r.get(0),
    )?)
}

fn store_label(kind: StoreKind, db_path: Option<&Path>) -> String {
    match db_path {
        Some(p) => format!("{} store '{}'", kind.as_str().to_uppercase(), p.display()),
        None => format!("{} store (in-memory)", kind.as_str().to_uppercase()),
    }
}

/// Snapshot the store with `VACUUM INTO` next to it:
/// `<file>.bak-<tag>-<unix-secs>[-n]`. Returns the backup path.
///
/// The destination is first created **empty with owner-only permissions**
/// (`create_new`, mode 0600 on unix) and only then filled — SQLite accepts an
/// existing empty file as a `VACUUM INTO` target — so the copy is never
/// readable under the process umask, not even briefly (P2R-9). `create_new`
/// also makes the name reservation race-free.
pub fn backup_store(conn: &Connection, db_path: &Path, tag: &str) -> Result<PathBuf> {
    let target = reserve_private_file(db_path, tag)?;
    let target_str = target
        .to_str()
        .context("backup path is not valid UTF-8")?
        .to_string();
    if let Err(e) = conn.execute("VACUUM INTO ?1", [&target_str]) {
        let _ = std::fs::remove_file(&target);
        return Err(e.into());
    }
    Ok(target)
}

/// Atomically create a new, empty, owner-only file named
/// `<db_path>.bak-<tag>-<unix-secs>[-n]` (the first free name) and return it.
pub(crate) fn reserve_private_file(db_path: &Path, tag: &str) -> Result<PathBuf> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let base = format!("{}.bak-{tag}-{secs}", db_path.display());
    for n in 0..1000 {
        let candidate = if n == 0 {
            PathBuf::from(&base)
        } else {
            PathBuf::from(format!("{base}-{n}"))
        };
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        match opts.open(&candidate) {
            Ok(_) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("creating backup file {}", candidate.display()));
            }
        }
    }
    bail!("no free backup file name for {base}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::database::init_db;

    fn ident(model: &str, dim: usize) -> EmbeddingIdentity {
        EmbeddingIdentity {
            provider: "test".into(),
            model: model.into(),
            dim,
        }
    }

    fn version(conn: &Connection) -> i64 {
        conn.query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap()
    }

    fn file_store(dir: &tempfile::TempDir, name: &str) -> (Connection, PathBuf) {
        let path = dir.path().join(name);
        (init_db(Some(&path)).unwrap(), path)
    }

    fn backups(dir: &tempfile::TempDir) -> Vec<String> {
        std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".bak-"))
            .collect()
    }

    fn insert_node(conn: &Connection, tenant: &str, fact: &str, emb: &[f32]) {
        conn.execute(
            "INSERT INTO nodes (tenant_id, payload) VALUES (?1, json_object('fact', ?2))",
            params![tenant, fact],
        )
        .unwrap();
        let id = conn.last_insert_rowid();
        let bytes: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();
        conn.execute(
            "INSERT INTO vec_nodes (node_id, embedding) VALUES (?1, ?2)",
            params![id, bytes],
        )
        .unwrap();
    }

    /// A fresh store is created at the latest version with full metadata, and
    /// no backup (nothing to protect).
    #[test]
    fn fresh_store_gets_latest_version_and_meta() {
        let dir = tempfile::tempdir().unwrap();
        let (conn, path) = file_store(&dir, "stm.sqlite");
        let report = prepare_store(&conn, StoreKind::Stm, Some(&path), &ident("m", 4)).unwrap();

        assert_eq!(
            (report.from_version, report.to_version),
            (0, STM_SCHEMA_VERSION)
        );
        assert_eq!(report.backup, None);
        assert_eq!(version(&conn), STM_SCHEMA_VERSION);
        assert_eq!(
            get_meta(&conn, "store_kind").unwrap().as_deref(),
            Some("stm")
        );
        assert_eq!(
            get_meta(&conn, "schema_version").unwrap().as_deref(),
            Some("2")
        );
        assert_eq!(
            get_meta(&conn, "embedding_model").unwrap().as_deref(),
            Some("m")
        );
        assert_eq!(
            get_meta(&conn, "embedding_dim").unwrap().as_deref(),
            Some("4")
        );
        assert_eq!(
            get_meta(&conn, "embedding_provider").unwrap().as_deref(),
            Some("test")
        );
        assert!(get_meta(&conn, "created_at").unwrap().is_some());
        assert_eq!(vec_table_dim(&conn, "vec_nodes").unwrap(), Some(4));
        assert!(backups(&dir).is_empty());

        // Re-opening is a no-op.
        let again = prepare_store(&conn, StoreKind::Stm, Some(&path), &ident("m", 4)).unwrap();
        assert_eq!((again.from_version, again.to_version), (2, 2));
    }

    /// A legacy (v0) STM store with several tenants is backed up, migrated to
    /// the single workspace tenant (with a warning), and its CCL duplicates
    /// collapse onto one definition per name.
    #[test]
    fn legacy_store_is_backed_up_and_tenants_merged() {
        let dir = tempfile::tempdir().unwrap();
        let (conn, path) = file_store(&dir, "stm.sqlite");
        init_schema(&conn, 4).unwrap(); // legacy: schema, no meta, user_version 0
        insert_node(&conn, "legacy", "a", &[1.0, 0.0, 0.0, 0.0]);
        insert_node(&conn, "t1", "b", &[0.0, 1.0, 0.0, 0.0]);
        conn.execute(
            "INSERT INTO episodes (tenant_id, session_id, raw_dialogue) VALUES ('t1', 's', 'x')",
            [],
        )
        .unwrap();
        for t in ["legacy", "t1"] {
            conn.execute(
                "INSERT INTO ccl_registry (tenant_id, name, description) VALUES (?1, 'dream', 'd')",
                [t],
            )
            .unwrap();
        }

        let report = prepare_store(&conn, StoreKind::Stm, Some(&path), &ident("m", 4)).unwrap();

        assert_eq!((report.from_version, report.to_version), (0, 2));
        let backup = report.backup.expect("existing store is backed up");
        assert!(backup.exists());
        // The backup is the pre-migration store: still two tenants, no meta.
        let old = Connection::open(&backup).unwrap();
        let old_tenants: i64 = old
            .query_row("SELECT COUNT(DISTINCT tenant_id) FROM nodes", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(old_tenants, 2);

        let tenants: Vec<String> = conn
            .prepare("SELECT DISTINCT tenant_id FROM nodes UNION SELECT tenant_id FROM episodes UNION SELECT tenant_id FROM ccl_registry")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(tenants, vec![WORKSPACE_TENANT.to_string()]);
        let ccl: i64 = conn
            .query_row("SELECT COUNT(*) FROM ccl_registry", [], |r| r.get(0))
            .unwrap();
        assert_eq!(ccl, 1);
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.contains("legacy") && w.contains("t1")),
            "{:?}",
            report.warnings
        );
        // The legacy vectors are adopted with a note.
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.contains("predates embedding metadata"))
        );
    }

    /// A single-tenant legacy store migrates without a merge warning.
    #[test]
    fn single_tenant_migration_does_not_warn_about_merging() {
        let conn = init_db(None as Option<&String>).unwrap();
        init_schema(&conn, 4).unwrap();
        insert_node(&conn, "legacy", "a", &[1.0, 0.0, 0.0, 0.0]);
        let report = prepare_store(&conn, StoreKind::Stm, None, &ident("m", 4)).unwrap();
        assert!(!report.warnings.iter().any(|w| w.contains("tenants")));
        let t: String = conn
            .query_row("SELECT tenant_id FROM nodes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(t, WORKSPACE_TENANT);
    }

    /// P2R-9: the backup destination exists as an empty 0600 file *before*
    /// any data is written into it, and backups get distinct names.
    #[cfg(unix)]
    #[test]
    fn backup_file_is_private_before_it_is_filled() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("stm.sqlite");

        let reserved = reserve_private_file(&db, "t").unwrap();
        let meta = std::fs::metadata(&reserved).unwrap();
        assert_eq!(meta.len(), 0, "reserved empty");
        assert_eq!(meta.permissions().mode() & 0o777, 0o600);
        let second = reserve_private_file(&db, "t").unwrap();
        assert_ne!(reserved, second, "a taken name is never reused");

        // The full backup fills a reserved file and keeps it private.
        let (conn, path) = file_store(&dir, "live.sqlite");
        prepare_store(&conn, StoreKind::Stm, Some(&path), &ident("m", 4)).unwrap();
        let backup = backup_store(&conn, &path, "manual").unwrap();
        let meta = std::fs::metadata(&backup).unwrap();
        assert!(meta.len() > 0);
        assert_eq!(meta.permissions().mode() & 0o777, 0o600);
        let copy = Connection::open(&backup).unwrap();
        assert_eq!(
            get_meta(&copy, "store_kind").unwrap().as_deref(),
            Some("stm")
        );
    }

    /// A store written by a newer binary is refused, untouched.
    #[test]
    fn newer_store_is_refused() {
        let conn = init_db(None as Option<&String>).unwrap();
        prepare_store(&conn, StoreKind::Ltm, None, &ident("m", 4)).unwrap();
        conn.pragma_update(None, "user_version", LTM_SCHEMA_VERSION + 1)
            .unwrap();

        let err = prepare_store(&conn, StoreKind::Ltm, None, &ident("m", 4))
            .unwrap_err()
            .to_string();
        assert!(err.contains("newer version"), "{err}");
    }

    /// A different embedding model or dimension is refused with a pointer at
    /// `neurolithe reembed`.
    #[test]
    fn embedding_mismatch_is_refused_with_reembed_hint() {
        let conn = init_db(None as Option<&String>).unwrap();
        prepare_store(&conn, StoreKind::Stm, None, &ident("model-a", 4)).unwrap();

        for other in [ident("model-b", 4), ident("model-a", 8)] {
            let err = prepare_store(&conn, StoreKind::Stm, None, &other)
                .unwrap_err()
                .to_string();
            assert!(err.contains("neurolithe reembed"), "{err}");
            assert!(err.contains("model-a"), "{err}");
        }
    }

    /// A legacy store holding vectors of another dimension is refused; an empty
    /// legacy index is simply rebuilt at the embedder's dimension.
    #[test]
    fn legacy_dimension_mismatch() {
        let with_vectors = init_db(None as Option<&String>).unwrap();
        init_schema(&with_vectors, 4).unwrap();
        insert_node(&with_vectors, "legacy", "a", &[1.0, 0.0, 0.0, 0.0]);
        let err = prepare_store(&with_vectors, StoreKind::Stm, None, &ident("m", 8))
            .unwrap_err()
            .to_string();
        assert!(err.contains("reembed") && err.contains("4-dim"), "{err}");

        let empty = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&empty, 768).unwrap();
        prepare_store(&empty, StoreKind::Ltm, None, &ident("m", 384)).unwrap();
        assert_eq!(vec_table_dim(&empty, "vec_ltm").unwrap(), Some(384));
        assert_eq!(vec_table_dim(&empty, "vec_leaves").unwrap(), Some(384));
    }

    /// Opening a store as the wrong kind is refused (both with and without meta).
    #[test]
    fn wrong_store_kind_is_refused() {
        let conn = init_db(None as Option<&String>).unwrap();
        prepare_store(&conn, StoreKind::Stm, None, &ident("m", 4)).unwrap();
        assert!(prepare_store(&conn, StoreKind::Ltm, None, &ident("m", 4)).is_err());

        let legacy_ltm = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&legacy_ltm, 4).unwrap();
        let err = prepare_store(&legacy_ltm, StoreKind::Stm, None, &ident("m", 4))
            .unwrap_err()
            .to_string();
        assert!(err.contains("legacy 'ltm'"), "{err}");
    }
}
