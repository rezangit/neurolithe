use crate::infrastructure::config::AppConfig;
use crate::infrastructure::schema::{init_ltm_schema, init_schema};
use anyhow::Context;
use rusqlite::{Connection, TransactionBehavior};
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
}

/// Open both memory stores from config and apply each store's schema at its own
/// vector dimension. The stores' dimensions are independent, so building one
/// never affects the other. Spine seeding is the LTM repository's job (called
/// by the daemon, slice 11), not the schema's.
///
/// Errors name the store and file so a corrupt or locked DB is diagnosable
/// (QA-15).
pub fn init_stores(config: &AppConfig) -> anyhow::Result<MemoryStores> {
    let stm = open_store("STM", config.stm.path.as_deref(), |c| {
        init_schema(c, config.stm.vector_dimension)
    })?;
    let ltm = open_store("LTM", config.ltm.path.as_deref(), |c| {
        init_ltm_schema(c, config.ltm.vector_dimension)
    })?;
    Ok(MemoryStores { stm, ltm })
}

/// How long [`open_store`] keeps retrying a store that is busy at startup.
const OPEN_RETRY_WINDOW: std::time::Duration = std::time::Duration::from_secs(10);

/// `SQLITE_BUSY` / `SQLITE_LOCKED` (any extended code).
fn is_busy(err: &rusqlite::Error) -> bool {
    matches!(
        err.sqlite_error_code(),
        Some(rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked)
    )
}

/// Open one store and apply its schema, attaching the store name + path to any
/// failure.
///
/// Retries the whole open while the file is busy. `busy_timeout` alone is not
/// enough at startup: when several processes open a *fresh* file at once, the
/// rollback→WAL conversion (and first-time DDL) can deadlock on lock upgrades,
/// and SQLite then returns SQLITE_BUSY immediately *without* invoking the busy
/// handler (QA-7). Opening is idempotent (IF NOT EXISTS DDL), so dropping the
/// connection and trying again with jittered backoff is safe.
fn open_store(
    label: &str,
    path: Option<&str>,
    schema: impl Fn(&Connection) -> rusqlite::Result<()>,
) -> anyhow::Result<Connection> {
    let shown = path.unwrap_or(":memory:");
    let deadline = std::time::Instant::now() + OPEN_RETRY_WINDOW;
    let mut attempt: u64 = 0;
    loop {
        let result =
            init_db(path.as_ref())
                .map_err(|e| (e, "opening"))
                .and_then(|conn| match schema(&conn) {
                    Ok(()) => Ok(conn),
                    Err(e) => Err((e, "initializing schema of")),
                });
        match result {
            Ok(conn) => return Ok(conn),
            Err((e, _)) if is_busy(&e) && std::time::Instant::now() < deadline => {
                attempt += 1;
                // 10–60 ms, varied per process + attempt so contenders desync.
                let jitter = (u64::from(std::process::id()) + attempt * 17) % 50;
                std::thread::sleep(std::time::Duration::from_millis(10 + jitter));
            }
            Err((e, what)) => {
                return Err(anyhow::Error::new(e))
                    .with_context(|| format!("{what} {label} store at '{shown}'"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::config::{
        BusQueryConfig, DecayConfig, FeederConfig, KafkaConfig, LlmConfig, LlmProvider,
        MetricsConfig, PithosConfig, StoreConfig, SweepConfig,
    };

    #[test]
    fn test_init_db_in_memory() {
        let _conn = init_db(None as Option<&String>).expect("Failed to init in-memory db");
    }

    /// A test config pointing the two stores at distinct paths with distinct
    /// vector dimensions.
    fn test_config(stm_path: String, ltm_path: String) -> AppConfig {
        AppConfig {
            llm: LlmConfig {
                provider: LlmProvider::Custom,
                model: "m".into(),
                embedding_model: "e".into(),
                base_url: None,
                embedding_provider: None,
                embedding_base_url: None,
                embedding_project: None,
                embedding_location: None,
                request_timeout_secs: 120,
            },
            stm: StoreConfig {
                vector_dimension: 1536,
                path: Some(stm_path),
            },
            ltm: StoreConfig {
                vector_dimension: 768,
                path: Some(ltm_path),
            },
            kafka: KafkaConfig {
                brokers: "localhost:9092".into(),
                group_id: "neurolithe".into(),
            },
            pithos: PithosConfig {
                base_url: "http://localhost:8080".into(),
                token: String::new(),
            },
            sweep: SweepConfig {
                interval_secs: 86_400,
            },
            metrics: MetricsConfig { interval_secs: 60 },
            feeder: FeederConfig { enabled: true },
            bus_query: BusQueryConfig { enabled: true },
            decay: DecayConfig::default(),
        }
    }

    /// Both stores init into their own files independently, and their vector
    /// dimensions don't interfere: STM's `vec_nodes` is built at the STM
    /// dimension while LTM carries a different one.
    #[test]
    fn test_init_stores_independent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let stm_path = dir.path().join("stm.sqlite");
        let ltm_path = dir.path().join("ltm.sqlite");

        let config = test_config(
            stm_path.to_string_lossy().into_owned(),
            ltm_path.to_string_lossy().into_owned(),
        );
        // Sanity: the two stores carry different dimensions.
        assert_ne!(config.stm.vector_dimension, config.ltm.vector_dimension);

        let stores = init_stores(&config).expect("init_stores should succeed");

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
        let embedding = vec![0.0_f32; config.stm.vector_dimension];
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
        let dir = tempfile::tempdir().expect("tempdir");
        let stm_path = dir.path().join("stm.sqlite");
        // Not a database: SQLite fails on first use.
        std::fs::write(
            &stm_path,
            b"this is definitely not a sqlite file, just junk bytes",
        )
        .unwrap();
        let config = test_config(
            stm_path.to_string_lossy().into_owned(),
            dir.path().join("ltm.sqlite").to_string_lossy().into_owned(),
        );
        let err = match init_stores(&config) {
            Ok(_) => panic!("corrupt store must fail"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("STM"), "error should name the store: {err}");
        assert!(
            err.contains(&*stm_path.to_string_lossy()),
            "error should name the file: {err}"
        );
    }
}
