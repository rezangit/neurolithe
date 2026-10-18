use rusqlite::Connection;

/// Add a column to an existing table only if it is not already present.
/// Idempotent, so it is safe to run on every startup: fresh stores already
/// carry the column from their `CREATE TABLE`, pre-existing stores get it once.
/// `col_def` is the SQL type/constraints (e.g. `"TEXT"`).
fn add_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
    col_def: &str,
) -> rusqlite::Result<()> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let existing: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<_, _>>()?;
    if !existing.iter().any(|c| c == column) {
        conn.execute(
            &format!("ALTER TABLE {table} ADD COLUMN {column} {col_def}"),
            [],
        )?;
    }
    Ok(())
}

/// The STM sqlite-vec index table.
pub const STM_VEC_TABLES: &[&str] = &["vec_nodes"];
/// The LTM sqlite-vec index tables (concepts, document leaves).
pub const LTM_VEC_TABLES: &[&str] = &["vec_ltm", "vec_leaves"];

/// `CREATE VIRTUAL TABLE IF NOT EXISTS <table>` as a vec0 index at `dim`.
fn create_vec_table(conn: &Connection, table: &str, dim: usize) -> rusqlite::Result<()> {
    conn.execute(
        &format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(
                node_id INTEGER PRIMARY KEY,
                embedding float[{dim}]
            )"
        ),
        [],
    )?;
    Ok(())
}

/// Create the STM vector index (`vec_nodes`) at `dim`, if absent.
pub fn create_stm_vec_table(conn: &Connection, dim: usize) -> rusqlite::Result<()> {
    create_vec_table(conn, "vec_nodes", dim)
}

/// Create the LTM vector indexes (`vec_ltm`, `vec_leaves`) at `dim`, if absent.
pub fn create_ltm_vec_tables(conn: &Connection, dim: usize) -> rusqlite::Result<()> {
    for table in LTM_VEC_TABLES {
        create_vec_table(conn, table, dim)?;
    }
    Ok(())
}

/// Drop the given vec0 tables (used to rebuild them at a new dimension).
pub fn drop_vec_tables(conn: &Connection, tables: &[&str]) -> rusqlite::Result<()> {
    for table in tables {
        conn.execute(&format!("DROP TABLE IF EXISTS {table}"), [])?;
    }
    Ok(())
}

/// The dimension a vec0 table was created with (parsed from its DDL), or
/// `None` if the table does not exist.
pub fn vec_table_dim(conn: &Connection, table: &str) -> rusqlite::Result<Option<usize>> {
    use rusqlite::OptionalExtension;
    let sql: Option<String> = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = ?1",
            [table],
            |r| r.get(0),
        )
        .optional()?;
    Ok(sql.and_then(|sql| {
        let start = sql.find("float[")? + "float[".len();
        let len = sql[start..].find(']')?;
        sql[start..start + len].trim().parse().ok()
    }))
}

/// Initialize the database schema for NeuroLithe memory service.
/// This creates the required `episodes`, `nodes`, and `edges` tables if they don't exist.
pub fn init_schema(conn: &Connection, vector_dimension: usize) -> rusqlite::Result<()> {
    // 0. The CCL Registry
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ccl_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            UNIQUE(tenant_id, name)
        )",
        [],
    )?;

    // 1. The Ground-Truth Episodic Logs
    conn.execute(
        "CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            raw_dialogue TEXT NOT NULL,
            ccl TEXT DEFAULT 'reality',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;

    // 2. The Graph Nodes (The Facts)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL,
            source_episode_id INTEGER,
            payload JSON NOT NULL,
            status TEXT DEFAULT 'active',
            ccl TEXT DEFAULT 'reality',
            is_explicit BOOLEAN DEFAULT 0,
            support_count INTEGER DEFAULT 1,
            relevance_score REAL DEFAULT 1.0,
            context_key TEXT,
            last_accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_decayed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(source_episode_id) REFERENCES episodes(id)
        )",
        [],
    )?;

    // 2b. Working-memory migration: pre-existing stores need the new columns
    // added (the CREATE above only applies to fresh DBs). Idempotent — guarded
    // on `pragma_table_info`. `context_key` backs the recency read (slice 3);
    // `last_decayed_at` backs *real-elapsed* decay (a note decays by its true
    // age since it was last decayed/read, not a fixed pass — so a sweep or
    // restart can't wipe fresh working notes). SQLite forbids a non-constant
    // default on ALTER, so the migrated column is nullable and the sweep
    // coalesces NULL → `last_accessed_at`.
    add_column_if_missing(conn, "nodes", "context_key", "TEXT")?;
    add_column_if_missing(conn, "nodes", "last_decayed_at", "DATETIME")?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_nodes_context_recency
             ON nodes(tenant_id, ccl, context_key, last_accessed_at DESC)",
        [],
    )?;

    // 3. The Graph Edges (The Relationships with Temporal Bounds)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS edges (
            source_id INTEGER,
            target_id INTEGER,
            relation TEXT NOT NULL,
            ccl TEXT DEFAULT 'reality',
            valid_from DATETIME,
            valid_until DATETIME,
            weight REAL DEFAULT 1.0,
            FOREIGN KEY(source_id) REFERENCES nodes(id),
            FOREIGN KEY(target_id) REFERENCES nodes(id)
        )",
        [],
    )?;
    // 4. The Vector Index (Semantic Search via sqlite-vec)
    // We use vec0 to store our configurable dimensional embeddings
    create_stm_vec_table(conn, vector_dimension)?;

    // Only active, meaningfully-embedded nodes belong in the KNN index (ARC-17).
    // Stores written before that rule may still hold vectors for archived nodes
    // and zero-vector graph anchors (`kind = subject`); they occupy top-k slots
    // and push real matches out. Purge them — idempotent, a no-op once clean.
    conn.execute(
        "DELETE FROM vec_nodes WHERE node_id IN (
            SELECT id FROM nodes
            WHERE status != 'active' OR json_extract(payload, '$.kind') = 'subject'
        )",
        [],
    )?;

    // 5. The FTS5 Index (Full-Text Keyword Search)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_nodes USING fts5(
            payload,
            content='nodes',
            content_rowid='id'
        )",
        [],
    )?;

    // FTS sync triggers
    conn.execute_batch(
        "
        CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
            INSERT INTO fts_nodes(rowid, payload) VALUES (new.id, new.payload);
        END;
        CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
            INSERT INTO fts_nodes(fts_nodes, rowid, payload) VALUES ('delete', old.id, old.payload);
        END;
        CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
            INSERT INTO fts_nodes(fts_nodes, rowid, payload) VALUES ('delete', old.id, old.payload);
            INSERT INTO fts_nodes(rowid, payload) VALUES (new.id, new.payload);
        END;
        ",
    )?;

    // 6. Idempotency ledger for `memory.command` writes. A `commandId` is
    // recorded after a remember/forget succeeds; a duplicate delivery is
    // skipped. Old rows are swept periodically (the ids only guard against
    // at-least-once redelivery, which is bounded in time).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS processed_commands (
            command_id TEXT PRIMARY KEY,
            processed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;

    Ok(())
}

/// Indexes for the STM graph: `edges` is read by endpoint on every recall
/// (1-hop expansion, a hit's connections, a turn's `about` subjects) and on
/// every node delete (foreign-key checks). Idempotent; STM schema v3.
pub fn create_stm_edge_indexes(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id, relation);
         CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);",
    )
}

/// The STM node expression index for `dataId` lookups (trace, forget by
/// dataId, working subjects, ingest upserts). Partial, so facts without a
/// `dataId` cost nothing. A query uses it when it compares exactly
/// `json_extract(payload, '$.dataId') = ?`. Building it evaluates
/// `json_extract` on every row, so it fails if any payload is not valid JSON.
/// Idempotent; STM schema v3.
pub fn create_stm_data_id_index(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_nodes_data_id ON nodes(json_extract(payload, '$.dataId'))
             WHERE json_extract(payload, '$.dataId') IS NOT NULL",
        [],
    )?;
    Ok(())
}

/// Initialize the Long-Term Memory schema (the permanent knowledge tree) in the
/// LTM database. Creates `tree_nodes`, `tree_edges` (multi-parent DAG),
/// `leaves` (`dataId` references), the `vec_ltm` sqlite-vec index at the LTM
/// dimension, `fts_ltm` over node summaries with sync triggers, and the
/// full-text document bodies ([`create_ltm_doc_bodies`]). Idempotent.
pub fn init_ltm_schema(conn: &Connection, vector_dimension: usize) -> rusqlite::Result<()> {
    // 0. Concept / leaf nodes. `permanent` is always true — LTM never decays.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tree_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL DEFAULT 'grown',
            permanent BOOLEAN NOT NULL DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;

    // 1. Parent -> child edges. The composite PK allows a node to sit under
    //    many parents (DAG) while preventing duplicate same-parent edges.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tree_edges (
            parent_id INTEGER NOT NULL,
            child_id INTEGER NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (parent_id, child_id),
            FOREIGN KEY (parent_id) REFERENCES tree_nodes(id),
            FOREIGN KEY (child_id) REFERENCES tree_nodes(id)
        )",
        [],
    )?;
    // The (parent_id, child_id) PK indexes child-lookups by parent (get_children).
    // Add the reverse index for parent-lookups by child (get_parents / ancestor
    // walks in LTM retrieval).
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tree_edges_child ON tree_edges(child_id)",
        [],
    )?;

    // 2. Document leaves — a tree node points at a dataId (the source document).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS leaves (
            tree_node_id INTEGER NOT NULL,
            data_id TEXT NOT NULL,
            provenance TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (tree_node_id, data_id),
            FOREIGN KEY (tree_node_id) REFERENCES tree_nodes(id)
        )",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_leaves_data_id ON leaves(data_id)",
        [],
    )?;

    // 3. Vector index over CONCEPT node embeddings (sqlite-vec, LTM dimension).
    // Concepts and leaves live in separate indexes: placement searches concepts
    // only, so a document's embedding can never crowd out a concept match (the
    // vec0 `k` cut is applied before any kind filter — a shared index would let
    // a near leaf displace the real concept and misroute the doc to the inbox).
    // 3b. `vec_leaves`: vector index over DOCUMENT LEAF embeddings — so recall
    // can land on a document directly by meaning (while it sits in the inbox).
    create_ltm_vec_tables(conn, vector_dimension)?;

    // 4. FTS5 over node summaries (content table = tree_nodes).
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_ltm USING fts5(
            summary,
            content='tree_nodes',
            content_rowid='id'
        )",
        [],
    )?;

    conn.execute_batch(
        "
        CREATE TRIGGER IF NOT EXISTS tree_nodes_ai AFTER INSERT ON tree_nodes BEGIN
            INSERT INTO fts_ltm(rowid, summary) VALUES (new.id, new.summary);
        END;
        CREATE TRIGGER IF NOT EXISTS tree_nodes_ad AFTER DELETE ON tree_nodes BEGIN
            INSERT INTO fts_ltm(fts_ltm, rowid, summary) VALUES ('delete', old.id, old.summary);
        END;
        CREATE TRIGGER IF NOT EXISTS tree_nodes_au AFTER UPDATE ON tree_nodes BEGIN
            INSERT INTO fts_ltm(fts_ltm, rowid, summary) VALUES ('delete', old.id, old.summary);
            INSERT INTO fts_ltm(rowid, summary) VALUES (new.id, new.summary);
        END;
        ",
    )?;

    create_ltm_doc_bodies(conn)?;
    Ok(())
}

/// LTM v2: each document's **full** text, keyed by `data_id`, with a
/// full-text index over it. A leaf's summary and embedding only see the
/// start of a long document (without a chat model, its first ~1,000
/// characters); recall's keyword leg searches this instead, so evidence deep
/// in a document is still found.
///
/// - `doc_bodies(id, data_id, title, body, backfilled)`: one row per document
///   (an upsert on re-filing, keyed by the unique `data_id`). Written in the
///   same transaction as the leaf, so a filed leaf always has the matching
///   text. `backfilled = 1` marks the stand-in text (title + summary) the v2
///   migration gave a document filed before bodies were kept; any real
///   write clears it.
/// - `fts_doc_bodies(title, body)`: FTS5 over it (external content keyed by
///   the declared `id`, which a VACUUM never renumbers), kept in sync by
///   triggers.
/// - Deleting the **last** leaf that carries a `data_id` drops its body. A
///   replacement is filed before the old copy goes, so an upsert keeps it.
///
/// Idempotent.
pub fn create_ltm_doc_bodies(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS doc_bodies (
            id INTEGER PRIMARY KEY,
            data_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL,
            backfilled INTEGER NOT NULL DEFAULT 0
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_doc_bodies USING fts5(
            title,
            body,
            content='doc_bodies',
            content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS doc_bodies_ai AFTER INSERT ON doc_bodies BEGIN
            INSERT INTO fts_doc_bodies(rowid, title, body)
            VALUES (new.id, new.title, new.body);
        END;
        CREATE TRIGGER IF NOT EXISTS doc_bodies_ad AFTER DELETE ON doc_bodies BEGIN
            INSERT INTO fts_doc_bodies(fts_doc_bodies, rowid, title, body)
            VALUES ('delete', old.id, old.title, old.body);
        END;
        CREATE TRIGGER IF NOT EXISTS doc_bodies_au AFTER UPDATE ON doc_bodies BEGIN
            INSERT INTO fts_doc_bodies(fts_doc_bodies, rowid, title, body)
            VALUES ('delete', old.id, old.title, old.body);
            INSERT INTO fts_doc_bodies(rowid, title, body)
            VALUES (new.id, new.title, new.body);
        END;
        CREATE TRIGGER IF NOT EXISTS leaves_ad_doc_body AFTER DELETE ON leaves
        WHEN NOT EXISTS (SELECT 1 FROM leaves WHERE data_id = old.data_id) BEGIN
            DELETE FROM doc_bodies WHERE data_id = old.data_id;
        END;
        ",
    )?;
    // A store made before the flag existed: its rows are real text (0).
    add_column_if_missing(
        conn,
        "doc_bodies",
        "backfilled",
        "INTEGER NOT NULL DEFAULT 0",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::database::init_db;

    /// W3R-6: `doc_bodies` rows have a declared `INTEGER PRIMARY KEY`, so
    /// the FTS index's rowids survive a VACUUM (an implicit rowid may be
    /// renumbered, which would point the index at the wrong documents).
    #[test]
    fn test_doc_bodies_index_survives_vacuum() {
        let dir = tempfile::tempdir().unwrap();
        let conn = init_db(Some(&dir.path().join("ltm.sqlite"))).unwrap();
        init_ltm_schema(&conn, 4).unwrap();
        for (i, (id, body)) in [
            ("a", "alpha apple"),
            ("b", "bravo banana"),
            ("c", "charlie cherry"),
        ]
        .iter()
        .enumerate()
        {
            conn.execute(
                "INSERT INTO tree_nodes (id, name, kind) VALUES (?1, 'n', 'leaf')",
                rusqlite::params![i as i64 + 1],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO leaves (tree_node_id, data_id) VALUES (?1, ?2)",
                rusqlite::params![i as i64 + 1, id],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO doc_bodies (data_id, title, body) VALUES (?1, '', ?2)",
                rusqlite::params![id, body],
            )
            .unwrap();
        }
        conn.execute("DELETE FROM leaves WHERE data_id = 'b'", [])
            .unwrap();
        conn.execute_batch("VACUUM").unwrap();
        let found: Vec<String> = conn
            .prepare(
                "SELECT b.data_id FROM fts_doc_bodies f JOIN doc_bodies b ON b.id = f.rowid
                 WHERE fts_doc_bodies MATCH 'cherry'",
            )
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(found, ["c"]);
    }

    /// E2: LTM keeps each document's full body in `doc_bodies`, full-text
    /// indexed by `fts_doc_bodies`. An upsert re-indexes it; the body goes
    /// once the last leaf carrying its `data_id` is deleted (not before, so
    /// a replacement filed before the old copy is dropped keeps it).
    #[test]
    fn test_doc_bodies_fts_sync_and_cleanup() {
        let conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&conn, 4).unwrap();
        let hits = |q: &str| -> Vec<String> {
            let mut stmt = conn
                .prepare(
                    "SELECT b.data_id FROM fts_doc_bodies f JOIN doc_bodies b ON b.id = f.rowid
                     WHERE fts_doc_bodies MATCH ?1 ORDER BY b.data_id",
                )
                .unwrap();
            stmt.query_map([q], |r| r.get(0))
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap()
        };
        let leaf = |id: i64, data_id: &str| {
            conn.execute(
                "INSERT INTO tree_nodes (id, name, kind) VALUES (?1, 'n', 'leaf')",
                rusqlite::params![id],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO leaves (tree_node_id, data_id) VALUES (?1, ?2)",
                rusqlite::params![id, data_id],
            )
            .unwrap();
        };
        let put = |data_id: &str, body: &str| {
            conn.execute(
                "INSERT INTO doc_bodies (data_id, title, body) VALUES (?1, 't', ?2)
                 ON CONFLICT(data_id) DO UPDATE SET title = excluded.title, body = excluded.body",
                rusqlite::params![data_id, body],
            )
            .unwrap();
        };

        leaf(1, "doc_a");
        put("doc_a", "a long letter about the harbour lighthouse");
        assert_eq!(hits("lighthouse"), ["doc_a"]);

        put("doc_a", "rewritten: now about the old windmill");
        assert!(hits("lighthouse").is_empty(), "upsert re-indexes");
        assert_eq!(hits("windmill"), ["doc_a"]);

        // A replacement leaf for the same document, then the old one goes:
        // the body stays while any leaf still carries the data_id.
        leaf(2, "doc_a");
        conn.execute("DELETE FROM leaves WHERE tree_node_id = 1", [])
            .unwrap();
        assert_eq!(hits("windmill"), ["doc_a"]);
        conn.execute("DELETE FROM leaves WHERE tree_node_id = 2", [])
            .unwrap();
        assert!(hits("windmill").is_empty(), "last leaf gone, body gone");
        let left: i64 = conn
            .query_row("SELECT COUNT(*) FROM doc_bodies", [], |r| r.get(0))
            .unwrap();
        assert_eq!(left, 0);
    }

    #[test]
    fn test_init_schema() {
        let conn = init_db(None as Option<&String>).expect("Failed to open DB");
        init_schema(&conn, 1536).expect("Failed to initialize schema");

        // Verify tables exist
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap();

        let table_names: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();

        assert!(table_names.contains(&"episodes".to_string()));
        assert!(table_names.contains(&"nodes".to_string()));
        assert!(table_names.contains(&"edges".to_string()));
        assert!(table_names.contains(&"ccl_registry".to_string()));
        assert!(table_names.contains(&"vec_nodes".to_string()));
        assert!(table_names.contains(&"fts_nodes".to_string()));
    }

    /// The `context_key` column must exist after init and the whole schema must
    /// be safe to run twice (idempotent migration).
    #[test]
    fn test_context_key_column_present_and_idempotent() {
        let conn = init_db(None as Option<&String>).expect("Failed to open DB");
        init_schema(&conn, 1536).expect("first init");
        // Running the full schema again must not error (ALTER guarded).
        init_schema(&conn, 1536).expect("second init should be idempotent");

        let cols: Vec<String> = conn
            .prepare("PRAGMA table_info(nodes)")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(
            cols.iter().any(|c| c == "context_key"),
            "nodes must carry context_key"
        );
    }

    /// A store created *before* the working-memory feature (nodes table with no
    /// `context_key`) must be migrated in place by `init_schema` — the column is
    /// added without dropping data.
    #[test]
    fn test_migrates_preexisting_store() {
        let conn = init_db(None as Option<&String>).expect("Failed to open DB");
        // Simulate an old store: a `nodes` table lacking `context_key`.
        conn.execute(
            "CREATE TABLE nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                source_episode_id INTEGER,
                payload JSON NOT NULL,
                status TEXT DEFAULT 'active',
                ccl TEXT DEFAULT 'reality',
                is_explicit BOOLEAN DEFAULT 0,
                support_count INTEGER DEFAULT 1,
                relevance_score REAL DEFAULT 1.0,
                last_accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO nodes (tenant_id, payload) VALUES ('t', '{}')",
            [],
        )
        .unwrap();

        // Running the current schema migrates the existing table.
        init_schema(&conn, 1536).expect("migration on pre-existing store");

        let cols: Vec<String> = conn
            .prepare("PRAGMA table_info(nodes)")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(cols.iter().any(|c| c == "context_key"));

        // Pre-existing row survives and reads back NULL for the new column.
        let ck: Option<String> = conn
            .query_row(
                "SELECT context_key FROM nodes WHERE tenant_id = 't'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(ck, None);
    }
}
