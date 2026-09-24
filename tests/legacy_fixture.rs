//! Legacy (v0.2.x, pre-workspace, pre-meta) store fixture in
//! `tests/fixtures/legacy-v0.2/`: `stm.sqlite`, `ltm.sqlite`, `manifest.json`.
//!
//! The fixture is a realistic input for the Phase 2 import/migration tests. It
//! was produced by the real v0.2.x binary (branch `master` before Phase 2),
//! driven over MCP with the same deterministic fake LLM/embedder the tests use
//! (64-dim, model "fake-embed"). It holds two tenants ("jarvis", the old
//! default, and "work"), explicit and extracted facts, entity nodes + edges,
//! episodes, and the old seeded LTM spine.
//!
//! Regenerate (only needed if the fixture must change):
//! ```sh
//! rm -rf target/legacy-src && mkdir -p target/legacy-src
//! git archive <v0.2.x ref> | tar -x -C target/legacy-src
//! NL_TARGET=qa-legacy scripts/cargo.sh build --manifest-path target/legacy-src/Cargo.toml
//! NL_TARGET=qa scripts/cargo.sh test --test legacy_fixture -- --ignored generate_legacy_fixture
//! ```
mod common;

use common::{API_KEY, DIM, FakeLlm, McpProcess, write_legacy_config};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/legacy-v0.2")
}

/// Explicit facts stored per tenant (`None` = the v0.2.x default tenant).
const EXPLICIT: &[(Option<&str>, &str)] = &[
    (None, "Alice works at Acme"),
    (None, "The vault code is 4711"),
    (None, "Bob likes green tea"),
    (Some("work"), "Quarterly report is due Friday"),
];

/// Dialogue turns (tenant, session, message); `[rel:X]` makes an edge to X.
const DIALOGUE: &[(Option<&str>, &str, &str)] = &[
    (None, "s1", "I moved to Oslo last spring [rel:Oslo]"),
    (
        Some("work"),
        "w1",
        "I have a meeting with Carol [rel:Carol]",
    ),
];

#[test]
#[ignore = "regenerates the committed fixture; needs the v0.2.x binary in target/qa-legacy"]
fn generate_legacy_fixture() {
    let legacy_bin =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("target/qa-legacy/debug/neurolithe");
    assert!(
        legacy_bin.exists(),
        "build the v0.2.x binary first (see module docs): {}",
        legacy_bin.display()
    );

    let llm = FakeLlm::start();
    let dir = tempfile::tempdir().unwrap();
    write_legacy_config(dir.path(), &llm.base_url());
    let mut server = McpProcess::spawn_with(
        &legacy_bin,
        &["mcp"],
        dir.path(),
        dir.path(),
        &[("OPENAI_API_KEY", API_KEY), ("NEUROLITHE_API_KEY", API_KEY)],
    );
    let init = server.initialize();
    let version = init["serverInfo"]["version"]
        .as_str()
        .unwrap_or("?")
        .to_string();

    for (tenant, fact) in EXPLICIT {
        let mut args = json!({"fact_text": fact, "tags": ["fixture"]});
        if let Some(t) = tenant {
            args["tenant_id"] = json!(t);
        }
        let res = server.call_tool("store_memory", args);
        assert!(!res.is_error, "store_memory: {}", res.text);
    }
    for (tenant, session, msg) in DIALOGUE {
        let mut args = json!({"session_id": session, "new_message": msg});
        if let Some(t) = tenant {
            args["tenant_id"] = json!(t);
        }
        let res = server.call_tool("push_dialogue", args);
        assert!(!res.is_error, "push_dialogue: {}", res.text);
    }

    let mut tenants = serde_json::Map::new();
    for tenant in ["jarvis", "work"] {
        let res = server.call_tool("export_tenant", json!({"tenant_id": tenant}));
        assert!(!res.is_error, "export_tenant: {}", res.text);
        let export: Value = serde_json::from_str(&res.text).unwrap();
        let mut facts: Vec<String> = export["extracted_facts"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|f| f["fact"].as_str().map(str::to_string))
            .collect();
        facts.sort();
        tenants.insert(tenant.to_string(), json!(facts));
    }

    let (code, stderr) = server.shutdown();
    assert_eq!(code, Some(0), "legacy server exit; stderr:\n{stderr}");

    // Fold any WAL back into the main files so each fixture is one file.
    for name in ["stm.sqlite", "ltm.sqlite"] {
        let conn = rusqlite::Connection::open(dir.path().join(name)).unwrap();
        conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))
            .unwrap();
        conn.pragma_update(None, "journal_mode", "DELETE").unwrap();
    }

    let out = fixture_dir();
    std::fs::create_dir_all(&out).unwrap();
    for name in ["stm.sqlite", "ltm.sqlite"] {
        std::fs::copy(dir.path().join(name), out.join(name)).unwrap();
    }
    let edges = count(&out.join("stm.sqlite"), "SELECT count(*) FROM edges");
    let episodes = count(&out.join("stm.sqlite"), "SELECT count(*) FROM episodes");
    let manifest = json!({
        "generated_by": format!("neurolithe v{version} (pre-Phase-2 master)"),
        "embedding_provider": "openai-compatible fake (tests/common)",
        "embedding_model": "fake-embed",
        "embedding_dim": DIM,
        "tenants": tenants,
        "edges": edges,
        "episodes": episodes,
    });
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap() + "\n",
    )
    .unwrap();
}

fn count(db: &Path, sql: &str) -> i64 {
    let conn =
        rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap();
    conn.query_row(sql, [], |r| r.get(0)).unwrap()
}

fn manifest() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("manifest.json")).unwrap())
        .unwrap()
}

/// Guards the committed fixture itself: it must stay a true legacy store.
#[test]
fn legacy_fixture_is_a_pre_phase2_store() {
    let m = manifest();
    for name in ["stm.sqlite", "ltm.sqlite"] {
        let db = fixture_dir().join(name);
        assert_eq!(count(&db, "PRAGMA user_version"), 0, "{name}: user_version");
        assert_eq!(
            count(
                &db,
                "SELECT count(*) FROM sqlite_master WHERE name = 'meta'"
            ),
            0,
            "{name}: legacy stores have no meta table"
        );
        assert!(
            !fixture_dir().join(format!("{name}-wal")).exists(),
            "{name}: stray WAL file"
        );
    }
    let stm = fixture_dir().join("stm.sqlite");
    // Two tenants, including the old "jarvis" default.
    assert_eq!(
        count(&stm, "SELECT count(DISTINCT tenant_id) FROM nodes"),
        2
    );
    assert!(
        count(
            &stm,
            "SELECT count(*) FROM nodes WHERE tenant_id = 'jarvis'"
        ) > 0
    );
    assert_eq!(
        count(&stm, "SELECT count(*) FROM edges"),
        m["edges"].as_i64().unwrap()
    );
    assert!(m["edges"].as_i64().unwrap() >= 2, "fixture needs edges");
    assert!(
        m["episodes"].as_i64().unwrap() >= 2,
        "fixture needs episodes"
    );
    // vec tables were created at the fixture's dimension.
    let conn =
        rusqlite::Connection::open_with_flags(&stm, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap();
    let vec_sql: String = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = 'vec_nodes'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        vec_sql.contains(&format!("float[{DIM}]")),
        "vec_nodes dim: {vec_sql}"
    );
    // The manifest lists the facts the import test must find again.
    let jarvis: Vec<String> = serde_json::from_value(m["tenants"]["jarvis"].clone()).unwrap();
    for (tenant, fact) in EXPLICIT {
        if tenant.is_none() {
            assert!(jarvis.contains(&fact.to_string()), "manifest lacks {fact}");
        }
    }
}
