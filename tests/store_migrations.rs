//! Phase 2 §4: store metadata, schema migrations, the embedding-identity
//! check (refuse to start on a model/dimension mismatch), `neurolithe reembed`,
//! and legacy-store import edge cases.
mod common;

use common::{FakeLlm, Home, McpProcess};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Config for the fake embedder with a given model name. The dimension is
/// not configured: it is probed from the embedder (`dim` must match the fake).
fn config(llm: &FakeLlm, model: &str, dim: usize) -> String {
    assert_eq!(llm.dim, dim, "the fake decides the dimension");
    let url = llm.base_url();
    format!(
        r#"[llm]
provider = "openai"
model = "fake-chat"
base_url = "{url}"
embedding_provider = "openai"
embedding_model = "{model}"
embedding_base_url = "{url}"
"#
    )
}

fn set_config(home: &Home, toml: &str) {
    std::fs::write(home.path().join("neurolithe.toml"), toml).unwrap();
}

fn meta(db: &Path, key: &str) -> Option<String> {
    let conn =
        rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap();
    conn.query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
        .ok()
}

fn files_with_prefix(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix))
        })
        .collect()
}

/// Start `mcp` and store one fact; returns after a clean shutdown.
fn seed_fact(home: &Home, fact: &str) {
    let mut server = home.spawn_mcp(&[], &[]);
    server.initialize();
    let res = server.call_tool("store_memory", json!({"fact_text": fact}));
    assert!(!res.is_error, "{} / {}", res.text, server.stderr());
    let (code, stderr) = server.shutdown();
    assert_eq!(code, Some(0), "{stderr}");
}

/// `mcp` must refuse to start: exits non-zero; returns stderr.
#[track_caller]
fn assert_refuses_to_start(home: &Home) -> String {
    let mut server: McpProcess = home.spawn_mcp(&[], &[]);
    let code = server.wait_exit(Duration::from_secs(30));
    let stderr = server.stderr();
    assert!(
        matches!(code, Some(c) if c != 0),
        "server did not refuse to start (exit {code:?}); stderr:\n{stderr}"
    );
    stderr
}

#[test]
fn fresh_stores_record_their_embedding_identity() {
    let llm = FakeLlm::start();
    let home = Home::with_config(&config(&llm, "fake-embed", 64));
    seed_fact(&home, "hello");
    for (f, kind) in [("stm.sqlite", "stm"), ("ltm.sqlite", "ltm")] {
        let db = home.workspace_dir("default").join(f);
        assert_eq!(
            meta(&db, "embedding_model").as_deref(),
            Some("openai:fake-embed"),
            "{f}"
        );
        assert_eq!(meta(&db, "embedding_dim").as_deref(), Some("64"), "{f}");
        assert_eq!(meta(&db, "store_kind").as_deref(), Some(kind), "{f}");
        assert!(meta(&db, "schema_version").is_some(), "{f}");
        assert!(meta(&db, "created_at").is_some(), "{f}");
    }
}

#[test]
fn refuses_to_start_on_embedding_model_mismatch_and_reembed_fixes_it() {
    let llm = FakeLlm::start();
    let home = Home::with_config(&config(&llm, "fake-embed", 64));
    seed_fact(&home, "Zoe owns a sailboat");

    set_config(&home, &config(&llm, "other-embed", 64));
    let stderr = assert_refuses_to_start(&home);
    assert!(
        stderr.contains("fake-embed"),
        "names the stored model:\n{stderr}"
    );
    assert!(
        stderr.contains("neurolithe reembed"),
        "points at reembed:\n{stderr}"
    );

    let out = home.cli(&["reembed"]);
    out.assert_success();
    assert!(
        out.stdout.contains("Re-embedded workspace"),
        "{}",
        out.stdout
    );
    let backups: Vec<&str> = out
        .stdout
        .lines()
        .filter_map(|l| l.strip_prefix("Backup: "))
        .collect();
    assert!(
        !backups.is_empty(),
        "reembed must back up first:\n{}",
        out.stdout
    );
    for b in &backups {
        assert!(Path::new(b.trim()).exists(), "backup missing: {b}");
    }

    let db = home.workspace_dir("default").join("stm.sqlite");
    assert_eq!(
        meta(&db, "embedding_model").as_deref(),
        Some("openai:other-embed")
    );
    let mut server = home.spawn_mcp(&[], &[]);
    server.initialize();
    let res = server.call_tool("query_memory", json!({"query": "Zoe sailboat"}));
    assert!(!res.is_error, "{}", res.text);
    assert!(
        res.text.contains("Zoe owns a sailboat"),
        "fact lost by reembed: {}",
        res.text
    );
}

#[test]
fn refuses_to_start_on_embedding_dimension_mismatch() {
    let llm64 = FakeLlm::start();
    let home = Home::with_config(&config(&llm64, "fake-embed", 64));
    seed_fact(&home, "dimension test");

    let llm32 = FakeLlm::start_with_dim(32);
    set_config(&home, &config(&llm32, "fake-embed", 32));
    let stderr = assert_refuses_to_start(&home);
    assert!(stderr.contains("64"), "names the stored dim:\n{stderr}");
    assert!(stderr.contains("neurolithe reembed"), "{stderr}");

    // reembed rebuilds at the new dimension; the fact survives.
    home.cli(&["reembed"]).assert_success();
    let db = home.workspace_dir("default").join("stm.sqlite");
    assert_eq!(meta(&db, "embedding_dim").as_deref(), Some("32"));
    let mut server = home.spawn_mcp(&[], &[]);
    server.initialize();
    let res = server.call_tool("query_memory", json!({"query": "dimension test"}));
    assert!(res.text.contains("dimension test"), "{}", res.text);
}

#[test]
fn refuses_a_store_written_by_a_newer_version() {
    let llm = FakeLlm::start();
    let home = Home::with_config(&config(&llm, "fake-embed", 64));
    seed_fact(&home, "from the future");
    let db = home.workspace_dir("default").join("stm.sqlite");
    let before = std::fs::read(&db).unwrap();
    {
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.pragma_update(None, "user_version", 999).unwrap();
    }
    let stderr = assert_refuses_to_start(&home);
    assert!(stderr.contains("newer"), "{stderr}");
    assert!(stderr.contains("999"), "{stderr}");
    // Refusal must not modify or back up the store.
    assert!(files_with_prefix(&home.workspace_dir("default"), "stm.sqlite.bak").is_empty());
    let after = std::fs::read(&db).unwrap();
    assert_eq!(before.len(), after.len());
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/legacy-v0.2")
        .join(name)
}

#[test]
fn legacy_import_backs_up_before_migrating_and_reports_tenant_merge() {
    let llm = FakeLlm::start();
    let home = Home::with_config(&config(&llm, "fake-embed", 64));
    let out = home.cli(&[
        "workspace",
        "import",
        "old",
        "--stm",
        fixture("stm.sqlite").to_str().unwrap(),
        "--ltm",
        fixture("ltm.sqlite").to_str().unwrap(),
    ]);
    out.assert_success();
    let all = format!("{}{}", out.stdout, out.stderr);
    assert!(all.contains("STM store migrated v0"), "{all}");
    assert!(all.contains("LTM store migrated v0"), "{all}");
    // The tenant collapse is reported, naming the legacy tenants.
    assert!(all.contains("2 tenants (jarvis, work)"), "{all}");

    let ws = home.workspace_dir("old");
    let backups = files_with_prefix(&ws, "stm.sqlite.bak-v0-");
    assert_eq!(backups.len(), 1, "pre-migration STM backup: {backups:?}");
    let conn = rusqlite::Connection::open_with_flags(
        &backups[0],
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    let v: i64 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .unwrap();
    assert_eq!(v, 0, "backup must be the pre-migration store");
    let tenants: i64 = conn
        .query_row("SELECT count(DISTINCT tenant_id) FROM nodes", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(tenants, 2, "backup must keep the original tenants");
    assert!(!files_with_prefix(&ws, "ltm.sqlite.bak-v0-").is_empty());
}

#[test]
fn legacy_import_with_another_embedding_dim_points_at_reembed() {
    let llm32 = FakeLlm::start_with_dim(32);
    let home = Home::with_config(&config(&llm32, "fake-embed", 32));
    let out = home.cli(&[
        "workspace",
        "import",
        "old",
        "--stm",
        fixture("stm.sqlite").to_str().unwrap(),
        "--ltm",
        fixture("ltm.sqlite").to_str().unwrap(),
    ]);
    let all = format!("{}{}", out.stdout, out.stderr);
    assert!(
        all.contains("neurolithe reembed --workspace old"),
        "import must point at reembed (exit {:?}):\n{all}",
        out.code
    );
    assert!(home.workspace_dir("old").join("stm.sqlite").exists());

    home.cli(&["--workspace", "old", "reembed"])
        .assert_success();
    let mut server = home.spawn_mcp(&["--workspace", "old"], &[]);
    server.initialize();
    let res = server.call_tool("query_memory", json!({"query": "Alice Acme"}));
    assert!(!res.is_error, "{}", res.text);
    assert!(res.text.contains("Alice works at Acme"), "{}", res.text);
}

#[test]
fn import_refuses_swapped_store_files() {
    let llm = FakeLlm::start();
    let home = Home::with_config(&config(&llm, "fake-embed", 64));
    let out = home.cli(&[
        "workspace",
        "import",
        "swapped",
        "--stm",
        fixture("ltm.sqlite").to_str().unwrap(),
        "--ltm",
        fixture("stm.sqlite").to_str().unwrap(),
    ]);
    assert_eq!(out.code, Some(1), "{out:?}");
    assert!(out.stderr.contains("Error:"), "{}", out.stderr);
}

/// Sorted (name, bytes) of every file in `dir`.
fn dir_snapshot(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .map(|p| {
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                std::fs::read(&p).unwrap(),
            )
        })
        .collect();
    files.sort();
    files
}

/// P2R-1: a real v0.2 store runs in WAL mode, often with frames not yet
/// checkpointed (e.g. the daemon is still running). Importing it must pick up
/// the WAL-only data and leave the source directory exactly as it was: same
/// files (no new or removed `-wal`/`-shm`), same bytes.
#[test]
fn p2r1_legacy_import_of_a_wal_mode_store_is_complete_and_leaves_source_untouched() {
    let src = tempfile::tempdir().unwrap();
    for f in ["stm.sqlite", "ltm.sqlite"] {
        std::fs::copy(fixture(f), src.path().join(f)).unwrap();
    }
    // Convert the copy to WAL and add a fact that lives only in the WAL. The
    // connection stays open (like a running v0.2 process), so the frame is not
    // checkpointed into the main file.
    let writer = rusqlite::Connection::open(src.path().join("stm.sqlite")).unwrap();
    let mode: String = writer
        .query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))
        .unwrap();
    assert_eq!(mode, "wal");
    writer.pragma_update(None, "wal_autocheckpoint", 0).unwrap();
    writer
        .execute(
            "INSERT INTO nodes (tenant_id, payload, status, ccl, is_explicit, support_count, relevance_score)
             VALUES ('jarvis', '{\"fact\":\"Fact only in the WAL\",\"tags\":[]}', 'active', 'reality', 1, 1, 1.0)",
            [],
        )
        .unwrap();
    let wal = src.path().join("stm.sqlite-wal");
    assert!(
        std::fs::metadata(&wal).map(|m| m.len()).unwrap_or(0) > 0,
        "precondition: the new fact must be in an uncheckpointed WAL"
    );
    let before = dir_snapshot(src.path());

    let llm = FakeLlm::start();
    let home = Home::with_config(&config(&llm, "fake-embed", 64));
    let out = home.cli(&[
        "workspace",
        "import",
        "walstore",
        "--stm",
        src.path().join("stm.sqlite").to_str().unwrap(),
        "--ltm",
        src.path().join("ltm.sqlite").to_str().unwrap(),
    ]);
    out.assert_success();

    // Source untouched: same file list, same bytes. (`-shm` is SQLite's shared
    // index; a reader may update its read marks, so only its presence counts.)
    let after = dir_snapshot(src.path());
    let names = |s: &[(String, Vec<u8>)]| s.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(
        names(&after),
        names(&before),
        "P2R-1: source file list changed"
    );
    for ((name, b), (_, a)) in before.iter().zip(after.iter()) {
        if !name.ends_with("-shm") {
            assert!(a == b, "P2R-1: source file {name} was modified");
        }
    }
    drop(writer);

    // Everything arrived, including the WAL-only fact.
    let mut server = home.spawn_mcp(&["--workspace", "walstore"], &[]);
    server.initialize();
    let export = server.call_tool("workspace_export", json!({}));
    assert!(!export.is_error, "{}", export.text);
    let facts: Vec<String> = export.json()["stm_facts"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|f| f["payload"]["fact"].as_str().map(str::to_string))
        .collect();
    assert!(
        facts.contains(&"Fact only in the WAL".to_string()),
        "P2R-1: WAL-only data lost on import: {facts:?}"
    );
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("manifest.json")).unwrap()).unwrap();
    for tenant_facts in manifest["tenants"].as_object().unwrap().values() {
        for fact in tenant_facts.as_array().unwrap() {
            let fact = fact.as_str().unwrap().to_string();
            assert!(facts.contains(&fact), "P2R-1: {fact:?} missing");
        }
    }
}

/// P2R-1: the common case: a WAL-mode v0.2 store that was closed cleanly (no
/// `-wal`/`-shm` next to it). Reading it for the import must not create
/// sidecar files in the source directory, or change any byte.
#[test]
fn p2r1_legacy_import_of_a_closed_wal_mode_store_creates_no_sidecars() {
    let src = tempfile::tempdir().unwrap();
    for f in ["stm.sqlite", "ltm.sqlite"] {
        std::fs::copy(fixture(f), src.path().join(f)).unwrap();
        let conn = rusqlite::Connection::open(src.path().join(f)).unwrap();
        let mode: String = conn
            .query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))
            .unwrap();
        assert_eq!(mode, "wal");
        // Closing the last connection checkpoints and removes the sidecars.
    }
    let before = dir_snapshot(src.path());
    assert_eq!(
        before.len(),
        2,
        "precondition: only the two store files: {:?}",
        before.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );

    let llm = FakeLlm::start();
    let home = Home::with_config(&config(&llm, "fake-embed", 64));
    let out = home.cli(&[
        "workspace",
        "import",
        "closedwal",
        "--stm",
        src.path().join("stm.sqlite").to_str().unwrap(),
        "--ltm",
        src.path().join("ltm.sqlite").to_str().unwrap(),
    ]);
    out.assert_success();
    let after = dir_snapshot(src.path());
    assert_eq!(
        after.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        before.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        "P2R-1: import left files behind in the source directory"
    );
    assert!(after == before, "P2R-1: source bytes changed");
    assert_eq!(
        home.cli(&["workspace", "export", "closedwal"]).json()["stm_facts"]
            .as_array()
            .map(|a| a.len()),
        Some(8),
        "all 8 legacy facts imported"
    );
}

/// P2R-2: a legacy store has no recorded embedding identity. Import adopts the
/// configured one; it must say so and tell the user how to fix it if wrong.
#[test]
fn p2r2_legacy_import_reports_assumed_embedding_and_reembed_hint() {
    let llm = FakeLlm::start();
    let home = Home::with_config(&config(&llm, "fake-embed", 64));
    let out = home.cli(&[
        "workspace",
        "import",
        "old",
        "--stm",
        fixture("stm.sqlite").to_str().unwrap(),
        "--ltm",
        fixture("ltm.sqlite").to_str().unwrap(),
    ]);
    out.assert_success();
    let all = format!("{}{}", out.stdout, out.stderr);
    assert!(
        all.contains("predates embedding metadata"),
        "P2R-2: no note about the assumed embedding identity:\n{all}"
    );
    assert!(
        all.contains("fake-embed"),
        "P2R-2: note must name the assumed model:\n{all}"
    );
    assert!(
        all.contains("neurolithe reembed"),
        "P2R-2: no reembed hint:\n{all}"
    );
}

fn copy_fixture_to(dir: &Path) {
    for f in ["stm.sqlite", "ltm.sqlite"] {
        std::fs::copy(fixture(f), dir.join(f)).unwrap();
    }
}

fn import(home: &Home, name: &str, src: &Path) -> common::CliOutput {
    home.cli(&[
        "workspace",
        "import",
        name,
        "--stm",
        src.join("stm.sqlite").to_str().unwrap(),
        "--ltm",
        src.join("ltm.sqlite").to_str().unwrap(),
    ])
}

/// N2: import opens with the "sources must not be in use" guidance, also
/// shown in `--help`.
#[test]
fn n2_import_states_the_not_in_use_precondition() {
    let llm = FakeLlm::start();
    let home = Home::with_config(&config(&llm, "fake-embed", 64));
    let src = tempfile::tempdir().unwrap();
    copy_fixture_to(src.path());
    let out = import(&home, "old", src.path());
    out.assert_success();
    let first = out.stdout.lines().next().unwrap_or_default();
    assert!(
        first.starts_with("Note: the source stores must not be in use"),
        "first line: {first:?}"
    );
    let help = home.cli(&["workspace", "import", "--help"]);
    help.assert_success();
    assert!(help.stdout.contains("in use"), "{}", help.stdout);
}

/// N2: a source written to during the copy is refused ("in use") and leaves
/// no workspace behind. Timing-dependent by nature, so a busy writer runs
/// through several attempts: every outcome must be clean (a complete import
/// or a refusal with nothing left), and at least one must be a refusal.
#[test]
fn n2_import_refuses_a_source_that_changes_during_the_copy() {
    let llm = FakeLlm::start();
    let home = Home::with_config(&config(&llm, "fake-embed", 64));
    let src = tempfile::tempdir().unwrap();
    copy_fixture_to(src.path());

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let writer = {
        let (stop, db) = (stop.clone(), src.path().join("stm.sqlite"));
        std::thread::spawn(move || {
            let conn = rusqlite::Connection::open(db).unwrap();
            conn.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))
                .unwrap();
            let mut i = 0u64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = conn.execute(
                    "INSERT INTO episodes (tenant_id, session_id, raw_dialogue, ccl) \
                     VALUES ('jarvis', 'busy', ?1, 'reality')",
                    [format!("turn {i}")],
                );
                i += 1;
            }
        })
    };

    let mut refused = 0;
    for attempt in 0..6 {
        let name = format!("try{attempt}");
        let out = import(&home, &name, src.path());
        if out.code == Some(0) {
            // A complete, consistent import (the copy happened between writes).
            assert!(home.workspace_dir(&name).join("stm.sqlite").exists());
        } else {
            assert_eq!(out.code, Some(1), "{out:?}");
            assert!(out.stderr.contains("is in use"), "N2: {}", out.stderr);
            assert!(
                !home.workspace_dir(&name).exists(),
                "N2: half-import left behind"
            );
            refused += 1;
        }
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    writer.join().unwrap();
    assert!(
        refused > 0,
        "N2: a constantly written source was never refused"
    );
}

/// N2: a damaged source is refused and nothing is left behind.
#[test]
fn n2_import_of_a_damaged_store_leaves_nothing_behind() {
    let llm = FakeLlm::start();
    let home = Home::with_config(&config(&llm, "fake-embed", 64));
    let src = tempfile::tempdir().unwrap();
    copy_fixture_to(src.path());
    // Scribble over the middle of the file: header intact, b-tree pages broken.
    let path = src.path().join("stm.sqlite");
    let mut bytes = std::fs::read(&path).unwrap();
    let page = 4096;
    for b in &mut bytes[page * 20..page * 40] {
        *b = 0xA5;
    }
    std::fs::write(&path, bytes).unwrap();

    let out = import(&home, "broken", src.path());
    assert_eq!(out.code, Some(1), "N2: damaged store imported: {out:?}");
    assert!(out.stderr.contains("Error:"), "{}", out.stderr);
    assert!(
        !home.workspace_dir("broken").exists(),
        "N2: damaged import left behind"
    );
}
