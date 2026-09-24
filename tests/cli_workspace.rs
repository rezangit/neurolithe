//! Phase 2 §2: the `neurolithe workspace …` CLI, including backup and the
//! import of a legacy (v0.2.x) store from `tests/fixtures/legacy-v0.2`.
mod common;

use common::{FakeLlm, Home};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn count(db: &Path, sql: &str) -> i64 {
    let conn =
        rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap_or_else(|e| panic!("open {}: {e}", db.display()));
    conn.query_row(sql, [], |r| r.get(0))
        .unwrap_or_else(|e| panic!("{sql} on {}: {e}", db.display()))
}

/// Store `facts` in workspace `ws` through a short-lived MCP session.
fn seed(home: &Home, ws: &str, facts: &[&str]) {
    let mut server = home.spawn_mcp(&["--workspace", ws], &[]);
    server.initialize();
    for fact in facts {
        let res = server.call_tool("store_memory", json!({"fact_text": fact}));
        assert!(!res.is_error, "{}", res.text);
    }
    let (code, stderr) = server.shutdown();
    assert_eq!(code, Some(0), "{stderr}");
}

#[test]
fn list_create_and_delete() {
    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());

    let out = home.cli(&["workspace", "list"]);
    out.assert_success();
    assert!(out.stdout.contains("No workspaces yet"), "{}", out.stdout);

    home.cli(&["workspace", "create", "alpha"]).assert_success();
    home.cli(&["workspace", "create", "default"])
        .assert_success();
    assert!(home.workspace_dir("alpha").is_dir());

    let out = home.cli(&["workspace", "list"]);
    out.assert_success();
    let lines: Vec<&str> = out
        .stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert_eq!(lines.len(), 2, "{}", out.stdout);
    let default_line = lines.iter().find(|l| l.contains("default")).unwrap();
    let alpha_line = lines.iter().find(|l| l.contains("alpha")).unwrap();
    assert!(
        default_line.trim_start().starts_with('*'),
        "configured ws not marked: {default_line}"
    );
    assert!(!alpha_line.trim_start().starts_with('*'), "{alpha_line}");
    assert!(
        alpha_line.contains("stm") && alpha_line.contains("ltm"),
        "{alpha_line}"
    );

    // Existing / invalid names are errors.
    let out = home.cli(&["workspace", "create", "alpha"]);
    assert_eq!(out.code, Some(1), "{out:?}");
    assert!(out.stderr.contains("Error:"), "{}", out.stderr);
    let out = home.cli(&["workspace", "create", "Bad/Name"]);
    assert_eq!(out.code, Some(1), "{out:?}");

    // Delete needs --yes and leaves everything in place without it.
    let out = home.cli(&["workspace", "delete", "alpha"]);
    assert_ne!(out.code, Some(0), "{out:?}");
    assert!(home.workspace_dir("alpha").exists());
    home.cli(&["workspace", "delete", "alpha", "--yes"])
        .assert_success();
    assert!(!home.workspace_dir("alpha").exists());
    let out = home.cli(&["workspace", "delete", "alpha", "--yes"]);
    assert_eq!(out.code, Some(1), "deleting a missing workspace: {out:?}");
}

#[test]
fn export_to_stdout_and_file() {
    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    seed(&home, "notes", &["Export me"]);

    let out = home.cli(&["workspace", "export", "notes"]);
    out.assert_success();
    let export = out.json();
    assert_eq!(export["workspace"], "notes", "{export}");
    let facts: Vec<&str> = export["stm_facts"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|f| f["payload"]["fact"].as_str())
        .collect();
    assert_eq!(facts, vec!["Export me"]);

    let file = home.cwd.path().join("out.json");
    home.cli(&[
        "workspace",
        "export",
        "notes",
        "--out",
        file.to_str().unwrap(),
    ])
    .assert_success();
    let from_file: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    assert_eq!(from_file, export);

    // A workspace created but never opened has no stores to back up.
    home.cli(&["workspace", "create", "empty"]).assert_success();
    let out = home.cli(&["workspace", "backup", "empty"]);
    assert_eq!(out.code, Some(1), "{out:?}");
    assert!(out.stderr.contains("no store files"), "{}", out.stderr);

    let out = home.cli(&["workspace", "export", "missing"]);
    assert_eq!(out.code, Some(1), "{out:?}");
}

#[test]
fn backup_writes_timestamped_consistent_copies() {
    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    seed(&home, "keep", &["one", "two", "three"]);
    let dest = home.cwd.path().join("backups");

    let out = home.cli(&[
        "workspace",
        "backup",
        "keep",
        "--out",
        dest.to_str().unwrap(),
    ]);
    out.assert_success();
    let written: Vec<PathBuf> = out
        .stdout
        .lines()
        .filter_map(|l| l.strip_prefix("Wrote "))
        .map(|p| PathBuf::from(p.trim()))
        .collect();
    assert_eq!(written.len(), 2, "{}", out.stdout);
    for path in &written {
        assert!(path.exists(), "{}", path.display());
        let name = path.file_name().unwrap().to_str().unwrap();
        // keep-YYYYMMDDTHHMMSSZ-{stm,ltm}.sqlite
        let rest = name
            .strip_prefix("keep-")
            .unwrap_or_else(|| panic!("{name}"));
        let (ts, kind) = rest.split_once('-').unwrap_or_else(|| panic!("{name}"));
        assert_eq!(ts.len(), 16, "{name}");
        assert!(ts.as_bytes()[8] == b'T' && ts.ends_with('Z'), "{name}");
        assert!(kind == "stm.sqlite" || kind == "ltm.sqlite", "{name}");
    }
    let stm_backup = written
        .iter()
        .find(|p| p.to_str().unwrap().ends_with("-stm.sqlite"))
        .unwrap();
    assert_eq!(count(stm_backup, "SELECT count(*) FROM nodes"), 3);
    for path in &written {
        let conn =
            rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
                .unwrap();
        let check: String = conn
            .query_row("PRAGMA integrity_check", [], |r| r.get(0))
            .unwrap();
        assert_eq!(check, "ok", "{}", path.display());
    }
}

#[test]
fn import_brings_a_legacy_store_into_a_workspace() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/legacy-v0.2");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture.join("manifest.json")).unwrap())
            .unwrap();
    let before_stm = std::fs::read(fixture.join("stm.sqlite")).unwrap();
    let before_ltm = std::fs::read(fixture.join("ltm.sqlite")).unwrap();

    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    let (stm, ltm) = (fixture.join("stm.sqlite"), fixture.join("ltm.sqlite"));
    let out = home.cli(&[
        "workspace",
        "import",
        "legacy",
        "--stm",
        stm.to_str().unwrap(),
        "--ltm",
        ltm.to_str().unwrap(),
    ]);
    out.assert_success();

    // The source files are never modified.
    assert_eq!(
        std::fs::read(fixture.join("stm.sqlite")).unwrap(),
        before_stm
    );
    assert_eq!(
        std::fs::read(fixture.join("ltm.sqlite")).unwrap(),
        before_ltm
    );

    // Every fact of every legacy tenant is now in the one workspace.
    let mut expected: Vec<String> = Vec::new();
    for facts in manifest["tenants"].as_object().unwrap().values() {
        expected.extend(serde_json::from_value::<Vec<String>>(facts.clone()).unwrap());
    }
    expected.sort();
    let mut server = home.spawn_mcp(&["--workspace", "legacy"], &[]);
    server.initialize();
    let export = server.call_tool("workspace_export", json!({}));
    assert!(!export.is_error, "{}", export.text);
    let mut facts: Vec<String> = export.json()["stm_facts"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|f| f["payload"]["fact"].as_str().map(str::to_string))
        .collect();
    facts.sort();
    assert_eq!(facts, expected, "imported facts");

    // Imported vectors are searchable with the (same) embedder.
    let res = server.call_tool("query_memory", json!({"query": "Alice Acme"}));
    assert!(!res.is_error, "{}", res.text);
    assert!(res.text.contains("Alice works at Acme"), "{}", res.text);
    let (code, stderr) = server.shutdown();
    assert_eq!(code, Some(0), "{stderr}");

    // Graph + episodes carried over; tenancy collapsed to one tenant.
    let db = home.workspace_dir("legacy").join("stm.sqlite");
    assert_eq!(
        count(&db, "SELECT count(*) FROM edges"),
        manifest["edges"].as_i64().unwrap()
    );
    assert_eq!(
        count(&db, "SELECT count(*) FROM episodes"),
        manifest["episodes"].as_i64().unwrap()
    );
    assert_eq!(count(&db, "SELECT count(DISTINCT tenant_id) FROM nodes"), 1);
    assert_eq!(
        count(&db, "SELECT count(*) FROM nodes WHERE tenant_id = 'jarvis'"),
        0
    );
    // Migrated: versioned, with store metadata.
    for f in ["stm.sqlite", "ltm.sqlite"] {
        let db = home.workspace_dir("legacy").join(f);
        assert!(count(&db, "PRAGMA user_version") > 0, "{f} not migrated");
        assert_eq!(
            count(
                &db,
                "SELECT count(*) FROM meta WHERE key = 'embedding_model'"
            ),
            1,
            "{f}"
        );
    }

    // Importing onto an existing name is refused.
    let out = home.cli(&[
        "workspace",
        "import",
        "legacy",
        "--stm",
        stm.to_str().unwrap(),
        "--ltm",
        ltm.to_str().unwrap(),
    ]);
    assert_eq!(out.code, Some(1), "{out:?}");
}

/// P2R-9: backups and exports hold the same data as the store, so they are
/// private (0600), including when `export --out` overwrites an existing file.
#[cfg(unix)]
#[test]
fn p2r9_backup_and_export_files_are_private() {
    use std::os::unix::fs::PermissionsExt;
    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;

    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    seed(&home, "priv", &["private fact"]);

    let dest = home.cwd.path().join("bk");
    let out = home.cli(&[
        "workspace",
        "backup",
        "priv",
        "--out",
        dest.to_str().unwrap(),
    ]);
    out.assert_success();
    for line in out.stdout.lines().filter_map(|l| l.strip_prefix("Wrote ")) {
        assert_eq!(mode(Path::new(line.trim())), 0o600, "P2R-9: backup {line}");
    }

    let file = home.cwd.path().join("export.json");
    std::fs::write(&file, "old, world-readable").unwrap();
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();
    home.cli(&[
        "workspace",
        "export",
        "priv",
        "--out",
        file.to_str().unwrap(),
    ])
    .assert_success();
    assert_eq!(mode(&file), 0o600, "P2R-9: overwritten export");
    assert!(
        std::fs::read_to_string(&file)
            .unwrap()
            .contains("private fact")
    );
}
