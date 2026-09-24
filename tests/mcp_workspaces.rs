//! Phase 2 §2: workspaces over MCP. Selection, the `workspace_*` tools, and
//! physical isolation between workspaces.
mod common;

use common::{FakeLlm, Harness, Home, home_config_toml_with};
use serde_json::{Value, json};

fn names(list: &Value) -> Vec<String> {
    list.as_array()
        .unwrap_or_else(|| panic!("not an array: {list}"))
        .iter()
        .map(|w| w["name"].as_str().unwrap().to_string())
        .collect()
}

fn query_facts(h: &mut Harness, query: &str) -> Vec<String> {
    let res = h.call("query_memory", json!({"query": query}));
    res.assert_ok();
    res.json()
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|r| r["fact"].as_str().map(str::to_string))
        .collect()
}

#[test]
fn fresh_home_starts_in_the_default_workspace() {
    let mut h = Harness::start();
    let res = h.call("workspace_current", json!({}));
    res.assert_ok();
    let cur = res.json();
    assert_eq!(cur["name"], "default", "{cur}");
    assert_eq!(cur["active"], true, "{cur}");
    assert!(cur["stm_bytes"].as_u64().is_some_and(|b| b > 0), "{cur}");
    assert!(cur["ltm_bytes"].as_u64().is_some_and(|b| b > 0), "{cur}");
    let path = cur["path"].as_str().unwrap();
    let expected = h.home.workspace_dir("default");
    assert_eq!(
        std::fs::canonicalize(path).unwrap(),
        std::fs::canonicalize(&expected).unwrap()
    );
    assert!(expected.join("stm.sqlite").exists());
    assert!(expected.join("ltm.sqlite").exists());
    // Nothing is written to the CWD.
    assert_eq!(
        std::fs::read_dir(h.home.cwd.path()).unwrap().count(),
        0,
        "files appeared in the CWD"
    );
}

#[test]
fn workspaces_are_physically_isolated() {
    let mut h = Harness::start();
    h.call(
        "store_memory",
        json!({"fact_text": "Alpha lives in default"}),
    )
    .assert_ok();

    let created = h.call("workspace_create", json!({"name": "work"}));
    created.assert_ok();
    assert_eq!(created.json()["name"], "work");
    // Creating does not switch.
    assert_eq!(h.active_workspace(), "default");

    let list = h.call("workspace_list", json!({}));
    list.assert_ok();
    let list = list.json();
    assert_eq!(names(&list), vec!["default", "work"], "sorted by name");
    let active: Vec<_> = list
        .as_array()
        .unwrap()
        .iter()
        .filter(|w| w["active"] == true)
        .collect();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0]["name"], "default");

    h.call("workspace_switch", json!({"name": "work"}))
        .assert_ok();
    assert_eq!(h.active_workspace(), "work");
    assert!(h.stm_facts().is_empty(), "work sees default's facts");
    assert!(query_facts(&mut h, "Alpha").is_empty());
    h.call("store_memory", json!({"fact_text": "Beta lives in work"}))
        .assert_ok();

    h.call("workspace_switch", json!({"name": "default"}))
        .assert_ok();
    assert_eq!(h.stm_facts(), vec!["Alpha lives in default".to_string()]);
    assert!(!query_facts(&mut h, "Beta").contains(&"Beta lives in work".to_string()));

    // Each workspace is its own pair of files.
    for ws in ["default", "work"] {
        let dir = h.home.workspace_dir(ws);
        assert!(dir.join("stm.sqlite").exists() && dir.join("ltm.sqlite").exists());
    }
}

#[test]
fn workspace_export_reads_a_named_workspace() {
    let mut h = Harness::start();
    h.call("workspace_create", json!({"name": "other"}))
        .assert_ok();
    h.call("workspace_switch", json!({"name": "other"}))
        .assert_ok();
    h.call("store_memory", json!({"fact_text": "Only in other"}))
        .assert_ok();
    h.call("workspace_switch", json!({"name": "default"}))
        .assert_ok();

    assert_eq!(
        h.exported_facts(Some("other")),
        vec!["Only in other".to_string()]
    );
    assert!(h.exported_facts(None).is_empty());
    assert!(
        h.try_call("workspace_export", json!({"name": "missing"}))
            .is_err()
    );
}

#[test]
fn workspace_names_are_validated() {
    let mut h = Harness::start();
    let too_long = "a".repeat(65);
    for bad in [
        "",
        "Upper",
        "-leading-dash",
        "_leading",
        "../escape",
        "a/b",
        "sp ace",
        "dot.name",
        too_long.as_str(),
    ] {
        for tool in ["workspace_create", "workspace_switch"] {
            let err = h.try_call(tool, json!({"name": bad}));
            assert!(err.is_err(), "{tool} accepted {bad:?}");
        }
    }
    // Nothing escaped the workspaces dir.
    assert!(!h.path().join("escape").exists());
    let list = h.call("workspace_list", json!({})).json();
    assert_eq!(names(&list), vec!["default"]);

    // Edge-of-regex names are fine.
    let max = format!("a{}", "b".repeat(63));
    for good in ["a", "0-9_x", max.as_str()] {
        h.call("workspace_create", json!({"name": good}))
            .assert_ok();
    }
}

#[test]
fn workspace_create_rejects_existing_and_switch_rejects_missing() {
    let mut h = Harness::start();
    let err = h
        .try_call("workspace_create", json!({"name": "default"}))
        .unwrap_err();
    assert!(err.contains("already exists"), "{err}");
    let err = h
        .try_call("workspace_switch", json!({"name": "ghost"}))
        .unwrap_err();
    assert!(err.contains("does not exist"), "{err}");
    assert!(!h.home.workspace_dir("ghost").exists(), "switch created it");
    assert_eq!(h.active_workspace(), "default");
}

#[test]
fn workspace_delete_requires_matching_confirm_and_inactive_target() {
    let mut h = Harness::start();
    h.call("workspace_create", json!({"name": "doomed"}))
        .assert_ok();
    let dir = h.home.workspace_dir("doomed");

    for args in [
        json!({"name": "doomed"}),
        json!({"name": "doomed", "confirm": ""}),
        json!({"name": "doomed", "confirm": "Doomed"}),
        json!({"name": "doomed", "confirm": "other"}),
        json!({"name": "doomed", "confirm": true}),
    ] {
        assert!(
            h.try_call("workspace_delete", args.clone()).is_err(),
            "accepted {args}"
        );
        assert!(dir.exists(), "deleted by rejected call {args}");
    }

    // The active workspace cannot be deleted.
    let err = h
        .try_call(
            "workspace_delete",
            json!({"name": "default", "confirm": "default"}),
        )
        .unwrap_err();
    assert!(err.contains("active"), "{err}");
    assert!(h.home.workspace_dir("default").exists());

    let res = h.call(
        "workspace_delete",
        json!({"name": "doomed", "confirm": "doomed"}),
    );
    res.assert_ok();
    assert!(res.text.contains("doomed"), "{}", res.text);
    assert!(!dir.exists(), "workspace dir survived delete");
    let list = h.call("workspace_list", json!({})).json();
    assert_eq!(names(&list), vec!["default"]);

    // Deleting a missing workspace is refused.
    assert!(
        h.try_call(
            "workspace_delete",
            json!({"name": "doomed", "confirm": "doomed"})
        )
        .is_err()
    );
}

#[test]
fn workspace_switch_resets_session_buffers() {
    let mut h = Harness::start();
    h.call(
        "push_dialogue",
        json!({"session_id": "s1", "new_message": "said in default"}),
    )
    .assert_ok();
    h.call("workspace_create", json!({"name": "fresh"}))
        .assert_ok();
    h.call("workspace_switch", json!({"name": "fresh"}))
        .assert_ok();
    let res = h.call(
        "push_dialogue",
        json!({"session_id": "s1", "new_message": "said in fresh"}),
    );
    res.assert_ok();
    let recent = res.json()["recent_messages"].clone();
    assert!(
        !recent.to_string().contains("said in default"),
        "session leaked across workspaces: {recent}"
    );
}

/// `[mcp] allow_workspace_switch = false` pins the session to its workspace:
/// no switching, and (P2R-4) no reaching into other workspaces either:
/// create, export, and delete of another workspace are refused. The pinned
/// workspace itself stays fully usable.
#[test]
fn p2r4_pinned_session_cannot_touch_other_workspaces() {
    let llm = FakeLlm::start();
    let home = Home::with_config(&home_config_toml_with(
        &llm.base_url(),
        "",
        "[mcp]\nallow_workspace_switch = false\n",
    ));
    // Another workspace with data, made outside the pinned session.
    {
        let mut other = home.spawn_mcp(&["--workspace", "other"], &[]);
        other.initialize();
        assert!(
            !other
                .call_tool("store_memory", json!({"fact_text": "other secret"}))
                .is_error
        );
        other.shutdown();
    }

    let mut server = home.spawn_mcp(&[], &[]);
    server.initialize();
    let err = server
        .try_call_tool("workspace_switch", json!({"name": "other"}))
        .expect_err("switch allowed despite allow_workspace_switch = false");
    assert!(err.contains("disabled"), "{err}");

    for (tool, args) in [
        ("workspace_export", json!({"name": "other"})),
        (
            "workspace_delete",
            json!({"name": "other", "confirm": "other"}),
        ),
        ("workspace_create", json!({"name": "third"})),
    ] {
        let err = server
            .try_call_tool(tool, args.clone())
            .expect_err("pinned session reached another workspace");
        assert!(err.contains("pinned"), "P2R-4: {tool} {args}: {err}");
        assert!(!err.contains("other secret"), "P2R-4: data leaked: {err}");
    }
    assert!(
        home.workspace_dir("other").join("stm.sqlite").exists(),
        "P2R-4: deleted"
    );
    assert!(!home.workspace_dir("third").exists(), "P2R-4: created");

    // The pinned workspace itself is fully usable, including exporting it.
    assert!(
        !server
            .call_tool("store_memory", json!({"fact_text": "mine"}))
            .is_error
    );
    for args in [json!({}), json!({"name": "default"})] {
        let export = server.call_tool("workspace_export", args.clone());
        assert!(!export.is_error, "{args}: {}", export.text);
        assert!(export.text.contains("mine"), "{}", export.text);
    }
    assert!(!server.call_tool("workspace_list", json!({})).is_error);
    let cur = server.call_tool("workspace_current", json!({}));
    assert_eq!(cur.json()["name"], "default");
}

// --- P2R-3: a workspace open in another process can't be deleted/reembedded ---

/// Start `mcp` on `ws` and make sure the workspace is actually open.
fn open_in_other_process(home: &Home, ws: &str) -> common::McpProcess {
    let mut holder = home.spawn_mcp(&["--workspace", ws], &[]);
    holder.initialize();
    let res = holder.call_tool("store_memory", json!({"fact_text": format!("held {ws}")}));
    assert!(!res.is_error, "{}", res.text);
    holder
}

#[test]
fn p2r3_cli_delete_and_reembed_refuse_a_workspace_open_elsewhere() {
    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    let holder = open_in_other_process(&home, "busy");

    let out = home.cli(&["workspace", "delete", "busy", "--yes"]);
    assert_eq!(
        out.code,
        Some(1),
        "P2R-3: delete of an open workspace: {out:?}"
    );
    assert!(out.stderr.contains("in use"), "{}", out.stderr);
    assert!(
        home.workspace_dir("busy").join("stm.sqlite").exists(),
        "P2R-3: deleted anyway"
    );

    let out = home.cli(&["--workspace", "busy", "reembed"]);
    assert_eq!(
        out.code,
        Some(1),
        "P2R-3: reembed of an open workspace: {out:?}"
    );
    assert!(out.stderr.contains("in use"), "{}", out.stderr);

    // Once the holder is gone, both work.
    let (code, stderr) = holder.shutdown();
    assert_eq!(code, Some(0), "{stderr}");
    home.cli(&["--workspace", "busy", "reembed"])
        .assert_success();
    home.cli(&["workspace", "delete", "busy", "--yes"])
        .assert_success();
    assert!(!home.workspace_dir("busy").exists());
}

#[test]
fn p2r3_mcp_delete_refuses_a_workspace_open_in_another_process() {
    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    let holder = open_in_other_process(&home, "busy");

    let mut server = home.spawn_mcp(&[], &[]);
    server.initialize();
    let err = server
        .try_call_tool(
            "workspace_delete",
            json!({"name": "busy", "confirm": "busy"}),
        )
        .expect_err("P2R-3: deleted a workspace open in another process");
    assert!(err.contains("in use"), "{err}");
    assert!(home.workspace_dir("busy").join("stm.sqlite").exists());

    drop(holder.shutdown());
    let res = server.call_tool(
        "workspace_delete",
        json!({"name": "busy", "confirm": "busy"}),
    );
    assert!(!res.is_error, "{}", res.text);
    assert!(!home.workspace_dir("busy").exists());
}

// --- selection: --workspace, NEUROLITHE_WORKSPACE, config ------------------

fn current_name(home: &Home, args: &[&str], env: &[(&str, &str)]) -> String {
    let mut server = home.spawn_mcp(args, env);
    server.initialize();
    let res = server.call_tool("workspace_current", json!({}));
    assert!(!res.is_error, "{} / stderr: {}", res.text, server.stderr());
    res.json()["name"].as_str().unwrap().to_string()
}

#[test]
fn workspace_selection_precedence() {
    let llm = FakeLlm::start();
    let home = Home::with_config(&home_config_toml_with(
        &llm.base_url(),
        "workspace = \"from-config\"",
        "",
    ));
    assert_eq!(current_name(&home, &[], &[]), "from-config");
    assert_eq!(
        current_name(&home, &[], &[("NEUROLITHE_WORKSPACE", "from-env")]),
        "from-env"
    );
    assert_eq!(
        current_name(
            &home,
            &["--workspace", "from-flag"],
            &[("NEUROLITHE_WORKSPACE", "from-env")]
        ),
        "from-flag"
    );
    // Each selected workspace was created on demand.
    for ws in ["from-config", "from-env", "from-flag"] {
        assert!(home.workspace_dir(ws).join("stm.sqlite").exists(), "{ws}");
    }
}

#[test]
fn invalid_workspace_selection_fails_at_startup() {
    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    let out = home.cli(&["--workspace", "../evil", "mcp"]);
    assert_eq!(
        out.code,
        Some(1),
        "stdout: {} stderr: {}",
        out.stdout,
        out.stderr
    );
    assert!(out.stderr.contains("Error"), "{}", out.stderr);
    assert!(!home.path().join("evil").exists());
}

#[test]
fn two_servers_on_different_workspaces_do_not_share_data() {
    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    let mut a = home.spawn_mcp(&["--workspace", "alpha"], &[]);
    let mut b = home.spawn_mcp(&["--workspace", "beta"], &[]);
    a.initialize();
    b.initialize();
    assert!(
        !a.call_tool("store_memory", json!({"fact_text": "alpha secret"}))
            .is_error
    );
    assert!(
        !b.call_tool("store_memory", json!({"fact_text": "beta secret"}))
            .is_error
    );
    let a_facts = a.call_tool("stm_list", json!({})).text;
    let b_facts = b.call_tool("stm_list", json!({})).text;
    assert!(a_facts.contains("alpha secret") && !a_facts.contains("beta secret"));
    assert!(b_facts.contains("beta secret") && !b_facts.contains("alpha secret"));
}
