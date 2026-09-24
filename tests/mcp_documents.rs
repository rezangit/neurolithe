//! Phase 2 §5: long-term memory in standalone MCP mode: `remember_document`
//! and `recall_ltm`.
mod common;

use common::{FakeLlm, Harness, Home};
use serde_json::{Value, json};

/// Recall hits for `query` (the `recall_ltm` array).
fn recall(h: &mut Harness, query: &str) -> Vec<Value> {
    let res = h.call("recall_ltm", json!({"query": query, "k": 50}));
    res.assert_ok();
    res.json().as_array().cloned().unwrap_or_default()
}

fn hit_ids(hits: &[Value]) -> Vec<String> {
    hits.iter()
        .filter_map(|h| h["dataId"].as_str().map(str::to_string))
        .collect()
}

/// `ltm_leaves` of the active workspace's export.
fn leaves(h: &mut Harness) -> Vec<Value> {
    let res = h.call("workspace_export", json!({}));
    res.assert_ok();
    res.json()["ltm_leaves"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

#[test]
fn remember_document_then_recall_it() {
    let mut h = Harness::start();
    let text = "Invoice 2026-114 from Acme GmbH for 3 desks, total 1,240 EUR.";
    let res = h.call(
        "remember_document",
        json!({"title": "Acme invoice", "text": text, "tags": ["invoice", "office"]}),
    );
    res.assert_ok();
    let placed = res.json();
    let data_id = placed["data_id"].as_str().expect("data_id").to_string();
    assert!(data_id.starts_with("doc_"), "minted id: {data_id}");
    assert!(placed["leaf_id"].is_i64(), "{placed}");
    assert_eq!(placed["updated"], false, "{placed}");
    assert_eq!(
        placed["summarized"], true,
        "chat LLM is configured: {placed}"
    );
    let path: Vec<String> = serde_json::from_value(placed["concept_path"].clone()).unwrap();
    assert_eq!(path.first().map(String::as_str), Some("root"), "{path:?}");

    // The summary went through the chat LLM, and the leaf was embedded.
    assert!(h.llm.requests().iter().any(|r| {
        r.path.ends_with("/chat/completions") && r.body.to_string().contains("Invoice 2026-114")
    }));

    let hits = recall(&mut h, "Acme invoice desks");
    assert!(
        hit_ids(&hits).contains(&data_id),
        "recall_ltm missed the document: {hits:?}"
    );
    let hit = hits
        .iter()
        .find(|x| x["dataId"] == data_id.as_str())
        .unwrap();
    assert_eq!(hit["provenance"]["source"], "remember_document", "{hit}");
    assert!(hit["ancestors"].is_array(), "{hit}");

    let trace = h.call("trace_dataId", json!({"dataId": data_id}));
    trace.assert_ok();
    assert!(!trace.json()["ltm_leaf"].is_null(), "{}", trace.text);

    let leaves = leaves(&mut h);
    assert_eq!(leaves.len(), 1, "{leaves:?}");
    assert_eq!(leaves[0]["data_id"], data_id.as_str());
    assert_eq!(leaves[0]["title"], "Acme invoice");
}

#[test]
fn remember_document_upserts_by_data_id() {
    let mut h = Harness::start();
    let first = h.call(
        "remember_document",
        json!({"title": "Lease", "text": "Lease v1.", "data_id": "lease-1"}),
    );
    first.assert_ok();
    assert_eq!(first.json()["data_id"], "lease-1");
    assert_eq!(first.json()["updated"], false);

    let second = h.call(
        "remember_document",
        json!({"title": "Lease (amended)", "text": "Lease v2, amended with a longer text.", "data_id": "lease-1"}),
    );
    second.assert_ok();
    assert_eq!(second.json()["updated"], true, "{}", second.text);

    let leaves = leaves(&mut h);
    let versions: Vec<&Value> = leaves
        .iter()
        .filter(|l| l["data_id"] == "lease-1")
        .collect();
    assert_eq!(versions.len(), 1, "upsert duplicated the leaf: {leaves:?}");
    assert_eq!(versions[0]["title"], "Lease (amended)");
    let hits = recall(&mut h, "Lease amended");
    assert_eq!(
        hit_ids(&hits).iter().filter(|id| *id == "lease-1").count(),
        1,
        "{hits:?}"
    );
}

#[test]
fn remember_document_validates_input() {
    let mut h = Harness::start();
    for args in [
        json!({}),
        json!({"title": "no text"}),
        json!({"text": ""}),
        json!({"text": "   \n "}),
        json!({"text": "x", "tags": "not-a-list"}),
    ] {
        assert!(
            h.try_call("remember_document", args.clone()).is_err(),
            "accepted {args}"
        );
    }
    assert!(leaves(&mut h).is_empty());
    // Untitled documents take their first line as the title.
    let res = h.call(
        "remember_document",
        json!({"text": "First line title\nbody text"}),
    );
    res.assert_ok();
    assert_eq!(leaves(&mut h)[0]["title"], "First line title");
}

#[test]
fn documents_are_isolated_per_workspace_and_survive_restart() {
    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    {
        let mut server = home.spawn_mcp(&[], &[]);
        server.initialize();
        let res = server.call_tool(
            "remember_document",
            json!({"title": "Passport", "text": "Passport renewal due 2027.", "data_id": "pp-1"}),
        );
        assert!(!res.is_error, "{}", res.text);
        let (code, stderr) = server.shutdown();
        assert_eq!(code, Some(0), "{stderr}");
    }

    // Restart: still recallable in the same workspace.
    let mut server = home.spawn_mcp(&[], &[]);
    server.initialize();
    let hits = server.call_tool("recall_ltm", json!({"query": "Passport renewal", "k": 50}));
    assert!(
        hits.text.contains("pp-1"),
        "lost after restart: {}",
        hits.text
    );

    // Another workspace does not see it.
    assert!(
        !server
            .call_tool("workspace_create", json!({"name": "other"}))
            .is_error
    );
    assert!(
        !server
            .call_tool("workspace_switch", json!({"name": "other"}))
            .is_error
    );
    let hits = server.call_tool("recall_ltm", json!({"query": "Passport renewal", "k": 50}));
    assert!(!hits.is_error, "{}", hits.text);
    assert!(
        !hits.text.contains("pp-1"),
        "leaked across workspaces: {}",
        hits.text
    );
    let trace = server.call_tool("trace_dataId", json!({"dataId": "pp-1"}));
    assert!(trace.json()["ltm_leaf"].is_null(), "{}", trace.text);
}

// --- LTM spine (§6): generic default, configurable via [[ltm.spine]] ---------

/// All concept names in an `ltm_map` result, depth-first.
fn concept_names(nodes: &Value, out: &mut Vec<String>) {
    for n in nodes.as_array().cloned().unwrap_or_default() {
        if let Some(name) = n["name"].as_str() {
            out.push(name.to_string());
        }
        concept_names(&n["children"], out);
    }
}

fn map_names(server: &mut common::McpProcess) -> Vec<String> {
    let res = server.call_tool("ltm_map", json!({"depth": 5}));
    assert!(!res.is_error, "{}", res.text);
    let mut names = Vec::new();
    concept_names(&res.json(), &mut names);
    names
}

#[test]
fn default_spine_is_generic() {
    let mut h = Harness::start();
    let names = map_names(&mut h.server);
    for expected in ["root", "notes", "documents", "inbox"] {
        assert!(
            names.iter().any(|n| n == expected),
            "missing {expected}: {names:?}"
        );
    }
    // The old hard-coded personal-life branches are gone.
    for gone in ["job", "investment", "self-improvement", "learning"] {
        assert!(
            !names.iter().any(|n| n == gone),
            "legacy spine branch {gone}: {names:?}"
        );
    }
}

#[test]
fn custom_spine_from_config_is_seeded_and_embedded() {
    let llm = FakeLlm::start();
    let home = Home::with_config(&common::home_config_toml_with(
        &llm.base_url(),
        "",
        r#"[[ltm.spine]]
path = "projects/alpha"
description = "Everything about project Alpha."

[[ltm.spine]]
path = "recipes"
description = "Cooking recipes and meal plans."
"#,
    ));
    let mut server = home.spawn_mcp(&[], &[]);
    server.initialize();
    let names = map_names(&mut server);
    for expected in ["root", "projects", "alpha", "recipes", "inbox"] {
        assert!(
            names.iter().any(|n| n == expected),
            "missing {expected}: {names:?}"
        );
    }
    assert!(
        !names.iter().any(|n| n == "notes"),
        "default spine mixed in: {names:?}"
    );

    // Spine concepts are embedded at startup: recall surfaces them before any
    // document exists.
    let res = server.call_tool("recall_ltm", json!({"query": "meal plans", "k": 50}));
    assert!(!res.is_error, "{}", res.text);
    let concepts: Vec<String> = res
        .json()
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|h| h["concept"].as_str().map(str::to_string))
        .collect();
    assert!(concepts.iter().any(|c| c == "recipes"), "{concepts:?}");
}
