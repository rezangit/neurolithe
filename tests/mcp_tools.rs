//! Happy-path coverage of the memory tools, black-box over STDIO against the
//! fake LLM, in a fresh home's default workspace. Workspace tools live in
//! `mcp_workspaces.rs`; known-bug regressions in `mcp_regressions.rs`.
mod common;

use common::{Harness, extracted_fact_text};
use serde_json::{Value, json};
use std::time::Duration;

fn facts_in(result: &Value) -> Vec<String> {
    result
        .as_array()
        .unwrap_or_else(|| panic!("expected a JSON array, got {result}"))
        .iter()
        .filter_map(|r| r["fact"].as_str().map(str::to_string))
        .collect()
}

#[test]
fn store_memory_then_query_memory_finds_it() {
    let mut h = Harness::start();
    h.call(
        "store_memory",
        json!({"fact_text": "Alice works at Acme", "tags": ["job"]}),
    )
    .assert_ok();

    let res = h.call("query_memory", json!({"query": "Alice"}));
    res.assert_ok();
    let facts = facts_in(&res.json());
    assert!(
        facts.contains(&"Alice works at Acme".to_string()),
        "{facts:?}"
    );

    // Result entries are token-optimized: no internal ids or scores.
    let first = &res.json()[0];
    assert!(first.get("fact").is_some() && first.get("ccl").is_some());
    assert!(first.get("id").is_none() && first.get("relevance_score").is_none());

    // The embedder was used for the write and the read.
    assert!(h.llm.count("/embeddings") >= 2);
}

#[test]
fn query_memory_respects_k() {
    let mut h = Harness::start();
    for i in 0..4 {
        h.call(
            "store_memory",
            json!({"fact_text": format!("Carol fact number {i}")}),
        )
        .assert_ok();
    }
    let res = h.call("query_memory", json!({"query": "Carol", "k": 1}));
    res.assert_ok();
    assert!(
        facts_in(&res.json()).len() <= 1,
        "k=1 ignored: {}",
        res.text
    );
}

#[test]
fn unicode_and_fts_operators_round_trip() {
    let mut h = Harness::start();
    let fact = r#"Ünïcödé 日本語 🧠 "quoted" OR NOT AND * ( ) -- ;DROP TABLE nodes"#;
    h.call("store_memory", json!({"fact_text": fact}))
        .assert_ok();
    let res = h.call(
        "query_memory",
        json!({"query": r#"日本語 🧠 " OR ( * NEAR"#}),
    );
    res.assert_ok();
    assert!(
        facts_in(&res.json()).contains(&fact.to_string()),
        "{}",
        res.text
    );
}

#[test]
fn push_dialogue_returns_context_window() {
    let mut h = Harness::start();
    let res = h.call(
        "push_dialogue",
        json!({"session_id": "s1", "new_message": "I moved to Berlin"}),
    );
    res.assert_ok();
    let ctx = res.json();
    assert!(ctx.get("summary").is_some(), "{ctx}");
    assert!(
        ctx["recent_messages"]
            .as_array()
            .is_some_and(|m| m.iter().any(|x| x == "I moved to Berlin")),
        "{ctx}"
    );
    assert!(ctx["relevant_facts"].is_array(), "{ctx}");
    // The turn was sent to the LLM for fact extraction.
    assert!(
        h.llm.requests().iter().any(|r| {
            r.path.ends_with("/chat/completions")
                && r.body.to_string().contains("I moved to Berlin")
        }),
        "no extraction request reached the LLM"
    );
}

#[test]
fn push_dialogue_keeps_session_history() {
    let mut h = Harness::start();
    for msg in ["first turn", "second turn"] {
        h.call(
            "push_dialogue",
            json!({"session_id": "s-hist", "new_message": msg}),
        )
        .assert_ok();
    }
    let res = h.call(
        "push_dialogue",
        json!({"session_id": "s-hist", "new_message": "third turn"}),
    );
    let recent = res.json()["recent_messages"].clone();
    for msg in ["first turn", "second turn", "third turn"] {
        assert!(
            recent.as_array().unwrap().iter().any(|m| m == msg),
            "missing {msg}: {recent}"
        );
    }
}

#[test]
fn recall_ltm_returns_a_json_array() {
    let mut h = Harness::start();
    let res = h.call("recall_ltm", json!({"query": "receipt"}));
    res.assert_ok();
    assert!(res.json().is_array(), "{}", res.text);
}

#[test]
fn stm_list_pages_and_filters() {
    let mut h = Harness::start();
    for fact in ["apple pie", "banana bread", "cherry tart"] {
        h.call("store_memory", json!({"fact_text": fact}))
            .assert_ok();
    }
    assert_eq!(h.stm_facts().len(), 3);

    let page = h.call("stm_list", json!({"limit": 2, "offset": 0}));
    page.assert_ok();
    assert_eq!(page.json().as_array().unwrap().len(), 2);
    let rest = h.call("stm_list", json!({"limit": 2, "offset": 2}));
    assert_eq!(rest.json().as_array().unwrap().len(), 1);

    let filtered = h.call("stm_list", json!({"contains": "BANANA"}));
    filtered.assert_ok();
    assert_eq!(facts_in(&filtered.json()), vec!["banana bread".to_string()]);

    let entry = &page.json()[0];
    for key in ["fact", "status", "relevance_score"] {
        assert!(
            entry.get(key).is_some(),
            "stm_list entry lacks {key}: {entry}"
        );
    }
}

#[test]
fn memory_stats_and_health_reflect_stored_facts() {
    let mut h = Harness::start();
    let before = h.call("memory_stats", json!({}));
    before.assert_ok();
    assert_eq!(before.json()["stm_active_nodes"], 0);

    h.call("store_memory", json!({"fact_text": "Gina"}))
        .assert_ok();

    let stats = h.call("memory_stats", json!({})).json();
    assert_eq!(stats["stm_active_nodes"], 1, "{stats}");
    assert!(stats["ltm_tree_nodes"].is_number(), "{stats}");

    let health = h.call("health", json!({}));
    health.assert_ok();
    assert_eq!(health.json()["stm_active_nodes"], 1);
}

#[test]
fn ltm_introspection_tools_walk_the_tree() {
    let mut h = Harness::start();
    let map = h.call("ltm_map", json!({"depth": 2}));
    map.assert_ok();
    let roots = map.json();
    assert!(roots.is_array(), "{roots}");

    if let Some(root_id) = roots.get(0).and_then(|r| r["id"].as_i64()) {
        let node = h.call("inspect_node", json!({"id": root_id}));
        node.assert_ok();
        assert!(node.json().is_object(), "{}", node.text);

        let sub = h.call("subtree", json!({"node": root_id, "depth": 1}));
        sub.assert_ok();
        assert_eq!(sub.json()["id"], root_id, "{}", sub.text);
    }

    let dbg = h.call("placement_debug", json!({"sample": 5}));
    dbg.assert_ok();
    let _ = dbg.json();
}

#[test]
fn inspect_node_and_subtree_validate_ids() {
    let mut h = Harness::start();
    assert!(h.call("inspect_node", json!({})).is_error);
    assert!(h.call("inspect_node", json!({"id": "1"})).is_error);
    assert!(h.call("subtree", json!({})).is_error);

    // A well-formed but unknown id must not crash the server.
    let _ = h.call("inspect_node", json!({"id": 999_999}));
    h.call("health", json!({})).assert_ok();
}

#[test]
fn trace_data_id_reports_unknown_document() {
    let mut h = Harness::start();
    let res = h.call("trace_dataId", json!({"dataId": "doc_does_not_exist"}));
    res.assert_ok();
    let trace = res.json();
    assert!(trace.is_object(), "{trace}");
    assert!(trace["ltm_leaf"].is_null(), "{trace}");
}

#[test]
fn extracted_facts_are_queryable_after_push_dialogue() {
    // End-to-end learning loop: dialogue → extraction → recall. Depends on QA-1.
    let mut h = Harness::start();
    h.call(
        "push_dialogue",
        json!({"session_id": "s1", "new_message": "Hank owns a red bike"}),
    )
    .assert_ok();
    let expected = extracted_fact_text("Hank owns a red bike");
    let found = common::eventually(Duration::from_secs(5), || {
        let res = h.call("query_memory", json!({"query": "Hank bike"}));
        !res.is_error && facts_in(&res.json()).contains(&expected)
    });
    assert!(found, "QA-1: extracted fact never became queryable");
}

#[test]
fn workspace_export_contains_stored_facts() {
    let mut h = Harness::start();
    h.call(
        "store_memory",
        json!({"fact_text": "Dana plays chess", "tags": ["hobby"]}),
    )
    .assert_ok();
    let res = h.call("workspace_export", json!({}));
    res.assert_ok();
    let export = res.json();
    assert_eq!(export["workspace"], "default", "{export}");
    assert!(export["ltm_leaves"].is_array(), "{export}");
    let fact = &export["stm_facts"][0];
    assert_eq!(fact["payload"]["fact"], "Dana plays chess", "{export}");
    assert_eq!(fact["payload"]["tags"], json!(["hobby"]), "{export}");
    for key in [
        "ccl",
        "status",
        "relevance_score",
        "support_count",
        "created_at",
    ] {
        assert!(fact.get(key).is_some(), "export fact lacks {key}: {fact}");
    }
}

/// Phase 2 removed tenancy: a stale `tenant_id` argument is ignored and the
/// data lands in (and is read from) the active workspace.
#[test]
fn stale_tenant_id_argument_is_ignored() {
    let mut h = Harness::start();
    h.call(
        "store_memory",
        json!({"tenant_id": "someone-else", "fact_text": "Ivy likes jazz"}),
    )
    .assert_ok();
    let res = h.call(
        "query_memory",
        json!({"tenant_id": "other", "query": "Ivy jazz"}),
    );
    res.assert_ok();
    assert!(
        facts_in(&res.json()).contains(&"Ivy likes jazz".to_string()),
        "{}",
        res.text
    );
    assert_eq!(h.exported_facts(None), vec!["Ivy likes jazz".to_string()]);
}
