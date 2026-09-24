//! Regression tests for defects found in the 2026-09 QA review
//! (`project-review/reports/03-qa.md`). Each test asserts the CORRECT behaviour
//! and is tagged with its issue ID; it fails until the matching fix lands.
mod common;

use common::{
    API_KEY, FakeLlm, Harness, MALFORMED_EXTRACTION, McpProcess, extracted_fact_text, write_config,
};
use serde_json::json;
use std::time::Duration;

const EVENTUALLY: Duration = Duration::from_secs(5);

// --- QA-1: push_dialogue must persist extracted facts -----------------------

/// QA-1: `push_dialogue` passed a placeholder episode id (0) into the sleep
/// pipeline, every `store_node` hit a FOREIGN KEY failure, and the error was
/// swallowed — nothing was ever learned.
#[test]
fn qa1_push_dialogue_persists_extracted_facts() {
    let mut h = Harness::start();
    h.call(
        "push_dialogue",
        json!({"tenant_id": "t1", "session_id": "s1", "new_message": "I moved to Berlin"}),
    )
    .assert_ok();

    let expected = extracted_fact_text("I moved to Berlin");
    let stored = common::eventually(EVENTUALLY, || h.stm_facts().contains(&expected));
    assert!(
        stored,
        "QA-1: extracted fact not persisted; stm_list = {:?}",
        h.stm_facts()
    );
    let stats = h.call("memory_stats", json!({})).json();
    assert!(
        stats["stm_active_nodes"].as_i64().unwrap_or(0) >= 1,
        "QA-1: {stats}"
    );
}

/// QA-1: extraction relationships become entity nodes + edges.
#[test]
fn qa1_push_dialogue_persists_relationship_entities() {
    let mut h = Harness::start();
    h.call(
        "push_dialogue",
        json!({"tenant_id": "t1", "session_id": "s1",
               "new_message": "I work with Ivy [rel:Ivy]"}),
    )
    .assert_ok();
    let ok = common::eventually(EVENTUALLY, || {
        let facts = h.stm_facts();
        facts.contains(&extracted_fact_text("I work with Ivy [rel:Ivy]"))
            && facts.contains(&"Ivy".to_string())
    });
    assert!(
        ok,
        "QA-1: fact/entity missing; stm_list = {:?}",
        h.stm_facts()
    );
}

// --- QA-2 / QA-3: delete_tenant --------------------------------------------

/// QA-2: `delete_tenant` without `tenant_id` silently deleted the built-in
/// default tenant. A destructive tool must require an explicit target.
#[test]
fn qa2_delete_tenant_requires_tenant_id() {
    let mut h = Harness::start();
    // Stored in the default tenant (no tenant_id).
    h.call("store_memory", json!({"fact_text": "Default-tenant fact"}))
        .assert_ok();

    let res = h.call("delete_tenant", json!({}));
    assert!(
        res.is_error,
        "QA-2: delete_tenant without tenant_id must be rejected, got: {}",
        res.text
    );
    assert!(
        h.stm_facts().contains(&"Default-tenant fact".to_string()),
        "QA-2: default tenant data was deleted"
    );
}

/// QA-2 / SEC-07: a destructive delete also needs `confirm` equal to
/// `tenant_id`. A missing, empty, mismatched or wrong-typed `confirm` is
/// rejected with `isError` and deletes nothing.
#[test]
fn qa2_delete_tenant_requires_matching_confirm() {
    let mut h = Harness::start();
    h.call(
        "store_memory",
        json!({"tenant_id": "t-guard", "fact_text": "Guarded fact"}),
    )
    .assert_ok();

    for args in [
        json!({"tenant_id": "t-guard"}),
        json!({"tenant_id": "t-guard", "confirm": ""}),
        json!({"tenant_id": "t-guard", "confirm": "t-other"}),
        json!({"tenant_id": "t-guard", "confirm": "T-GUARD"}),
        json!({"tenant_id": "t-guard", "confirm": true}),
    ] {
        let res = h.call("delete_tenant", args.clone());
        assert!(
            res.is_error,
            "QA-2: delete_tenant accepted {args}: {}",
            res.text
        );
        assert_eq!(
            h.exported_facts("t-guard"),
            vec!["Guarded fact".to_string()],
            "QA-2: data deleted by rejected call {args}"
        );
        let q = h.call(
            "query_memory",
            json!({"tenant_id": "t-guard", "query": "Guarded"}),
        );
        q.assert_ok();
        assert!(
            q.text.contains("Guarded fact"),
            "QA-2: fact no longer queryable after rejected call {args}: {}",
            q.text
        );
    }

    // The matching confirm still works.
    h.call(
        "delete_tenant",
        json!({"tenant_id": "t-guard", "confirm": "t-guard"}),
    )
    .assert_ok();
    assert!(h.exported_facts("t-guard").is_empty());
}

/// QA-3: `delete_tenant` never deleted `edges`, so with foreign keys ON any
/// tenant owning an edge failed to delete (whole transaction rolled back).
#[test]
fn qa3_delete_tenant_with_edges_succeeds() {
    let mut h = Harness::start();
    h.call(
        "push_dialogue",
        json!({"tenant_id": "t-edges", "session_id": "s1",
               "new_message": "Jack lives in Oslo [rel:Oslo]"}),
    )
    .assert_ok();
    h.call(
        "store_memory",
        json!({"tenant_id": "t-keep", "fact_text": "Kim stays"}),
    )
    .assert_ok();

    // Precondition (needs QA-1), via the public surface: the extracted fact
    // is recallable and carries the RELATED_TO → Oslo edge as a connection.
    let fact = extracted_fact_text("Jack lives in Oslo [rel:Oslo]");
    let has_edge = common::eventually(EVENTUALLY, || {
        oslo_connections(&mut h, &fact)
            .iter()
            .any(|c| c["relation"] == "RELATED_TO" && c["entity"] == "Oslo")
    });
    assert!(
        has_edge,
        "QA-3 precondition (blocked by QA-1): no Oslo edge; connections = {:?}",
        oslo_connections(&mut h, &fact)
    );
    // Ground truth in the store (REV-8): an edge row exists before the delete.
    let edges_before = stm_edge_count(&h);
    assert!(edges_before >= 1, "QA-3 precondition: {edges_before} edges");

    let res = h.call(
        "delete_tenant",
        json!({"tenant_id": "t-edges", "confirm": "t-edges"}),
    );
    assert!(!res.is_error, "QA-3: delete failed: {}", res.text);
    assert!(
        h.exported_facts("t-edges").is_empty(),
        "QA-3: tenant data survived delete"
    );
    assert!(
        oslo_connections(&mut h, &fact).is_empty(),
        "QA-3: connections survived delete"
    );
    // Only t-edges owned edges (store_memory creates none), so none may remain.
    assert_eq!(stm_edge_count(&h), 0, "QA-3: orphan edges left behind");
    assert_eq!(h.exported_facts("t-keep"), vec!["Kim stays".to_string()]);
}

/// Connections on the `query_memory` hit whose fact is `fact` (tenant t-edges).
fn oslo_connections(h: &mut Harness, fact: &str) -> Vec<serde_json::Value> {
    let res = h.call(
        "query_memory",
        json!({"tenant_id": "t-edges", "query": "Jack Oslo"}),
    );
    if res.is_error {
        return Vec::new();
    }
    res.json()
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r["fact"] == fact)
        .flat_map(|r| r["connections"].as_array().cloned().unwrap_or_default())
        .collect()
}

/// Rows in the STM `edges` table, read-only (safe alongside the live server
/// under WAL). The public surface can show an edge exists but cannot prove
/// that none are left after a delete, so this is the ground truth.
fn stm_edge_count(h: &Harness) -> i64 {
    let conn = rusqlite::Connection::open_with_flags(
        h.path().join("stm.sqlite"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .expect("open stm.sqlite read-only");
    conn.query_row("SELECT count(*) FROM edges", [], |r| r.get(0))
        .expect("count edges")
}

// --- QA-9: input validation --------------------------------------------------

/// QA-9: an empty or missing `fact_text` was stored as an empty fact.
#[test]
fn qa9_store_memory_rejects_empty_fact() {
    let mut h = Harness::start();
    for args in [
        json!({"tenant_id": "t1", "fact_text": ""}),
        json!({"tenant_id": "t1", "fact_text": "   "}),
        json!({"tenant_id": "t1"}),
    ] {
        let res = h.call("store_memory", args.clone());
        assert!(res.is_error, "QA-9: accepted {args}: {}", res.text);
    }
    assert!(h.stm_facts().is_empty(), "QA-9: {:?}", h.stm_facts());
}

/// QA-9: an empty query leaked a sqlite-vec SQL error instead of a clean
/// validation error.
#[test]
fn qa9_query_memory_rejects_empty_query_cleanly() {
    let mut h = Harness::start();
    for args in [
        json!({"tenant_id": "t1", "query": ""}),
        json!({"tenant_id": "t1"}),
    ] {
        let res = h.call("query_memory", args.clone());
        assert!(res.is_error, "QA-9: accepted {args}: {}", res.text);
        let lower = res.text.to_lowercase();
        assert!(
            !lower.contains("vec0") && !lower.contains("order by") && !lower.contains("sql"),
            "QA-9: leaks SQL internals: {}",
            res.text
        );
    }
}

/// QA-9: `push_dialogue` without its required arguments stored an empty turn.
#[test]
fn qa9_push_dialogue_requires_session_and_message() {
    let mut h = Harness::start();
    for args in [
        json!({}),
        json!({"session_id": "s1"}),
        json!({"session_id": "s1", "new_message": ""}),
        json!({"new_message": "hello"}),
    ] {
        let res = h.call("push_dialogue", args.clone());
        assert!(res.is_error, "QA-9: accepted {args}: {}", res.text);
    }
}

// --- QA-12: input size caps --------------------------------------------------

/// QA-12: a multi-megabyte fact was accepted, stored, and echoed in every
/// recall. Oversized input must be rejected.
#[test]
fn qa12_store_memory_rejects_oversized_fact() {
    let mut h = Harness::start();
    let huge = "x".repeat(2 * 1024 * 1024);
    let res = h.call(
        "store_memory",
        json!({"tenant_id": "t1", "fact_text": huge}),
    );
    assert!(res.is_error, "QA-12: 2 MiB fact accepted");
    assert!(h.stm_facts().is_empty());
}

// --- QA-10: missing API key ----------------------------------------------------

/// QA-10: with no API key configured the binary sent the literal `dummy_key`
/// to the provider. It must fail fast (startup error) or return a clear
/// "key missing" tool error, and never send a placeholder credential.
#[test]
fn qa10_missing_api_key_never_sends_placeholder() {
    let llm = FakeLlm::start();
    let dir = tempfile::tempdir().unwrap();
    write_config(dir.path(), &llm.base_url());
    let mut server = McpProcess::spawn(dir.path(), &[]);

    let exited = server.wait_exit(Duration::from_millis(500));
    let failed_clearly = match exited {
        Some(code) => {
            assert_ne!(code, 0, "QA-10: exited 0 without a key");
            server.stderr().to_lowercase().contains("key")
        }
        None => {
            server.initialize();
            let res = server.call_tool("store_memory", json!({"fact_text": "needs a key"}));
            res.is_error && res.text.to_lowercase().contains("key")
        }
    };

    let placeholder_sent = llm.requests().iter().any(|r| {
        r.authorization
            .as_deref()
            .is_some_and(|a| a.contains("dummy_key"))
    });
    assert!(!placeholder_sent, "QA-10: placeholder key sent to the LLM");
    assert!(
        failed_clearly,
        "QA-10: missing key not reported clearly (stderr: {})",
        server.stderr()
    );
}

/// Sanity for QA-10: with a key configured, it is what reaches the provider.
#[test]
fn qa10_configured_api_key_is_sent() {
    let mut h = Harness::start();
    h.call(
        "store_memory",
        json!({"tenant_id": "t1", "fact_text": "Lena"}),
    )
    .assert_ok();
    let auths: Vec<_> = h
        .llm
        .requests()
        .iter()
        .filter_map(|r| r.authorization.clone())
        .collect();
    assert!(!auths.is_empty());
    assert!(
        auths.iter().all(|a| a == &format!("Bearer {API_KEY}")),
        "{auths:?}"
    );
}

// --- QA-7: concurrent processes on one store ---------------------------------

/// QA-7: several processes sharing one store crashed at startup with
/// "database is locked" (`busy_timeout` was set after `journal_mode=WAL`) and
/// individual writes failed under contention.
/// Runs several rounds on fresh stores because the startup race is timing-dependent.
#[test]
fn qa7_concurrent_processes_share_one_store() {
    const ROUNDS: usize = 3;
    for round in 0..ROUNDS {
        qa7_round(round);
    }
}

fn qa7_round(round: usize) {
    const PROCS: usize = 4;
    const WRITES: usize = 15;
    let llm = FakeLlm::start();
    let dir = tempfile::tempdir().unwrap();
    write_config(dir.path(), &llm.base_url());

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(PROCS));
    let workers: Vec<_> = (0..PROCS)
        .map(|w| {
            let path = dir.path().to_path_buf();
            let barrier = barrier.clone();
            std::thread::spawn(move || -> Result<(), String> {
                barrier.wait();
                let mut server = McpProcess::spawn_with_key(&path);
                if let Some(code) = server.wait_exit(Duration::from_millis(300)) {
                    return Err(format!(
                        "round {round} worker {w} exited at startup ({code:?}): {}",
                        server.stderr()
                    ));
                }
                server.initialize();
                for i in 0..WRITES {
                    let res = server.call_tool(
                        "store_memory",
                        json!({"tenant_id": "t1", "fact_text": format!("worker {w} fact {i}")}),
                    );
                    if res.is_error {
                        return Err(format!("round {round} worker {w} write {i}: {}", res.text));
                    }
                }
                Ok(())
            })
        })
        .collect();

    let errors: Vec<String> = workers
        .into_iter()
        .filter_map(|t| t.join().unwrap().err())
        .collect();
    assert!(errors.is_empty(), "QA-7: {errors:#?}");

    let mut check = McpProcess::spawn_with_key(dir.path());
    check.initialize();
    let stats = check.call_tool("memory_stats", json!({})).json();
    assert_eq!(
        stats["stm_active_nodes"],
        json!(PROCS * WRITES),
        "QA-7: round {round} lost writes: {stats}"
    );
}

// --- REV-1: push_dialogue learning errors ------------------------------------

/// REV-1: when fact extraction fails after the turn was archived,
/// `push_dialogue` succeeds (isError=false), still returns the context window,
/// and reports the failure in `learning_error`. Nothing is learned from the turn.
#[test]
fn rev1_push_dialogue_reports_learning_error_but_succeeds() {
    let mut h = Harness::start();
    let msg = format!("Remember the vault code {MALFORMED_EXTRACTION}");
    let res = h.call(
        "push_dialogue",
        json!({"tenant_id": "t1", "session_id": "s-rev1", "new_message": msg}),
    );
    assert!(
        !res.is_error,
        "REV-1: extraction failure must not fail the call: {}",
        res.text
    );
    let ctx = res.json();
    let err = ctx["learning_error"]
        .as_str()
        .unwrap_or_else(|| panic!("REV-1: learning_error missing: {ctx}"));
    assert!(!err.trim().is_empty(), "REV-1: empty learning_error");
    assert!(
        ctx["recent_messages"]
            .as_array()
            .is_some_and(|m| m.iter().any(|x| x == msg.as_str())),
        "REV-1: context window lacks the turn: {ctx}"
    );
    assert!(ctx["relevant_facts"].is_array(), "{ctx}");
    assert!(
        h.stm_facts().is_empty(),
        "REV-1: facts stored from a failed extraction: {:?}",
        h.stm_facts()
    );

    // The turn was archived: it is still in the session on the next push,
    // and that successful push carries no learning_error.
    let next = h.call(
        "push_dialogue",
        json!({"tenant_id": "t1", "session_id": "s-rev1", "new_message": "next turn"}),
    );
    next.assert_ok();
    let next_ctx = next.json();
    assert!(
        next_ctx["recent_messages"]
            .as_array()
            .is_some_and(|m| m.iter().any(|x| x == msg.as_str())),
        "REV-1: archived turn missing from session: {next_ctx}"
    );
    assert!(
        next_ctx.get("learning_error").is_none(),
        "REV-1: learning_error on success: {next_ctx}"
    );
}

/// REV-1: on a fully successful push, the optional `learning_error` and
/// `warnings` keys are omitted entirely.
#[test]
fn rev1_successful_push_dialogue_has_no_error_keys() {
    let mut h = Harness::start();
    let res = h.call(
        "push_dialogue",
        json!({"tenant_id": "t1", "session_id": "s1", "new_message": "All good here"}),
    );
    res.assert_ok();
    let ctx = res.json();
    assert!(ctx.get("learning_error").is_none(), "REV-1: {ctx}");
    assert!(ctx.get("warnings").is_none(), "REV-1: {ctx}");
    assert!(
        h.stm_facts()
            .contains(&extracted_fact_text("All good here")),
        "{:?}",
        h.stm_facts()
    );
}

/// REV-1: at most 32 facts are accepted from one extraction.
#[test]
fn rev1_extraction_is_capped_at_32_facts() {
    let mut h = Harness::start();
    h.call(
        "push_dialogue",
        json!({"tenant_id": "t1", "session_id": "s1", "new_message": "dump [many-facts:40]"}),
    )
    .assert_ok();
    let bulk = h
        .stm_facts()
        .into_iter()
        .filter(|f| f.starts_with("Bulk fact "))
        .count();
    assert!(bulk > 0, "no bulk facts stored at all");
    assert!(
        bulk <= 32,
        "REV-1: {bulk} facts accepted from one extraction"
    );
}
