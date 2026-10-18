//! Phase 1 regressions (review findings DEV-2/QA-3, DEV-8, ARC-17, ARC-2).

use super::*;
use crate::domain::models::SessionId;
use crate::infrastructure::database::init_db;
use crate::infrastructure::embedding_codec::as_blob;
use crate::infrastructure::schema::init_schema;
use rusqlite::params;
use serde_json::json;

fn repo() -> SqliteMemoryRepository {
    let conn = init_db(None as Option<&String>).unwrap();
    init_schema(&conn, 4).unwrap();
    SqliteMemoryRepository::new(conn)
}

fn node(tenant: &str, payload: serde_json::Value, status: &str) -> MemoryNode {
    MemoryNode {
        id: None,
        tenant_id: TenantId(tenant.into()),
        source_episode_id: None,
        payload,
        status: status.into(),
        ccl: "reality".into(),
        is_explicit: false,
        support_count: 1,
        relevance_score: 1.0,
        context_key: None,
    }
}

fn fact(tenant: &str, text: &str) -> MemoryNode {
    node(tenant, json!({ "fact": text }), "active")
}

fn count(repo: &SqliteMemoryRepository, sql: &str) -> i64 {
    repo.conn.query_row(sql, [], |r| r.get(0)).unwrap()
}

fn edge(source_id: i64, target_id: i64) -> Edge {
    Edge {
        source_id,
        target_id,
        relation: "knows".into(),
        ccl: "reality".into(),
        valid_from: None,
        valid_until: None,
        weight: 1.0,
    }
}

fn search(repo: &SqliteMemoryRepository, tenant: &str, k: usize) -> Vec<MemoryResult> {
    repo.query_with_graph(
        "zzz",
        &[1.0, 0.0, 0.0, 0.0],
        &TenantId(tenant.into()),
        &TimeFilter::default(),
        &["reality".to_string()],
        k,
    )
    .unwrap()
}

/// DEV-2/QA-3: a tenant with edges (both directions, incl. an edge whose
/// source is another tenant's node pointing in) is fully erased — edges,
/// vectors, nodes, episodes, CCL registry — and nothing of tenant B is lost.
#[test]
fn delete_tenant_with_edges_removes_everything_and_spares_others() {
    let r = repo();
    let ep = r
        .store_episode(&Episode {
            id: None,
            tenant_id: TenantId("A".into()),
            session_id: SessionId("s".into()),
            raw_dialogue: "hi".into(),
            ccl: "reality".into(),
            created_at: None,
        })
        .unwrap();
    let mut a1 = fact("A", "alice");
    a1.source_episode_id = Some(ep);
    let a1 = r.store_node(&a1, &[0.1, 0.0, 0.0, 0.0]).unwrap();
    let a2 = r
        .store_node(&fact("A", "bob"), &[0.2, 0.0, 0.0, 0.0])
        .unwrap();
    let b1 = r
        .store_node(&fact("B", "carol"), &[0.3, 0.0, 0.0, 0.0])
        .unwrap();
    let b2 = r
        .store_node(&fact("B", "dave"), &[0.4, 0.0, 0.0, 0.0])
        .unwrap();
    r.store_edge(&edge(a1, a2)).unwrap(); // A -> A
    r.store_edge(&edge(a2, a1)).unwrap(); // reverse direction
    r.store_edge(&edge(b1, b2)).unwrap(); // B -> B (must survive)
    for t in ["A", "B"] {
        r.store_ccl_definition(&CclDefinition {
            id: None,
            tenant_id: TenantId(t.into()),
            name: "dream".into(),
            description: "d".into(),
        })
        .unwrap();
    }

    r.delete_tenant(&TenantId("A".into()))
        .expect("delete_tenant must succeed with edges present");

    assert_eq!(
        count(&r, "SELECT COUNT(*) FROM nodes WHERE tenant_id='A'"),
        0
    );
    assert_eq!(
        count(&r, "SELECT COUNT(*) FROM episodes WHERE tenant_id='A'"),
        0
    );
    assert_eq!(
        count(&r, "SELECT COUNT(*) FROM ccl_registry WHERE tenant_id='A'"),
        0
    );
    assert_eq!(
        count(
            &r,
            &format!("SELECT COUNT(*) FROM vec_nodes WHERE node_id IN ({a1},{a2})")
        ),
        0
    );
    // Tenant B is intact.
    assert_eq!(
        count(&r, "SELECT COUNT(*) FROM nodes WHERE tenant_id='B'"),
        2
    );
    assert_eq!(count(&r, "SELECT COUNT(*) FROM edges"), 1);
    assert_eq!(
        count(&r, "SELECT COUNT(*) FROM ccl_registry WHERE tenant_id='B'"),
        1
    );
}

/// DEV-8: `k` above 5 is honoured (the old hard-coded `LIMIT 5` capped it).
#[test]
fn query_with_graph_honours_k_above_five() {
    let r = repo();
    for i in 0..10 {
        r.store_node(
            &fact("A", &format!("fact{i}")),
            &[1.0, i as f32 * 0.01, 0.0, 0.0],
        )
        .unwrap();
    }
    assert_eq!(search(&r, "A", 10).len(), 10);
    assert_eq!(search(&r, "A", 3).len(), 3);
}

/// ARC-17: zero-vector graph anchors are not indexed, so a crowd of them
/// can't push the real (farther) match out of the KNN top-k.
#[test]
fn zero_vector_anchors_do_not_starve_recall() {
    let r = repo();
    for i in 0..12 {
        r.store_node(
            &node(
                "A",
                json!({ "fact": format!("anchor{i}"), "kind": "subject" }),
                "active",
            ),
            &[0.0; 4],
        )
        .unwrap();
    }
    r.store_node(&fact("A", "the real fact"), &[0.0, 1.0, 0.0, 0.0])
        .unwrap();

    assert_eq!(count(&r, "SELECT COUNT(*) FROM vec_nodes"), 1);
    let out = search(&r, "A", 5);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].fact, "the real fact");
}

/// ARC-17: archived nodes are never indexed, and a sweep that archives a
/// node drops its vector — so archived rows can't consume top-k slots.
#[test]
fn archived_nodes_leave_the_vector_index() {
    let r = repo();
    for i in 0..12 {
        r.store_node(
            &node("A", json!({ "fact": format!("old{i}") }), "archived"),
            &[1.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
    }
    let live = r
        .store_node(&fact("A", "live fact"), &[0.9, 0.3, 0.0, 0.0])
        .unwrap();
    assert_eq!(count(&r, "SELECT COUNT(*) FROM vec_nodes"), 1);
    assert_eq!(search(&r, "A", 5).len(), 1);

    // A sweep that archives the live node removes its vector too.
    r.conn
        .execute(
            "UPDATE nodes SET relevance_score = 0.05 WHERE id = ?1",
            params![live],
        )
        .unwrap();
    r.sweep_decay(&crate::domain::decay::DecayEngine::new(7.0))
        .unwrap();
    assert_eq!(count(&r, "SELECT COUNT(*) FROM vec_nodes"), 0);
}

/// ARC-17: re-initialising the schema purges vectors that older stores kept
/// for archived nodes and `subject` anchors.
#[test]
fn init_schema_purges_legacy_unindexable_vectors() {
    let r = repo();
    let anchor = r
        .store_node(
            &node(
                "A",
                json!({ "fact": "anchor", "kind": "subject" }),
                "active",
            ),
            &[0.0; 4],
        )
        .unwrap();
    let old = r
        .store_node(&fact("A", "old"), &[1.0, 0.0, 0.0, 0.0])
        .unwrap();
    let keep = r
        .store_node(&fact("A", "keep"), &[0.0, 1.0, 0.0, 0.0])
        .unwrap();
    // Simulate a legacy store: anchor vector present, `old` archived with vector.
    r.conn
        .execute(
            "INSERT INTO vec_nodes(node_id, embedding) VALUES (?1, ?2)",
            params![anchor, as_blob(&[0.0; 4])],
        )
        .unwrap();
    r.conn
        .execute(
            "UPDATE nodes SET status = 'archived' WHERE id = ?1",
            params![old],
        )
        .unwrap();
    assert_eq!(count(&r, "SELECT COUNT(*) FROM vec_nodes"), 3);

    init_schema(&r.conn, 4).unwrap();

    assert_eq!(
        count(&r, "SELECT COUNT(*) FROM vec_nodes"),
        1,
        "only the active, embedded node stays indexed"
    );
    assert_eq!(
        count(
            &r,
            &format!("SELECT COUNT(*) FROM vec_nodes WHERE node_id = {keep}")
        ),
        1
    );
}

/// ARC-2/DEV-10: `update_node_content` replaces the vector with the payload,
/// so the node is found by its new meaning and no longer by its old one.
#[test]
fn update_node_content_reembeds() {
    let r = repo();
    let id = r
        .store_node(&fact("A", "old text"), &[1.0, 0.0, 0.0, 0.0])
        .unwrap();
    r.update_node_content(id, &json!({ "fact": "new text" }), &[0.0, 0.0, 1.0, 0.0])
        .unwrap();

    let near_new = r
        .find_similar_nodes(&[0.0, 0.0, 1.0, 0.0], &TenantId("A".into()), 0.01, 1)
        .unwrap();
    assert_eq!(near_new.len(), 1);
    assert_eq!(near_new[0].0.payload["fact"], "new text");
    let near_old = r
        .find_similar_nodes(&[1.0, 0.0, 0.0, 0.0], &TenantId("A".into()), 0.01, 1)
        .unwrap();
    assert!(near_old.is_empty(), "the stale vector must be gone");
}

/// DEV-4 mitigation: `find_similar_nodes` over-fetches, so nearer nodes of
/// another tenant don't hide this tenant's match.
#[test]
fn find_similar_nodes_survives_other_tenant_crowding() {
    let r = repo();
    for i in 0..3 {
        r.store_node(&fact("B", &format!("b{i}")), &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
    }
    r.store_node(&fact("A", "mine"), &[0.9, 0.1, 0.0, 0.0])
        .unwrap();
    let hits = r
        .find_similar_nodes(&[1.0, 0.0, 0.0, 0.0], &TenantId("A".into()), 2.0, 1)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0.payload["fact"], "mine");
}

/// A query with no keyword terms (only punctuation) runs vector-only recall:
/// the FTS leg is left out of the hybrid query. It must still return the
/// nearest facts, not fail inside sqlite-vec.
#[test]
fn query_without_keyword_terms_runs_vector_only() {
    let r = repo();
    r.store_node(&fact("A", "the real fact"), &[1.0, 0.0, 0.0, 0.0])
        .unwrap();
    let out = r
        .query_with_graph(
            "?! -- ()",
            &[1.0, 0.0, 0.0, 0.0],
            &TenantId("A".into()),
            &TimeFilter::default(),
            &["reality".to_string()],
            5,
        )
        .expect("vector-only recall must not error");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].fact, "the real fact");
    let hits = r
        .hybrid_search("?!", &[1.0, 0.0, 0.0, 0.0], &TenantId("A".into()), 5)
        .expect("vector-only hybrid search must not error");
    assert_eq!(hits.len(), 1);
}

/// DEV-19: the sweep reads each batch inside its own write transaction. A
/// boost committed by another connection while the sweep waits for the write
/// lock must survive: the old sweep read every score up front, then
/// overwrote the boosted row with the stale, decayed value.
#[test]
fn sweep_does_not_overwrite_a_concurrent_boost() {
    use crate::application::migrations::{StoreKind, migrate_store};
    use crate::domain::decay::DecayEngine;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stm.sqlite");
    let conn = init_db(Some(&path)).unwrap();
    migrate_store(&conn, StoreKind::Stm, None, 4).unwrap();
    let sweeper = SqliteMemoryRepository::new(conn);
    let id = sweeper
        .store_node(&fact("A", "old"), &[1.0, 0.0, 0.0, 0.0])
        .unwrap();
    // Aged two days at a 1-day half-life: a sweep alone would decay it to 0.25.
    sweeper
        .conn
        .execute(
            "UPDATE nodes SET last_decayed_at = datetime('now', '-2 days') WHERE id = ?1",
            params![id],
        )
        .unwrap();

    // Another connection takes the write lock and boosts the node, then
    // commits only after the sweep has started waiting for the lock.
    let other = init_db(Some(&path)).unwrap();
    other.execute_batch("BEGIN IMMEDIATE").unwrap();
    other
        .execute(
            "UPDATE nodes SET relevance_score = 1.0, last_decayed_at = CURRENT_TIMESTAMP WHERE id = ?1",
            params![id],
        )
        .unwrap();
    let sweep = std::thread::spawn(move || {
        sweeper.sweep_decay(&DecayEngine::new(1.0)).unwrap();
        sweeper
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    other.execute_batch("COMMIT").unwrap();
    let sweeper = sweep.join().unwrap();

    let score: f64 = sweeper
        .conn
        .query_row(
            "SELECT relevance_score FROM nodes WHERE id = ?1",
            params![id],
            |r| r.get(0),
        )
        .unwrap();
    assert!(score > 0.99, "the concurrent boost was lost: score {score}");
}

/// The sweep runs in batches (one transaction each); every active node is
/// decayed exactly once across batch boundaries, and nodes archived in an
/// early batch leave the vector index.
#[test]
fn sweep_covers_every_node_across_batches() {
    use super::lifecycle::SWEEP_BATCH;
    use crate::domain::decay::DecayEngine;

    let r = repo();
    let total = SWEEP_BATCH * 2 + 7;
    for i in 0..total {
        // Every third node starts faint enough to archive after one half-life.
        let mut n = fact("A", &format!("f{i}"));
        n.relevance_score = if i % 3 == 0 { 0.15 } else { 1.0 };
        r.store_node(&n, &[1.0, i as f32, 0.0, 0.0]).unwrap();
    }
    r.conn
        .execute(
            "UPDATE nodes SET last_decayed_at = datetime('now', '-1 day')",
            [],
        )
        .unwrap();

    r.sweep_decay(&DecayEngine::new(1.0)).unwrap();

    let faint = total.div_ceil(3) as i64;
    assert_eq!(
        count(&r, "SELECT COUNT(*) FROM nodes WHERE status = 'archived'"),
        faint
    );
    assert_eq!(
        count(
            &r,
            "SELECT COUNT(*) FROM nodes WHERE status = 'active' AND abs(relevance_score - 0.5) < 0.001"
        ),
        total as i64 - faint,
        "each surviving node decayed exactly once (1.0 -> 0.5)"
    );
    assert_eq!(
        count(&r, "SELECT COUNT(*) FROM vec_nodes"),
        total as i64 - faint
    );
}

/// W2R-4: the sweep opens its batch transactions as IMMEDIATE itself, so
/// DEV-19 holds even on a connection whose default behaviour is DEFERRED
/// (e.g. one opened differently by a future store thread). A deferred batch
/// would read a stale snapshot first and then fail or overwrite on upgrade.
#[test]
fn sweep_is_immediate_even_on_a_deferred_connection() {
    use crate::application::migrations::{StoreKind, migrate_store};
    use crate::domain::decay::DecayEngine;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stm.sqlite");
    let mut conn = init_db(Some(&path)).unwrap();
    migrate_store(&conn, StoreKind::Stm, None, 4).unwrap();
    conn.set_transaction_behavior(rusqlite::TransactionBehavior::Deferred);
    let sweeper = SqliteMemoryRepository::new(conn);
    let id = sweeper
        .store_node(&fact("A", "old"), &[1.0, 0.0, 0.0, 0.0])
        .unwrap();
    sweeper
        .conn
        .execute(
            "UPDATE nodes SET last_decayed_at = datetime('now', '-2 days') WHERE id = ?1",
            params![id],
        )
        .unwrap();

    let other = init_db(Some(&path)).unwrap();
    other.execute_batch("BEGIN IMMEDIATE").unwrap();
    other
        .execute(
            "UPDATE nodes SET relevance_score = 1.0, last_decayed_at = CURRENT_TIMESTAMP WHERE id = ?1",
            params![id],
        )
        .unwrap();
    let sweep = std::thread::spawn(move || {
        let swept = sweeper.sweep_decay(&DecayEngine::new(1.0));
        (sweeper, swept)
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    other.execute_batch("COMMIT").unwrap();
    let (sweeper, swept) = sweep.join().unwrap();
    swept.expect("the sweep waits for the lock instead of failing");

    let score: f64 = sweeper
        .conn
        .query_row(
            "SELECT relevance_score FROM nodes WHERE id = ?1",
            params![id],
            |r| r.get(0),
        )
        .unwrap();
    assert!(score > 0.99, "the concurrent boost was lost: score {score}");
}
