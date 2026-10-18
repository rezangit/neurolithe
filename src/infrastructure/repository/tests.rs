//! Repository behaviour: storage, search ranking, context recall, the decay
//! sweep, read reinforcement and soft reset.

use super::*;
use crate::domain::models::SessionId;
use crate::infrastructure::database::init_db;
use crate::infrastructure::schema::init_schema;
use rusqlite::params;
use serde_json::json;

/// A punctuation-heavy query must not error the hybrid search, and a query
/// lifted verbatim from a stored fact must find it (the field-report scenario).
#[test]
fn test_query_with_graph_is_punctuation_safe_and_finds_verbatim() {
    let repo = setup_mem_repo();
    let t = TenantId("legacy".into());
    let node = MemoryNode {
        id: None,
        tenant_id: t.clone(),
        source_episode_id: None,
        payload: json!({"fact": "Maple Court Annual Budget Report (Cedar Advisors, 2022)"}),
        status: "active".into(),
        ccl: "reality".into(),
        is_explicit: true,
        support_count: 1,
        relevance_score: 0.99,
        context_key: None,
    };
    repo.store_node(&node, &vec![0.1f32; 1536]).unwrap();

    // A punctuation-laden query must not error, and must match on keywords.
    let out = repo
        .query_with_graph(
            "budget report: Cedar-Advisors (2022)",
            &vec![0.9f32; 1536],
            &t,
            &TimeFilter::default(),
            &["reality".to_string()],
            10,
        )
        .unwrap();
    assert!(
        out.iter().any(|r| r.fact.contains("Budget Report")),
        "keyword leg should surface the verbatim fact"
    );
    // The archive reference is surfaced so the hit can be traced/fetched.
    let hit = out
        .iter()
        .find(|r| r.fact.contains("Budget Report"))
        .unwrap();
    assert_eq!(hit.data_id.as_deref(), None, "no dataId in this fixture");
}

/// Results rank by SEARCH quality, not decay score: a stronger keyword match
/// outranks a weaker one even when the weaker one was reinforced to a higher
/// relevance_score. Also verifies `data_id` is surfaced from the payload.
#[test]
fn test_ranking_prefers_match_quality_and_surfaces_data_id() {
    let repo = setup_mem_repo();
    let t = TenantId("legacy".into());

    // Strong match: contains both query terms; modest relevance.
    let strong = MemoryNode {
        id: None,
        tenant_id: t.clone(),
        source_episode_id: None,
        payload: json!({"fact": "Annual budget reserve fund", "dataId": "doc_dep"}),
        status: "active".into(),
        ccl: "reality".into(),
        is_explicit: true,
        support_count: 1,
        relevance_score: 0.5,
        context_key: None,
    };
    // Weak match (no query terms) but reinforced to the top relevance score.
    let weak = MemoryNode {
        id: None,
        tenant_id: t.clone(),
        source_episode_id: None,
        payload: json!({"fact": "Newsletter holiday hours", "dataId": "doc_news"}),
        status: "active".into(),
        ccl: "reality".into(),
        is_explicit: true,
        support_count: 1,
        relevance_score: 1.0,
        context_key: None,
    };
    // Near-identical embeddings so the keyword leg — not vector distance —
    // decides the order.
    repo.store_node(&strong, &vec![0.5f32; 1536]).unwrap();
    repo.store_node(&weak, &vec![0.5f32; 1536]).unwrap();

    let out = repo
        .query_with_graph(
            "budget reserve",
            &vec![0.5f32; 1536],
            &t,
            &TimeFilter::default(),
            &["reality".to_string()],
            10,
        )
        .unwrap();

    assert_eq!(
        out[0].fact, "Annual budget reserve fund",
        "the stronger keyword match ranks first despite lower relevance_score"
    );
    assert_eq!(
        out[0].data_id.as_deref(),
        Some("doc_dep"),
        "the archive reference is surfaced on the hit"
    );
}

fn setup_mem_repo() -> Box<dyn MemoryRepository> {
    let conn = init_db(None as Option<&String>).unwrap();
    init_schema(&conn, 1536).unwrap();
    Box::new(SqliteMemoryRepository::new(conn))
}

#[test]
fn test_store_episode_and_node() {
    let repo = setup_mem_repo();

    let episode = Episode {
        id: None,
        tenant_id: TenantId("test-tenant".into()),
        session_id: SessionId("session-1".into()),
        raw_dialogue: "I live in Berlin".into(),
        ccl: "reality".into(),
        created_at: None,
    };

    let episode_id = repo.store_episode(&episode).unwrap();
    assert!(episode_id > 0);

    let node = MemoryNode {
        id: None,
        tenant_id: TenantId("test-tenant".into()),
        source_episode_id: Some(episode_id),
        payload: json!({"fact": "User lives in Berlin"}),
        status: "active".into(),
        ccl: "reality".into(),
        is_explicit: true,
        support_count: 1,
        relevance_score: 1.0,
        context_key: None,
    };

    let dummy_embedding = vec![0.1f32; 1536]; // fake 1536d vector
    let node_id = repo.store_node(&node, &dummy_embedding).unwrap();
    assert!(node_id > 0);

    // Since repo is now a Box<dyn MemoryRepository>, we cannot directly access repo.conn.
    // For testing the internal state (trigger effects), we need another connection or a specialized test method.
    // For now, testing the repository's returned behavior is sufficient since FTS trigger setups are tested
    // separately in schema tests.
}

/// A node's `context_key` round-trips through store + read (slice 1). A
/// situational note carries its key; a feeder/knowledge-path node leaves it
/// NULL. Reads (here via `find_similar_nodes`) surface the key faithfully.
#[test]
fn test_context_key_round_trips() {
    let repo = setup_mem_repo();
    let tenant = TenantId("ctx-tenant".into());

    // Situational note: distinct embedding + a context key.
    let mut emb_a = vec![0.0f32; 1536];
    emb_a[0] = 1.0;
    let noted = MemoryNode {
        id: None,
        tenant_id: tenant.clone(),
        source_episode_id: None,
        payload: json!({"fact": "found report = doc_1"}),
        status: "active".into(),
        ccl: "working".into(),
        is_explicit: true,
        support_count: 1,
        relevance_score: 1.0,
        context_key: Some("chat.jid:123".into()),
    };
    repo.store_node(&noted, &emb_a).unwrap();

    // Feeder-style knowledge node: different embedding, no context key.
    let mut emb_b = vec![0.0f32; 1536];
    emb_b[1] = 1.0;
    let knowledge = MemoryNode {
        id: None,
        tenant_id: tenant.clone(),
        source_episode_id: None,
        payload: json!({"fact": "invoice total 42"}),
        status: "active".into(),
        ccl: "reality".into(),
        is_explicit: true,
        support_count: 1,
        relevance_score: 1.0,
        context_key: None,
    };
    repo.store_node(&knowledge, &emb_b).unwrap();

    // Reading nearest to emb_a returns the note with its key preserved.
    let near_a = repo.find_similar_nodes(&emb_a, &tenant, 2.0, 1).unwrap();
    assert_eq!(near_a.len(), 1);
    assert_eq!(near_a[0].0.context_key.as_deref(), Some("chat.jid:123"));

    // Reading nearest to emb_b returns the knowledge node with NULL key.
    let near_b = repo.find_similar_nodes(&emb_b, &tenant, 2.0, 1).unwrap();
    assert_eq!(near_b.len(), 1);
    assert_eq!(near_b[0].0.context_key, None);
}

#[test]
fn test_hybrid_search() {
    let repo = setup_mem_repo();

    // Let's create an episode
    let episode = Episode {
        id: None,
        tenant_id: TenantId("tenant-X".into()),
        session_id: SessionId("session-1".into()),
        raw_dialogue: "I have a dog named Rust.".into(),
        ccl: "reality".into(),
        created_at: None,
    };
    let ep_id = repo.store_episode(&episode).unwrap();

    // Node 1: Contains the keyword "dog" explicitly
    let node1 = MemoryNode {
        id: None,
        tenant_id: TenantId("tenant-X".into()),
        source_episode_id: Some(ep_id),
        payload: json!({"fact": "User owns a dog"}),
        status: "active".into(),
        ccl: "reality".into(),
        is_explicit: true,
        support_count: 1,
        relevance_score: 1.0,
        context_key: None,
    };
    // Node 2: Contains another keyword but vector might be close
    let node2 = MemoryNode {
        id: None,
        tenant_id: TenantId("tenant-X".into()),
        source_episode_id: Some(ep_id),
        payload: json!({"fact": "User is a programmer"}),
        status: "active".into(),
        ccl: "reality".into(),
        is_explicit: true,
        support_count: 1,
        relevance_score: 0.8,
        context_key: None,
    };

    let emb1 = vec![0.9f32; 1536];
    let emb2 = vec![0.1f32; 1536]; // different embedding

    repo.store_node(&node1, &emb1).unwrap();
    repo.store_node(&node2, &emb2).unwrap();

    // Query combining text and an embedding close to emb1
    let query_emb = vec![0.85f32; 1536];

    let results = repo
        .hybrid_search("dog", &query_emb, &TenantId("tenant-X".into()), 5)
        .unwrap();
    assert!(!results.is_empty());

    // Should rank node1 highest because it matches FTS "dog" AND vector distance is closer
    let ranked_top = &results[0];
    assert_eq!(
        ranked_top.payload.get("fact").unwrap().as_str().unwrap(),
        "User owns a dog"
    );
}

#[test]
fn test_tenant_isolation_delete_and_export() {
    let repo = setup_mem_repo();

    let t1 = TenantId("tenant-A".into());
    let t2 = TenantId("tenant-B".into());

    // Setup tenant A
    let ep1 = repo
        .store_episode(&Episode {
            id: None,
            tenant_id: t1.clone(),
            session_id: SessionId("s1".into()),
            raw_dialogue: "secret A".into(),
            ccl: "reality".into(),
            created_at: None,
        })
        .unwrap();

    repo.store_node(
        &MemoryNode {
            id: None,
            tenant_id: t1.clone(),
            source_episode_id: Some(ep1),
            payload: json!({"fact": "A fact"}),
            status: "active".into(),
            ccl: "reality".into(),
            is_explicit: false,
            support_count: 1,
            relevance_score: 1.0,
            context_key: None,
        },
        &vec![0.1; 1536],
    )
    .unwrap();

    // Setup tenant B
    let ep2 = repo
        .store_episode(&Episode {
            id: None,
            tenant_id: t2.clone(),
            session_id: SessionId("s2".into()),
            raw_dialogue: "secret B".into(),
            ccl: "reality".into(),
            created_at: None,
        })
        .unwrap();

    repo.store_node(
        &MemoryNode {
            id: None,
            tenant_id: t2.clone(),
            source_episode_id: Some(ep2),
            payload: json!({"fact": "B fact"}),
            status: "active".into(),
            ccl: "reality".into(),
            is_explicit: false,
            support_count: 1,
            relevance_score: 1.0,
            context_key: None,
        },
        &vec![0.2; 1536],
    )
    .unwrap();

    // Test Export
    let export_a = repo.export_tenant(&t1).unwrap();
    assert!(export_a.contains("A fact"));
    assert!(!export_a.contains("B fact"));

    // Test Deletion
    repo.delete_tenant(&t1).unwrap();

    let after_delete_a = repo.export_tenant(&t1).unwrap();
    assert!(!after_delete_a.contains("A fact"));

    let export_b = repo.export_tenant(&t2).unwrap();
    assert!(export_b.contains("B fact")); // B untouched
}

// --- Slice 2: decay sweep + boost + soft-reset (STM) ---

/// Concrete repo (small vector dim) so tests can read scores/counts back
/// off the private connection.
fn setup_concrete() -> SqliteMemoryRepository {
    let conn = init_db(None as Option<&String>).unwrap();
    init_schema(&conn, 4).unwrap();
    SqliteMemoryRepository::new(conn)
}

fn active_node(tenant: &TenantId, fact: &str, score: f64) -> MemoryNode {
    MemoryNode {
        id: None,
        tenant_id: tenant.clone(),
        source_episode_id: None,
        payload: json!({ "fact": fact }),
        status: "active".into(),
        ccl: "reality".into(),
        is_explicit: false,
        support_count: 1,
        relevance_score: score,
        context_key: None,
    }
}

fn score_status(repo: &SqliteMemoryRepository, id: i64) -> (f64, String) {
    repo.conn
        .query_row(
            "SELECT relevance_score, status FROM nodes WHERE id = ?1",
            params![id],
            |r| Ok((r.get::<_, f64>(0)?, r.get::<_, String>(1)?)),
        )
        .unwrap()
}

fn count(conn: &Connection, table: &str) -> i64 {
    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
        .unwrap()
}

/// Force a node's decay clock into the past so a sweep sees real elapsed time.
fn age_decay_clock(repo: &SqliteMemoryRepository, id: i64, spec: &str) {
    repo.conn
        .execute(
            "UPDATE nodes SET last_decayed_at = datetime('now', ?2) WHERE id = ?1",
            params![id, spec],
        )
        .unwrap();
}

/// A sweep decays each node by its *real* elapsed age. A node aged one
/// half-life halves; one below 0.1 flips to `archived`. A *fresh* node (no
/// elapsed time) is untouched — the property that stops a sweep/restart from
/// wiping just-written working notes.
#[test]
fn test_sweep_decays_by_real_elapsed_age() {
    use crate::domain::decay::DecayEngine;
    let repo = setup_concrete();
    let t = TenantId("t".into());

    // Aged one full day at half_life = 1 day -> factor 0.5.
    let healthy = repo
        .store_node(&active_node(&t, "healthy", 1.0), &[0.0; 4])
        .unwrap();
    age_decay_clock(&repo, healthy, "-1 day");
    // 0.15 aged one day -> 0.075 < 0.1 -> archived.
    let faint = repo
        .store_node(&active_node(&t, "faint", 0.15), &[0.1; 4])
        .unwrap();
    age_decay_clock(&repo, faint, "-1 day");
    // Fresh node (clock = now) must NOT decay on this sweep.
    let fresh = repo
        .store_node(&active_node(&t, "fresh", 1.0), &[0.2; 4])
        .unwrap();

    repo.sweep_decay(&DecayEngine::new(1.0)).unwrap();

    let (h_score, h_status) = score_status(&repo, healthy);
    assert!((h_score - 0.5).abs() < 1e-3, "healthy ~0.5, got {h_score}");
    assert_eq!(h_status, "active");

    let (f_score, f_status) = score_status(&repo, faint);
    assert!(f_score < 0.1, "faint decayed below threshold");
    assert_eq!(f_status, "archived");

    let (fr_score, fr_status) = score_status(&repo, fresh);
    assert!(fr_score > 0.99, "fresh node barely moves, got {fr_score}");
    assert_eq!(
        fr_status, "active",
        "a sweep never wipes a just-written note"
    );
}

/// Real-elapsed decay resolves the half-life per CCL: at the *same age* a
/// `working` note (30-min half-life) archives while a `reality` fact (7-day)
/// survives — the core of the working regime.
#[test]
fn test_sweep_resolves_half_life_per_ccl() {
    use crate::domain::decay::DecayEngine;
    let repo = setup_concrete();
    let t = TenantId("t".into());

    let reality = repo
        .store_node(&active_node(&t, "durable fact", 1.0), &[0.0; 4])
        .unwrap();
    age_decay_clock(&repo, reality, "-2 hours");

    let mut working_node = active_node(&t, "found report = doc_1", 1.0);
    working_node.ccl = "working".into();
    working_node.context_key = Some("chat.jid:1".into());
    let working = repo.store_node(&working_node, &[0.1; 4]).unwrap();
    age_decay_clock(&repo, working, "-2 hours");

    // reality 7 days, working 30 minutes.
    repo.sweep_decay(&DecayEngine::with_half_lives(7.0, 30.0 / 1440.0))
        .unwrap();

    let (r_score, r_status) = score_status(&repo, reality);
    assert_eq!(r_status, "active", "reality fact survives 2h");
    assert!(
        r_score > 0.9,
        "reality barely decays over hours, got {r_score}"
    );

    let (w_score, w_status) = score_status(&repo, working);
    assert_eq!(w_status, "archived", "working note archived after 2h");
    assert!(w_score < 0.1, "working note decayed hard, got {w_score}");
}

/// A boost restarts the decay clock: a note read mid-session survives the
/// next sweep even if it was written long ago.
#[test]
fn test_boost_restarts_decay_clock() {
    use crate::domain::decay::DecayEngine;
    let repo = setup_concrete();
    let t = TenantId("t".into());

    let mut n = active_node(&t, "reused note", 1.0);
    n.ccl = "working".into();
    let id = repo.store_node(&n, &[0.1; 4]).unwrap();
    age_decay_clock(&repo, id, "-2 hours"); // stale...

    repo.boost_relevance(&[id]).unwrap(); // ...but just read → clock resets

    repo.sweep_decay(&DecayEngine::with_half_lives(7.0, 30.0 / 1440.0))
        .unwrap();

    let (score, status) = score_status(&repo, id);
    assert_eq!(
        status, "active",
        "a just-read working note survives the sweep"
    );
    assert!(score > 0.9, "boost + reset clock keeps it hot, got {score}");
}

/// The recency backbone (slice 3): `recent_in_context` returns only the
/// active notes of the requested tenant + context, newest first — excluding
/// other contexts, NULL-context knowledge facts, and archived notes.
#[test]
fn test_recent_in_context_scopes_and_orders() {
    let repo = setup_concrete();
    let t = TenantId("t".into());

    // Helper: insert a working note in a context and stamp its recency.
    let insert = |fact: &str, ctx: Option<&str>, ccl: &str, secs_ago: i64, archived: bool| {
        let mut n = active_node(&t, fact, 1.0);
        n.ccl = ccl.into();
        n.context_key = ctx.map(|c| c.to_string());
        let id = repo.store_node(&n, &[0.1; 4]).unwrap();
        repo.conn
            .execute(
                "UPDATE nodes SET last_accessed_at = datetime('now', ?2),
                                 status = ?3 WHERE id = ?1",
                params![
                    id,
                    format!("-{secs_ago} seconds"),
                    if archived { "archived" } else { "active" }
                ],
            )
            .unwrap();
        id
    };

    // Active thread chat.1: two notes, B newer than A.
    insert("A first", Some("chat.1"), "working", 10, false);
    insert("B second", Some("chat.1"), "working", 1, false);
    // Noise that must be excluded:
    insert("other thread", Some("chat.2"), "working", 0, false); // different context
    insert("knowledge fact", None, "reality", 0, false); // NULL context
    insert("archived note", Some("chat.1"), "working", 0, true); // archived, same ctx

    let out = repo
        .recent_in_context(&t, &["working".to_string()], "chat.1", 10)
        .unwrap();

    let facts: Vec<&str> = out.iter().map(|r| r.fact.as_str()).collect();
    assert_eq!(
        facts,
        vec!["B second", "A first"],
        "only chat.1 active working notes, newest first"
    );
}

/// STM-GRAPH recall (N2): a turn is returned with its `about` subjects
/// attached as connections (label + dataId), and the subject anchor node is
/// NOT itself listed as a turn.
#[test]
fn test_recent_in_context_attaches_about_subjects() {
    let repo = setup_concrete();
    let t = TenantId("t".into());

    // A turn node (kind:turn) in the thread.
    let turn = MemoryNode {
        id: None,
        tenant_id: t.clone(),
        source_episode_id: None,
        payload: json!({ "fact": "find my lululemon purchase", "kind": "turn" }),
        status: "active".into(),
        ccl: "working".into(),
        is_explicit: true,
        support_count: 1,
        relevance_score: 1.0,
        context_key: Some("chat.1".into()),
    };
    let turn_id = repo.store_node(&turn, &[0.1; 4]).unwrap();

    // A subject anchor (kind:subject, dataId) + an `about` edge.
    let subject = MemoryNode {
        id: None,
        tenant_id: t.clone(),
        source_episode_id: None,
        payload: json!({ "fact": "Lululemon order", "kind": "subject", "dataId": "doc_LL" }),
        status: "active".into(),
        ccl: "working".into(),
        is_explicit: true,
        support_count: 1,
        relevance_score: 1.0,
        context_key: Some("chat.1".into()),
    };
    let subject_id = repo.store_node(&subject, &[0.0; 4]).unwrap();
    repo.store_edge(&crate::domain::models::Edge {
        source_id: turn_id,
        target_id: subject_id,
        relation: "about".into(),
        ccl: "working".into(),
        valid_from: None,
        valid_until: None,
        weight: 1.0,
    })
    .unwrap();

    let out = repo
        .recent_in_context(&t, &["working".to_string()], "chat.1", 10)
        .unwrap();

    // Exactly one TURN (the subject anchor is not listed as a turn).
    assert_eq!(out.len(), 1, "only the turn is returned, not the subject");
    assert_eq!(out[0].fact, "find my lululemon purchase");
    // …carrying its about-subject with the dataId the agent needs.
    assert_eq!(out[0].connections.len(), 1);
    assert_eq!(out[0].connections[0].relation, "about");
    assert_eq!(out[0].connections[0].entity, "Lululemon order (doc_LL)");
}

/// `recent_in_context` respects the limit and an empty ccl filter matches any
/// layer within the context.
#[test]
fn test_recent_in_context_limit_and_any_ccl() {
    let repo = setup_concrete();
    let t = TenantId("t".into());

    for i in 0..5 {
        let mut n = active_node(&t, &format!("note {i}"), 1.0);
        n.ccl = "working".into();
        n.context_key = Some("chat.1".into());
        let id = repo.store_node(&n, &[0.1; 4]).unwrap();
        repo.conn
            .execute(
                "UPDATE nodes SET last_accessed_at = datetime('now', ?2) WHERE id = ?1",
                params![id, format!("-{i} seconds")],
            )
            .unwrap();
    }

    // Empty ccl filter → any layer; limit caps the result.
    let out = repo.recent_in_context(&t, &[], "chat.1", 3).unwrap();
    assert_eq!(out.len(), 3, "limit respected");
    assert_eq!(out[0].fact, "note 0", "newest first");
}

/// Reads reinforce a `working` note exactly as they do facts: a boost resets
/// a decayed situational note to active/1.0, keeping an in-use session hot
/// between sweeps.
#[test]
fn test_boost_keeps_working_note_alive() {
    let repo = setup_concrete();
    let t = TenantId("t".into());

    let mut note = active_node(&t, "found report = doc_1", 0.2);
    note.ccl = "working".into();
    note.context_key = Some("chat.jid:1".into());
    let id = repo.store_node(&note, &[0.1; 4]).unwrap();

    repo.boost_relevance(&[id]).unwrap();

    let (score, status) = score_status(&repo, id);
    assert!(
        (score - 1.0).abs() < 1e-9,
        "boost reinforces the working note"
    );
    assert_eq!(status, "active");
}

/// Reading a node (boost) resets its relevance score to 1.0 — querying is
/// reinforcement, the counterforce to decay.
#[test]
fn test_boost_relevance_resets_score() {
    let repo = setup_concrete();
    let t = TenantId("t".into());
    let id = repo
        .store_node(&active_node(&t, "faded", 0.2), &[0.0; 4])
        .unwrap();

    repo.boost_relevance(&[id]).unwrap();

    let (score, status) = score_status(&repo, id);
    assert!((score - 1.0).abs() < 1e-9, "boost resets to 1.0");
    assert_eq!(status, "active");
}

/// Soft reset wipes the STM store (all tables) but a separate store
/// connection (standing in for LTM) keeps its rows — proof the reset is
/// scoped to one store.
#[test]
fn test_reset_store_empties_stm_only() {
    let repo = setup_concrete();
    let t = TenantId("t".into());

    let ep = repo
        .store_episode(&Episode {
            id: None,
            tenant_id: t.clone(),
            session_id: crate::domain::models::SessionId("s".into()),
            raw_dialogue: "hello".into(),
            ccl: "reality".into(),
            created_at: None,
        })
        .unwrap();
    let mut node = active_node(&t, "stm fact", 1.0);
    node.source_episode_id = Some(ep);
    repo.store_node(&node, &[0.0; 4]).unwrap();
    repo.store_ccl_definition(&crate::domain::models::CclDefinition {
        id: None,
        tenant_id: t.clone(),
        name: "dream".into(),
        description: "d".into(),
    })
    .unwrap();

    // A second, independent store connection with its own row (the "LTM").
    let ltm = init_db(None as Option<&String>).unwrap();
    ltm.execute("CREATE TABLE keep (v TEXT)", []).unwrap();
    ltm.execute("INSERT INTO keep (v) VALUES ('ltm-row')", [])
        .unwrap();

    assert!(count(&repo.conn, "nodes") > 0);

    repo.reset_store().unwrap();

    for table in ["nodes", "episodes", "edges", "ccl_registry", "vec_nodes"] {
        assert_eq!(count(&repo.conn, table), 0, "{table} should be empty");
    }
    // The other store is untouched.
    let ltm_rows: i64 = ltm
        .query_row("SELECT COUNT(*) FROM keep", [], |r| r.get(0))
        .unwrap();
    assert_eq!(ltm_rows, 1, "LTM store survives a soft reset");
}
