//! STM search ranking: reciprocal rank fusion of the vector and keyword legs
//! (`[retrieval]`), the stop-word filter on the keyword query, and scores.

use super::*;
use crate::domain::models::{Edge, TimeFilter};
use crate::infrastructure::database::init_db;
use crate::infrastructure::schema::init_schema;
use serde_json::json;

fn repo_with(ranking: Ranking) -> SqliteMemoryRepository {
    let conn = init_db(None as Option<&String>).unwrap();
    init_schema(&conn, 4).unwrap();
    SqliteMemoryRepository::new(conn).with_ranking(ranking)
}

fn store(r: &SqliteMemoryRepository, fact: &str, embedding: [f32; 4]) -> i64 {
    r.store_node(
        &MemoryNode {
            id: None,
            tenant_id: TenantId("A".into()),
            source_episode_id: None,
            payload: json!({ "fact": fact }),
            status: "active".into(),
            ccl: "reality".into(),
            is_explicit: true,
            support_count: 1,
            relevance_score: 1.0,
            context_key: None,
        },
        &embedding,
    )
    .unwrap()
}

fn search(
    r: &SqliteMemoryRepository,
    query: &str,
    embedding: [f32; 4],
    k: usize,
) -> Vec<MemoryResult> {
    r.query_with_graph(
        query,
        &embedding,
        &TenantId("A".into()),
        &TimeFilter::default(),
        &["reality".to_string()],
        k,
    )
    .unwrap()
}

/// Unrelated facts (far vectors, no shared words) so BM25 sees a realistic
/// corpus: with only two documents every term's IDF is near zero and keyword
/// scores vanish, which would hide the old summed-score failure.
fn filler(r: &SqliteMemoryRepository, n: usize) {
    for i in 0..n {
        store(
            r,
            &format!("filler note number {i}"),
            [0.0, 0.0, 1.0, i as f32],
        );
    }
}

fn facts(results: &[MemoryResult]) -> Vec<&str> {
    results.iter().map(|r| r.fact.as_str()).collect()
}

/// The eval's failure mode: a question shares only stop-words with an
/// unrelated turn and no words with the turn that answers it. The answer is
/// the nearest vector, so it must rank first; it used to lose to any keyword
/// hit (vector distance and BM25 were summed).
#[test]
fn semantic_match_beats_a_stop_word_only_keyword_match() {
    let r = repo_with(Ranking::default());
    store(&r, "Sony A7R IV camera flash", [1.0, 0.0, 0.0, 0.0]);
    store(&r, "what did I do with it", [0.0, 1.0, 0.0, 0.0]);
    filler(&r, 20);

    let hits = search(
        &r,
        "what did I get for my photography setup",
        [0.9, 0.1, 0.0, 0.0],
        2,
    );
    assert_eq!(facts(&hits)[0], "Sony A7R IV camera flash");
}

/// With one candidate per leg, the leg weights alone decide the order.
#[test]
fn leg_weights_decide_between_a_vector_and_a_keyword_hit() {
    let one_each = |vector_weight: f64, keyword_weight: f64| {
        let r = repo_with(Ranking {
            vector_weight,
            keyword_weight,
            leg_k_factor: 1,
            ..Ranking::default()
        });
        store(&r, "nearest by meaning", [1.0, 0.0, 0.0, 0.0]);
        store(&r, "zebra", [0.0, 0.0, 0.0, 1.0]);
        // k = 2, leg_k = 2: both legs see both nodes; give the keyword leg a
        // single match and the vector leg a clear order.
        let hits = search(&r, "zebra", [1.0, 0.0, 0.0, 0.0], 1);
        facts(&hits).first().map(|s| s.to_string())
    };
    assert_eq!(one_each(2.0, 1.0).as_deref(), Some("nearest by meaning"));
    assert_eq!(one_each(1.0, 2.0).as_deref(), Some("zebra"));
    // Keyword leg off: pure vector order.
    assert_eq!(one_each(1.0, 0.0).as_deref(), Some("nearest by meaning"));
}

/// `rrf_k` trades top-rank dominance against agreement between legs: a node
/// ranked 3rd by vector and 2nd by keyword beats the vector's 1st at k = 60,
/// but not at k = 0.
#[test]
fn rrf_k_changes_how_much_agreement_counts() {
    let top = |rrf_k: f64| {
        let r = repo_with(Ranking {
            rrf_k,
            keyword_weight: 1.0,
            leg_k_factor: 1,
            ..Ranking::default()
        });
        store(&r, "alpha", [1.0, 0.0, 0.0, 0.0]); // vector 1st
        store(&r, "gamma", [0.9, 0.3, 0.0, 0.0]); // vector 2nd
        store(
            &r,
            "zebra plus a long tail of other unrelated filler words",
            [0.7, 0.7, 0.0, 0.0],
        ); // vector 3rd, keyword 2nd
        store(&r, "zebra zebra zebra", [0.0, 0.0, 0.0, 1.0]); // keyword 1st only
        let hits = search(&r, "zebra", [1.0, 0.0, 0.0, 0.0], 3);
        facts(&hits)[0].to_string()
    };
    assert_eq!(
        top(60.0),
        "zebra plus a long tail of other unrelated filler words"
    );
    assert_eq!(top(0.0), "alpha");
}

/// A query made only of stop-words still searches by keyword (the filter
/// keeps them rather than leaving nothing to match).
#[test]
fn stop_word_only_query_still_uses_the_keyword_leg() {
    let r = repo_with(Ranking::default());
    store(&r, "it is what it is", [0.0, 0.0, 0.0, 1.0]);
    store(&r, "unrelated", [0.0, 1.0, 0.0, 0.0]);
    let hits = search(&r, "what is it", [0.0, 1.0, 0.0, 0.0], 2);
    assert_eq!(facts(&hits)[0], "it is what it is");
}

/// `k` above the old cap of 10 is honoured end to end in the repository.
#[test]
fn k_above_ten_returns_k_hits() {
    let r = repo_with(Ranking::default());
    for i in 0..15 {
        store(&r, &format!("fact {i}"), [1.0, i as f32 * 0.01, 0.0, 0.0]);
    }
    assert_eq!(
        search(&r, "nothing matches", [1.0, 0.0, 0.0, 0.0], 15).len(),
        15
    );
}

/// Scores are off by default; when on, direct hits carry their fused score
/// in descending order and graph neighbours carry none.
#[test]
fn scores_are_reported_only_when_enabled() {
    let off = repo_with(Ranking::default());
    store(&off, "alpha", [1.0, 0.0, 0.0, 0.0]);
    assert!(
        search(&off, "alpha", [1.0, 0.0, 0.0, 0.0], 5)
            .iter()
            .all(|h| h.score.is_none())
    );

    let on = repo_with(Ranking {
        include_scores: true,
        ..Ranking::default()
    });
    let a = store(&on, "alpha", [1.0, 0.0, 0.0, 0.0]);
    store(&on, "beta", [0.8, 0.6, 0.0, 0.0]);
    // A neighbour with no vector and no keyword match, reached only by edge.
    let n = store(&on, "neighbour", [0.0, 0.0, 0.0, 0.0]);
    on.store_edge(&Edge {
        source_id: a,
        target_id: n,
        relation: "knows".into(),
        ccl: "reality".into(),
        valid_from: None,
        valid_until: None,
        weight: 1.0,
    })
    .unwrap();

    let hits = search(&on, "alpha", [1.0, 0.0, 0.0, 0.0], 5);
    assert_eq!(facts(&hits), ["alpha", "beta", "neighbour"]);
    let (s0, s1) = (hits[0].score.unwrap(), hits[1].score.unwrap());
    assert!(s0 > s1, "{s0} > {s1}");
    // alpha: vector 1st + keyword 1st.
    let d = Ranking::default();
    let expected = (d.vector_weight + d.keyword_weight) / (d.rrf_k + 1.0);
    assert!((s0 - expected).abs() < 1e-9, "{s0} vs {expected}");
    assert_eq!(hits[2].score, None, "graph neighbours carry no score");
}

/// `hybrid_search` fuses the same way as `query_with_graph`.
#[test]
fn hybrid_search_uses_the_same_fusion() {
    let r = repo_with(Ranking::default());
    store(&r, "Sony A7R IV camera flash", [1.0, 0.0, 0.0, 0.0]);
    store(&r, "what did I do with it", [0.0, 1.0, 0.0, 0.0]);
    filler(&r, 20);
    let hits = r
        .hybrid_search(
            "what did I get for my photography setup",
            &[0.9, 0.1, 0.0, 0.0],
            &TenantId("A".into()),
            2,
        )
        .unwrap();
    assert_eq!(hits[0].payload["fact"], "Sony A7R IV camera flash");
}

/// W3R-4: with `keyword_weight = 0` the keyword leg is not run at all, so
/// keyword-only facts (here: no vector, so the vector leg comes back short)
/// don't fill the result with score-0 hits. Pure vector, as LTM recall does.
#[test]
fn zero_keyword_weight_runs_no_keyword_leg() {
    let r = repo_with(Ranking {
        keyword_weight: 0.0,
        ..Ranking::default()
    });
    store(&r, "alpha", [1.0, 0.0, 0.0, 0.0]);
    for i in 0..3 {
        // A zero vector is not indexed: reachable by keyword only.
        store(&r, &format!("zebra {i}"), [0.0, 0.0, 0.0, 0.0]);
    }
    let hits = search(&r, "zebra", [1.0, 0.0, 0.0, 0.0], 5);
    assert_eq!(facts(&hits), ["alpha"]);
}

/// W3R-4, symmetric: with `vector_weight = 0` only the keyword leg runs, so
/// vector-only neighbours don't fill slots; with no keyword terms there is
/// nothing to rank.
#[test]
fn zero_vector_weight_runs_no_vector_leg() {
    let r = repo_with(Ranking {
        vector_weight: 0.0,
        ..Ranking::default()
    });
    for i in 0..3 {
        store(&r, &format!("near {i}"), [1.0, i as f32 * 0.1, 0.0, 0.0]);
    }
    store(&r, "zebra", [0.0, 0.0, 0.0, 1.0]);
    let hits = search(&r, "zebra", [1.0, 0.0, 0.0, 0.0], 5);
    assert_eq!(facts(&hits), ["zebra"]);
    assert!(search(&r, "?!", [1.0, 0.0, 0.0, 0.0], 5).is_empty());
}
