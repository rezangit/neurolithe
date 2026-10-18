//! The repository's hot queries use the STM v3 indexes (`EXPLAIN QUERY PLAN`
//! over the exact SQL the repository runs, on a store migrated to latest).

use super::graph::DOCUMENT_FACT;
use super::inspect::COUNT_BY_DATA_ID_SQL;
use super::lifecycle::DELETE_BY_DATA_ID_SQL;
use super::search::{CONNECTIONS_SQL, Legs, fused_hits_sql, graph_query_sql};
use super::working::{ABOUT_SUBJECTS_SQL, FIND_WORKING_SUBJECT_SQL};
use crate::application::migrations::{StoreKind, migrate_store};
use crate::infrastructure::database::init_db;
use rusqlite::Connection;

/// An in-memory STM store brought to the latest schema version.
fn migrated_stm() -> Connection {
    let conn = init_db(None as Option<&String>).unwrap();
    migrate_store(&conn, StoreKind::Stm, None, 4).unwrap();
    conn
}

/// The `EXPLAIN QUERY PLAN` details of `sql`, one line per plan step.
fn plan(conn: &Connection, sql: &str) -> String {
    let mut stmt = conn
        .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
        .unwrap_or_else(|e| panic!("{e}: {sql}"));
    // Plans don't depend on values: bind NULL to every parameter.
    let nulls = vec![rusqlite::types::Null; stmt.parameter_count()];
    let rows = stmt
        .query_map(rusqlite::params_from_iter(nulls), |r| r.get::<_, String>(3))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    rows.join("\n")
}

/// Every combination of legs a search can run (vector-only, keyword-only,
/// both).
const ALL_LEG_SETS: [Legs; 3] = [
    Legs {
        vector: true,
        keyword: false,
    },
    Legs {
        vector: false,
        keyword: true,
    },
    Legs {
        vector: true,
        keyword: true,
    },
];

/// No plan step reads the whole `edges` table (aliased `e` or not).
fn assert_no_edge_scan(plan: &str) {
    for line in plan.lines() {
        assert!(
            !(line.starts_with("SCAN e") || line.starts_with("SCAN edges")),
            "full edges scan: {line}\n{plan}"
        );
    }
}

#[test]
fn connections_use_both_edge_indexes() {
    let p = plan(&migrated_stm(), CONNECTIONS_SQL);
    assert!(p.contains("idx_edges_source"), "{p}");
    assert!(p.contains("idx_edges_target"), "{p}");
    assert_no_edge_scan(&p);
}

#[test]
fn about_subjects_use_the_source_index() {
    let p = plan(&migrated_stm(), ABOUT_SUBJECTS_SQL);
    assert!(p.contains("idx_edges_source"), "{p}");
    assert_no_edge_scan(&p);
}

#[test]
fn graph_expansion_uses_both_edge_indexes() {
    let conn = migrated_stm();
    // Vector-only and hybrid (with the FTS keyword leg).
    for legs in ALL_LEG_SETS {
        let p = plan(&conn, &graph_query_sql(legs));
        assert!(p.contains("idx_edges_source"), "{p}");
        assert!(p.contains("idx_edges_target"), "{p}");
        assert_no_edge_scan(&p);
    }
}

#[test]
fn data_id_lookups_use_the_expression_index() {
    let conn = migrated_stm();
    let p = plan(&conn, COUNT_BY_DATA_ID_SQL);
    assert!(p.contains("idx_nodes_data_id"), "{p}");
    for sql in DELETE_BY_DATA_ID_SQL {
        let p = plan(&conn, sql);
        assert!(p.contains("idx_nodes_data_id"), "{sql}\n{p}");
        assert_no_edge_scan(&p);
    }
}

#[test]
fn working_subject_lookup_avoids_a_node_scan() {
    let p = plan(&migrated_stm(), FIND_WORKING_SUBJECT_SQL);
    assert!(!p.contains("SCAN nodes"), "{p}");
}

/// Document-fact upserts (Kafka ingest idempotency) look facts up by dataId:
/// the [`DOCUMENT_FACT`] predicate must hit `idx_nodes_data_id`. The
/// statements are built from the same predicate as `find_document_nodes` and
/// `replace_document_node`; `n.*` stands in for the node column list, which
/// does not affect the plan.
#[test]
fn document_fact_lookups_use_the_expression_index() {
    let conn = migrated_stm();
    let old = format!("SELECT id FROM nodes WHERE {DOCUMENT_FACT}");
    let statements = [
        // find_document_nodes
        format!("SELECT n.* FROM nodes n WHERE {DOCUMENT_FACT} ORDER BY n.id"),
        // replace_document_node: edges, vectors, then the old facts.
        format!("DELETE FROM edges WHERE source_id IN ({old}) OR target_id IN ({old})"),
        format!("DELETE FROM vec_nodes WHERE node_id IN ({old})"),
        format!("DELETE FROM nodes WHERE {DOCUMENT_FACT}"),
    ];
    for sql in &statements {
        let p = plan(&conn, sql);
        assert!(p.contains("idx_nodes_data_id"), "{sql}\n{p}");
        assert!(
            !p.lines()
                .any(|l| l == "SCAN n" || l.starts_with("SCAN nodes")),
            "full nodes scan: {sql}\n{p}"
        );
        assert_no_edge_scan(&p);
    }
}

/// The fused ranking reads each leg's candidate nodes by primary key, one
/// lookup per hit; it never walks `nodes` (by scan or by a tenant index), so
/// its cost follows the hits, not the store size. The keyword leg is an
/// FTS5 index query.
#[test]
fn fused_ranking_reads_nodes_by_key() {
    let conn = migrated_stm();
    for legs in ALL_LEG_SETS {
        let sql = format!("{} SELECT node_id FROM fused", fused_hits_sql(legs));
        let p = plan(&conn, &sql);
        let node_reads: Vec<&str> = p
            .lines()
            .filter(|l| l.contains(" rn ") || l.ends_with(" rn") || l.contains(" nodes"))
            .collect();
        let legs_run = usize::from(legs.vector) + usize::from(legs.keyword);
        assert_eq!(node_reads.len(), legs_run, "{legs:?}\n{p}");
        assert!(
            node_reads
                .iter()
                .all(|l| *l == "SEARCH rn USING INTEGER PRIMARY KEY (rowid=?)"),
            "{legs:?}\n{p}"
        );
        if legs.keyword {
            assert!(p.contains("VIRTUAL TABLE INDEX"), "{p}");
        }
    }
}
