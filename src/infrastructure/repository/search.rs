//! Recall: hybrid vector + keyword search, the 1-hop graph expansion around
//! direct hits, and the nearest-neighbour lookup the conflict resolver uses
//! for dedup.

use super::graph::boost_relevance;
use super::vector_index::knn_k;
use crate::domain::models::{MemoryConnection, MemoryNode, MemoryResult, TenantId, TimeFilter};
use crate::domain::ranking::{fts_match_query, leg_depth};
use crate::domain::tuning::Ranking;
use crate::infrastructure::embedding_codec::as_blob;
use anyhow::Result;
use rusqlite::types::ToSql;
use rusqlite::{Connection, params};

/// The `nodes` columns [`row_to_node`] maps, in order.
const NODE_COLUMNS: &str = "n.id, n.tenant_id, n.source_episode_id, n.payload, n.status, n.ccl, \
     n.is_explicit, n.support_count, n.relevance_score, n.context_key";

/// Map the first ten columns of a row selected with [`NODE_COLUMNS`] to a
/// [`MemoryNode`]. An unparseable payload reads as JSON `null`.
fn row_to_node(row: &rusqlite::Row) -> rusqlite::Result<MemoryNode> {
    let payload_str: String = row.get(3)?;
    Ok(MemoryNode {
        id: Some(row.get(0)?),
        tenant_id: TenantId(row.get(1)?),
        source_episode_id: row.get::<_, Option<i64>>(2)?,
        payload: serde_json::from_str(&payload_str).unwrap_or(serde_json::Value::Null),
        status: row.get(4)?,
        ccl: row.get(5)?,
        is_explicit: row.get(6)?,
        support_count: row.get(7)?,
        relevance_score: row.get(8)?,
        context_key: row.get(9)?,
    })
}

/// The document facts for `data_id` (see [`super::graph::DOCUMENT_FACT`]), any
/// status, oldest first.
pub(super) fn find_document_nodes(conn: &Connection, data_id: &str) -> Result<Vec<MemoryNode>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {NODE_COLUMNS} FROM nodes n WHERE {} ORDER BY n.id",
        super::graph::DOCUMENT_FACT
    ))?;
    let rows = stmt.query_map(params![data_id], row_to_node)?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// The string at `key` in a JSON payload, if present.
fn payload_str(payload: &serde_json::Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Visibility of a candidate node `rn`: this tenant, active, in the CCL
/// layers `:ccl` (a JSON array; NULL = any layer) and inside the time window
/// `:after`/`:before` (NULL = open). Applied inside each leg, before its
/// ranks are assigned, so rows the caller can't see never take a rank.
const VISIBLE: &str = "rn.tenant_id = :tenant AND rn.status = 'active'
      AND (:ccl IS NULL OR rn.ccl IN (SELECT value FROM json_each(:ccl)))
      AND (rn.created_at >= :after OR :after IS NULL)
      AND (rn.created_at <= :before OR :before IS NULL)";

/// The fused direct hits as CTEs ending in `fused(node_id, fused_score)`,
/// best first, at most `:limit` rows. Reciprocal rank fusion: the vector leg
/// (the `:knn_k` nearest by sqlite-vec distance) and the keyword leg (FTS5
/// BM25 over `:fts`) each rank their visible candidates, cut at `:leg_k`;
/// a node scores `Σ weight / (:rrf_k + rank)` over the legs it is in
/// (`:w_vec`, `:w_kw`): the same fusion as [`crate::domain::ranking::rrf_fuse`],
/// ties included (better vector rank, then better keyword rank; node id
/// last, so results are deterministic). A leg left out of `legs` is not run
/// and takes no parameters: no `:embedding`/`:knn_k` without the vector leg,
/// no `:fts` without the keyword leg (see [`Legs`]).
///
/// Both raw legs are MATERIALIZED: a lone vec0 KNN select would otherwise be
/// flattened into the outer query, handing it an ORDER BY that vec0 rejects
/// (P3-2). Each leg then reads its hits' nodes with a CROSS JOIN, which in
/// SQLite fixes the hits as the outer loop: the nodes are fetched by primary
/// key, rather than the planner walking every node of the tenant by index
/// and probing the hits.
pub(super) fn fused_hits_sql(legs: Legs) -> String {
    let vector_leg = if legs.vector {
        format!(
            "knn AS MATERIALIZED (
                SELECT node_id, distance FROM vec_nodes WHERE embedding MATCH :embedding AND k = :knn_k
            ),
            vector_leg AS (
                SELECT knn.node_id, ROW_NUMBER() OVER (ORDER BY knn.distance, knn.node_id) AS leg_rank
                FROM knn CROSS JOIN nodes rn ON rn.id = knn.node_id
                WHERE {VISIBLE}
                ORDER BY leg_rank LIMIT :leg_k
            )"
        )
    } else {
        "vector_leg AS (SELECT NULL AS node_id, NULL AS leg_rank WHERE 0)".to_string()
    };
    let keyword_leg = if legs.keyword {
        format!(
            "keyword_hits AS MATERIALIZED (
                SELECT rowid AS node_id, rank AS bm25 FROM fts_nodes WHERE fts_nodes MATCH :fts
            ),
            keyword_leg AS (
                SELECT h.node_id, ROW_NUMBER() OVER (ORDER BY h.bm25, h.node_id) AS leg_rank
                FROM keyword_hits h CROSS JOIN nodes rn ON rn.id = h.node_id
                WHERE {VISIBLE}
                ORDER BY leg_rank LIMIT :leg_k
            )"
        )
    } else {
        "keyword_leg AS (SELECT NULL AS node_id, NULL AS leg_rank WHERE 0)".to_string()
    };
    format!(
        "
            WITH {vector_leg},
            {keyword_leg},
            fused AS (
                SELECT node_id, SUM(score) AS fused_score,
                       MIN(vector_rank) AS vector_rank, MIN(keyword_rank) AS keyword_rank
                FROM (
                    SELECT node_id, :w_vec / (:rrf_k + leg_rank) AS score,
                           leg_rank AS vector_rank, NULL AS keyword_rank
                    FROM vector_leg
                    UNION ALL
                    SELECT node_id, :w_kw / (:rrf_k + leg_rank) AS score,
                           NULL AS vector_rank, leg_rank AS keyword_rank
                    FROM keyword_leg
                )
                GROUP BY node_id
                ORDER BY fused_score DESC, vector_rank NULLS LAST, keyword_rank NULLS LAST, node_id
                LIMIT :limit
            )"
    )
}

/// Which legs a search runs. A leg with weight 0 is not run at all, so its
/// candidates can't take result slots with a zero score (W3R-4; LTM recall
/// does the same); nor is the keyword leg when the query has no searchable
/// terms. Validation keeps at least one weight above 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Legs {
    pub(super) vector: bool,
    pub(super) keyword: bool,
}

/// The parameters of [`fused_hits_sql`] for one search, owned so a caller
/// can add its own and bind them all by name.
struct FusedParams {
    legs: Legs,
    embedding: Vec<u8>,
    fts: Option<String>,
    tenant: String,
    ccl: Option<String>,
    after: Option<String>,
    before: Option<String>,
    limit: i64,
    leg_k: i64,
    knn_k: i64,
    rrf_k: f64,
    w_vec: f64,
    w_kw: f64,
}

impl FusedParams {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ranking: &Ranking,
        query_text: &str,
        query_embedding: &[f32],
        tenant_id: &TenantId,
        ccl: Option<String>,
        time_filter: &TimeFilter,
        limit: usize,
    ) -> Self {
        let leg_k = leg_depth(limit, ranking);
        let fts = fts_match_query(query_text, ranking.fts_stopwords)
            .filter(|_| ranking.keyword_weight > 0.0);
        Self {
            legs: Legs {
                vector: ranking.vector_weight > 0.0,
                keyword: fts.is_some(),
            },
            embedding: as_blob(query_embedding).to_vec(),
            fts,
            tenant: tenant_id.0.clone(),
            ccl,
            after: time_filter.after.clone(),
            before: time_filter.before.clone(),
            limit: limit as i64,
            leg_k: leg_k as i64,
            knn_k: knn_k(leg_k),
            rrf_k: ranking.rrf_k,
            w_vec: ranking.vector_weight,
            w_kw: ranking.keyword_weight,
        }
    }

    /// Whether there is anything to rank (at least one leg runs).
    fn any_leg(&self) -> bool {
        self.legs.vector || self.legs.keyword
    }

    /// Named bindings for every parameter the SQL uses, given [`Self::legs`].
    /// Only valid when [`Self::any_leg`].
    fn bindings(&self) -> Vec<(&str, &dyn ToSql)> {
        let mut out: Vec<(&str, &dyn ToSql)> = vec![
            (":tenant", &self.tenant),
            (":ccl", &self.ccl),
            (":after", &self.after),
            (":before", &self.before),
            (":limit", &self.limit),
            (":leg_k", &self.leg_k),
            (":rrf_k", &self.rrf_k),
            (":w_vec", &self.w_vec),
            (":w_kw", &self.w_kw),
        ];
        if self.legs.vector {
            out.push((":embedding", &self.embedding));
            out.push((":knn_k", &self.knn_k));
        }
        if let Some(fts) = &self.fts {
            out.push((":fts", fts));
        }
        out
    }
}

pub(super) fn hybrid_search(
    conn: &Connection,
    ranking: &Ranking,
    query_text: &str,
    query_embedding: &[f32],
    tenant_id: &TenantId,
    limit: usize,
) -> Result<Vec<MemoryNode>> {
    let p = FusedParams::new(
        ranking,
        query_text,
        query_embedding,
        tenant_id,
        None,
        &TimeFilter::default(),
        limit,
    );
    if !p.any_leg() {
        return Ok(Vec::new());
    }
    let query = format!(
        "{}
            SELECT {NODE_COLUMNS}
            FROM fused rm JOIN nodes n ON n.id = rm.node_id
            ORDER BY rm.fused_score DESC, rm.vector_rank NULLS LAST, rm.keyword_rank NULLS LAST, n.id",
        fused_hits_sql(p.legs)
    );
    let mut stmt = conn.prepare(&query)?;
    let nodes = stmt
        .query_map(p.bindings().as_slice(), row_to_node)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(nodes)
}

/// The hybrid + graph + temporal recall query (blueprint section 2.5): the
/// fused direct hits of [`fused_hits_sql`] plus their 1-hop graph neighbours
/// (edges valid in the time window, in the requested layers). Direct hits
/// come first by fused score, then the neighbours by relevance. Columns: id,
/// payload, ccl, updated_at, fused score (NULL for a neighbour). The 1-hop
/// expansion is served by `idx_edges_source` / `idx_edges_target`.
pub(super) fn graph_query_sql(legs: Legs) -> String {
    format!(
        "{},
            graph_context AS (
                SELECT node_id FROM fused
                UNION
                SELECT target_id AS node_id FROM edges
                WHERE source_id IN (SELECT node_id FROM fused)
                  AND (valid_until IS NULL OR :after IS NULL OR valid_until >= :after)
                  AND (valid_from IS NULL OR :before IS NULL OR valid_from <= :before)
                  AND ccl IN (SELECT value FROM json_each(:ccl))
                UNION
                SELECT source_id AS node_id FROM edges
                WHERE target_id IN (SELECT node_id FROM fused)
                  AND (valid_until IS NULL OR :after IS NULL OR valid_until >= :after)
                  AND (valid_from IS NULL OR :before IS NULL OR valid_from <= :before)
                  AND ccl IN (SELECT value FROM json_each(:ccl))
            )
            SELECT n.id, n.payload, n.ccl, n.updated_at, rm.fused_score
            FROM nodes n
            JOIN graph_context gc ON n.id = gc.node_id
            LEFT JOIN fused rm ON rm.node_id = n.id
            WHERE n.tenant_id = :tenant AND n.status = 'active'
              AND (n.created_at >= :after OR :after IS NULL)
              AND (n.created_at <= :before OR :before IS NULL)
              AND n.ccl IN (SELECT value FROM json_each(:ccl))
            -- Rank by SEARCH quality, not decay score: direct hits first by
            -- fused score, then the 1-hop neighbours, relevance as a tiebreak
            -- (field-report round 2, ranking nit), id last for determinism.
            ORDER BY (rm.fused_score IS NULL), rm.fused_score DESC,
                     rm.vector_rank NULLS LAST, rm.keyword_rank NULLS LAST,
                     n.relevance_score DESC, n.id
            LIMIT :limit",
        fused_hits_sql(legs)
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn query_with_graph(
    conn: &Connection,
    ranking: &Ranking,
    query_text: &str,
    query_embedding: &[f32],
    tenant_id: &TenantId,
    time_filter: &TimeFilter,
    ccl_filter: &[String],
    limit: usize,
) -> Result<Vec<MemoryResult>> {
    let ccl_json = serde_json::to_string(ccl_filter)?;
    // The keyword leg runs only when the query yields safe FTS terms (see
    // `fts_match_query`); otherwise vector recall carries the search alone.
    let p = FusedParams::new(
        ranking,
        query_text,
        query_embedding,
        tenant_id,
        Some(ccl_json.clone()),
        time_filter,
        limit,
    );
    if !p.any_leg() {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare(&graph_query_sql(p.legs))?;

    let rows: Vec<(i64, String, String, String, Option<f64>)> = stmt
        .query_map(p.bindings().as_slice(), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<f64>>(4)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    // Collect node IDs for relevance boost
    let node_ids: Vec<i64> = rows.iter().map(|r| r.0).collect();
    if !node_ids.is_empty() {
        boost_relevance(conn, &node_ids)?;
    }

    // Build token-optimized output with 1-hop connections
    let mut results = Vec::new();
    for (node_id, payload_json, ccl, updated_at, score) in rows {
        let payload: serde_json::Value =
            serde_json::from_str(&payload_json).unwrap_or(serde_json::Value::Null);
        results.push(MemoryResult {
            fact: payload_str(&payload, "fact").unwrap_or_default(),
            ccl,
            last_updated: updated_at,
            connections: connections(conn, node_id, &ccl_json)?,
            // Surface the archive reference so a hit can be traced/fetched.
            data_id: payload_str(&payload, "dataId"),
            context_key: None,
            score: score.filter(|_| ranking.include_scores),
        });
    }

    Ok(results)
}

/// A node's 1-hop edges in both directions (`?1` node id, `?2` CCL layers as a
/// JSON array), each with the neighbour's payload. Served by
/// `idx_edges_source` / `idx_edges_target`.
pub(super) const CONNECTIONS_SQL: &str = "
    SELECT e.relation, e.ccl, e.valid_from, e.valid_until, n2.payload
    FROM edges e
    JOIN nodes n2 ON n2.id = e.target_id
    WHERE e.source_id = ?1
      AND e.ccl IN (SELECT value FROM json_each(?2))
    UNION ALL
    SELECT e.relation, e.ccl, e.valid_from, e.valid_until, n2.payload
    FROM edges e
    JOIN nodes n2 ON n2.id = e.source_id
    WHERE e.target_id = ?1
      AND e.ccl IN (SELECT value FROM json_each(?2))";

/// A node's 1-hop connections (both edge directions) within the CCL layers in
/// `ccl_json`, each naming the neighbour by its fact text.
fn connections(conn: &Connection, node_id: i64, ccl_json: &str) -> Result<Vec<MemoryConnection>> {
    let mut edge_stmt = conn.prepare(CONNECTIONS_SQL)?;

    let connections = edge_stmt
        .query_map(params![node_id, ccl_json], |row| {
            let entity_payload: String = row.get(4)?;
            let ep: serde_json::Value =
                serde_json::from_str(&entity_payload).unwrap_or(serde_json::Value::Null);
            Ok(MemoryConnection {
                relation: row.get(0)?,
                entity: payload_str(&ep, "fact").unwrap_or_default(),
                ccl: row.get(1)?,
                valid_from: row.get(2)?,
                valid_until: row.get(3)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(connections)
}

pub(super) fn find_similar_nodes(
    conn: &Connection,
    embedding: &[f32],
    tenant_id: &TenantId,
    threshold: f64,
    limit: usize,
) -> Result<Vec<(MemoryNode, f64)>> {
    let query = format!(
        "
            SELECT {NODE_COLUMNS}, v.distance
            FROM vec_nodes v
            JOIN nodes n ON n.id = v.node_id
            WHERE v.embedding MATCH ?1 AND k = ?2
              AND n.tenant_id = ?3 AND n.status = 'active'
              AND v.distance <= ?4
            ORDER BY v.distance ASC
            LIMIT ?5;
        "
    );

    let mut stmt = conn.prepare(&query)?;
    let hits = stmt
        .query_map(
            params![
                as_blob(embedding),
                knn_k(limit),
                tenant_id.0,
                threshold,
                limit as i64
            ],
            |row| Ok((row_to_node(row)?, row.get::<_, f64>(10)?)),
        )?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ranking::rrf_fuse;
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::schema::init_schema;

    /// The SQL fusion and the shared `rrf_fuse` must not drift: over the same
    /// leg rankings (read back from the query's own leg CTEs) they give the
    /// same candidates, order and scores, for several settings.
    #[test]
    fn sql_fusion_matches_rrf_fuse() {
        let conn = init_db(None as Option<&String>).unwrap();
        init_schema(&conn, 4).unwrap();
        let facts = [
            ("alpha zebra", [1.0, 0.0, 0.0, 0.0]),
            ("beta", [0.9, 0.3, 0.0, 0.0]),
            ("gamma zebra zebra", [0.7, 0.7, 0.0, 0.0]),
            ("delta zebra zebra zebra", [0.0, 0.0, 0.0, 1.0]),
            ("epsilon", [0.5, 0.5, 0.5, 0.5]),
            ("zeta zebra", [0.2, 0.9, 0.1, 0.0]),
        ];
        for (fact, emb) in facts {
            super::super::graph::store_node(
                &conn,
                &MemoryNode {
                    id: None,
                    tenant_id: TenantId("A".into()),
                    source_episode_id: None,
                    payload: serde_json::json!({ "fact": fact }),
                    status: "active".into(),
                    ccl: "reality".into(),
                    is_explicit: true,
                    support_count: 1,
                    relevance_score: 1.0,
                    context_key: None,
                },
                &emb,
            )
            .unwrap();
        }
        let ids = |sql: &str, p: &FusedParams| -> Vec<i64> {
            let mut stmt = conn.prepare(sql).unwrap();
            stmt.query_map(p.bindings().as_slice(), |r| r.get(0))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap()
        };
        let settings = [
            Ranking::default(),
            Ranking {
                rrf_k: 0.0,
                ..Ranking::default()
            },
            Ranking {
                vector_weight: 2.0,
                keyword_weight: 0.5,
                leg_k_factor: 1,
                ..Ranking::default()
            },
            Ranking {
                keyword_weight: 0.0,
                ..Ranking::default()
            },
        ];
        for ranking in settings {
            let p = FusedParams::new(
                &ranking,
                "zebra",
                &[1.0, 0.0, 0.0, 0.0],
                &TenantId("A".into()),
                None,
                &TimeFilter::default(),
                4,
            );
            let ctes = fused_hits_sql(p.legs);
            let vector = ids(
                &format!("{ctes} SELECT node_id FROM vector_leg ORDER BY leg_rank"),
                &p,
            );
            let keyword = ids(
                &format!("{ctes} SELECT node_id FROM keyword_leg ORDER BY leg_rank"),
                &p,
            );
            assert!(!vector.is_empty(), "{ranking:?}");
            // A zero-weight keyword leg isn't run (W3R-4).
            assert_eq!(keyword.is_empty(), !p.legs.keyword, "{ranking:?}");

            let mut stmt = conn
                .prepare(&format!("{ctes} SELECT node_id, fused_score FROM fused"))
                .unwrap();
            let sql: Vec<(i64, f64)> = stmt
                .query_map(p.bindings().as_slice(), |r| Ok((r.get(0)?, r.get(1)?)))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();

            let mut expected = rrf_fuse(&vector, &keyword, &ranking);
            expected.truncate(4);
            let order = |v: &[(i64, f64)]| v.iter().map(|x| x.0).collect::<Vec<_>>();
            assert_eq!(order(&sql), order(&expected), "{ranking:?}");
            for ((_, a), (_, b)) in sql.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-12, "{ranking:?}: {a} vs {b}");
            }
        }
    }
}
