//! Recall: hybrid vector + keyword search, the 1-hop graph expansion around
//! direct hits, and the nearest-neighbour lookup the conflict resolver uses
//! for dedup.

use super::graph::boost_relevance;
use super::vector_index::knn_k;
use crate::domain::models::{MemoryConnection, MemoryNode, MemoryResult, TenantId, TimeFilter};
use crate::infrastructure::embedding_codec::as_blob;
use anyhow::Result;
use rusqlite::{Connection, params};

/// Turn a raw user query into a **safe** FTS5 MATCH expression, or `None` if it
/// carries no searchable terms.
///
/// Two problems this solves (field-report §1): (1) passing the raw string to
/// `fts5 MATCH` treats it as an FTS query *expression*, so any punctuation
/// (`:`, `-`, `(`, `"`, `.`) is a syntax error that fails the whole hybrid
/// search; (2) bare space-separated tokens are implicitly AND-ed, so one word
/// the document happens not to contain yields zero rows. We tokenize on
/// non-alphanumerics, quote each token as a phrase (neutralizing operators), and
/// OR-join them — so search degrades to keyword-any, never to an error or an
/// over-strict all-terms match. Vector recall runs regardless; this only governs
/// the keyword leg.
fn fts_or_query(raw: &str) -> Option<String> {
    let terms: Vec<String> = raw
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .take(32) // bound the expression size for pathological inputs
        .map(|t| format!("\"{t}\""))
        .collect();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

/// The FTS keyword leg of a hybrid query (`?2` is the MATCH expression), or
/// nothing when the query yields no safe terms — the search is then
/// vector-only (still returns nearest neighbours).
fn keyword_leg(fts: &Option<String>) -> &'static str {
    if fts.is_some() {
        "UNION ALL
                SELECT rowid as node_id, rank as score FROM fts_nodes WHERE fts_nodes MATCH ?2"
    } else {
        ""
    }
}

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

/// The string at `key` in a JSON payload, if present.
fn payload_str(payload: &serde_json::Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

pub(super) fn hybrid_search(
    conn: &Connection,
    query_text: &str,
    query_embedding: &[f32],
    tenant_id: &TenantId,
    limit: usize,
) -> Result<Vec<MemoryNode>> {
    let fts = fts_or_query(query_text);
    let keyword_leg = keyword_leg(&fts);
    let query = format!(
        "
            WITH hybrid_matches AS (
                -- Semantic
                SELECT node_id, distance as score FROM vec_nodes WHERE embedding MATCH ?1 AND k = ?5
                {keyword_leg}
            ),
            -- Tenant/status are filtered BEFORE the limit, so rows the caller
            -- can never see (other tenants, archived) don't take result slots.
            ranked_matches AS (
                SELECT h.node_id, SUM(h.score) as combined_score
                FROM hybrid_matches h JOIN nodes rn ON rn.id = h.node_id
                WHERE rn.tenant_id = ?4 AND rn.status = 'active'
                GROUP BY h.node_id ORDER BY combined_score LIMIT ?3
            )
            SELECT {NODE_COLUMNS}
            FROM nodes n
            JOIN ranked_matches rm ON n.id = rm.node_id
            WHERE n.tenant_id = ?4 AND n.status = 'active'
            ORDER BY rm.combined_score ASC;
        "
    );

    let mut stmt = conn.prepare(&query)?;
    let nodes = stmt
        .query_map(
            params![
                as_blob(query_embedding),
                fts.unwrap_or_default(),
                limit as i64,
                tenant_id.0,
                knn_k(limit)
            ],
            row_to_node,
        )?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(nodes)
}

pub(super) fn query_with_graph(
    conn: &Connection,
    query_text: &str,
    query_embedding: &[f32],
    tenant_id: &TenantId,
    time_filter: &TimeFilter,
    ccl_filter: &[String],
    limit: usize,
) -> Result<Vec<MemoryResult>> {
    let ccl_json = serde_json::to_string(ccl_filter)?;

    // Blueprint section 2.5: Hybrid + Graph + Temporal query.
    // Keyword leg is included only when the query yields safe FTS terms
    // (see `fts_or_query`); otherwise vector recall carries the search alone.
    let fts = fts_or_query(query_text);
    let keyword_leg = keyword_leg(&fts);
    let query = format!(
        "
            WITH hybrid_matches AS (
                SELECT node_id, distance as score FROM vec_nodes WHERE embedding MATCH ?1 AND k = ?8
                {keyword_leg}
            ),
            -- Direct hits honour the caller's breadth (`?4`); the old hard-coded
            -- `LIMIT 5` silently capped every k > 5 (DEV-8).
            -- Visibility filters (tenant, status, layer, time) run BEFORE the
            -- limit so invisible rows never take one of the k slots.
            ranked_matches AS (
                SELECT h.node_id, SUM(h.score) as combined_score
                FROM hybrid_matches h JOIN nodes rn ON rn.id = h.node_id
                WHERE rn.tenant_id = ?3 AND rn.status = 'active'
                  AND rn.ccl IN (SELECT value FROM json_each(?7))
                  AND (rn.created_at >= ?5 OR ?5 IS NULL)
                  AND (rn.created_at <= ?6 OR ?6 IS NULL)
                GROUP BY h.node_id ORDER BY combined_score LIMIT ?4
            ),
            graph_context AS (
                SELECT node_id FROM ranked_matches
                UNION
                SELECT target_id AS node_id FROM edges
                WHERE source_id IN (SELECT node_id FROM ranked_matches)
                  AND (valid_until IS NULL OR ?5 IS NULL OR valid_until >= ?5)
                  AND (valid_from IS NULL OR ?6 IS NULL OR valid_from <= ?6)
                  AND ccl IN (SELECT value FROM json_each(?7))
                UNION
                SELECT source_id AS node_id FROM edges
                WHERE target_id IN (SELECT node_id FROM ranked_matches)
                  AND (valid_until IS NULL OR ?5 IS NULL OR valid_until >= ?5)
                  AND (valid_from IS NULL OR ?6 IS NULL OR valid_from <= ?6)
                  AND ccl IN (SELECT value FROM json_each(?7))
            )
            SELECT
                n.id, n.payload, n.ccl, n.relevance_score, n.updated_at
            FROM nodes n
            JOIN graph_context gc ON n.id = gc.node_id
            LEFT JOIN ranked_matches rm ON rm.node_id = n.id
            WHERE n.tenant_id = ?3 AND n.status = 'active'
              AND (n.created_at >= ?5 OR ?5 IS NULL)
              AND (n.created_at <= ?6 OR ?6 IS NULL)
              AND n.ccl IN (SELECT value FROM json_each(?7))
            -- Rank by SEARCH quality, not decay score: direct hybrid matches
            -- first (ordered by their combined vector+keyword score), then the
            -- 1-hop graph neighbours, with relevance as a final tiebreak. The old
            -- `ORDER BY relevance_score DESC` ignored match quality entirely, so
            -- once reads boosted several facts to 1.0 the best match no longer
            -- ranked first (field-report round 2, ranking nit).
            ORDER BY (rm.combined_score IS NULL), rm.combined_score ASC, n.relevance_score DESC
            LIMIT ?4;
        "
    );

    let mut stmt = conn.prepare(&query)?;

    let rows: Vec<(i64, String, String, String)> = stmt
        .query_map(
            params![
                as_blob(query_embedding),
                fts.unwrap_or_default(),
                tenant_id.0,
                limit as i64,
                time_filter.after,
                time_filter.before,
                &ccl_json,
                knn_k(limit),
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(4)?,
                ))
            },
        )?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    // Collect node IDs for relevance boost
    let node_ids: Vec<i64> = rows.iter().map(|(id, _, _, _)| *id).collect();
    if !node_ids.is_empty() {
        boost_relevance(conn, &node_ids)?;
    }

    // Build token-optimized output with 1-hop connections
    let mut results = Vec::new();
    for (node_id, payload_json, ccl, updated_at) in rows {
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
        });
    }

    Ok(results)
}

/// A node's 1-hop connections (both edge directions) within the CCL layers in
/// `ccl_json`, each naming the neighbour by its fact text.
fn connections(conn: &Connection, node_id: i64, ccl_json: &str) -> Result<Vec<MemoryConnection>> {
    let mut edge_stmt = conn.prepare(
        "SELECT e.relation, e.ccl, e.valid_from, e.valid_until, n2.payload
         FROM edges e
         JOIN nodes n2 ON n2.id = e.target_id
         WHERE e.source_id = ?1
           AND e.ccl IN (SELECT value FROM json_each(?2))
         UNION ALL
         SELECT e.relation, e.ccl, e.valid_from, e.valid_until, n2.payload
         FROM edges e
         JOIN nodes n2 ON n2.id = e.source_id
         WHERE e.target_id = ?1
           AND e.ccl IN (SELECT value FROM json_each(?2))",
    )?;

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
    use super::fts_or_query;

    /// The FTS sanitizer OR-joins quoted terms (any-term match, not all-terms)
    /// and neutralizes punctuation that would otherwise be an FTS5 syntax error.
    #[test]
    fn test_fts_or_query_sanitizes() {
        assert_eq!(
            fts_or_query("annual budget report"),
            Some("\"annual\" OR \"budget\" OR \"report\"".to_string())
        );
        // Punctuation (colons, parens, hyphens, quotes) becomes delimiters, so no
        // FTS operator leaks through to cause a syntax error.
        assert_eq!(
            fts_or_query("Acme-Corp (2022): \"report\""),
            Some("\"Acme\" OR \"Corp\" OR \"2022\" OR \"report\"".to_string())
        );
        // No alphanumeric terms → None (keyword leg is skipped, vector-only).
        assert_eq!(fts_or_query("   -:()  "), None);
    }
}
