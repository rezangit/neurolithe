use crate::domain::models::{Episode, MemoryNode, MemoryResult, TenantId};
use crate::domain::ports::MemoryRepository;
use crate::infrastructure::database::db_size_bytes;
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};

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

/// How many vector candidates to pull per requested result. The vec0 `k` cut
/// runs before the tenant/status filters (no partition key yet — Phase 2
/// workspaces remove the tenant filter entirely), so over-fetching keeps the
/// filters from starving the result set.
const KNN_OVERFETCH: usize = 4;

/// sqlite-vec's default upper bound on `k` for a vec0 KNN query.
const KNN_MAX_K: usize = 4096;

/// KNN `k` for a request of `limit` results: over-fetched, clamped to vec0's bound.
fn knn_k(limit: usize) -> i64 {
    limit.saturating_mul(KNN_OVERFETCH).clamp(1, KNN_MAX_K) as i64
}

/// View an `f32` embedding as the little-endian byte blob sqlite-vec expects.
fn vec_bytes(embedding: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding or invalid bit patterns; the byte slice
    // borrows the same memory for the same lifetime with the exact byte length.
    unsafe {
        std::slice::from_raw_parts(
            embedding.as_ptr() as *const u8,
            std::mem::size_of_val(embedding),
        )
    }
}

/// Whether an embedding carries meaning worth indexing. Empty or all-zero
/// vectors (graph-anchor `subject` nodes) are equidistant from everything, so
/// indexing them only burns KNN slots that real matches need (ARC-17).
fn is_indexable(embedding: &[f32]) -> bool {
    embedding.iter().any(|x| *x != 0.0)
}

pub struct SqliteMemoryRepository {
    conn: Connection,
    /// Keeps the workspace marked "open" (its shared advisory lock) for
    /// exactly as long as this connection lives — i.e. as long as anything
    /// still holds the repository.
    _lease: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
}

impl SqliteMemoryRepository {
    pub fn new(conn: Connection) -> Self {
        Self { conn, _lease: None }
    }

    /// Tie a workspace lease to this repository's connection lifetime.
    pub fn with_lease(mut self, lease: std::sync::Arc<dyn std::any::Any + Send + Sync>) -> Self {
        self._lease = Some(lease);
        self
    }

    /// The `about` subjects of a turn (STM-GRAPH), as connections: one per
    /// `turn —about→ subject` edge, entity = "label (dataId)" so the caller sees
    /// both the human label and the id it needs to act on.
    fn about_subjects(&self, turn_id: i64) -> Result<Vec<crate::domain::models::MemoryConnection>> {
        let mut stmt = self.conn.prepare(
            "SELECT json_extract(s.payload, '$.fact'), json_extract(s.payload, '$.dataId')
             FROM edges e JOIN nodes s ON s.id = e.target_id
             WHERE e.source_id = ?1 AND e.relation = 'about'
             ORDER BY s.id",
        )?;
        let rows = stmt.query_map(params![turn_id], |row| {
            let label: Option<String> = row.get(0)?;
            let data_id: Option<String> = row.get(1)?;
            Ok((label.unwrap_or_default(), data_id.unwrap_or_default()))
        })?;

        let mut out = Vec::new();
        for row in rows {
            let (label, data_id) = row?;
            let entity = if data_id.is_empty() {
                label
            } else if label.is_empty() {
                data_id
            } else {
                format!("{label} ({data_id})")
            };
            out.push(crate::domain::models::MemoryConnection {
                relation: "about".into(),
                entity,
                ccl: "working".into(),
                valid_from: None,
                valid_until: None,
            });
        }
        Ok(out)
    }
}

impl MemoryRepository for SqliteMemoryRepository {
    fn store_ccl_definition(
        &self,
        definition: &crate::domain::models::CclDefinition,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO ccl_registry (tenant_id, name, description) VALUES (?1, ?2, ?3)",
            params![
                definition.tenant_id.0,
                definition.name,
                definition.description
            ],
        )?;
        Ok(())
    }

    fn get_ccl_definitions(
        &self,
        tenant_id: &TenantId,
    ) -> Result<Vec<crate::domain::models::CclDefinition>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, description FROM ccl_registry WHERE tenant_id = ?1")?;
        let def_iter = stmt.query_map(params![tenant_id.0], |row| {
            Ok(crate::domain::models::CclDefinition {
                id: Some(row.get(0)?),
                tenant_id: tenant_id.clone(),
                name: row.get(1)?,
                description: row.get(2)?,
            })
        })?;
        let mut results = Vec::new();
        for def in def_iter {
            results.push(def?);
        }
        Ok(results)
    }

    fn store_episode(&self, episode: &Episode) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO episodes (tenant_id, session_id, raw_dialogue, ccl) VALUES (?1, ?2, ?3, ?4)",
            params![
                episode.tenant_id.0,
                episode.session_id.0,
                episode.raw_dialogue,
                episode.ccl
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    fn store_node(&self, node: &MemoryNode, embedding: &[f32]) -> Result<i64> {
        let payload_json = serde_json::to_string(&node.payload)?;

        let tx = self.conn.unchecked_transaction()?;

        tx.execute(
            "INSERT INTO nodes (tenant_id, source_episode_id, payload, status, ccl, is_explicit, support_count, relevance_score, context_key)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                node.tenant_id.0,
                node.source_episode_id,
                payload_json,
                node.status,
                node.ccl,
                node.is_explicit,
                node.support_count,
                node.relevance_score,
                node.context_key
            ],
        )?;
        let node_id = tx.last_insert_rowid();

        // Only active nodes with a meaningful vector enter the KNN index:
        // zero-vector anchors and archived nodes would otherwise occupy top-k
        // slots and push real matches out (ARC-17).
        if node.status == "active" && is_indexable(embedding) {
            tx.execute(
                "INSERT INTO vec_nodes(node_id, embedding) VALUES (?1, ?2)",
                params![node_id, vec_bytes(embedding)],
            )?;
        }

        tx.commit()?;
        Ok(node_id)
    }

    fn store_edge(&self, edge: &crate::domain::models::Edge) -> Result<()> {
        self.conn.execute(
            "INSERT INTO edges (source_id, target_id, relation, ccl, valid_from, valid_until, weight) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                edge.source_id,
                edge.target_id,
                edge.relation,
                edge.ccl,
                edge.valid_from,
                edge.valid_until,
                edge.weight
            ],
        )?;
        Ok(())
    }

    fn hybrid_search(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        tenant_id: &TenantId,
        limit: usize,
    ) -> Result<Vec<MemoryNode>> {
        let embedding_bytes = vec_bytes(query_embedding);

        // The keyword leg is included only when the query yields safe FTS terms;
        // otherwise the search is vector-only (still returns nearest neighbors).
        let fts = fts_or_query(query_text);
        let keyword_leg = if fts.is_some() {
            "UNION ALL
                SELECT rowid as node_id, rank as score FROM fts_nodes WHERE fts_nodes MATCH ?2"
        } else {
            ""
        };
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
            SELECT
                n.id, n.tenant_id, n.source_episode_id, n.payload, n.status, n.ccl, n.is_explicit, n.support_count, n.relevance_score, n.context_key
            FROM nodes n
            JOIN ranked_matches rm ON n.id = rm.node_id
            WHERE n.tenant_id = ?4 AND n.status = 'active'
            ORDER BY rm.combined_score ASC;
        "
        );

        let mut stmt = self.conn.prepare(&query)?;

        let node_iter = stmt.query_map(
            params![
                embedding_bytes,
                fts.unwrap_or_default(),
                limit as i64,
                tenant_id.0,
                knn_k(limit)
            ],
            |row| {
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
            },
        )?;

        let mut results = Vec::new();
        for node in node_iter {
            results.push(node?);
        }

        Ok(results)
    }

    fn query_with_graph(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        tenant_id: &crate::domain::models::TenantId,
        time_filter: &crate::domain::models::TimeFilter,
        ccl_filter: &[String],
        limit: usize,
    ) -> Result<Vec<crate::domain::models::MemoryResult>> {
        let ccl_json = serde_json::to_string(ccl_filter)?;

        let embedding_bytes = vec_bytes(query_embedding);

        // Blueprint section 2.5: Hybrid + Graph + Temporal query.
        // Keyword leg is included only when the query yields safe FTS terms
        // (see `fts_or_query`); otherwise vector recall carries the search alone.
        let fts = fts_or_query(query_text);
        let keyword_leg = if fts.is_some() {
            "UNION ALL
                SELECT rowid as node_id, rank as score FROM fts_nodes WHERE fts_nodes MATCH ?2"
        } else {
            ""
        };
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

        let mut stmt = self.conn.prepare(&query)?;

        let rows: Vec<(i64, String, String, f64, String)> = stmt
            .query_map(
                params![
                    embedding_bytes,
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
                        row.get::<_, f64>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                },
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        // Collect node IDs for relevance boost
        let node_ids: Vec<i64> = rows.iter().map(|(id, _, _, _, _)| *id).collect();
        if !node_ids.is_empty() {
            self.boost_relevance(&node_ids)?;
        }

        // Build token-optimized output with 1-hop connections
        let mut results = Vec::new();
        for (node_id, payload_str, ccl, _, updated_at) in rows {
            let payload: serde_json::Value =
                serde_json::from_str(&payload_str).unwrap_or(serde_json::Value::Null);
            let fact = payload
                .get("fact")
                .and_then(|f| f.as_str())
                .unwrap_or("")
                .to_string();
            // Surface the archive reference so a hit can be traced/fetched.
            let data_id = payload
                .get("dataId")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string());

            // Get 1-hop connections for this node
            let mut edge_stmt = self.conn.prepare(
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

            let connections: Vec<crate::domain::models::MemoryConnection> = edge_stmt
                .query_map(params![node_id, &ccl_json], |row| {
                    let rel: String = row.get(0)?;
                    let edge_ccl: String = row.get(1)?;
                    let vf: Option<String> = row.get(2)?;
                    let vu: Option<String> = row.get(3)?;
                    let entity_payload: String = row.get(4)?;
                    let ep: serde_json::Value =
                        serde_json::from_str(&entity_payload).unwrap_or(serde_json::Value::Null);
                    let entity = ep
                        .get("fact")
                        .and_then(|f| f.as_str())
                        .unwrap_or("")
                        .to_string();
                    Ok(crate::domain::models::MemoryConnection {
                        relation: rel,
                        entity,
                        ccl: edge_ccl,
                        valid_from: vf,
                        valid_until: vu,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;

            results.push(crate::domain::models::MemoryResult {
                fact,
                ccl,
                last_updated: updated_at,
                connections,
                data_id,
                context_key: None,
            });
        }

        Ok(results)
    }

    fn recent_in_context(
        &self,
        tenant_id: &TenantId,
        ccl_filter: &[String],
        context_key: &str,
        limit: usize,
    ) -> Result<Vec<MemoryResult>> {
        let ccl_json = serde_json::to_string(ccl_filter)?;

        // Newest-touched active TURN nodes in this exact context. `context_key =
        // ?` excludes NULL-context knowledge facts and other threads; the `kind`
        // guard excludes graph-anchor `subject` nodes (they're reached via edges,
        // never listed as turns). Old flat notes have no `kind` → treated as
        // turns. An empty ccl filter matches any layer.
        let mut stmt = self.conn.prepare(
            "SELECT id, json_extract(payload, '$.fact'), ccl, updated_at, context_key,
                    json_extract(payload, '$.dataId')
             FROM nodes
             WHERE tenant_id = ?1
               AND status = 'active'
               AND context_key = ?2
               AND COALESCE(json_extract(payload, '$.kind'), 'turn') != 'subject'
               AND (json_array_length(?3) = 0 OR ccl IN (SELECT value FROM json_each(?3)))
             ORDER BY last_accessed_at DESC, updated_at DESC
             LIMIT ?4",
        )?;

        // (id, MemoryResult) — collect first so the connection queries below
        // don't re-borrow the prepared statement's connection.
        let turns: Vec<(i64, crate::domain::models::MemoryResult)> = stmt
            .query_map(
                params![tenant_id.0, context_key, ccl_json, limit as i64],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        crate::domain::models::MemoryResult {
                            fact: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                            ccl: row.get(2)?,
                            last_updated: row.get(3)?,
                            connections: Vec::new(),
                            data_id: row.get(5)?,
                            context_key: row.get(4)?,
                        },
                    ))
                },
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        // Attach each turn's `about` subjects (STM-GRAPH) as connections, so the
        // caller sees what each turn was about — and the newest turn's subjects
        // are the current focus.
        let mut results = Vec::with_capacity(turns.len());
        for (turn_id, mut turn) in turns {
            turn.connections = self.about_subjects(turn_id)?;
            results.push(turn);
        }
        Ok(results)
    }

    fn boost_relevance(&self, node_ids: &[i64]) -> Result<()> {
        let placeholders: Vec<String> = node_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect();
        // A read reinforces: score back to 1.0, and the decay clock restarts
        // (last_decayed_at = now) so an actively-used note stays hot — decay
        // measures inactivity, not wall-clock age.
        let query = format!(
            "UPDATE nodes SET relevance_score = 1.0, last_accessed_at = CURRENT_TIMESTAMP, last_decayed_at = CURRENT_TIMESTAMP WHERE id IN ({})",
            placeholders.join(", ")
        );
        let params: Vec<Box<dyn rusqlite::types::ToSql>> = node_ids
            .iter()
            .map(|id| Box::new(*id) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();
        self.conn.execute(&query, param_refs.as_slice())?;
        Ok(())
    }

    fn find_similar_nodes(
        &self,
        embedding: &[f32],
        tenant_id: &TenantId,
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<(MemoryNode, f64)>> {
        let query = "
            SELECT n.id, n.tenant_id, n.source_episode_id, n.payload, n.status, n.ccl, n.is_explicit, n.support_count, n.relevance_score, n.context_key, v.distance
            FROM vec_nodes v
            JOIN nodes n ON n.id = v.node_id
            WHERE v.embedding MATCH ?1 AND k = ?2
              AND n.tenant_id = ?3 AND n.status = 'active'
              AND v.distance <= ?4
            ORDER BY v.distance ASC
            LIMIT ?5;
        ";

        let mut stmt = self.conn.prepare(query)?;

        let node_iter = stmt.query_map(
            params![
                vec_bytes(embedding),
                knn_k(limit),
                tenant_id.0,
                threshold,
                limit as i64
            ],
            |row| {
                let payload_str: String = row.get(3)?;
                let node = MemoryNode {
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
                };
                Ok((node, row.get::<_, f64>(10)?))
            },
        )?;

        Ok(node_iter.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn update_node_support(
        &self,
        node_id: i64,
        new_payload: Option<&serde_json::Value>,
    ) -> Result<()> {
        if let Some(payload) = new_payload {
            let payload_json = serde_json::to_string(payload)?;
            self.conn.execute(
                "UPDATE nodes SET support_count = support_count + 1, relevance_score = 1.0, updated_at = CURRENT_TIMESTAMP, payload = ?1 WHERE id = ?2",
                params![payload_json, node_id],
            )?;
        } else {
            self.conn.execute(
                "UPDATE nodes SET support_count = support_count + 1, relevance_score = 1.0, updated_at = CURRENT_TIMESTAMP WHERE id = ?1",
                params![node_id],
            )?;
        }
        Ok(())
    }

    fn update_node_content(
        &self,
        node_id: i64,
        payload: &serde_json::Value,
        embedding: &[f32],
    ) -> Result<()> {
        let payload_json = serde_json::to_string(payload)?;
        let tx = self.conn.unchecked_transaction()?;
        let status: String = tx.query_row(
            "SELECT status FROM nodes WHERE id = ?1",
            params![node_id],
            |r| r.get(0),
        )?;
        tx.execute(
            "UPDATE nodes SET support_count = support_count + 1, relevance_score = 1.0, updated_at = CURRENT_TIMESTAMP, payload = ?1 WHERE id = ?2",
            params![payload_json, node_id],
        )?;
        // The text changed, so the vector must follow it — a stale embedding
        // would keep matching the *old* meaning (ARC-2/DEV-10).
        tx.execute("DELETE FROM vec_nodes WHERE node_id = ?1", params![node_id])?;
        if status == "active" && is_indexable(embedding) {
            tx.execute(
                "INSERT INTO vec_nodes(node_id, embedding) VALUES (?1, ?2)",
                params![node_id, vec_bytes(embedding)],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    fn delete_tenant(&self, tenant_id: &TenantId) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;

        // FK-safe order (foreign_keys = ON): edges touching the tenant's nodes
        // in either direction, then vectors, nodes, episodes, and finally the
        // tenant's CCL registry (DEV-2/QA-3).
        tx.execute(
            "DELETE FROM edges
             WHERE source_id IN (SELECT id FROM nodes WHERE tenant_id = ?1)
                OR target_id IN (SELECT id FROM nodes WHERE tenant_id = ?1)",
            params![tenant_id.0],
        )?;
        tx.execute(
            "DELETE FROM vec_nodes WHERE node_id IN (SELECT id FROM nodes WHERE tenant_id = ?1)",
            params![tenant_id.0],
        )?;
        tx.execute(
            "DELETE FROM nodes WHERE tenant_id = ?1",
            params![tenant_id.0],
        )?;
        tx.execute(
            "DELETE FROM episodes WHERE tenant_id = ?1",
            params![tenant_id.0],
        )?;
        tx.execute(
            "DELETE FROM ccl_registry WHERE tenant_id = ?1",
            params![tenant_id.0],
        )?;

        tx.commit()?;
        Ok(())
    }

    fn export_tenant(&self, tenant_id: &TenantId) -> Result<String> {
        let mut stmt = self
            .conn
            .prepare("SELECT payload FROM nodes WHERE tenant_id = ?1")?;
        let payload_iter = stmt.query_map(params![tenant_id.0], |row| {
            let p: String = row.get(0)?;
            Ok(p)
        })?;

        let mut all_facts = Vec::new();
        for p in payload_iter {
            let val: serde_json::Value = serde_json::from_str(&p?)?;
            all_facts.push(val);
        }

        let export_json = serde_json::json!({
            "tenant_id": tenant_id.0,
            "extracted_facts": all_facts
        });

        Ok(serde_json::to_string_pretty(&export_json)?)
    }

    fn sweep_decay(&self, engine: &crate::domain::decay::DecayEngine) -> Result<()> {
        // Real-elapsed decay: each node decays by its *true* age in days since it
        // was last decayed (or last read/created for never-swept rows), not a
        // fixed pass. This decouples decay from sweep cadence — a note written
        // seconds ago survives a sweep or restart, and `working` notes archive
        // only after their real (short) half-life. `julianday` yields days as a
        // float; a clamp guards against clock skew making elapsed negative.
        let mut stmt = self.conn.prepare(
            "SELECT id, relevance_score, ccl,
                    MAX(0.0, julianday('now') - julianday(COALESCE(last_decayed_at, last_accessed_at))) AS elapsed_days
             FROM nodes WHERE status = 'active'",
        )?;

        let nodes: Vec<(i64, f64, String, f64)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(3)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let tx = self.conn.unchecked_transaction()?;
        for (id, current_score, ccl, elapsed_days) in nodes {
            let new_score = engine.calculate_decay_for(current_score, elapsed_days, &ccl);
            let new_status = if new_score < 0.1 {
                "archived"
            } else {
                "active"
            };

            // Advance the decay clock so the next sweep measures only *new*
            // elapsed time (no double-counting).
            tx.execute(
                "UPDATE nodes SET relevance_score = ?1, status = ?2, last_decayed_at = CURRENT_TIMESTAMP WHERE id = ?3",
                params![new_score, new_status, id],
            )?;
            // Archival is terminal in STM: drop the vector so archived nodes
            // stop consuming KNN slots (ARC-17).
            if new_status == "archived" {
                tx.execute("DELETE FROM vec_nodes WHERE node_id = ?1", params![id])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn stm_stats(&self) -> Result<crate::domain::ports::StmStats> {
        let active: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM nodes WHERE status = 'active'",
            [],
            |r| r.get(0),
        )?;
        let archived: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM nodes WHERE status = 'archived'",
            [],
            |r| r.get(0),
        )?;
        let avg_relevance: f64 = self.conn.query_row(
            "SELECT COALESCE(AVG(relevance_score), 0) FROM nodes WHERE status = 'active'",
            [],
            |r| r.get(0),
        )?;

        // 5 relevance bins; score 1.0 clamps into the top bin.
        let mut decay_histogram = vec![0i64; 5];
        let mut stmt = self.conn.prepare(
            "SELECT MIN(4, CAST(relevance_score * 5 AS INTEGER)) AS bin, COUNT(*)
             FROM nodes WHERE status = 'active' GROUP BY bin",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?;
        for row in rows {
            let (bin, count) = row?;
            if (0..5).contains(&bin) {
                decay_histogram[bin as usize] = count;
            }
        }

        Ok(crate::domain::ports::StmStats {
            active_nodes: active,
            archived_nodes: archived,
            avg_relevance,
            decay_histogram,
            db_size_bytes: db_size_bytes(&self.conn)?,
        })
    }

    fn find_working_subject(
        &self,
        tenant_id: &TenantId,
        context_key: &str,
        data_id: &str,
    ) -> Result<Option<i64>> {
        let id = self
            .conn
            .query_row(
                "SELECT id FROM nodes
                 WHERE tenant_id = ?1 AND context_key = ?2 AND ccl = 'working' AND status = 'active'
                   AND json_extract(payload, '$.kind') = 'subject'
                   AND json_extract(payload, '$.dataId') = ?3
                 ORDER BY id DESC LIMIT 1",
                params![tenant_id.0, context_key, data_id],
                |r| r.get::<_, i64>(0),
            )
            .optional()?;
        Ok(id)
    }

    fn list_node_summaries(
        &self,
        limit: usize,
        offset: usize,
        status: Option<&str>,
        contains: Option<&str>,
    ) -> Result<Vec<crate::domain::ports::StmNodeSummary>> {
        // Most-relevant first; optional status + substring filters, paginated.
        // NULL-guarded filters (`?N IS NULL OR …`) keep a single prepared
        // statement. `contains` matches the fact text case-insensitively (LIKE is
        // case-insensitive for ASCII); `%` in the term is passed through as-is.
        let mut stmt = self.conn.prepare(
            "SELECT json_extract(payload, '$.fact'), status, relevance_score, support_count,
                    ccl, last_accessed_at, json_extract(payload, '$.dataId')
             FROM nodes
             WHERE (?1 IS NULL OR status = ?1)
               AND (?2 IS NULL OR json_extract(payload, '$.fact') LIKE '%' || ?2 || '%')
             ORDER BY relevance_score DESC, last_accessed_at DESC
             LIMIT ?3 OFFSET ?4",
        )?;
        let rows = stmt.query_map(
            params![status, contains, limit as i64, offset as i64],
            |row| {
                Ok(crate::domain::ports::StmNodeSummary {
                    fact: row.get::<_, Option<String>>(0)?.unwrap_or_default(),
                    status: row.get(1)?,
                    relevance_score: row.get(2)?,
                    support_count: row.get(3)?,
                    ccl: row.get(4)?,
                    last_accessed_at: row.get(5)?,
                    data_id: row.get(6)?,
                })
            },
        )?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn count_by_data_id(&self, data_id: &str) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM nodes WHERE json_extract(payload, '$.dataId') = ?1",
            params![data_id],
            |r| r.get(0),
        )?)
    }

    fn is_command_processed(&self, command_id: &str) -> Result<bool> {
        Ok(self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM processed_commands WHERE command_id = ?1)",
            params![command_id],
            |r| r.get(0),
        )?)
    }

    fn mark_command_processed(&self, command_id: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO processed_commands (command_id) VALUES (?1)",
            params![command_id],
        )?;
        Ok(())
    }

    fn sweep_processed_commands(&self, older_than_days: i64) -> Result<usize> {
        let cutoff = format!("-{older_than_days} days");
        let removed = self.conn.execute(
            "DELETE FROM processed_commands WHERE processed_at < datetime('now', ?1)",
            params![cutoff],
        )?;
        Ok(removed)
    }

    fn delete_nodes_by_data_id(&self, data_id: &str) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        // Match nodes whose payload carries this dataId. Edges + vectors go
        // first (FK / index hygiene); deleting nodes fires the fts delete trigger.
        tx.execute(
            "DELETE FROM edges WHERE source_id IN
                (SELECT id FROM nodes WHERE json_extract(payload, '$.dataId') = ?1)
             OR target_id IN
                (SELECT id FROM nodes WHERE json_extract(payload, '$.dataId') = ?1)",
            params![data_id],
        )?;
        tx.execute(
            "DELETE FROM vec_nodes WHERE node_id IN
                (SELECT id FROM nodes WHERE json_extract(payload, '$.dataId') = ?1)",
            params![data_id],
        )?;
        tx.execute(
            "DELETE FROM nodes WHERE json_extract(payload, '$.dataId') = ?1",
            params![data_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn reset_store(&self) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        // FK-safe order: edges reference nodes, nodes reference episodes.
        // Deleting from `nodes` fires the `nodes_ad` trigger, keeping the
        // `fts_nodes` FTS index in sync — so it is not cleared directly.
        tx.execute("DELETE FROM edges", [])?;
        tx.execute("DELETE FROM vec_nodes", [])?;
        tx.execute("DELETE FROM nodes", [])?;
        tx.execute("DELETE FROM episodes", [])?;
        tx.execute("DELETE FROM ccl_registry", [])?;
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::models::{SessionId, TimeFilter};
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::schema::init_schema;
    use serde_json::json;

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
}

/// Phase 1 regressions (review findings DEV-2/QA-3, DEV-8, ARC-17, ARC-2).
#[cfg(test)]
mod phase1_regression_tests {
    use super::*;
    use crate::domain::models::{CclDefinition, Edge, SessionId, TimeFilter};
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::schema::init_schema;
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
                params![anchor, vec_bytes(&[0.0; 4])],
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
}
