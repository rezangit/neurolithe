//! The `vec_nodes` KNN index: which nodes enter it, how vectors are written
//! and removed, and how wide a KNN query reaches.

use crate::infrastructure::embedding_codec::as_blob;
use anyhow::Result;
use rusqlite::{Connection, params};

/// How many vector candidates to pull per requested result. The vec0 `k` cut
/// runs before the tenant/status filters (no partition key yet — Phase 2
/// workspaces remove the tenant filter entirely), so over-fetching keeps the
/// filters from starving the result set.
const KNN_OVERFETCH: usize = 4;

/// sqlite-vec's default upper bound on `k` for a vec0 KNN query.
const KNN_MAX_K: usize = 4096;

/// KNN `k` for a request of `limit` results: over-fetched, clamped to vec0's bound.
pub(super) fn knn_k(limit: usize) -> i64 {
    limit.saturating_mul(KNN_OVERFETCH).clamp(1, KNN_MAX_K) as i64
}

/// Whether an embedding carries meaning worth indexing. Empty or all-zero
/// vectors (graph-anchor `subject` nodes) are equidistant from everything, so
/// indexing them only burns KNN slots that real matches need (ARC-17).
fn is_indexable(embedding: &[f32]) -> bool {
    embedding.iter().any(|x| *x != 0.0)
}

/// Index a node's vector — but only an `active` node with a meaningful
/// vector: zero-vector anchors and archived nodes would otherwise occupy
/// top-k slots and push real matches out (ARC-17).
pub(super) fn index_if_eligible(
    conn: &Connection,
    node_id: i64,
    status: &str,
    embedding: &[f32],
) -> Result<()> {
    if status == "active" && is_indexable(embedding) {
        conn.execute(
            "INSERT INTO vec_nodes(node_id, embedding) VALUES (?1, ?2)",
            params![node_id, as_blob(embedding)],
        )?;
    }
    Ok(())
}

/// Drop a node's vector (no-op if it has none).
pub(super) fn unindex(conn: &Connection, node_id: i64) -> Result<()> {
    conn.execute("DELETE FROM vec_nodes WHERE node_id = ?1", params![node_id])?;
    Ok(())
}
