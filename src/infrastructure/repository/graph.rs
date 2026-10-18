//! Writes to the memory graph: episodes, nodes (with their KNN vector), edges,
//! support/content updates, and the read reinforcement that resets decay.

use super::vector_index::{index_if_eligible, unindex};
use crate::domain::models::{Edge, Episode, MemoryNode};
use anyhow::Result;
use rusqlite::{Connection, params};

pub(super) fn store_episode(conn: &Connection, episode: &Episode) -> Result<i64> {
    conn.execute(
        "INSERT INTO episodes (tenant_id, session_id, raw_dialogue, ccl) VALUES (?1, ?2, ?3, ?4)",
        params![
            episode.tenant_id.0,
            episode.session_id.0,
            episode.raw_dialogue,
            episode.ccl
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

pub(super) fn store_node(conn: &Connection, node: &MemoryNode, embedding: &[f32]) -> Result<i64> {
    let payload_json = serde_json::to_string(&node.payload)?;

    let tx = conn.unchecked_transaction()?;

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
    index_if_eligible(&tx, node_id, &node.status, embedding)?;

    tx.commit()?;
    Ok(node_id)
}

pub(super) fn store_edge(conn: &Connection, edge: &Edge) -> Result<()> {
    conn.execute(
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

pub(super) fn boost_relevance(conn: &Connection, node_ids: &[i64]) -> Result<()> {
    let placeholders: Vec<String> = (1..=node_ids.len()).map(|i| format!("?{i}")).collect();
    // A read reinforces: score back to 1.0, and the decay clock restarts
    // (last_decayed_at = now) so an actively-used note stays hot — decay
    // measures inactivity, not wall-clock age.
    let query = format!(
        "UPDATE nodes SET relevance_score = 1.0, last_accessed_at = CURRENT_TIMESTAMP, last_decayed_at = CURRENT_TIMESTAMP WHERE id IN ({})",
        placeholders.join(", ")
    );
    conn.execute(&query, rusqlite::params_from_iter(node_ids))?;
    Ok(())
}

pub(super) fn update_node_support(
    conn: &Connection,
    node_id: i64,
    new_payload: Option<&serde_json::Value>,
) -> Result<()> {
    if let Some(payload) = new_payload {
        let payload_json = serde_json::to_string(payload)?;
        conn.execute(
            "UPDATE nodes SET support_count = support_count + 1, relevance_score = 1.0, updated_at = CURRENT_TIMESTAMP, payload = ?1 WHERE id = ?2",
            params![payload_json, node_id],
        )?;
    } else {
        conn.execute(
            "UPDATE nodes SET support_count = support_count + 1, relevance_score = 1.0, updated_at = CURRENT_TIMESTAMP WHERE id = ?1",
            params![node_id],
        )?;
    }
    Ok(())
}

pub(super) fn update_node_content(
    conn: &Connection,
    node_id: i64,
    payload: &serde_json::Value,
    embedding: &[f32],
) -> Result<()> {
    let payload_json = serde_json::to_string(payload)?;
    let tx = conn.unchecked_transaction()?;
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
    unindex(&tx, node_id)?;
    index_if_eligible(&tx, node_id, &status, embedding)?;
    tx.commit()?;
    Ok(())
}
