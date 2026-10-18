//! Node lifecycle: the decay sweep that archives faded nodes, deletion (by
//! tenant, by dataId, or the whole store) and tenant export.

use super::vector_index::unindex;
use crate::domain::decay::{DecayEngine, should_archive};
use crate::domain::models::TenantId;
use anyhow::Result;
use rusqlite::{Connection, Transaction, TransactionBehavior, params};

/// Active nodes decayed per sweep transaction. Small enough that a batch
/// holds the write lock for milliseconds, so other writers (another process
/// sharing the store, or queued work) interleave between batches.
pub(super) const SWEEP_BATCH: usize = 500;

/// Decay every active node, [`SWEEP_BATCH`] nodes per transaction.
pub(super) fn sweep_decay(conn: &Connection, engine: &DecayEngine) -> Result<()> {
    let mut after_id = 0;
    loop {
        let (last_id, rows) = sweep_decay_batch(conn, engine, after_id, SWEEP_BATCH)?;
        if rows < SWEEP_BATCH {
            return Ok(());
        }
        after_id = last_id;
    }
}

/// Decay the next `batch` active nodes with an id above `after_id`, in one
/// write transaction. Returns the last id handled and how many rows were.
///
/// Real-elapsed decay: each node decays by its *true* age in days since it
/// was last decayed (or last read/created for never-swept rows), not a fixed
/// pass. This decouples decay from sweep cadence — a note written seconds ago
/// survives a sweep or restart, and `working` notes archive only after their
/// real (short) half-life. `julianday` yields days as a float; a clamp guards
/// against clock skew making elapsed negative.
///
/// The batch is read *inside* the transaction (BEGIN IMMEDIATE), so a boost
/// committed by another connection can't be overwritten with a stale score
/// (DEV-19).
fn sweep_decay_batch(
    conn: &Connection,
    engine: &DecayEngine,
    after_id: i64,
    batch: usize,
) -> Result<(i64, usize)> {
    // IMMEDIATE explicitly, not the connection's default: DEV-19 depends on
    // holding the write lock before the batch is read.
    let tx = Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    let nodes: Vec<(i64, f64, String, f64)> = tx
        .prepare_cached(
            "SELECT id, relevance_score, ccl,
                    MAX(0.0, julianday('now') - julianday(COALESCE(last_decayed_at, last_accessed_at))) AS elapsed_days
             FROM nodes WHERE status = 'active' AND id > ?1
             ORDER BY id LIMIT ?2",
        )?
        .query_map(params![after_id, batch as i64], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, f64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    {
        let mut update = tx.prepare_cached(
            "UPDATE nodes SET relevance_score = ?1, status = ?2, last_decayed_at = CURRENT_TIMESTAMP WHERE id = ?3",
        )?;
        for (id, current_score, ccl, elapsed_days) in &nodes {
            let new_score = engine.calculate_decay_for(*current_score, *elapsed_days, ccl);
            let archive = should_archive(new_score);
            let new_status = if archive { "archived" } else { "active" };

            // Advance the decay clock so the next sweep measures only *new*
            // elapsed time (no double-counting).
            update.execute(params![new_score, new_status, id])?;
            // Archival is terminal in STM: drop the vector so archived nodes
            // stop consuming KNN slots (ARC-17).
            if archive {
                unindex(&tx, *id)?;
            }
        }
    }
    tx.commit()?;
    let last_id = nodes.last().map_or(after_id, |n| n.0);
    Ok((last_id, nodes.len()))
}

pub(super) fn delete_tenant(conn: &Connection, tenant_id: &TenantId) -> Result<()> {
    let tx = conn.unchecked_transaction()?;

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

pub(super) fn export_tenant(conn: &Connection, tenant_id: &TenantId) -> Result<String> {
    let mut stmt = conn.prepare("SELECT payload FROM nodes WHERE tenant_id = ?1")?;
    let payload_iter = stmt.query_map(params![tenant_id.0], |row| row.get::<_, String>(0))?;

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

/// The statements of [`delete_nodes_by_data_id`], in FK-safe order (`?1`
/// dataId): edges and vectors first, then the nodes. The dataId match is
/// served by `idx_nodes_data_id`, the edge endpoints by the edge indexes.
pub(super) const DELETE_BY_DATA_ID_SQL: [&str; 3] = [
    "DELETE FROM edges WHERE source_id IN
        (SELECT id FROM nodes WHERE json_extract(payload, '$.dataId') = ?1)
     OR target_id IN
        (SELECT id FROM nodes WHERE json_extract(payload, '$.dataId') = ?1)",
    "DELETE FROM vec_nodes WHERE node_id IN
        (SELECT id FROM nodes WHERE json_extract(payload, '$.dataId') = ?1)",
    "DELETE FROM nodes WHERE json_extract(payload, '$.dataId') = ?1",
];

pub(super) fn delete_nodes_by_data_id(conn: &Connection, data_id: &str) -> Result<()> {
    let tx = conn.unchecked_transaction()?;
    // Match nodes whose payload carries this dataId. Edges + vectors go
    // first (FK / index hygiene); deleting nodes fires the fts delete trigger.
    for sql in DELETE_BY_DATA_ID_SQL {
        tx.execute(sql, params![data_id])?;
    }
    tx.commit()?;
    Ok(())
}

pub(super) fn reset_store(conn: &Connection) -> Result<()> {
    let tx = conn.unchecked_transaction()?;
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
