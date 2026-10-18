//! The working-memory view of one context (STM-GRAPH): its recent turns,
//! what each turn was `about`, and lookup of a context's subject anchors.

use crate::domain::models::{MemoryConnection, MemoryResult, TenantId};
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};

pub(super) fn recent_in_context(
    conn: &Connection,
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
    let mut stmt = conn.prepare(
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
    let turns: Vec<(i64, MemoryResult)> = stmt
        .query_map(
            params![tenant_id.0, context_key, ccl_json, limit as i64],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    MemoryResult {
                        fact: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                        ccl: row.get(2)?,
                        last_updated: row.get(3)?,
                        connections: Vec::new(),
                        data_id: row.get(5)?,
                        context_key: row.get(4)?,
                        score: None,
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
        turn.connections = about_subjects(conn, turn_id)?;
        results.push(turn);
    }
    Ok(results)
}

/// A turn's `about` edges (`?1` turn id) with each subject's label and
/// dataId. Served by `idx_edges_source (source_id, relation)`.
pub(super) const ABOUT_SUBJECTS_SQL: &str = "
    SELECT json_extract(s.payload, '$.fact'), json_extract(s.payload, '$.dataId')
    FROM edges e JOIN nodes s ON s.id = e.target_id
    WHERE e.source_id = ?1 AND e.relation = 'about'
    ORDER BY s.id";

/// The newest active `working` subject anchor for a dataId in one context
/// (`?1` tenant, `?2` context key, `?3` dataId).
pub(super) const FIND_WORKING_SUBJECT_SQL: &str = "
    SELECT id FROM nodes
    WHERE tenant_id = ?1 AND context_key = ?2 AND ccl = 'working' AND status = 'active'
      AND json_extract(payload, '$.kind') = 'subject'
      AND json_extract(payload, '$.dataId') = ?3
    ORDER BY id DESC LIMIT 1";

/// The `about` subjects of a turn (STM-GRAPH), as connections: one per
/// `turn —about→ subject` edge, entity = "label (dataId)" so the caller sees
/// both the human label and the id it needs to act on.
fn about_subjects(conn: &Connection, turn_id: i64) -> Result<Vec<MemoryConnection>> {
    let mut stmt = conn.prepare(ABOUT_SUBJECTS_SQL)?;
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
        out.push(MemoryConnection {
            relation: "about".into(),
            entity,
            ccl: "working".into(),
            valid_from: None,
            valid_until: None,
        });
    }
    Ok(out)
}

pub(super) fn find_working_subject(
    conn: &Connection,
    tenant_id: &TenantId,
    context_key: &str,
    data_id: &str,
) -> Result<Option<i64>> {
    let id = conn
        .query_row(
            FIND_WORKING_SUBJECT_SQL,
            params![tenant_id.0, context_key, data_id],
            |r| r.get::<_, i64>(0),
        )
        .optional()?;
    Ok(id)
}
