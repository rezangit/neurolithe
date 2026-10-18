//! Read-only views of the STM store for introspection: aggregate statistics,
//! a paginated node listing, and dataId counts.

use crate::domain::ports::{StmNodeSummary, StmStats};
use crate::infrastructure::database::db_size_bytes;
use anyhow::Result;
use rusqlite::{Connection, params};

/// Bins in the relevance histogram of [`stm_stats`] (equal-width over 0–1).
const DECAY_HISTOGRAM_BINS: i64 = 5;

pub(super) fn stm_stats(conn: &Connection) -> Result<StmStats> {
    let active: i64 = conn.query_row(
        "SELECT COUNT(*) FROM nodes WHERE status = 'active'",
        [],
        |r| r.get(0),
    )?;
    let archived: i64 = conn.query_row(
        "SELECT COUNT(*) FROM nodes WHERE status = 'archived'",
        [],
        |r| r.get(0),
    )?;
    let avg_relevance: f64 = conn.query_row(
        "SELECT COALESCE(AVG(relevance_score), 0) FROM nodes WHERE status = 'active'",
        [],
        |r| r.get(0),
    )?;

    // Equal-width relevance bins; score 1.0 clamps into the top bin.
    let mut decay_histogram = vec![0i64; DECAY_HISTOGRAM_BINS as usize];
    let mut stmt = conn.prepare(
        "SELECT MIN(?1 - 1, CAST(relevance_score * ?1 AS INTEGER)) AS bin, COUNT(*)
         FROM nodes WHERE status = 'active' GROUP BY bin",
    )?;
    let rows = stmt.query_map(params![DECAY_HISTOGRAM_BINS], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
    })?;
    for row in rows {
        let (bin, count) = row?;
        if (0..DECAY_HISTOGRAM_BINS).contains(&bin) {
            decay_histogram[bin as usize] = count;
        }
    }

    Ok(StmStats {
        active_nodes: active,
        archived_nodes: archived,
        avg_relevance,
        decay_histogram,
        db_size_bytes: db_size_bytes(conn)?,
    })
}

pub(super) fn list_node_summaries(
    conn: &Connection,
    limit: usize,
    offset: usize,
    status: Option<&str>,
    contains: Option<&str>,
) -> Result<Vec<StmNodeSummary>> {
    // Most-relevant first; optional status + substring filters, paginated.
    // NULL-guarded filters (`?N IS NULL OR …`) keep a single prepared
    // statement. `contains` matches the fact text case-insensitively (LIKE is
    // case-insensitive for ASCII); `%` in the term is passed through as-is.
    let mut stmt = conn.prepare(
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
            Ok(StmNodeSummary {
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

/// Nodes carrying a dataId (`?1`). Served by `idx_nodes_data_id`.
pub(super) const COUNT_BY_DATA_ID_SQL: &str =
    "SELECT COUNT(*) FROM nodes WHERE json_extract(payload, '$.dataId') = ?1";

pub(super) fn count_by_data_id(conn: &Connection, data_id: &str) -> Result<i64> {
    Ok(conn.query_row(COUNT_BY_DATA_ID_SQL, params![data_id], |r| r.get(0))?)
}
