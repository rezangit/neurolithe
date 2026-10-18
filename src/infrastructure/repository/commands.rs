//! The processed-command ledger: ids of bus commands already applied, so a
//! redelivered command is skipped (at-least-once delivery, applied once).

use anyhow::Result;
use rusqlite::{Connection, params};

pub(super) fn is_command_processed(conn: &Connection, command_id: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM processed_commands WHERE command_id = ?1)",
        params![command_id],
        |r| r.get(0),
    )?)
}

pub(super) fn mark_command_processed(conn: &Connection, command_id: &str) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO processed_commands (command_id) VALUES (?1)",
        params![command_id],
    )?;
    Ok(())
}

pub(super) fn sweep_processed_commands(conn: &Connection, older_than_days: i64) -> Result<usize> {
    let cutoff = format!("-{older_than_days} days");
    let removed = conn.execute(
        "DELETE FROM processed_commands WHERE processed_at < datetime('now', ?1)",
        params![cutoff],
    )?;
    Ok(removed)
}
