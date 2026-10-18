//! The CCL layer registry: custom cognitive layers registered per tenant.

use crate::domain::models::{CclDefinition, TenantId};
use anyhow::Result;
use rusqlite::{Connection, params};

pub(super) fn store_ccl_definition(conn: &Connection, definition: &CclDefinition) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO ccl_registry (tenant_id, name, description) VALUES (?1, ?2, ?3)",
        params![
            definition.tenant_id.0,
            definition.name,
            definition.description
        ],
    )?;
    Ok(())
}

pub(super) fn get_ccl_definitions(
    conn: &Connection,
    tenant_id: &TenantId,
) -> Result<Vec<CclDefinition>> {
    let mut stmt =
        conn.prepare("SELECT id, name, description FROM ccl_registry WHERE tenant_id = ?1")?;
    let definitions = stmt
        .query_map(params![tenant_id.0], |row| {
            Ok(CclDefinition {
                id: Some(row.get(0)?),
                tenant_id: tenant_id.clone(),
                name: row.get(1)?,
                description: row.get(2)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(definitions)
}
