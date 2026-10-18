//! SQLite implementation of the STM [`MemoryRepository`].
//!
//! One `rusqlite::Connection` per store. The trait impl below is a dispatch
//! table; the SQL lives in one submodule per responsibility:
//!
//! - [`graph`]: writes to the memory graph (episodes, nodes, edges, support,
//!   content, read reinforcement).
//! - [`search`]: hybrid vector + keyword recall, graph expansion, dedup lookup.
//! - [`working`]: the per-context working-memory view (recent turns and their
//!   `about` subjects).
//! - [`lifecycle`]: decay sweep, deletion, export and reset.
//! - [`inspect`]: read-only statistics and listings.
//! - [`registry`]: the CCL layer registry.
//! - [`commands`]: the processed-command ledger (idempotent bus commands).
//! - [`vector_index`]: the `vec_nodes` KNN index rules shared by the above.

mod commands;
mod graph;
mod inspect;
mod lifecycle;
mod registry;
mod search;
mod vector_index;
mod working;

#[cfg(test)]
mod index_tests;
#[cfg(test)]
mod ranking_tests;
#[cfg(test)]
mod regression_tests;
#[cfg(test)]
mod tests;

use crate::domain::decay::DecayEngine;
use crate::domain::models::{
    CclDefinition, Edge, Episode, MemoryNode, MemoryResult, TenantId, TimeFilter,
};
use crate::domain::ports::{MemoryRepository, StmNodeSummary, StmStats};
use crate::domain::tuning::Ranking;
use anyhow::Result;
use rusqlite::Connection;

pub struct SqliteMemoryRepository {
    conn: Connection,
    /// Keeps the workspace marked "open" (its shared advisory lock) for
    /// exactly as long as this connection lives — i.e. as long as anything
    /// still holds the repository.
    _lease: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    /// How search fuses its vector and keyword legs (`[retrieval]`).
    ranking: Ranking,
}

impl SqliteMemoryRepository {
    pub fn new(conn: Connection) -> Self {
        Self {
            conn,
            _lease: None,
            ranking: Ranking::default(),
        }
    }

    /// Use `ranking` for search instead of the defaults.
    pub fn with_ranking(mut self, ranking: Ranking) -> Self {
        self.ranking = ranking;
        self
    }

    /// Tie a workspace lease to this repository's connection lifetime.
    pub fn with_lease(mut self, lease: std::sync::Arc<dyn std::any::Any + Send + Sync>) -> Self {
        self._lease = Some(lease);
        self
    }
}

impl MemoryRepository for SqliteMemoryRepository {
    fn store_ccl_definition(&self, definition: &CclDefinition) -> Result<()> {
        registry::store_ccl_definition(&self.conn, definition)
    }

    fn get_ccl_definitions(&self, tenant_id: &TenantId) -> Result<Vec<CclDefinition>> {
        registry::get_ccl_definitions(&self.conn, tenant_id)
    }

    fn store_episode(&self, episode: &Episode) -> Result<i64> {
        graph::store_episode(&self.conn, episode)
    }

    fn store_node(&self, node: &MemoryNode, embedding: &[f32]) -> Result<i64> {
        graph::store_node(&self.conn, node, embedding)
    }

    fn store_edge(&self, edge: &Edge) -> Result<()> {
        graph::store_edge(&self.conn, edge)
    }

    fn hybrid_search(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        tenant_id: &TenantId,
        limit: usize,
    ) -> Result<Vec<MemoryNode>> {
        search::hybrid_search(
            &self.conn,
            &self.ranking,
            query_text,
            query_embedding,
            tenant_id,
            limit,
        )
    }

    fn query_with_graph(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        tenant_id: &TenantId,
        time_filter: &TimeFilter,
        ccl_filter: &[String],
        limit: usize,
    ) -> Result<Vec<MemoryResult>> {
        search::query_with_graph(
            &self.conn,
            &self.ranking,
            query_text,
            query_embedding,
            tenant_id,
            time_filter,
            ccl_filter,
            limit,
        )
    }

    fn recent_in_context(
        &self,
        tenant_id: &TenantId,
        ccl_filter: &[String],
        context_key: &str,
        limit: usize,
    ) -> Result<Vec<MemoryResult>> {
        working::recent_in_context(&self.conn, tenant_id, ccl_filter, context_key, limit)
    }

    fn boost_relevance(&self, node_ids: &[i64]) -> Result<()> {
        graph::boost_relevance(&self.conn, node_ids)
    }

    fn find_similar_nodes(
        &self,
        embedding: &[f32],
        tenant_id: &TenantId,
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<(MemoryNode, f64)>> {
        search::find_similar_nodes(&self.conn, embedding, tenant_id, threshold, limit)
    }

    fn update_node_support(
        &self,
        node_id: i64,
        new_payload: Option<&serde_json::Value>,
    ) -> Result<()> {
        graph::update_node_support(&self.conn, node_id, new_payload)
    }

    fn update_node_content(
        &self,
        node_id: i64,
        payload: &serde_json::Value,
        embedding: &[f32],
    ) -> Result<()> {
        graph::update_node_content(&self.conn, node_id, payload, embedding)
    }

    fn delete_tenant(&self, tenant_id: &TenantId) -> Result<()> {
        lifecycle::delete_tenant(&self.conn, tenant_id)
    }

    fn export_tenant(&self, tenant_id: &TenantId) -> Result<String> {
        lifecycle::export_tenant(&self.conn, tenant_id)
    }

    fn sweep_decay(&self, engine: &DecayEngine) -> Result<()> {
        lifecycle::sweep_decay(&self.conn, engine)
    }

    fn find_document_nodes(&self, data_id: &str) -> Result<Vec<MemoryNode>> {
        search::find_document_nodes(&self.conn, data_id)
    }

    fn replace_document_node(
        &self,
        data_id: &str,
        node: &MemoryNode,
        embedding: &[f32],
    ) -> Result<i64> {
        graph::replace_document_node(&self.conn, data_id, node, embedding)
    }

    fn stm_stats(&self) -> Result<StmStats> {
        inspect::stm_stats(&self.conn)
    }

    fn find_working_subject(
        &self,
        tenant_id: &TenantId,
        context_key: &str,
        data_id: &str,
    ) -> Result<Option<i64>> {
        working::find_working_subject(&self.conn, tenant_id, context_key, data_id)
    }

    fn list_node_summaries(
        &self,
        limit: usize,
        offset: usize,
        status: Option<&str>,
        contains: Option<&str>,
    ) -> Result<Vec<StmNodeSummary>> {
        inspect::list_node_summaries(&self.conn, limit, offset, status, contains)
    }

    fn count_by_data_id(&self, data_id: &str) -> Result<i64> {
        inspect::count_by_data_id(&self.conn, data_id)
    }

    fn is_command_processed(&self, command_id: &str) -> Result<bool> {
        commands::is_command_processed(&self.conn, command_id)
    }

    fn mark_command_processed(&self, command_id: &str) -> Result<()> {
        commands::mark_command_processed(&self.conn, command_id)
    }

    fn sweep_processed_commands(&self, older_than_days: i64) -> Result<usize> {
        commands::sweep_processed_commands(&self.conn, older_than_days)
    }

    fn delete_nodes_by_data_id(&self, data_id: &str) -> Result<()> {
        lifecycle::delete_nodes_by_data_id(&self.conn, data_id)
    }

    fn reset_store(&self) -> Result<()> {
        lifecycle::reset_store(&self.conn)
    }
}
