use crate::domain::models::{MemoryResult, TenantId, TimeFilter};
use crate::domain::ports::{Embedder, MemoryRepository};
use anyhow::Result;
use std::sync::Arc;

pub struct RetrievalService {
    embedder: Arc<dyn Embedder>,
    memory_repo: Arc<dyn MemoryRepository>,
}

impl RetrievalService {
    pub fn new(embedder: Arc<dyn Embedder>, memory_repo: Arc<dyn MemoryRepository>) -> Self {
        Self {
            embedder,
            memory_repo,
        }
    }

    /// Retrieve up to `limit` relevant facts, with graph traversal and
    /// temporal filtering. Returns token-optimized MemoryResult (no internal
    /// IDs; a fused score only when `[retrieval] include_scores` is on).
    pub async fn query(
        &self,
        tenant_id: &TenantId,
        raw_query: &str,
        time_filter: &TimeFilter,
        ccl_filter: &[String],
        limit: usize,
    ) -> Result<Vec<MemoryResult>> {
        // 1. Generate embedding for the raw query string
        let embedding = self.embedder.embed_text(raw_query).await?;

        // 2. Perform hybrid search with graph traversal + temporal filtering
        let results = self.memory_repo.query_with_graph(
            raw_query,
            &embedding,
            tenant_id,
            time_filter,
            ccl_filter,
            limit,
        )?;

        Ok(results)
    }

    /// Recency backbone for the working-memory map (STM-WORKING-MEMORY slice 3):
    /// the most-recent situational notes in one context, newest first. Passive —
    /// no embedding, no boost.
    pub fn recent_in_context(
        &self,
        tenant_id: &TenantId,
        ccl_filter: &[String],
        context_key: &str,
        limit: usize,
    ) -> Result<Vec<MemoryResult>> {
        self.memory_repo
            .recent_in_context(tenant_id, ccl_filter, context_key, limit)
    }
}
