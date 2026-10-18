use crate::domain::models::{MemoryResult, TenantId, TimeFilter};
use crate::domain::ports::{LlmClient, MemoryRepository};
use crate::domain::tuning::STM_RECALL_K;
use anyhow::Result;
use std::sync::Arc;

pub struct RetrievalService {
    llm_client: Arc<dyn LlmClient>,
    memory_repo: Arc<dyn MemoryRepository>,
}

impl RetrievalService {
    pub fn new(llm_client: Arc<dyn LlmClient>, memory_repo: Arc<dyn MemoryRepository>) -> Self {
        Self {
            llm_client,
            memory_repo,
        }
    }

    /// Retrieve relevant memory context with graph traversal and temporal filtering
    /// Returns token-optimized MemoryResult (no internal IDs/scores)
    pub async fn query(
        &self,
        tenant_id: &TenantId,
        raw_query: &str,
        time_filter: &TimeFilter,
        ccl_filter: &[String],
    ) -> Result<Vec<MemoryResult>> {
        // 1. Generate embedding for the raw query string
        let embedding = self.llm_client.embed_text(raw_query).await?;

        // 2. Perform hybrid search with graph traversal + temporal filtering
        let results = self.memory_repo.query_with_graph(
            raw_query,
            &embedding,
            tenant_id,
            time_filter,
            ccl_filter,
            STM_RECALL_K,
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
