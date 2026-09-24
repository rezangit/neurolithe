use crate::domain::models::{Edge, Episode, MemoryNode, MemoryResult, TenantId, TimeFilter};
use anyhow::Result;

pub trait MemoryRepository {
    /// Store a CCL definition in the registry
    fn store_ccl_definition(&self, definition: &crate::domain::models::CclDefinition)
    -> Result<()>;

    /// Retrieve all configured CCL definitions for a tenant
    fn get_ccl_definitions(
        &self,
        tenant_id: &TenantId,
    ) -> Result<Vec<crate::domain::models::CclDefinition>>;

    /// Store raw episodic dialogue
    fn store_episode(&self, episode: &Episode) -> Result<i64>;

    /// Store a structured fact (Node) along with its embedding
    fn store_node(&self, node: &MemoryNode, embedding: &[f32]) -> Result<i64>;

    /// Store a relationship edge between two nodes
    fn store_edge(&self, edge: &Edge) -> Result<()>;

    /// Search via hybrid (Vector + FTS) search
    fn hybrid_search(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        tenant_id: &TenantId,
        limit: usize,
    ) -> Result<Vec<MemoryNode>>;

    /// Full hybrid search with 1-hop graph traversal and temporal filtering (blueprint spec)
    fn query_with_graph(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        tenant_id: &TenantId,
        time_filter: &TimeFilter,
        ccl_filter: &[String],
        limit: usize,
    ) -> Result<Vec<MemoryResult>>;

    /// Recency-ordered working-memory read (STM-WORKING-MEMORY slice 3): the
    /// most-recently-touched **active** notes in one tenant + `context_key`,
    /// newest first. No embedding hop — the cheap, always-works backbone of
    /// RECALL. Scoped to `context_key`, so NULL-context knowledge facts and
    /// other contexts/threads are excluded; `ccl_filter` narrows further (empty
    /// = any layer). Does **not** boost — orientation is passive; only facts the
    /// agent actually uses get reinforced.
    fn recent_in_context(
        &self,
        tenant_id: &TenantId,
        ccl_filter: &[String],
        context_key: &str,
        limit: usize,
    ) -> Result<Vec<MemoryResult>>;

    /// Boost relevance score back to 1.0 on read (blueprint: reading resets decay)
    fn boost_relevance(&self, node_ids: &[i64]) -> Result<()>;

    /// Find an existing active `working` **subject** node for a `dataId` in one
    /// tenant + thread (STM-GRAPH): a graph anchor reused across the session's
    /// turns so they connect through it. `None` if none exists yet.
    fn find_working_subject(
        &self,
        tenant_id: &TenantId,
        context_key: &str,
        data_id: &str,
    ) -> Result<Option<i64>>;

    /// Find active nodes of one tenant within `threshold` vector distance of
    /// `embedding`, nearest first, each paired with its distance (for conflict
    /// resolution). Returns at most `limit` nodes.
    fn find_similar_nodes(
        &self,
        embedding: &[f32],
        tenant_id: &TenantId,
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<(MemoryNode, f64)>>;

    /// Reinforce a node (support_count + 1, relevance back to 1.0), optionally
    /// replacing its payload. The payload must keep the same fact text — the
    /// stored embedding is left untouched. Use [`Self::update_node_content`]
    /// when the text changes.
    fn update_node_support(
        &self,
        node_id: i64,
        new_payload: Option<&serde_json::Value>,
    ) -> Result<()>;

    /// Replace a node's payload **and** its embedding atomically (reinforcing
    /// it like [`Self::update_node_support`]). Used when a merge changes the
    /// fact text, so the vector keeps describing what the node now says.
    fn update_node_content(
        &self,
        node_id: i64,
        payload: &serde_json::Value,
        embedding: &[f32],
    ) -> Result<()>;

    /// Delete all data for a given tenant
    fn delete_tenant(&self, tenant_id: &TenantId) -> Result<()>;

    /// Export all data for a given tenant as structured JSON string
    fn export_tenant(&self, tenant_id: &TenantId) -> Result<String>;

    /// Apply decay sweep across all active memory nodes
    fn sweep_decay(&self, engine: &crate::domain::decay::DecayEngine) -> Result<()>;

    /// Wipe every record in this store, keeping the schema intact.
    ///
    /// Used by V2 reset: a soft reset wipes the STM store only (this call on
    /// the STM connection); LTM lives in a separate connection/file and is
    /// untouched. All tenants and CCL layers are cleared.
    fn reset_store(&self) -> Result<()>;

    /// Delete STM nodes carrying a given `dataId` in their payload (and their
    /// vectors/edges). Used by the feeder's tombstone path to forget a document
    /// from working memory. Best-effort: the conflict resolver may have merged a
    /// document into a shared node, in which case nothing distinct remains here.
    fn delete_nodes_by_data_id(&self, data_id: &str) -> Result<()>;

    /// CT-scan statistics for the STM store (slice 9).
    fn stm_stats(&self) -> Result<StmStats>;

    /// List STM facts (most-relevant first) for introspection, with pagination
    /// (`limit`/`offset`), an optional status filter (`active`/`archived`), and
    /// an optional case-insensitive substring filter on the fact text
    /// (`contains`) so a client can find facts without pulling the whole store
    /// (field-report §5).
    fn list_node_summaries(
        &self,
        limit: usize,
        offset: usize,
        status: Option<&str>,
        contains: Option<&str>,
    ) -> Result<Vec<StmNodeSummary>>;

    /// How many STM nodes carry a given `dataId` (for `trace_dataId`).
    fn count_by_data_id(&self, data_id: &str) -> Result<i64>;

    /// Idempotency: has a `memory.command` with this `commandId` already been
    /// applied? Guards against Kafka at-least-once redelivery of writes.
    fn is_command_processed(&self, command_id: &str) -> Result<bool>;

    /// Record a `commandId` as applied (no-op if already present).
    fn mark_command_processed(&self, command_id: &str) -> Result<()>;

    /// Delete idempotency rows older than `older_than_days`; returns how many
    /// were removed. Called from the periodic sweep.
    fn sweep_processed_commands(&self, older_than_days: i64) -> Result<usize>;
}

/// Which embedder produced a store's vectors. Recorded in each store's `meta`
/// table; a store embedded by a different model or at a different dimension
/// cannot be searched with the current embedder and must be re-embedded
/// (`neurolithe reembed`). `dim` is what the embedder actually outputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingIdentity {
    pub provider: String,
    pub model: String,
    pub dim: usize,
}

/// A single STM fact, flattened for the introspection CT scan.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StmNodeSummary {
    pub fact: String,
    pub status: String,
    pub relevance_score: f64,
    pub support_count: i32,
    pub ccl: String,
    pub last_accessed_at: Option<String>,
    pub data_id: Option<String>,
}

/// A snapshot of STM health for the metrics CT scan.
#[derive(Debug, Clone, PartialEq)]
pub struct StmStats {
    pub active_nodes: i64,
    pub archived_nodes: i64,
    /// Mean relevance over active nodes (0.0 if none).
    pub avg_relevance: f64,
    /// Active-node counts bucketed by relevance into 5 bins
    /// ([0,0.2),[0.2,0.4),[0.4,0.6),[0.6,0.8),[0.8,1.0]).
    pub decay_histogram: Vec<i64>,
    pub db_size_bytes: i64,
}

use serde::{Deserialize, Serialize};

pub fn default_ccl() -> String {
    "reality".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExtractedFact {
    pub fact: String,
    #[serde(default = "default_ccl")]
    pub ccl: String,
    pub tags: Vec<String>,
    #[serde(default)]
    pub relationships: Vec<ExtractedRelationship>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    pub target_entity: String,
    pub relation: String,
    #[serde(default = "default_ccl")]
    pub ccl: String,
    #[serde(default)]
    pub valid_from: Option<String>,
    #[serde(default)]
    pub valid_until: Option<String>,
}

// `Send + Sync` so `Arc<dyn LlmClient>` can be shared across tasks and composed
// (e.g. the chat/embedding split in `SplitLlmClient`). Every implementor —
// reqwest-based provider clients and test stubs — is genuinely thread-safe.
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    /// Extract factual statements from given raw dialogue
    async fn extract_facts(
        &self,
        dialogue: &str,
        valid_ccls: &[crate::domain::models::CclDefinition],
    ) -> Result<Vec<ExtractedFact>>;

    /// Generate a short description for a new cognitive context layer
    async fn generate_ccl_description(&self, ccl_name: &str, context: &str) -> Result<String>;

    /// Generate a 1536d float vector for text
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>>;

    /// Compress/summarize old dialogue messages into a dense summary
    async fn compress_context(&self, messages: &str) -> Result<String>;

    /// The embedder's output dimension. The default probes by embedding a short
    /// text; providers that know it statically (e.g. local models) override.
    async fn embedding_dim(&self) -> Result<usize> {
        Ok(self.embed_text("dimension probe").await?.len())
    }

    /// Stable identifier of the embedding model, `"<provider>:<model>"` (e.g.
    /// `"local:bge-small-en-v1.5"`). Recorded in store metadata; a change means
    /// the stores must be re-embedded. Default: `"unknown"`.
    fn embedding_model_id(&self) -> String {
        "unknown".to_string()
    }

    /// Whether a chat model is configured (fact extraction, compression,
    /// summaries). Embeddings are independent of this. Default: `true`.
    fn chat_available(&self) -> bool {
        true
    }
}

/// The [`EmbeddingIdentity`] of `llm`'s embedder: provider and model from
/// [`LlmClient::embedding_model_id`] (`"provider:model"`; without a `:` the
/// provider is `"unknown"`), dimension from [`LlmClient::embedding_dim`].
pub async fn embedding_identity(llm: &dyn LlmClient) -> Result<EmbeddingIdentity> {
    let id = llm.embedding_model_id();
    let provider = match id.split_once(':') {
        Some((provider, _)) if !provider.is_empty() => provider.to_string(),
        _ => "unknown".to_string(),
    };
    Ok(EmbeddingIdentity {
        provider,
        model: id,
        dim: llm.embedding_dim().await?,
    })
}
