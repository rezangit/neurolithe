//! Query service — the read use-case behind the `memory.query` bus door.
//!
//! Dispatches a parsed request by [`QueryScope`] to the existing retrieval
//! primitives ([`RetrievalService`] for STM, [`LtmRetrieval`] for LTM) and
//! returns application-layer results ([`QueryOutcome`]). The delivery layer
//! ([`crate::interfaces::bus_query`]) maps that outcome onto the wire envelope,
//! so this module has zero knowledge of Kafka or JSON shapes.
//!
//! The headline `ltm_via_stm` scope (design §5) is server-side compounding:
//! recall STM facts, fold them into a single seed embedding, and use that to
//! recall LTM — one round trip, reusing the recalled meaning.

use crate::application::ltm_retrieval::{LtmRetrieval, RecallResult};
use crate::application::retrieval::RetrievalService;
use crate::domain::models::{MemoryResult, TenantId, TimeFilter, WORKING_CCL};
use crate::domain::ports::LlmClient;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// How many top STM facts seed the LTM search in `ltm_via_stm`.
const SEED_FACT_COUNT: usize = 3;

/// CCL applied when a request omits one. The STM store's hybrid query filters
/// with `n.ccl IN (…)`, so an *empty* list matches **nothing** — it cannot mean
/// "all layers". We therefore default to the base `reality` layer, matching both
/// the MCP `query_memory` default and the agent-writes-default-`reality`
/// convention (design §7 / plan finding). Callers wanting another layer pass it
/// explicitly.
const DEFAULT_CCL: &str = "reality";

/// What kind of read a `memory.query` asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryScope {
    /// Recent, decaying facts only.
    Stm,
    /// Permanent knowledge tree only.
    Ltm,
    /// Both stores, independently.
    Both,
    /// STM recall seeds the LTM search (the headline, design §5).
    LtmViaStm,
    /// Working-memory map: recency-first situational notes for a `context_key`,
    /// with optional semantic enrichment (STM-WORKING-MEMORY §5). No `query`
    /// text required — recency is the backbone.
    StmMap,
}

/// A parsed, defaulted read request (application-layer — no wire types).
#[derive(Debug, Clone)]
pub struct QueryRequest {
    pub scope: QueryScope,
    pub tenant: String,
    pub query: String,
    pub k: usize,
    pub time_filter: TimeFilter,
    pub ccl: Vec<String>,
    /// Working-memory thread to orient by (`stm_map` only). `None` on ordinary
    /// reads.
    pub context_key: Option<String>,
}

/// The labelled result of a read, before wire mapping.
#[derive(Debug, Default)]
pub struct QueryOutcome {
    pub stm: Vec<MemoryResult>,
    pub ltm: Vec<RecallResult>,
    /// STM fact texts that seeded the LTM search (`ltm_via_stm` only).
    pub seeded_by: Vec<String>,
}

/// Reads STM + LTM for the bus query door.
pub struct QueryService {
    retrieval: RetrievalService,
    ltm: LtmRetrieval,
    embedder: Arc<dyn LlmClient>,
}

impl QueryService {
    pub fn new(
        retrieval: RetrievalService,
        ltm: LtmRetrieval,
        embedder: Arc<dyn LlmClient>,
    ) -> Self {
        Self {
            retrieval,
            ltm,
            embedder,
        }
    }

    /// Dispatch a request to the right store(s). Always returns a populated or
    /// empty outcome; the caller turns an `Err` into an error reply.
    pub async fn execute(&self, req: &QueryRequest) -> Result<QueryOutcome> {
        match req.scope {
            QueryScope::Stm => {
                let stm = self.stm_search(req).await?;
                Ok(QueryOutcome {
                    stm,
                    ..Default::default()
                })
            }
            QueryScope::Ltm => {
                let ltm = self.ltm_recall(&req.query, req.k).await?;
                Ok(QueryOutcome {
                    ltm,
                    ..Default::default()
                })
            }
            QueryScope::Both => {
                let stm = self.stm_search(req).await?;
                let ltm = self.ltm_recall(&req.query, req.k).await?;
                Ok(QueryOutcome {
                    stm,
                    ltm,
                    ..Default::default()
                })
            }
            QueryScope::LtmViaStm => self.ltm_via_stm(req).await,
            QueryScope::StmMap => {
                let stm = self.stm_map(req).await?;
                Ok(QueryOutcome {
                    stm,
                    ..Default::default()
                })
            }
        }
    }

    /// Working-memory map (STM-WORKING-MEMORY §5):
    /// **Pure context-scoped recency** — the only thing that reliably resolves
    /// an information-poor follow-up ("what is *its* id?"). It returns the most
    /// recent working notes *in this exact thread* and nothing else.
    ///
    /// It deliberately does **no** vector/semantic search. An earlier design
    /// enriched the map with a similarity search, but live use showed that fuzzy
    /// "find something similar" is wrong for a situational map: it dragged in
    /// unrelated documents (reality) and other threads' notes, so the agent
    /// answered about the wrong thing. Durable/semantic recall stays where it
    /// belongs — behind the explicit `memory.query` tool the agent invokes with
    /// its own precise query. With no `context_key` there is no thread to orient
    /// by, so the map is empty.
    async fn stm_map(&self, req: &QueryRequest) -> Result<Vec<MemoryResult>> {
        let Some(context_key) = req.context_key.as_deref() else {
            return Ok(Vec::new());
        };
        let tenant = TenantId(req.tenant.clone());
        // `stm_map` defaults to the working layer; an explicit ccl overrides.
        let ccl_filter: Vec<String> = if req.ccl.is_empty() {
            vec![WORKING_CCL.to_string()]
        } else {
            req.ccl.clone()
        };
        self.retrieval
            .recent_in_context(&tenant, &ccl_filter, context_key, req.k)
    }

    /// STM hybrid search, truncated to the requested breadth `k`. An empty CCL
    /// list falls back to [`DEFAULT_CCL`] (the store cannot express match-all).
    async fn stm_search(&self, req: &QueryRequest) -> Result<Vec<MemoryResult>> {
        let tenant = TenantId(req.tenant.clone());
        let ccl_filter: Vec<String> = if req.ccl.is_empty() {
            vec![DEFAULT_CCL.to_string()]
        } else {
            req.ccl.clone()
        };
        let mut results = self
            .retrieval
            .query(&tenant, &req.query, &req.time_filter, &ccl_filter)
            .await?;
        results.truncate(req.k);
        Ok(results)
    }

    /// Embed `text` and recall the `k` nearest LTM nodes within the recall
    /// distance cap (possibly none).
    async fn ltm_recall(&self, text: &str, k: usize) -> Result<Vec<RecallResult>> {
        let embedding = self.embedder.embed_text(text).await?;
        self.ltm.recall(&embedding, k)
    }

    /// The compound path: STM recall → seed(query ⊕ top-k fact texts) → one
    /// embed → LTM recall. Degrades to plain LTM recall when STM is empty.
    async fn ltm_via_stm(&self, req: &QueryRequest) -> Result<QueryOutcome> {
        let stm = self.stm_search(req).await?;

        let seed_facts: Vec<String> = stm
            .iter()
            .take(SEED_FACT_COUNT)
            .map(|m| m.fact.clone())
            .collect();

        // Seed reuses the recalled meaning; with no facts it's just the query,
        // so the compound degrades to a plain LTM recall (seededBy stays empty).
        let seed = if seed_facts.is_empty() {
            req.query.clone()
        } else {
            format!("{}\n{}", req.query, seed_facts.join("\n"))
        };

        let embedding = self.embedder.embed_text(&seed).await?;
        let ltm = self.ltm.recall(&embedding, req.k)?;

        Ok(QueryOutcome {
            stm,
            ltm,
            seeded_by: seed_facts,
        })
    }
}
