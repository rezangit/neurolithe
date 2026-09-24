//! Query consumer — the queries → results request/reply loop
//! (`[kafka.topics] queries` / `results`, defaults `memory.query` /
//! `memory.result`).
//!
//! Mirrors the feeder/command loops (a `StreamConsumer` + `FutureProducer` on a
//! single-threaded `LocalSet`). Each request is parsed ([`parse_query`]),
//! dispatched to [`QueryService`], and answered on the results topic keyed by
//! its `correlationId`. Routing:
//!
//! - a well-formed request → execute → **always** reply `ok`/`empty`/`error`
//!   (an internal failure is an `error` reply, plus a copy to the DLQ topic for
//!   the operator — never an auto-retry, never a hang);
//! - an addressable-but-malformed request → `error` reply to its id;
//! - un-addressable bytes (no `correlationId`) → the parking topic.
//!
//! Every query reads the process's single workspace; a legacy `tenant` field is
//! ignored. The rdkafka plumbing needs a live broker; the dispatch + reply logic
//! is unit-tested via [`QueryConsumer::respond`].

use crate::application::query_service::{QueryOutcome, QueryRequest, QueryService};
use crate::domain::models::WORKSPACE_TENANT;
use crate::infrastructure::config::KafkaConfig;
use crate::infrastructure::kafka_client::{
    StopSignal, base_config, commit_on_shutdown, consumer_config, flush_on_shutdown, query_group,
    recv_or_stop,
};
use crate::interfaces::bus_query::{
    MemoryQuery, MemoryReply, ParseOutcome, ReplyStatus, StmEntry, flatten_recall, parse_query,
};
use anyhow::{Context, Result};
use rdkafka::consumer::{CommitMode, Consumer, StreamConsumer};
use rdkafka::message::{Header, Message, OwnedHeaders};
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use std::sync::Arc;

/// Map an application-layer [`QueryOutcome`] onto the wire reply: STM entries
/// stay token-optimized, LTM recalls flatten to one entry per document leaf.
fn to_reply(correlation_id: &str, outcome: QueryOutcome) -> MemoryReply {
    let stm = outcome
        .stm
        .iter()
        .map(StmEntry::from_memory_result)
        .collect();
    let ltm = outcome.ltm.iter().flat_map(flatten_recall).collect();
    MemoryReply::success(correlation_id, stm, ltm, outcome.seeded_by)
}

pub struct QueryConsumer {
    consumer: StreamConsumer,
    producer: FutureProducer,
    query_service: Arc<QueryService>,
    topic: String,
    result_topic: String,
    dlq_topic: String,
    parking_topic: String,
    group_id: String,
}

impl QueryConsumer {
    pub fn new(kafka: &KafkaConfig, query_service: Arc<QueryService>) -> Result<Self> {
        let group_id = query_group(kafka);
        // Requests are live traffic, not a replay source: start at the end.
        let consumer: StreamConsumer = consumer_config(kafka, &group_id, "latest")
            .create()
            .context("creating queries consumer")?;

        let producer: FutureProducer = base_config(kafka)
            .create()
            .context("creating results producer")?;

        Ok(Self {
            consumer,
            producer,
            query_service,
            topic: kafka.topics.queries.clone(),
            result_topic: kafka.topics.results.clone(),
            dlq_topic: kafka.topics.dlq.clone(),
            parking_topic: kafka.topics.parking.clone(),
            group_id,
        })
    }

    /// Consume until `stop` fires: parse → dispatch → reply → commit, then
    /// commit offsets synchronously and flush the reply producer.
    pub async fn run(&self, mut stop: StopSignal) -> Result<()> {
        self.consumer
            .subscribe(&[&self.topic])
            .with_context(|| format!("subscribing to {}", self.topic))?;

        while let Some(next) = recv_or_stop(&self.consumer, &mut stop).await {
            match next {
                Err(e) => tracing::warn!("query consumer error: {e}"),
                Ok(msg) => {
                    self.process(&msg).await;
                    if let Err(e) = self.consumer.commit_message(&msg, CommitMode::Async) {
                        tracing::warn!("query commit failed: {e}");
                    }
                }
            }
        }
        commit_on_shutdown(&self.consumer, "query consumer");
        flush_on_shutdown(&self.producer, "query consumer");
        Ok(())
    }

    async fn process(&self, msg: &rdkafka::message::BorrowedMessage<'_>) {
        match parse_query(msg.payload().unwrap_or_default()) {
            ParseOutcome::Query(q) => {
                let reply = self.respond(q.as_ref()).await;
                // On an internal error, still reply — and copy to the DLQ topic so
                // an operator sees it. Never auto-retry.
                if reply.status == ReplyStatus::Error {
                    let reason = reply.error.clone().unwrap_or_else(|| "query error".into());
                    self.send_aside(msg, &self.dlq_topic, &reason).await;
                }
                self.publish_result(&reply).await;
            }
            ParseOutcome::Rejected {
                correlation_id,
                reason,
            } => {
                // Addressable → reply an error rather than silently parking.
                self.publish_result(&MemoryReply::error(correlation_id, reason))
                    .await;
            }
            ParseOutcome::Unroutable { reason } => {
                // No correlationId to reply to → park the raw bytes.
                self.send_aside(msg, &self.parking_topic, &reason).await;
            }
        }
    }

    /// Execute a parsed query and build its reply. Always returns a reply — an
    /// execution error becomes an `error` reply, never a hang.
    async fn respond(&self, q: &MemoryQuery) -> MemoryReply {
        let req = QueryRequest {
            scope: q.scope,
            tenant: WORKSPACE_TENANT.to_string(),
            query: q.query.clone(),
            k: q.k,
            time_filter: q.time_filter.clone().unwrap_or_default(),
            ccl: q.ccl.clone(),
            context_key: q.context_key.clone(),
        };
        match self.query_service.execute(&req).await {
            Ok(outcome) => to_reply(&q.correlation_id, outcome),
            Err(e) => MemoryReply::error(&q.correlation_id, e.to_string()),
        }
    }

    async fn publish_result(&self, reply: &MemoryReply) {
        let payload = match serde_json::to_vec(reply) {
            Ok(bytes) => bytes,
            Err(e) => {
                tracing::warn!("failed to serialize query result: {e}");
                return;
            }
        };
        let record = FutureRecord::to(&self.result_topic)
            .key(&reply.correlation_id)
            .payload(&payload);
        if let Err((e, _)) = self.producer.send(record, Timeout::Never).await {
            tracing::warn!("failed to publish query result: {e}");
        }
    }

    /// Forward the original message to a dead-letter / parking topic with the
    /// context headers (reason, consumer, source topic/partition/offset).
    async fn send_aside(
        &self,
        msg: &rdkafka::message::BorrowedMessage<'_>,
        topic: &str,
        reason: &str,
    ) {
        let partition = msg.partition().to_string();
        let offset = msg.offset().to_string();
        let headers = OwnedHeaders::new()
            .insert(Header {
                key: "x-reason",
                value: Some(reason),
            })
            .insert(Header {
                key: "x-consumer",
                value: Some(self.group_id.as_str()),
            })
            .insert(Header {
                key: "x-source-topic",
                value: Some(self.topic.as_str()),
            })
            .insert(Header {
                key: "x-source-partition",
                value: Some(partition.as_str()),
            })
            .insert(Header {
                key: "x-source-offset",
                value: Some(offset.as_str()),
            });

        let key = msg.key().unwrap_or_default();
        let payload = msg.payload().unwrap_or_default();
        let record = FutureRecord::to(topic)
            .key(key)
            .payload(payload)
            .headers(headers);
        if let Err((e, _)) = self.producer.send(record, Timeout::Never).await {
            tracing::warn!("failed to forward to {topic}: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORKSPACE: &str = WORKSPACE_TENANT;
    use crate::application::ltm_retrieval::LtmRetrieval;
    use crate::application::query_service::QueryScope;
    use crate::application::retrieval::RetrievalService;
    use crate::domain::ltm::LtmRepository;
    use crate::domain::models::{CclDefinition, MemoryNode, TenantId};
    use crate::domain::ports::{ExtractedFact, LlmClient, MemoryRepository};
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::ltm_repository::SqliteLtmRepository;
    use crate::infrastructure::repository::SqliteMemoryRepository;
    use crate::infrastructure::schema::{init_ltm_schema, init_schema};
    use async_trait::async_trait;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const DIM: usize = 8;

    /// Records every `embed_text` input + a call count; returns a constant
    /// vector so hybrid search matches whatever was seeded with the same stub.
    struct RecordingLlm {
        embeds: Mutex<Vec<String>>,
        count: AtomicUsize,
        fail_embed: bool,
    }

    impl RecordingLlm {
        fn new(fail_embed: bool) -> Self {
            Self {
                embeds: Mutex::new(Vec::new()),
                count: AtomicUsize::new(0),
                fail_embed,
            }
        }
        fn embed_count(&self) -> usize {
            self.count.load(Ordering::SeqCst)
        }
        fn embed_inputs(&self) -> Vec<String> {
            self.embeds.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl LlmClient for RecordingLlm {
        async fn extract_facts(
            &self,
            _dialogue: &str,
            _valid_ccls: &[CclDefinition],
        ) -> anyhow::Result<Vec<ExtractedFact>> {
            Ok(vec![])
        }
        async fn generate_ccl_description(&self, _n: &str, _c: &str) -> anyhow::Result<String> {
            Ok("desc".into())
        }
        async fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            self.count.fetch_add(1, Ordering::SeqCst);
            self.embeds.lock().unwrap().push(text.to_string());
            if self.fail_embed {
                anyhow::bail!("embed failed");
            }
            Ok(vec![0.1_f32; DIM])
        }
        async fn compress_context(&self, _m: &str) -> anyhow::Result<String> {
            Ok("summary".into())
        }
    }

    struct Fixture {
        consumer: QueryConsumer,
        llm: Arc<RecordingLlm>,
        stm: Arc<SqliteMemoryRepository>,
    }

    fn fixture(fail_embed: bool) -> Fixture {
        let stm_conn = init_db(None as Option<&String>).unwrap();
        init_schema(&stm_conn, DIM).unwrap();
        let stm = Arc::new(SqliteMemoryRepository::new(stm_conn));

        let ltm_conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&ltm_conn, DIM).unwrap();
        let ltm = Arc::new(SqliteLtmRepository::new(ltm_conn));
        ltm.seed_spine().unwrap();

        let llm = Arc::new(RecordingLlm::new(fail_embed));

        let retrieval =
            RetrievalService::new(llm.clone(), stm.clone() as Arc<dyn MemoryRepository>);
        let ltm_retrieval = LtmRetrieval::new(ltm.clone() as Arc<dyn LtmRepository>);
        let query_service = Arc::new(QueryService::new(
            retrieval,
            ltm_retrieval,
            llm.clone() as Arc<dyn LlmClient>,
        ));

        // A broker-less QueryConsumer: the loop/producer are never driven in unit
        // tests (respond() is the tested seam); construction points at a bogus
        // broker but never connects.
        let consumer = QueryConsumer::new(
            &crate::infrastructure::kafka_client::tests::test_kafka(&[]),
            query_service,
        )
        .unwrap();
        Fixture { consumer, llm, stm }
    }

    /// Seed one STM fact directly through the repository (no LLM extraction).
    fn seed_stm_fact(stm: &SqliteMemoryRepository, tenant: &str, fact: &str) {
        let node = MemoryNode {
            id: None,
            tenant_id: TenantId(tenant.into()),
            source_episode_id: None,
            payload: serde_json::json!({ "fact": fact, "tags": [] }),
            status: "active".into(),
            ccl: "reality".into(),
            is_explicit: true,
            support_count: 1,
            relevance_score: 1.0,
            context_key: None,
        };
        stm.store_node(&node, &[0.1_f32; DIM]).unwrap();
    }

    fn query(scope: QueryScope, text: &str) -> MemoryQuery {
        serde_json::from_value(serde_json::json!({
            "correlationId": "c1",
            "scope": scope,
            "query": text,
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn stm_scope_returns_facts_and_no_ltm() {
        let f = fixture(false);
        seed_stm_fact(&f.stm, WORKSPACE, "User likes tea");

        let reply = f.consumer.respond(&query(QueryScope::Stm, "tea")).await;

        assert_eq!(reply.status, ReplyStatus::Ok);
        assert_eq!(reply.stm.len(), 1);
        assert_eq!(reply.stm[0].fact, "User likes tea");
        assert!(reply.ltm.is_empty());
    }

    /// One workspace per process: a legacy `tenant` field is ignored, so a
    /// query naming any tenant reads the workspace's facts.
    #[tokio::test]
    async fn legacy_tenant_field_is_ignored() {
        let f = fixture(false);
        seed_stm_fact(&f.stm, WORKSPACE, "User likes tea");

        for tenant in [None, Some("someone-else")] {
            let mut raw = serde_json::json!({
                "correlationId": "c1", "scope": "stm", "query": "tea"
            });
            if let Some(t) = tenant {
                raw["tenant"] = serde_json::json!(t);
            }
            let q: MemoryQuery = serde_json::from_value(raw).unwrap();
            let reply = f.consumer.respond(&q).await;
            assert_eq!(reply.stm.len(), 1, "tenant {tenant:?}: {reply:?}");
        }
    }

    /// Seed a working-memory note under a context key (bypassing the LLM).
    fn seed_working_note(stm: &SqliteMemoryRepository, tenant: &str, fact: &str, context: &str) {
        let node = MemoryNode {
            id: None,
            tenant_id: TenantId(tenant.into()),
            source_episode_id: None,
            payload: serde_json::json!({ "fact": fact, "tags": [] }),
            status: "active".into(),
            ccl: "working".into(),
            is_explicit: true,
            support_count: 1,
            relevance_score: 1.0,
            context_key: Some(context.into()),
        };
        stm.store_node(&node, &[0.1_f32; DIM]).unwrap();
    }

    /// Build an `stm_map` request with a context key and no query text.
    fn stm_map_query(context: &str) -> MemoryQuery {
        serde_json::from_value(serde_json::json!({
            "correlationId": "c1",
            "scope": "stm_map",
            "contextKey": context,
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn stm_map_recency_backbone_does_zero_embeds() {
        let f = fixture(false);
        seed_working_note(&f.stm, WORKSPACE, "found report = doc_42", "chat.1");
        seed_working_note(&f.stm, WORKSPACE, "other-thread note", "chat.2");

        let reply = f.consumer.respond(&stm_map_query("chat.1")).await;

        // Only the chat.1 note; carries its context key; no embedding hop at all.
        assert_eq!(reply.stm.len(), 1);
        assert!(reply.stm[0].fact.contains("doc_42"));
        assert_eq!(reply.stm[0].context_key.as_deref(), Some("chat.1"));
        assert_eq!(f.llm.embed_count(), 0, "recency backbone must not embed");
    }

    #[tokio::test]
    async fn stm_map_is_pure_context_recency_no_vector_no_leaks() {
        let f = fixture(false);
        // This thread's note; ANOTHER thread's working note; and a reality fact
        // — all mention "inspection" so a vector/keyword pass WOULD match each.
        seed_working_note(&f.stm, WORKSPACE, "inspection report = doc_42", "chat.1");
        seed_working_note(
            &f.stm,
            WORKSPACE,
            "inspection follow-up from chat 2",
            "chat.2",
        );
        seed_stm_fact(&f.stm, WORKSPACE, "inspection checklist tips"); // reality

        // Even WITH a query, the map is pure recency in-thread: the query is not
        // used for retrieval (no embedding, no cross-context, no reality).
        let with_query: MemoryQuery = serde_json::from_value(serde_json::json!({
            "correlationId": "c1",
            "scope": "stm_map",
            "contextKey": "chat.1",
            "query": "inspection",
        }))
        .unwrap();
        let reply = f.consumer.respond(&with_query).await;

        let facts: Vec<&str> = reply.stm.iter().map(|e| e.fact.as_str()).collect();
        assert_eq!(
            facts,
            vec!["inspection report = doc_42"],
            "only this thread's notes"
        );
        assert_eq!(reply.stm[0].context_key.as_deref(), Some("chat.1"));
        assert_eq!(
            f.llm.embed_count(),
            0,
            "the situational map never runs a vector search"
        );
    }

    #[tokio::test]
    async fn stm_map_without_context_is_empty() {
        let f = fixture(false);
        seed_working_note(&f.stm, WORKSPACE, "some working note", "chat.9");
        seed_stm_fact(&f.stm, WORKSPACE, "inspection checklist tips");

        // No contextKey → no thread to orient by → empty (never a global
        // similarity search).
        let no_ctx: MemoryQuery = serde_json::from_value(serde_json::json!({
            "correlationId": "c1",
            "scope": "stm_map",
            "query": "inspection",
        }))
        .unwrap();
        let reply = f.consumer.respond(&no_ctx).await;

        assert_eq!(reply.status, ReplyStatus::Empty);
        assert!(reply.stm.is_empty());
        assert_eq!(f.llm.embed_count(), 0);
    }

    #[tokio::test]
    async fn ltm_scope_skips_stm_and_embeds_the_query() {
        let f = fixture(false);
        seed_stm_fact(&f.stm, WORKSPACE, "User likes tea");

        let reply = f.consumer.respond(&query(QueryScope::Ltm, "tea")).await;

        // LTM-only: STM section stays empty even though a fact exists…
        assert!(reply.stm.is_empty());
        // …and the query was embedded for the LTM recall.
        assert_eq!(f.llm.embed_count(), 1);
    }

    #[tokio::test]
    async fn both_scope_runs_stm_and_ltm() {
        let f = fixture(false);
        seed_stm_fact(&f.stm, WORKSPACE, "User likes tea");

        let reply = f.consumer.respond(&query(QueryScope::Both, "tea")).await;

        assert_eq!(reply.stm.len(), 1);
        // One embed for STM search + one for LTM recall.
        assert_eq!(f.llm.embed_count(), 2);
    }

    #[tokio::test]
    async fn empty_store_yields_empty_status() {
        let f = fixture(false);
        let reply = f
            .consumer
            .respond(&query(QueryScope::Stm, "anything"))
            .await;
        assert_eq!(reply.status, ReplyStatus::Empty);
        assert!(reply.stm.is_empty() && reply.ltm.is_empty());
    }

    #[tokio::test]
    async fn execution_error_always_replies_with_error_status() {
        let f = fixture(true); // embedder fails
        let reply = f.consumer.respond(&query(QueryScope::Ltm, "tea")).await;
        assert_eq!(reply.status, ReplyStatus::Error);
        assert!(reply.error.is_some());
        assert_eq!(reply.correlation_id, "c1");
    }

    #[tokio::test]
    async fn ltm_via_stm_seeds_ltm_with_stm_fact_texts() {
        let f = fixture(false);
        seed_stm_fact(&f.stm, WORKSPACE, "User likes tea");

        let reply = f
            .consumer
            .respond(&query(QueryScope::LtmViaStm, "beverage"))
            .await;

        // The recalled STM fact is reported as the seed…
        assert_eq!(reply.seeded_by, vec!["User likes tea".to_string()]);
        // …and the LTM seed embedding folds the query together with that fact.
        let seed_input = f.llm.embed_inputs().last().cloned().unwrap();
        assert!(seed_input.contains("beverage"));
        assert!(seed_input.contains("User likes tea"));
    }

    #[tokio::test]
    async fn ltm_via_stm_degrades_when_stm_empty() {
        let f = fixture(false);
        let reply = f
            .consumer
            .respond(&query(QueryScope::LtmViaStm, "beverage"))
            .await;
        // No STM facts → no seed, but still a valid (empty) reply.
        assert!(reply.seeded_by.is_empty());
        // The seed embed was just the raw query.
        let seed_input = f.llm.embed_inputs().last().cloned().unwrap();
        assert_eq!(seed_input, "beverage");
    }
}
