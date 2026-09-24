//! Ingestion — turn a document event into memory, written to BOTH stores: a
//! permanent LTM leaf and a decaying STM working-memory fact.
//!
//! This is the broker-independent core of the Kafka feeder (the rdkafka loop in
//! `interfaces::kafka_feeder` drives it). The event carries the document text
//! itself — distill -> place in LTM + resolve into STM. Tombstone -> forget in
//! both.
//!
//! Dimension note: the feeder writes the same distilled embedding to both
//! stores, so both must share the embedder's dimension. A mismatch surfaces as
//! a loud store error rather than silent corruption.

use crate::application::distill::Distiller;
use crate::application::ltm_placement::{DocumentToPlace, LtmPlacement};
use crate::domain::cognition::conflict_resolver::{AdaptationResult, ConflictResolver};
use crate::domain::ltm::{LtmRepository, Provenance};
use crate::domain::models::{MemoryNode, TenantId};
use crate::domain::ports::{LlmClient, MemoryRepository};
use anyhow::{Result, anyhow};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

/// A document event on the documents topic:
/// `{"data_id": "...", "title"?: "...", "text": "...", "tags"?: [...], "ts"?: "..."}`.
///
/// Self-contained: the text travels in the event, nothing is fetched. `dataId`
/// is accepted as an alias for `data_id`. Unknown fields are ignored.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DocumentEvent {
    #[serde(alias = "dataId", default)]
    pub data_id: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Producer timestamp (informational; not interpreted).
    #[serde(default)]
    pub ts: Option<String>,
}

impl DocumentEvent {
    /// The document id, if present and non-blank.
    pub fn document_id(&self) -> Option<&str> {
        self.data_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }

    /// The document text, if present and non-blank.
    pub fn body(&self) -> Option<&str> {
        self.text.as_deref().filter(|t| !t.trim().is_empty())
    }
}

/// What ingesting one document did.
#[derive(Debug, Clone, PartialEq)]
pub enum IngestOutcome {
    Ingested {
        data_id: String,
        leaf_node_id: i64,
        matched: bool,
    },
    /// No usable text in the event — nothing learned, but the offset still
    /// advances.
    Skipped { data_id: String },
}

/// Dual-writes documents into LTM (placement) + STM (conflict resolver).
pub struct IngestionService {
    stm: Arc<dyn MemoryRepository>,
    placement: LtmPlacement,
    distiller: Distiller,
    conflict_resolver: ConflictResolver,
    tenant: TenantId,
}

impl IngestionService {
    pub fn new(
        stm: Arc<dyn MemoryRepository>,
        ltm: Arc<dyn LtmRepository>,
        llm: Arc<dyn LlmClient>,
        embedding_dim: usize,
        tenant: impl Into<String>,
    ) -> Self {
        Self {
            placement: LtmPlacement::new(ltm),
            distiller: Distiller::new(llm, embedding_dim),
            conflict_resolver: ConflictResolver::new(),
            stm,
            tenant: TenantId(tenant.into()),
        }
    }

    /// Ingest one document event: distill its text, then write a permanent LTM
    /// leaf and a decaying STM fact. An event without text is `Skipped`.
    pub async fn ingest(&self, event: &DocumentEvent) -> Result<IngestOutcome> {
        let document_id = event
            .document_id()
            .ok_or_else(|| anyhow!("document event has no data_id"))?
            .to_string();

        let Some(text) = event.body() else {
            return Ok(IngestOutcome::Skipped {
                data_id: document_id,
            });
        };

        let mut tags: Vec<String> = Vec::new();
        for t in &event.tags {
            if !tags.contains(t) {
                tags.push(t.clone());
            }
        }

        let distillate = self.distiller.distill(event.title.as_deref(), text).await?;

        // A human-readable leaf title: the event's own title when given, else
        // the first heading/line of the summary, else the dataId.
        let title = match event.title.as_deref().map(str::trim) {
            Some(t) if !t.is_empty() => derive_title(t, &document_id),
            _ => derive_title(&distillate.summary, &document_id),
        };

        // LTM: file a permanent leaf under the best concept (or inbox).
        let placement = self.placement.place(&DocumentToPlace {
            name: title,
            summary: distillate.summary.clone(),
            embedding: distillate.embedding.clone(),
            data_id: document_id.clone(),
            provenance: Provenance {
                source: "kafka".into(),
                ingested_at: None,
                confidence: 1.0,
            },
        })?;

        // STM: a decaying working-memory fact, merged via the conflict resolver.
        let payload = json!({ "fact": distillate.summary, "tags": tags, "dataId": document_id });
        if let AdaptationResult::AccommodateCreate = self.conflict_resolver.resolve(
            &self.stm,
            &distillate.embedding,
            &self.tenant,
            &payload,
        )? {
            let node = MemoryNode {
                id: None,
                tenant_id: self.tenant.clone(),
                source_episode_id: None,
                payload,
                status: "active".into(),
                ccl: "reality".into(),
                is_explicit: true,
                support_count: 1,
                relevance_score: 1.0,
                context_key: None,
            };
            self.stm.store_node(&node, &distillate.embedding)?;
        }

        Ok(IngestOutcome::Ingested {
            data_id: document_id,
            leaf_node_id: placement.leaf_node_id,
            matched: placement.matched,
        })
    }

    /// Re-home inbox documents under concepts using their stored embeddings (no
    /// LLM) — e.g. after a placement-threshold change or a new spine branch.
    /// Returns how many were re-filed.
    pub fn garden_inbox(&self) -> Result<usize> {
        self.placement.garden_inbox()
    }

    /// Tombstone: forget a document from BOTH stores.
    pub async fn forget(&self, document_id: &str) -> Result<()> {
        self.placement.forget(document_id)?;
        self.stm.delete_nodes_by_data_id(document_id)?;
        Ok(())
    }
}

/// Derive a short leaf title from a distilled summary: its first non-empty line,
/// stripped of a leading Markdown heading marker and capped to ~80 chars. Falls
/// back to `fallback` (the dataId) when the summary has no usable text.
fn derive_title(summary: &str, fallback: &str) -> String {
    let line = summary
        .lines()
        .map(|l| l.trim().trim_start_matches('#').trim())
        .find(|l| !l.is_empty());
    match line {
        Some(l) => {
            let capped: String = l.chars().take(80).collect();
            if l.chars().count() > 80 {
                format!("{capped}…")
            } else {
                capped
            }
        }
        None => fallback.to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::domain::models::CclDefinition;
    use crate::domain::ports::ExtractedFact;
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::ltm_repository::SqliteLtmRepository;
    use crate::infrastructure::repository::SqliteMemoryRepository;
    use crate::infrastructure::schema::{init_ltm_schema, init_schema};
    use async_trait::async_trait;

    const DIM: usize = 8;
    const TENANT: &str = "default";

    struct StubLlm;
    #[async_trait]
    impl LlmClient for StubLlm {
        async fn extract_facts(
            &self,
            _d: &str,
            _c: &[CclDefinition],
        ) -> Result<Vec<ExtractedFact>> {
            Ok(vec![])
        }
        async fn generate_ccl_description(&self, _n: &str, _c: &str) -> Result<String> {
            Ok(String::new())
        }
        async fn embed_text(&self, _t: &str) -> Result<Vec<f32>> {
            Ok(vec![0.25; DIM])
        }
        async fn compress_context(&self, m: &str) -> Result<String> {
            Ok(format!("summary[{}]", m.len()))
        }
    }

    /// A document event with text (shared with other modules' tests).
    pub(crate) fn text_event(data_id: &str) -> DocumentEvent {
        DocumentEvent {
            data_id: Some(data_id.into()),
            title: None,
            text: Some(format!("the text of {data_id}")),
            tags: vec!["letter".into()],
            ts: None,
        }
    }

    struct Harness {
        svc: IngestionService,
        stm: Arc<SqliteMemoryRepository>,
        ltm: Arc<SqliteLtmRepository>,
        tenant: TenantId,
    }

    fn harness() -> Harness {
        harness_with(Arc::new(StubLlm))
    }

    /// Embeds, but has no chat model (the default no-LLM setup).
    struct NoChatLlm;
    #[async_trait]
    impl LlmClient for NoChatLlm {
        async fn extract_facts(
            &self,
            _d: &str,
            _c: &[CclDefinition],
        ) -> Result<Vec<ExtractedFact>> {
            anyhow::bail!("LLM not configured")
        }
        async fn generate_ccl_description(&self, _n: &str, _c: &str) -> Result<String> {
            anyhow::bail!("LLM not configured")
        }
        async fn embed_text(&self, _t: &str) -> Result<Vec<f32>> {
            Ok(vec![0.25; DIM])
        }
        async fn compress_context(&self, _m: &str) -> Result<String> {
            anyhow::bail!("LLM not configured")
        }
        fn chat_available(&self) -> bool {
            false
        }
    }

    fn harness_with(llm: Arc<dyn LlmClient>) -> Harness {
        let stm_conn = init_db(None as Option<&String>).unwrap();
        init_schema(&stm_conn, DIM).unwrap();
        let stm = Arc::new(SqliteMemoryRepository::new(stm_conn));

        let ltm_conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&ltm_conn, DIM).unwrap();
        let ltm = Arc::new(SqliteLtmRepository::new(ltm_conn));
        ltm.seed_spine().unwrap(); // provides the inbox fallback

        let svc = IngestionService::new(
            stm.clone() as Arc<dyn MemoryRepository>,
            ltm.clone() as Arc<dyn LtmRepository>,
            llm,
            DIM,
            TENANT,
        );
        Harness {
            svc,
            stm,
            ltm,
            tenant: TenantId(TENANT.into()),
        }
    }

    /// The new wire shape parses: snake_case `data_id`, optional title/tags/ts,
    /// and `dataId` is accepted as an alias.
    #[test]
    fn test_event_wire_shape() {
        let e: DocumentEvent = serde_json::from_str(
            r#"{"data_id":"d1","title":"T","text":"body","tags":["a"],"ts":"2026-01-01T00:00:00Z","extra":1}"#,
        )
        .unwrap();
        assert_eq!(e.document_id(), Some("d1"));
        assert_eq!(e.body(), Some("body"));
        assert_eq!(e.tags, vec!["a".to_string()]);

        let e: DocumentEvent = serde_json::from_str(r#"{"dataId":"d2","text":"x"}"#).unwrap();
        assert_eq!(e.document_id(), Some("d2"));

        let e: DocumentEvent = serde_json::from_str(r#"{"data_id":"  ","text":"  "}"#).unwrap();
        assert_eq!(e.document_id(), None);
        assert_eq!(e.body(), None);
    }

    /// A document lands a leaf in LTM AND a node in STM; a tombstone removes it
    /// from both. The text comes from the event itself.
    #[tokio::test]
    async fn test_ingest_dual_writes_then_tombstone_forgets_both() {
        let h = harness();

        let outcome = h.svc.ingest(&text_event("doc_1")).await.unwrap();
        match outcome {
            IngestOutcome::Ingested { data_id, .. } => assert_eq!(data_id, "doc_1"),
            other => panic!("expected Ingested, got {other:?}"),
        }

        assert!(h.ltm.get_node_by_data_id("doc_1").unwrap().is_some());
        let export = h.stm.export_tenant(&h.tenant).unwrap();
        assert!(export.contains("doc_1"), "STM has the doc fact");
        assert!(export.contains("letter"), "event tags reach the STM fact");

        h.svc.forget("doc_1").await.unwrap();
        assert!(h.ltm.get_node_by_data_id("doc_1").unwrap().is_none());
        let export = h.stm.export_tenant(&h.tenant).unwrap();
        assert!(!export.contains("doc_1"), "STM fact removed");
    }

    /// An event without text (missing or blank) is Skipped; nothing is written.
    #[tokio::test]
    async fn test_event_without_text_is_skipped() {
        let h = harness();
        for text in [None, Some("   ".to_string())] {
            let e = DocumentEvent {
                text,
                ..text_event("doc_2")
            };
            let outcome = h.svc.ingest(&e).await.unwrap();
            assert_eq!(
                outcome,
                IngestOutcome::Skipped {
                    data_id: "doc_2".into()
                }
            );
        }
        assert!(h.ltm.get_node_by_data_id("doc_2").unwrap().is_none());
    }

    /// `derive_title` takes the first meaningful line (stripping a Markdown
    /// heading) and falls back to the dataId when the summary is blank.
    #[test]
    fn test_derive_title() {
        assert_eq!(
            derive_title("# Annual Budget Report\n\nBody...", "doc_x"),
            "Annual Budget Report"
        );
        assert_eq!(derive_title("   \n\n", "doc_x"), "doc_x");
        let long = "x".repeat(200);
        let title = derive_title(&long, "doc_x");
        assert!(title.chars().count() <= 81 && title.ends_with('…'));
    }

    /// Without an event title, the leaf is named from the summary (not the raw
    /// dataId), and its provenance carries a populated `ingested_at`.
    #[tokio::test]
    async fn test_leaf_gets_title_and_ingested_at() {
        let h = harness();
        h.svc.ingest(&text_event("doc_title")).await.unwrap();

        let leaf = h.ltm.get_node_by_data_id("doc_title").unwrap().unwrap();
        assert!(leaf.name.starts_with("summary["));
        let stored = h.ltm.get_leaf(leaf.id.unwrap()).unwrap().unwrap();
        assert!(stored.provenance.ingested_at.is_some());
    }

    /// An explicit event title names the leaf.
    #[tokio::test]
    async fn test_event_title_names_the_leaf() {
        let h = harness();
        let e = DocumentEvent {
            title: Some("# Lease agreement".into()),
            ..text_event("doc_lease")
        };
        h.svc.ingest(&e).await.unwrap();
        let leaf = h.ltm.get_node_by_data_id("doc_lease").unwrap().unwrap();
        assert_eq!(leaf.name, "Lease agreement");
    }

    /// An event without a data_id is an error (the feeder dead-letters it).
    #[tokio::test]
    async fn test_event_without_id_errors() {
        let h = harness();
        let bad = DocumentEvent {
            data_id: None,
            ..text_event("x")
        };
        assert!(h.svc.ingest(&bad).await.is_err());
    }

    /// No chat model: the feeder still files the document into LTM and STM,
    /// using the title + start of the text as the summary.
    #[tokio::test]
    async fn test_ingest_without_chat_llm_still_dual_writes() {
        let h = harness_with(Arc::new(NoChatLlm));
        let e = DocumentEvent {
            title: Some("Lease agreement".into()),
            text: Some("The lease runs to 2027.".into()),
            ..text_event("doc_nochat")
        };
        let outcome = h.svc.ingest(&e).await.expect("ingest without a chat LLM");
        assert!(matches!(outcome, IngestOutcome::Ingested { .. }));

        let leaf = h.ltm.get_node_by_data_id("doc_nochat").unwrap().unwrap();
        assert_eq!(leaf.name, "Lease agreement");
        assert!(
            leaf.summary.contains("The lease runs to 2027."),
            "{}",
            leaf.summary
        );
        let export = h.stm.export_tenant(&h.tenant).unwrap();
        assert!(
            export.contains("doc_nochat") && export.contains("lease runs"),
            "{export}"
        );
    }
}
