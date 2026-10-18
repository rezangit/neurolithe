//! Ingestion — turn a document event into memory, written to BOTH stores: a
//! permanent LTM leaf and a decaying STM working-memory fact.
//!
//! This is the broker-independent core of the Kafka feeder (the rdkafka loop in
//! `interfaces::kafka_feeder` drives it). The event carries the document text
//! itself — distill -> place in LTM + store in STM. Tombstone -> forget in
//! both.
//!
//! **Idempotent by `dataId`.** The feeder is an at-least-once consumer, so the
//! same event can arrive more than once (redelivery after a crash or a
//! rebalance, a replay). Ingest is an upsert keyed by the event's `dataId`:
//!
//! - The full text is stored for keyword recall with the leaf (one
//!   transaction). An unchanged redelivery only fills in a missing or
//!   backfilled text, so a document filed before bodies were kept gets one,
//!   but a newer text (e.g. from remember_document) is never overwritten.
//! - The event content (title, text, tags; not `ts`) is fingerprinted. If the
//!   STM fact for the `dataId` already carries that fingerprint and LTM holds
//!   exactly one leaf for it, the event is a duplicate: nothing is written and
//!   nothing is distilled ([`IngestOutcome::Unchanged`]).
//! - Otherwise the new version **replaces** the old one in both stores: the LTM
//!   leaf via [`LtmPlacement::replace`], the STM fact via
//!   [`MemoryRepository::replace_document_node`]. A changed payload never adds
//!   to the old content.
//!
//! **Crash safety.** STM and LTM are separate SQLite files, so one transaction
//! can't span both. Instead every step is re-runnable and ordered so the STM
//! fact (with the fingerprint) is the commit marker, written last: LTM first
//! (filing the new leaf is one transaction; dropping old copies follows and
//! is idempotent), then STM (one transaction). A crash anywhere before the STM
//! write leaves no matching fingerprint, so the redelivered event redoes the
//! whole upsert and heals any duplicate leaf. Because the feeder commits the
//! offset only after `ingest` returns, the redelivery is guaranteed.
//!
//! Dimension note: the feeder writes the same distilled embedding to both
//! stores, so both must share the embedder's dimension. A mismatch surfaces as
//! a loud store error rather than silent corruption.

use crate::application::distill::Distiller;
use crate::application::ltm_placement::{DocumentToPlace, LtmPlacement};
use crate::domain::ltm::{DocumentBody, LtmRepository, Provenance};
use crate::domain::models::{MemoryNode, REALITY_CCL, TenantId};
use crate::domain::ports::{MemoryRepository, Models};
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

/// Payload key of the STM document fact that holds the event's content
/// fingerprint (see the module docs).
pub const CONTENT_HASH_KEY: &str = "contentHash";

/// What ingesting one document did.
#[derive(Debug, Clone, PartialEq)]
pub enum IngestOutcome {
    /// Written to both stores. `replaced` is true when an earlier version (or
    /// a half-written copy) with this `dataId` was replaced.
    Ingested {
        data_id: String,
        leaf_node_id: i64,
        matched: bool,
        replaced: bool,
    },
    /// A redelivery of content already stored in both stores: nothing written.
    Unchanged { data_id: String, leaf_node_id: i64 },
    /// No usable text in the event — nothing learned, but the offset still
    /// advances.
    Skipped { data_id: String },
}

/// Upserts documents into LTM (placement) + STM (one fact per `dataId`).
pub struct IngestionService {
    stm: Arc<dyn MemoryRepository>,
    ltm: Arc<dyn LtmRepository>,
    placement: LtmPlacement,
    distiller: Distiller,
    tenant: TenantId,
}

impl IngestionService {
    pub fn new(
        stm: Arc<dyn MemoryRepository>,
        ltm: Arc<dyn LtmRepository>,
        models: Models,
        embedding_dim: usize,
        tenant: impl Into<String>,
        thresholds: &crate::domain::thresholds::Thresholds,
    ) -> Self {
        Self {
            placement: LtmPlacement::from_thresholds(ltm.clone(), thresholds),
            distiller: Distiller::with_models(models, embedding_dim),
            ltm,
            stm,
            tenant: TenantId(tenant.into()),
        }
    }

    /// Ingest one document event (an upsert by `dataId`; see the module docs):
    /// distill its text, then file the LTM leaf and store the STM fact, each
    /// replacing any earlier version. A redelivered duplicate is `Unchanged`;
    /// an event without text is `Skipped`.
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

        // The full text for recall's keyword leg. A new or changed version
        // stores it with its leaf, in one transaction (W3R-2).
        let body = DocumentBody {
            title: event.title.clone().unwrap_or_default(),
            text: text.to_string(),
        };

        let hash = content_fingerprint(event.title.as_deref(), text, &tags);
        if let Some(leaf_node_id) = self.stored_version(&document_id, &hash)? {
            // Identical content is already filed. Fill in its text if the
            // document has none or only the v2 backfill's stand-in, never
            // overwriting a newer text (another writer may have upserted
            // this data_id since, W3R-7).
            self.ltm
                .fill_document_body(&document_id, &body.title, &body.text)?;
            tracing::debug!("document '{document_id}' unchanged; redelivery ignored");
            return Ok(IngestOutcome::Unchanged {
                data_id: document_id,
                leaf_node_id,
            });
        }

        let distillate = self.distiller.distill(event.title.as_deref(), text).await?;

        // A human-readable leaf title: the event's own title when given, else
        // the first heading/line of the summary, else the dataId.
        let title = match event.title.as_deref().map(str::trim) {
            Some(t) if !t.is_empty() => derive_title(t, &document_id),
            _ => derive_title(&distillate.summary, &document_id),
        };

        // 1. LTM: file the new leaf under the best concept (or inbox), then drop
        //    every older copy of this dataId.
        let (placement, replaced_leaves) = self.placement.replace_with_body(
            &DocumentToPlace {
                name: title,
                summary: distillate.summary.clone(),
                embedding: distillate.embedding.clone(),
                data_id: document_id.clone(),
                provenance: Provenance {
                    source: "kafka".into(),
                    ingested_at: None,
                    confidence: 1.0,
                },
            },
            Some(&body),
        )?;

        // 2. STM, last: the fact carrying the fingerprint marks the upsert
        //    complete. It replaces the previous version of this document's fact
        //    (one transaction) instead of going through the conflict resolver,
        //    so a new version neither reinforces nor merges with the old one.
        let payload = json!({
            "fact": distillate.summary,
            "tags": tags,
            "dataId": document_id,
            CONTENT_HASH_KEY: hash,
        });
        let node = MemoryNode {
            id: None,
            tenant_id: self.tenant.clone(),
            source_episode_id: None,
            payload,
            status: "active".into(),
            ccl: REALITY_CCL.into(),
            is_explicit: true,
            support_count: 1,
            relevance_score: 1.0,
            context_key: None,
        };
        let had_fact = !self.stm.find_document_nodes(&document_id)?.is_empty();
        self.stm
            .replace_document_node(&document_id, &node, &distillate.embedding)?;

        Ok(IngestOutcome::Ingested {
            data_id: document_id,
            leaf_node_id: placement.leaf_node_id,
            matched: placement.matched,
            replaced: replaced_leaves > 0 || had_fact,
        })
    }

    /// The LTM leaf id if this exact content (`hash`) is already fully stored
    /// for `data_id`: one STM document fact carrying the fingerprint, and one
    /// LTM leaf. Anything else (no fact, an older fingerprint, a missing or
    /// duplicated leaf) means the upsert must run.
    fn stored_version(&self, data_id: &str, hash: &str) -> Result<Option<i64>> {
        let facts = self.stm.find_document_nodes(data_id)?;
        let [fact] = facts.as_slice() else {
            return Ok(None);
        };
        if fact.payload.get(CONTENT_HASH_KEY).and_then(|h| h.as_str()) != Some(hash) {
            return Ok(None);
        }
        let leaves = self.ltm.get_nodes_by_data_id(data_id)?;
        Ok(match leaves.as_slice() {
            [leaf] => leaf.id,
            _ => None,
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

/// A stable fingerprint of a document event's content: title, text and tags
/// (the producer's `ts` is informational and excluded, so a re-publish of the
/// same content is still a duplicate). Fields are length-prefixed so they
/// can't run into each other. FNV-1a (64-bit): it only has to tell one version
/// of a document from the next, and its output never changes between builds.
pub fn content_fingerprint(title: Option<&str>, text: &str, tags: &[String]) -> String {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    let mut feed = |bytes: &[u8]| {
        for b in (bytes.len() as u64).to_le_bytes().iter().chain(bytes) {
            h ^= u64::from(*b);
            h = h.wrapping_mul(PRIME);
        }
    };
    feed(title.map(str::trim).unwrap_or("").as_bytes());
    feed(text.as_bytes());
    for tag in tags {
        feed(tag.as_bytes());
    }
    format!("fnv1a64:{h:016x}")
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
    use crate::domain::ports::{ChatModel, Embedder, ExtractedFact};
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::ltm_repository::SqliteLtmRepository;
    use crate::infrastructure::repository::SqliteMemoryRepository;
    use crate::infrastructure::schema::{init_ltm_schema, init_schema};
    use async_trait::async_trait;

    const DIM: usize = 8;
    const TENANT: &str = "default";

    struct StubLlm;
    #[async_trait]
    impl ChatModel for StubLlm {
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
        async fn compress_context(&self, m: &str) -> Result<String> {
            Ok(format!("summary[{}]", m.len()))
        }
    }

    #[async_trait]
    impl Embedder for StubLlm {
        async fn embed_text(&self, _t: &str) -> Result<Vec<f32>> {
            Ok(vec![0.25; DIM])
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

    /// An ingestion service over fresh in-memory stores (shared with the
    /// feeder's broker-free tests).
    pub(crate) struct Harness {
        pub(crate) svc: IngestionService,
        pub(crate) stm: Arc<SqliteMemoryRepository>,
        pub(crate) ltm: Arc<SqliteLtmRepository>,
        pub(crate) tenant: TenantId,
    }

    fn harness() -> Harness {
        harness_with(Models::new(Arc::new(StubLlm), Some(Arc::new(StubLlm))))
    }

    /// Embeds, but has no chat model (the default no-LLM setup).
    pub(crate) struct NoChatLlm;
    #[async_trait]
    impl Embedder for NoChatLlm {
        async fn embed_text(&self, _t: &str) -> Result<Vec<f32>> {
            Ok(vec![0.25; DIM])
        }
    }

    pub(crate) fn harness_with(models: Models) -> Harness {
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
            models,
            DIM,
            TENANT,
            &crate::domain::thresholds::Thresholds::text_embedding_004(),
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
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
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

    // --- Idempotent ingest (at-least-once delivery) ---------------------------

    /// Counts LLM calls so a test can tell a no-op redelivery from a re-ingest.
    /// Embeds like [`NoChatLlm`] (no chat model), so the summary is the text.
    struct CountingLlm {
        embeds: std::sync::atomic::AtomicUsize,
    }
    #[async_trait]
    impl Embedder for CountingLlm {
        async fn embed_text(&self, _t: &str) -> Result<Vec<f32>> {
            self.embeds
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(vec![0.25; DIM])
        }
    }

    fn versioned(data_id: &str, text: &str) -> DocumentEvent {
        DocumentEvent {
            data_id: Some(data_id.into()),
            title: Some("Lease".into()),
            text: Some(text.into()),
            tags: vec!["letter".into()],
            ts: None,
        }
    }

    /// LTM leaves carrying `data_id`.
    fn leaves(h: &Harness, data_id: &str) -> Vec<crate::domain::ltm::TreeNode> {
        h.ltm.get_nodes_by_data_id(data_id).unwrap()
    }

    /// STM facts carrying `data_id`: (fact text, support count).
    fn stm_facts(h: &Harness, data_id: &str) -> Vec<(String, i32)> {
        h.stm
            .list_node_summaries(1000, 0, None, None)
            .unwrap()
            .into_iter()
            .filter(|n| n.data_id.as_deref() == Some(data_id))
            .map(|n| (n.fact, n.support_count))
            .collect()
    }

    /// Exactly one LTM leaf and one STM fact for `data_id`, both saying `text`.
    fn assert_single_version(h: &Harness, data_id: &str, text: &str) {
        let leaves = leaves(h, data_id);
        assert_eq!(leaves.len(), 1, "one LTM leaf for {data_id}: {leaves:?}");
        assert!(
            leaves[0].summary.contains(text),
            "leaf holds the latest text: {}",
            leaves[0].summary
        );
        let facts = stm_facts(h, data_id);
        assert_eq!(facts.len(), 1, "one STM fact for {data_id}: {facts:?}");
        assert!(
            facts[0].0.contains(text),
            "STM holds the latest text: {facts:?}"
        );
        assert_eq!(facts[0].1, 1, "a redelivery must not reinforce: {facts:?}");
    }

    /// The same event delivered N times leaves exactly the state of one
    /// delivery: one leaf (same id), one unreinforced STM fact, and the
    /// duplicates don't even re-run the embedder.
    #[tokio::test]
    async fn test_duplicate_delivery_is_a_no_op() {
        let llm = Arc::new(CountingLlm {
            embeds: Default::default(),
        });
        let h = harness_with(Models::embed_only(llm.clone()));
        let e = versioned("doc_dup", "The lease runs to 2027.");

        let first = h.svc.ingest(&e).await.unwrap();
        let embeds_after_first = llm.embeds.load(std::sync::atomic::Ordering::SeqCst);
        for _ in 0..3 {
            h.svc.ingest(&e).await.unwrap();
        }

        assert_single_version(&h, "doc_dup", "The lease runs to 2027.");
        let IngestOutcome::Ingested { leaf_node_id, .. } = first else {
            panic!("expected Ingested, got {first:?}");
        };
        assert_eq!(leaves(&h, "doc_dup")[0].id, Some(leaf_node_id), "leaf kept");
        assert_eq!(
            llm.embeds.load(std::sync::atomic::Ordering::SeqCst),
            embeds_after_first,
            "a duplicate is detected before distilling"
        );
    }

    /// W3R-7: an unchanged redelivery only fills in a missing or backfilled
    /// body; it never overwrites a text written since (e.g. an MCP upsert of
    /// the same data_id), even when that text is short enough to equal its
    /// summary.
    #[tokio::test]
    async fn test_redelivery_never_overwrites_a_newer_body() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        let body = |id: &str| -> String {
            h.ltm
                .conn_for_tests()
                .query_row(
                    "SELECT body FROM doc_bodies WHERE data_id = ?1",
                    [id],
                    |r| r.get(0),
                )
                .unwrap()
        };
        let v1 = versioned("doc_k", "kafka version text");
        h.svc.ingest(&v1).await.unwrap();
        // Another writer (remember_document) has since stored its own text.
        h.ltm
            .put_document_body("doc_k", "Lease", "mcp version")
            .unwrap();
        let again = h.svc.ingest(&v1).await.unwrap();
        assert!(
            matches!(again, IngestOutcome::Unchanged { .. }),
            "{again:?}"
        );
        assert_eq!(body("doc_k"), "mcp version", "the newer text is kept");

        // A body the v2 migration backfilled (title + summary) is replaced
        // by the redelivered full text.
        h.ltm
            .conn_for_tests()
            .execute(
                "UPDATE doc_bodies SET body = 'backfilled summary', backfilled = 1
                 WHERE data_id = 'doc_k'",
                [],
            )
            .unwrap();
        h.svc.ingest(&v1).await.unwrap();
        assert_eq!(body("doc_k"), "kafka version text");
    }

    /// Embeds like [`NoChatLlm`], but fails on any text containing "FAIL".
    struct FailOnMarker;
    #[async_trait]
    impl Embedder for FailOnMarker {
        async fn embed_text(&self, t: &str) -> Result<Vec<f32>> {
            if t.contains("FAIL") {
                anyhow::bail!("embedder down");
            }
            Ok(vec![0.25; DIM])
        }
    }

    /// W3R-2: a changed event whose filing fails leaves the stored text
    /// matching the stored (old) leaf: the new words find nothing, the old
    /// ones still find the document.
    #[tokio::test]
    async fn test_failed_update_keeps_the_body_in_step_with_the_leaf() {
        let h = harness_with(Models::embed_only(Arc::new(FailOnMarker)));
        h.svc
            .ingest(&versioned("doc_x", "alpha original lighthouse"))
            .await
            .unwrap();
        let leaf = leaves(&h, "doc_x")[0].id.unwrap();
        assert!(
            h.svc
                .ingest(&versioned("doc_x", "FAIL beta windmill changed"))
                .await
                .is_err()
        );
        let hits = |q: &str| h.ltm.search_document_bodies(q, 10).unwrap();
        assert!(
            hits("\"windmill\"").is_empty(),
            "the unfiled version's text"
        );
        assert_eq!(hits("\"lighthouse\""), [leaf], "the filed version's text");
    }

    /// W3R-2: a brand-new document whose filing fails leaves no text behind.
    #[tokio::test]
    async fn test_failed_first_ingest_leaves_no_body() {
        let h = harness_with(Models::embed_only(Arc::new(FailOnMarker)));
        assert!(
            h.svc
                .ingest(&versioned("doc_y", "FAIL secret salary letter"))
                .await
                .is_err()
        );
        let left: i64 = h
            .ltm
            .conn_for_tests()
            .query_row("SELECT COUNT(*) FROM doc_bodies", [], |r| r.get(0))
            .unwrap();
        assert_eq!(left, 0);
    }

    /// W3R-1 (privacy): forget removes a document's full text even when no
    /// leaf carries it (an orphan, e.g. from an ingest that failed before
    /// filing).
    #[tokio::test]
    async fn test_forget_removes_an_orphan_body() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        h.ltm
            .put_document_body("doc_secret", "", "secret salary letter")
            .unwrap();
        h.svc.forget("doc_secret").await.unwrap();
        let left: i64 = h
            .ltm
            .conn_for_tests()
            .query_row(
                "SELECT COUNT(*) FROM doc_bodies WHERE data_id = 'doc_secret'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(left, 0, "the full text must not outlive forget");
    }

    /// E2: every ingest stores the document's full text for keyword recall,
    /// including a duplicate redelivery (so a document filed before bodies
    /// existed gets one), and a changed event overwrites it.
    #[tokio::test]
    async fn test_ingest_keeps_the_full_text_searchable() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        let filler = "lorem ipsum dolor sit amet ".repeat(80);
        let leaf_hits = |q: &str| h.ltm.search_document_bodies(q, 10).unwrap();

        let v1 = versioned("doc_b", &format!("{filler} the lighthouse keeper"));
        h.svc.ingest(&v1).await.unwrap();
        let leaf = leaves(&h, "doc_b")[0].id.unwrap();
        assert_eq!(leaf_hits("\"lighthouse\""), [leaf], "text past the excerpt");

        // A pre-v2 document: filed, but no body. Its unchanged redelivery
        // still writes one.
        h.ltm
            .conn_for_tests()
            .execute("DELETE FROM doc_bodies", [])
            .unwrap();
        assert!(leaf_hits("\"lighthouse\"").is_empty());
        let again = h.svc.ingest(&v1).await.unwrap();
        assert!(
            matches!(again, IngestOutcome::Unchanged { .. }),
            "{again:?}"
        );
        assert_eq!(leaf_hits("\"lighthouse\""), [leaf]);

        // A changed event overwrites the body.
        h.svc
            .ingest(&versioned("doc_b", &format!("{filler} the old windmill")))
            .await
            .unwrap();
        let leaf = leaves(&h, "doc_b")[0].id.unwrap();
        assert!(leaf_hits("\"lighthouse\"").is_empty());
        assert_eq!(leaf_hits("\"windmill\""), [leaf]);
    }

    /// A changed payload with the same dataId replaces the old content in both
    /// stores; it doesn't add to it.
    #[tokio::test]
    async fn test_changed_payload_replaces_the_old_version() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        h.svc
            .ingest(&versioned("doc_v", "The lease runs to 2027."))
            .await
            .unwrap();
        h.svc
            .ingest(&versioned("doc_v", "The lease was extended to 2030."))
            .await
            .unwrap();

        assert_single_version(&h, "doc_v", "extended to 2030");
        let export = h.stm.export_tenant(&h.tenant).unwrap();
        assert!(
            !export.contains("runs to 2027"),
            "old STM text gone: {export}"
        );
        assert!(!leaves(&h, "doc_v")[0].summary.contains("runs to 2027"));
    }

    /// Tags are part of the content: a tag-only change is a new version, and
    /// the STM fact carries exactly the new tags (not a union).
    #[tokio::test]
    async fn test_tag_change_is_a_new_version() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        let v1 = versioned("doc_t", "Same text.");
        h.svc.ingest(&v1).await.unwrap();
        let v2 = DocumentEvent {
            tags: vec!["contract".into()],
            ..v1
        };
        h.svc.ingest(&v2).await.unwrap();

        assert_single_version(&h, "doc_t", "Same text.");
        let export = h.stm.export_tenant(&h.tenant).unwrap();
        assert!(export.contains("contract"), "{export}");
        assert!(!export.contains("letter"), "old tag replaced: {export}");
    }

    /// Crash after the LTM write, before the STM write: the redelivered event
    /// completes the ingest without duplicating the leaf.
    #[tokio::test]
    async fn test_redelivery_after_crash_between_ltm_and_stm() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        let e = versioned("doc_crash", "The lease runs to 2027.");
        h.svc.ingest(&e).await.unwrap();
        // The state a crash between the two writes leaves: LTM has the leaf,
        // STM never got its fact.
        h.stm.delete_nodes_by_data_id("doc_crash").unwrap();
        assert_eq!(leaves(&h, "doc_crash").len(), 1);

        h.svc.ingest(&e).await.unwrap();
        assert_single_version(&h, "doc_crash", "runs to 2027");
    }

    /// Crash while a NEW version was half-written (LTM already has v2, STM
    /// still has v1): redelivering v2 finishes the job — one leaf, one fact,
    /// both v2.
    #[tokio::test]
    async fn test_redelivery_after_crash_mid_update() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        h.svc
            .ingest(&versioned("doc_mid", "The lease runs to 2027."))
            .await
            .unwrap();
        // Simulate the crash: v2 reached LTM (placed beside v1, the old copy
        // not yet removed) and STM never saw it.
        let v2 = versioned("doc_mid", "The lease was extended to 2030.");
        let placement = LtmPlacement::new(h.ltm.clone() as Arc<dyn LtmRepository>, 1.10);
        placement
            .place(&DocumentToPlace {
                name: "Lease".into(),
                summary: "Lease\n\nThe lease was extended to 2030.".into(),
                embedding: vec![0.25; DIM],
                data_id: "doc_mid".into(),
                provenance: Provenance {
                    source: "kafka".into(),
                    ingested_at: None,
                    confidence: 1.0,
                },
            })
            .unwrap();
        assert_eq!(leaves(&h, "doc_mid").len(), 2);

        h.svc.ingest(&v2).await.unwrap();
        assert_single_version(&h, "doc_mid", "extended to 2030");
    }

    /// A stray duplicate leaf (e.g. a crash inside an earlier LTM replace) is
    /// cleaned up by the next delivery of that document, even an unchanged one.
    #[tokio::test]
    async fn test_redelivery_repairs_a_duplicate_leaf() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        let e = versioned("doc_two", "The lease runs to 2027.");
        h.svc.ingest(&e).await.unwrap();
        let placement = LtmPlacement::new(h.ltm.clone() as Arc<dyn LtmRepository>, 1.10);
        placement
            .place(&DocumentToPlace {
                name: "Lease".into(),
                summary: "Lease\n\nThe lease runs to 2027.".into(),
                embedding: vec![0.25; DIM],
                data_id: "doc_two".into(),
                provenance: Provenance {
                    source: "kafka".into(),
                    ingested_at: None,
                    confidence: 1.0,
                },
            })
            .unwrap();

        h.svc.ingest(&e).await.unwrap();
        assert_single_version(&h, "doc_two", "runs to 2027");
    }

    /// Re-ingesting a document replaces only its own STM fact: a working-memory
    /// `subject` anchor that shares the dataId (a dialogue turn about the
    /// document) survives.
    #[tokio::test]
    async fn test_reingest_keeps_subject_nodes_with_the_same_data_id() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        h.svc
            .ingest(&versioned("doc_s", "The lease runs to 2027."))
            .await
            .unwrap();
        h.stm
            .store_node(
                &MemoryNode {
                    id: None,
                    tenant_id: h.tenant.clone(),
                    source_episode_id: None,
                    payload: json!({"fact": "Lease", "kind": "subject", "dataId": "doc_s"}),
                    status: "active".into(),
                    ccl: "working".into(),
                    is_explicit: true,
                    support_count: 1,
                    relevance_score: 1.0,
                    context_key: Some("thread-1".into()),
                },
                &[0.0; DIM],
            )
            .unwrap();

        h.svc
            .ingest(&versioned("doc_s", "The lease was extended to 2030."))
            .await
            .unwrap();
        assert_eq!(
            h.stm.count_by_data_id("doc_s").unwrap(),
            2,
            "fact + subject"
        );
        let export = h.stm.export_tenant(&h.tenant).unwrap();
        assert!(export.contains("\"subject\""), "{export}");
        assert!(!export.contains("runs to 2027"), "{export}");
    }

    /// A tombstone is idempotent too: forgetting twice (or a document that was
    /// never ingested) succeeds and leaves nothing behind.
    #[tokio::test]
    async fn test_repeated_tombstone_is_idempotent() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        h.svc
            .ingest(&versioned("doc_f", "The lease runs to 2027."))
            .await
            .unwrap();
        for _ in 0..2 {
            h.svc.forget("doc_f").await.unwrap();
        }
        h.svc.forget("never_seen").await.unwrap();
        assert!(leaves(&h, "doc_f").is_empty());
        assert!(stm_facts(&h, "doc_f").is_empty());
    }

    /// The fingerprint is stable (a fixed value for fixed input, so it survives
    /// upgrades), covers title, text and tags, and keeps fields apart.
    #[test]
    fn test_content_fingerprint() {
        let tags = vec!["a".to_string()];
        let fp = content_fingerprint(Some("T"), "body", &tags);
        assert_eq!(fp, content_fingerprint(Some(" T "), "body", &tags));
        assert!(fp.starts_with("fnv1a64:") && fp.len() == 24, "{fp}");
        // FNV-1a 64 of sixteen zero bytes (two empty length-prefixed fields).
        assert_eq!(
            content_fingerprint(None, "", &[]),
            "fnv1a64:88201fb960ff6465"
        );
        for other in [
            content_fingerprint(None, "body", &tags),
            content_fingerprint(Some("T"), "body!", &tags),
            content_fingerprint(Some("T"), "body", &[]),
            content_fingerprint(Some("Tb"), "ody", &tags),
        ] {
            assert_ne!(fp, other);
        }
    }

    /// Only the producer timestamp changed: still a duplicate.
    #[tokio::test]
    async fn test_new_ts_alone_is_a_duplicate() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        let v1 = DocumentEvent {
            ts: Some("2026-01-01T00:00:00Z".into()),
            ..versioned("doc_ts", "Same text.")
        };
        h.svc.ingest(&v1).await.unwrap();
        let v2 = DocumentEvent {
            ts: Some("2026-02-01T00:00:00Z".into()),
            ..v1
        };
        assert!(matches!(
            h.svc.ingest(&v2).await.unwrap(),
            IngestOutcome::Unchanged { .. }
        ));
        assert_single_version(&h, "doc_ts", "Same text.");
    }
}
