//! `remember_document`: file a document into long-term memory from the
//! standalone MCP server (Phase 2 §5).
//!
//! The document is summarized (with the chat LLM if one is configured,
//! otherwise by truncation), embedded, and placed in the knowledge tree under
//! its best-matching concept (or the inbox). Idempotent by `data_id`: storing a
//! document again with the same id replaces the previous version.

use crate::application::ltm_placement::{DocumentToPlace, LtmPlacement};
use crate::domain::ltm::{LtmRepository, Provenance, TreeNodeKind};
use crate::domain::ports::LlmClient;
use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Without a chat LLM (or if summarization fails) the leaf summary is the
/// first this-many characters of the text.
pub const SUMMARY_FALLBACK_CHARS: usize = 1000;
/// Leaf titles are clipped to this many characters.
pub const TITLE_MAX_CHARS: usize = 120;
/// Provenance `source` recorded on leaves filed through this service.
pub const DOCUMENT_SOURCE: &str = "remember_document";

/// A document to remember.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RememberDocument {
    /// Human-facing title (becomes the leaf name). Empty → first line of text.
    #[serde(default)]
    pub title: String,
    pub text: String,
    /// Stable id for upserts. Omitted → a new `doc_<uuidv7>` is minted.
    #[serde(default)]
    pub data_id: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Where the document landed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RememberedDocument {
    pub data_id: String,
    pub leaf_id: i64,
    /// Concept names from the root down to the leaf's parent,
    /// e.g. `["root", "documents"]` or `["root", "inbox"]`.
    pub concept_path: Vec<String>,
    /// True if an earlier version with this `data_id` was replaced.
    pub updated: bool,
    /// True if the chat LLM summarized the text; false if it was truncated.
    pub summarized: bool,
}

/// Files documents into LTM (see module docs).
pub struct DocumentService {
    ltm: Arc<dyn LtmRepository>,
    /// Resolved placement threshold (see `domain::thresholds`).
    placement_max_distance: f64,
    /// Embeddings are required; summaries are used only when
    /// [`LlmClient::chat_available`].
    llm: Arc<dyn LlmClient>,
}

impl DocumentService {
    pub fn new(
        ltm: Arc<dyn LtmRepository>,
        llm: Arc<dyn LlmClient>,
        thresholds: &crate::domain::thresholds::Thresholds,
    ) -> Self {
        Self {
            ltm,
            placement_max_distance: thresholds.placement_max_distance.value,
            llm,
        }
    }

    pub async fn remember(&self, doc: RememberDocument) -> Result<RememberedDocument> {
        let text = doc.text.trim();
        if text.is_empty() {
            bail!("remember_document requires non-empty 'text'");
        }
        let title = clip(
            if doc.title.trim().is_empty() {
                text.lines().next().unwrap_or(text).trim()
            } else {
                doc.title.trim()
            },
            TITLE_MAX_CHARS,
        );
        let data_id = match doc.data_id.as_deref().map(str::trim) {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => format!("doc_{}", Uuid::now_v7()),
        };

        let (mut summary, summarized) = self.summarize(text).await;
        if !doc.tags.is_empty() {
            summary = format!("{summary}\nTags: {}", doc.tags.join(", "));
        }
        let embedding = self
            .llm
            .embed_text(&format!("{title}\n\n{summary}"))
            .await?;

        // Upsert: file the new version, then drop every older copy and re-roll
        // the parents they leave (see `LtmPlacement::replace`).
        let (placed, replaced) = LtmPlacement::new(self.ltm.clone(), self.placement_max_distance)
            .replace(&DocumentToPlace {
            name: title,
            summary,
            embedding,
            data_id: data_id.clone(),
            provenance: Provenance {
                source: DOCUMENT_SOURCE.to_string(),
                ingested_at: None,
                confidence: 1.0,
            },
        })?;

        Ok(RememberedDocument {
            data_id,
            leaf_id: placed.leaf_node_id,
            concept_path: self.concept_path(placed.parent_id)?,
            updated: replaced > 0,
            summarized,
        })
    }

    /// Summary via the chat LLM when available; otherwise (or on failure) the
    /// clipped text. Returns (summary, summarized_by_llm).
    async fn summarize(&self, text: &str) -> (String, bool) {
        if self.llm.chat_available() {
            match self.llm.compress_context(text).await {
                Ok(s) if !s.trim().is_empty() => return (s.trim().to_string(), true),
                Ok(_) => tracing::warn!("document summary was empty; storing an excerpt"),
                Err(e) => {
                    tracing::warn!("document summary failed ({e:#}); storing an excerpt")
                }
            }
        }
        (clip(text, SUMMARY_FALLBACK_CHARS), false)
    }

    /// Names from the root down to `node_id` (following the first parent at
    /// each step; bounded against cycles).
    fn concept_path(&self, node_id: i64) -> Result<Vec<String>> {
        let mut path = Vec::new();
        let mut current = self.ltm.get_node(node_id)?;
        while let Some(node) = current {
            if node.kind != TreeNodeKind::Leaf {
                path.push(node.name.clone());
            }
            if path.len() > 64 {
                break;
            }
            let id = node.id.ok_or_else(|| anyhow!("stored node without id"))?;
            current = self.ltm.get_parents(id)?.into_iter().next();
        }
        path.reverse();
        Ok(path)
    }
}

/// `s` clipped to `max` characters on a char boundary (with an ellipsis).
fn clip(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ltm::TreeNode;
    use crate::domain::models::CclDefinition;
    use crate::domain::ports::ExtractedFact;
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::ltm_repository::SqliteLtmRepository;
    use crate::infrastructure::schema::init_ltm_schema;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const DIM: usize = 4;

    /// Embeds texts mentioning "invoice" near the `documents` concept, everything
    /// else far away; summarizes to a fixed marker (or fails).
    struct Stub {
        chat: bool,
        summarize_fails: bool,
        summaries: AtomicUsize,
    }

    #[async_trait]
    impl LlmClient for Stub {
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
        async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            Ok(if text.contains("invoice") {
                vec![1.0, 0.0, 0.0, 0.0]
            } else if text.contains("receipt") {
                // L2 0.5 from the `documents` concept.
                vec![1.0, 0.5, 0.0, 0.0]
            } else {
                vec![0.0, 0.0, 0.0, 1.0]
            })
        }
        async fn compress_context(&self, m: &str) -> Result<String> {
            self.summaries.fetch_add(1, Ordering::SeqCst);
            if self.summarize_fails {
                bail!("LLM not configured");
            }
            Ok(format!("SUMMARY of {} chars", m.len()))
        }
        fn chat_available(&self) -> bool {
            self.chat
        }
    }

    fn setup(
        chat: bool,
        summarize_fails: bool,
    ) -> (DocumentService, Arc<SqliteLtmRepository>, Arc<Stub>) {
        let conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&conn, DIM).unwrap();
        let repo = Arc::new(SqliteLtmRepository::new(conn));
        repo.seed_spine().unwrap();
        // Give the `documents` concept a placement vector near "invoice" texts.
        let root = repo.get_roots().unwrap().remove(0).id.unwrap();
        let documents = repo
            .get_children(root)
            .unwrap()
            .into_iter()
            .find(|n| n.name == "documents")
            .unwrap();
        repo.set_concept_embedding(documents.id.unwrap(), &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        let stub = Arc::new(Stub {
            chat,
            summarize_fails,
            summaries: AtomicUsize::new(0),
        });
        (
            DocumentService::new(
                repo.clone(),
                stub.clone(),
                &crate::domain::thresholds::Thresholds::text_embedding_004(),
            ),
            repo,
            stub,
        )
    }

    fn doc(title: &str, text: &str, data_id: Option<&str>) -> RememberDocument {
        RememberDocument {
            title: title.into(),
            text: text.into(),
            data_id: data_id.map(str::to_string),
            tags: vec![],
        }
    }

    fn leaf_count(repo: &SqliteLtmRepository, data_id: &str) -> usize {
        repo.get_roots()
            .unwrap()
            .iter()
            .flat_map(|r| all_leaves(repo, r.id.unwrap()))
            .filter(|d| d == data_id)
            .count()
    }

    fn all_leaves(repo: &SqliteLtmRepository, id: i64) -> Vec<String> {
        let mut out: Vec<String> = repo
            .get_child_leaves(id)
            .unwrap()
            .into_iter()
            .map(|l| l.data_id)
            .collect();
        for c in repo.get_children(id).unwrap() {
            if c.kind != TreeNodeKind::Leaf {
                out.extend(all_leaves(repo, c.id.unwrap()));
            }
        }
        out
    }

    /// A matching document is summarized, placed under its concept, and the
    /// response carries data_id, leaf_id and the concept path.
    #[tokio::test]
    async fn remembers_and_places_under_matching_concept() {
        let (svc, repo, _) = setup(true, false);
        let out = svc
            .remember(doc(
                "ACME invoice",
                "An invoice from ACME for 3 widgets.",
                Some("inv-1"),
            ))
            .await
            .unwrap();

        assert_eq!(out.data_id, "inv-1");
        assert_eq!(out.concept_path, vec!["root", "documents"]);
        assert!(out.summarized);
        assert!(!out.updated);
        let leaf: TreeNode = repo.get_node(out.leaf_id).unwrap().unwrap();
        assert_eq!(leaf.name, "ACME invoice");
        assert!(leaf.summary.starts_with("SUMMARY of"));
    }

    /// Without a chat LLM the text is stored as a clipped excerpt (no LLM call);
    /// an unmatched document goes to the inbox; a data_id is minted.
    #[tokio::test]
    async fn without_chat_llm_truncates_and_mints_id() {
        let (svc, repo, stub) = setup(false, false);
        let long = "x".repeat(SUMMARY_FALLBACK_CHARS + 50);
        let out = svc.remember(doc("", &long, None)).await.unwrap();

        assert_eq!(stub.summaries.load(Ordering::SeqCst), 0, "no chat call");
        assert!(!out.summarized);
        assert!(out.data_id.starts_with("doc_"));
        assert_eq!(out.concept_path, vec!["root", "inbox"]);
        let leaf = repo.get_node(out.leaf_id).unwrap().unwrap();
        assert_eq!(leaf.summary.chars().count(), SUMMARY_FALLBACK_CHARS + 1); // + '…'
        assert_eq!(leaf.name.chars().count(), TITLE_MAX_CHARS + 1);
    }

    /// A failing summarizer degrades to truncation instead of failing.
    #[tokio::test]
    async fn summary_failure_falls_back_to_excerpt() {
        let (svc, _, _) = setup(true, true);
        let out = svc.remember(doc("t", "some text", None)).await.unwrap();
        assert!(!out.summarized);
    }

    /// Same data_id twice → one leaf (the new version), reported as updated.
    #[tokio::test]
    async fn upsert_by_data_id_replaces_previous_version() {
        let (svc, repo, _) = setup(false, false);
        let first = svc
            .remember(doc("draft", "first version", Some("note-7")))
            .await
            .unwrap();
        let second = svc
            .remember(doc("final", "an invoice, second version", Some("note-7")))
            .await
            .unwrap();

        assert!(second.updated);
        assert_ne!(first.leaf_id, second.leaf_id);
        assert!(
            repo.get_node(first.leaf_id).unwrap().is_none(),
            "old leaf removed"
        );
        assert_eq!(leaf_count(&repo, "note-7"), 1);
        assert_eq!(
            second.concept_path,
            vec!["root", "documents"],
            "re-placed by new content"
        );
        let found = repo.get_node_by_data_id("note-7").unwrap().unwrap();
        assert_eq!(found.name, "final");
    }

    fn children_named(repo: &SqliteLtmRepository, parent: &str) -> TreeNode {
        let root = repo.get_roots().unwrap().remove(0).id.unwrap();
        repo.get_children(root)
            .unwrap()
            .into_iter()
            .find(|n| n.name == parent)
            .unwrap()
    }

    /// P2R-7: when an upsert moves a document to another concept, the old
    /// parent's rolled-up summary no longer lists it.
    #[tokio::test]
    async fn upsert_rerolls_the_old_parent() {
        let (svc, repo, _) = setup(false, false);
        svc.remember(doc("Draft memo", "plain text", Some("d-1")))
            .await
            .unwrap(); // → inbox
        assert!(
            children_named(&repo, "inbox")
                .summary
                .contains("Draft memo")
        );

        let out = svc
            .remember(doc("Final invoice", "an invoice now", Some("d-1")))
            .await
            .unwrap(); // → documents
        assert_eq!(out.concept_path, vec!["root", "documents"]);

        let inbox = children_named(&repo, "inbox");
        assert!(
            !inbox.summary.contains("Draft memo"),
            "stale inbox summary: {}",
            inbox.summary
        );
        assert!(
            children_named(&repo, "documents")
                .summary
                .contains("Final invoice")
        );
    }

    /// P2R-7: duplicates left by an interrupted replace (two leaves with one
    /// data_id) are all removed by the next upsert.
    #[tokio::test]
    async fn upsert_heals_duplicate_leaves() {
        let (svc, repo, _) = setup(false, false);
        let placement = LtmPlacement::new(
            repo.clone(),
            crate::domain::thresholds::TEXT_EMBEDDING_004.placement_max_distance,
        );
        for name in ["copy a", "copy b"] {
            placement
                .place(&DocumentToPlace {
                    name: name.into(),
                    summary: name.into(),
                    embedding: vec![0.0, 0.0, 0.0, 1.0],
                    data_id: "dup".into(),
                    provenance: Provenance {
                        source: "t".into(),
                        ingested_at: None,
                        confidence: 1.0,
                    },
                })
                .unwrap();
        }
        assert_eq!(repo.get_nodes_by_data_id("dup").unwrap().len(), 2);

        let out = svc
            .remember(doc("fresh", "new text", Some("dup")))
            .await
            .unwrap();

        let left = repo.get_nodes_by_data_id("dup").unwrap();
        assert_eq!(left.len(), 1, "all stale copies removed");
        assert_eq!(left[0].id, Some(out.leaf_id));
        assert!(out.updated);
        assert!(!children_named(&repo, "inbox").summary.contains("copy a"));
    }

    /// The resolved placement threshold decides filing: a document at distance
    /// 0.5 from `documents` files there under 1.10, but into the inbox under 0.3.
    #[tokio::test]
    async fn placement_threshold_reaches_remember_document() {
        let (loose, _, _) = setup(false, false);
        let out = loose
            .remember(doc("r", "a receipt", Some("r1")))
            .await
            .unwrap();
        assert_eq!(out.concept_path, vec!["root", "documents"]);

        let (svc, repo, stub) = setup(false, false);
        let strict = crate::domain::thresholds::Thresholds::resolve(
            "unknown:model",
            crate::domain::thresholds::ThresholdOverrides {
                placement_max_distance: Some(0.3),
                ..Default::default()
            },
        )
        .unwrap()
        .0;
        drop(svc);
        let svc = DocumentService::new(repo.clone(), stub.clone(), &strict);
        let out = svc
            .remember(doc("r", "a receipt", Some("r2")))
            .await
            .unwrap();
        assert_eq!(out.concept_path, vec!["root", "inbox"]);
    }

    #[tokio::test]
    async fn empty_text_is_rejected() {
        let (svc, _, _) = setup(false, false);
        assert!(svc.remember(doc("t", "   ", None)).await.is_err());
    }
}
