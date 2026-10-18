//! Meaning extraction — distill a document's text into the form the memory
//! stores need: a concise summary + an embedding (+ forward-looking concept
//! hints). The Kafka feeder runs this on the text carried by each document event.

use crate::domain::ports::Models;
use crate::infrastructure::embedding_codec::check_dim;
use anyhow::Result;

/// The distilled meaning of a document.
#[derive(Debug, Clone, PartialEq)]
pub struct Distillate {
    /// Concise summary — the leaf's summary and roll-up input.
    pub summary: String,
    /// Candidate concept labels for placement. Reserved for smart growth
    /// (deferred); V2 placement uses the embedding, so this is empty for now.
    pub concept_hints: Vec<String>,
    /// Embedding of the summary, locked to the LTM vector dimension.
    pub embedding: Vec<f32>,
}

/// How much of a document's text becomes its summary when no chat model is
/// configured (characters, not bytes).
pub const FALLBACK_SUMMARY_CHARS: usize = 1500;

/// The summary used without a chat model: the title (if any), a blank line, and
/// the first [`FALLBACK_SUMMARY_CHARS`] characters of the text (whitespace-
/// trimmed, cut on a char boundary, `…` when truncated).
pub fn fallback_summary(title: Option<&str>, text: &str) -> String {
    let body = text.trim();
    let mut head: String = body.chars().take(FALLBACK_SUMMARY_CHARS).collect();
    if body.chars().count() > FALLBACK_SUMMARY_CHARS {
        head.push('…');
    }
    match title.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => format!("{t}\n\n{head}"),
        None => head,
    }
}

/// Distills document text into a summary (chat model, or the no-LLM fallback)
/// plus an embedding.
pub struct Distiller {
    /// The embedder, and the chat model that summarizes (`None`: fallback).
    models: Models,
    /// Expected embedding length (LTM vector dimension). A mismatch is a hard
    /// error — a wrong-dimension vector would corrupt `vec_ltm`.
    expected_dim: usize,
}

impl Distiller {
    pub fn with_models(models: Models, expected_dim: usize) -> Self {
        Self {
            models,
            expected_dim,
        }
    }

    /// Summarize `text` (with the chat model, or — when none is configured —
    /// [`fallback_summary`] of `title` + the start of the text), then embed the
    /// summary. Errors if the embedding's length does not match the configured
    /// LTM dimension.
    pub async fn distill(&self, title: Option<&str>, text: &str) -> Result<Distillate> {
        let summary = match &self.models.chat {
            Some(chat) => chat.compress_context(text).await?,
            None => fallback_summary(title, text),
        };
        let embedding = self.models.embedder.embed_text(&summary).await?;

        check_dim(&embedding, self.expected_dim, "embedder (LTM summary)")?;

        Ok(Distillate {
            summary,
            concept_hints: Vec::new(),
            embedding,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::models::CclDefinition;
    use crate::domain::ports::{ChatModel, Embedder, ExtractedFact};
    use async_trait::async_trait;
    use std::sync::Arc;

    const DIM: usize = 8;

    /// A stub LLM: summary is a fixed marker, embedding is a fixed-length vector
    /// (length controlled per-test to exercise the dimension check).
    struct StubLlm {
        embed_dim: usize,
    }

    /// The stub as both models (it summarizes and embeds).
    fn stub(embed_dim: usize) -> Models {
        Models::new(
            Arc::new(StubLlm { embed_dim }),
            Some(Arc::new(StubLlm { embed_dim })),
        )
    }

    /// No chat model: an embedder only.
    struct NoChatLlm;

    #[async_trait]
    impl Embedder for NoChatLlm {
        async fn embed_text(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![0.5; DIM])
        }
    }

    #[async_trait]
    impl ChatModel for StubLlm {
        async fn extract_facts(
            &self,
            _dialogue: &str,
            _valid_ccls: &[CclDefinition],
        ) -> Result<Vec<ExtractedFact>> {
            Ok(vec![])
        }
        async fn generate_ccl_description(&self, _name: &str, _ctx: &str) -> Result<String> {
            Ok(String::new())
        }
        async fn compress_context(&self, messages: &str) -> Result<String> {
            Ok(format!("SUMMARY OF: {messages}"))
        }
    }

    #[async_trait]
    impl Embedder for StubLlm {
        async fn embed_text(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![0.5; self.embed_dim])
        }
    }

    #[tokio::test]
    async fn test_distill_returns_summary_and_embedding() {
        let distiller = Distiller::with_models(stub(DIM), DIM);

        let out = distiller
            .distill(Some("Title"), "a long document body")
            .await
            .unwrap();
        assert_eq!(out.summary, "SUMMARY OF: a long document body");
        assert_eq!(out.embedding.len(), DIM, "embedding is the configured dim");
        assert!(out.concept_hints.is_empty());
    }

    #[tokio::test]
    async fn test_distill_rejects_wrong_dimension() {
        // Model returns DIM+1; the distiller must refuse it (would corrupt vec_ltm).
        let distiller = Distiller::with_models(stub(DIM + 1), DIM);
        assert!(distiller.distill(None, "text").await.is_err());
    }

    /// Without a chat model the summary is the title + the first 1,500 chars of
    /// the text, and it is still embedded.
    #[tokio::test]
    async fn test_distill_without_chat_uses_title_and_text_head() {
        let distiller = Distiller::with_models(Models::embed_only(Arc::new(NoChatLlm)), DIM);

        let out = distiller
            .distill(Some("Lease agreement"), "  The lease runs to 2027.  ")
            .await
            .unwrap();
        assert_eq!(out.summary, "Lease agreement\n\nThe lease runs to 2027.");
        assert_eq!(out.embedding.len(), DIM);

        let untitled = distiller.distill(None, "body only").await.unwrap();
        assert_eq!(untitled.summary, "body only");
    }

    /// Long text is cut to FALLBACK_SUMMARY_CHARS characters (on a char
    /// boundary, multi-byte safe) and marked with an ellipsis.
    #[test]
    fn test_fallback_summary_truncates_on_char_boundary() {
        let text = "é".repeat(FALLBACK_SUMMARY_CHARS + 10);
        let s = fallback_summary(Some("  "), &text);
        assert_eq!(s.chars().count(), FALLBACK_SUMMARY_CHARS + 1);
        assert!(s.ends_with('…'));
        let exact = "x".repeat(FALLBACK_SUMMARY_CHARS);
        assert_eq!(fallback_summary(None, &exact), exact);
    }
}
