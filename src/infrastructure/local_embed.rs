//! Local, offline text embeddings — the `local` embedding provider.
//!
//! Backed by [fastembed-rs](https://github.com/Anush008/fastembed-rs), which
//! runs quantised/ONNX sentence-embedding models through ONNX Runtime (`ort`).
//! No API key, no network after the one-time model download. The model files
//! are cached under a caller-supplied directory (`<home>/models`), never the
//! CWD (fastembed's own default is `./.fastembed_cache`, which we override).
//!
//! The model is loaded lazily on the first `embed_text` call (a cold load of
//! `bge-small-en-v1.5` downloads ~130 MB once, then loads in well under a
//! second). The dimension is known up front from the model table, so stores
//! can be sized without touching the network — see [`LocalEmbedder::dim`].
//!
//! Only the embedding half of [`LlmClient`] is implemented; the chat methods
//! fail. `create_llm_client` pairs this with a chat provider (or with the
//! "LLM not configured" stub) via `SplitLlmClient`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use crate::domain::ports::{ExtractedFact, LlmClient};

/// Default local model: small (384-d), fast on CPU, strong English retrieval.
pub const DEFAULT_LOCAL_MODEL: &str = "bge-small-en-v1.5";

/// Friendly, stable names for the models we document. The name is what goes
/// into config and into the store's `embedding_model` meta key, so it must
/// never change for a given model. Any other fastembed model can still be
/// selected by its enum name (e.g. `SnowflakeArcticEmbedS`); its canonical id
/// is then the lower-cased enum name.
const ALIASES: &[(&str, EmbeddingModel)] = &[
    ("bge-small-en-v1.5", EmbeddingModel::BGESmallENV15),
    ("bge-small-en-v1.5-q", EmbeddingModel::BGESmallENV15Q),
    ("bge-base-en-v1.5", EmbeddingModel::BGEBaseENV15),
    ("bge-large-en-v1.5", EmbeddingModel::BGELargeENV15),
    ("all-minilm-l6-v2", EmbeddingModel::AllMiniLML6V2),
    ("multilingual-e5-small", EmbeddingModel::MultilingualE5Small),
    ("bge-m3", EmbeddingModel::BGEM3),
    ("embeddinggemma-300m", EmbeddingModel::EmbeddingGemma300M),
    ("mxbai-embed-large-v1", EmbeddingModel::MxbaiEmbedLargeV1),
];

/// A resolved local model: the fastembed enum, its canonical config name and
/// its output dimension.
#[derive(Debug, Clone)]
pub struct LocalModelSpec {
    pub model: EmbeddingModel,
    pub name: String,
    pub dim: usize,
}

/// Resolve a configured model name (a friendly alias, case-insensitive, or a
/// fastembed enum name) to a spec. Pure: no I/O.
pub fn resolve_model(name: &str) -> Result<LocalModelSpec> {
    let wanted = name.trim();
    let model = ALIASES
        .iter()
        .find(|(alias, _)| alias.eq_ignore_ascii_case(wanted))
        .map(|(_, m)| m.clone())
        .or_else(|| wanted.parse::<EmbeddingModel>().ok())
        .ok_or_else(|| {
            let known: Vec<&str> = ALIASES.iter().map(|(a, _)| *a).collect();
            anyhow!(
                "unknown local embedding model '{wanted}'; use one of: {} \
                 (or any fastembed EmbeddingModel name)",
                known.join(", ")
            )
        })?;
    let dim = TextEmbedding::get_model_info(&model)
        .map_err(|e| anyhow!("no model info for {model:?}: {e}"))?
        .dim;
    let canonical = ALIASES
        .iter()
        .find(|(_, m)| *m == model)
        .map(|(a, _)| (*a).to_string())
        .unwrap_or_else(|| format!("{model:?}").to_ascii_lowercase());
    Ok(LocalModelSpec {
        model,
        name: canonical,
        dim,
    })
}

/// The `local` embedder. Cheap to construct (no I/O); the ONNX session is
/// created on first use and shared behind a mutex (fastembed's `embed` takes
/// `&mut self`). Inference runs on the blocking pool so it never stalls the
/// async runtime.
pub struct LocalEmbedder {
    spec: LocalModelSpec,
    cache_dir: PathBuf,
    engine: tokio::sync::OnceCell<Arc<Mutex<TextEmbedding>>>,
}

impl LocalEmbedder {
    /// `cache_dir` is where model files are downloaded/cached — pass
    /// `<home>/models`. It is created if missing.
    pub fn new(model_name: &str, cache_dir: PathBuf) -> Result<Self> {
        Ok(Self {
            spec: resolve_model(model_name)?,
            cache_dir,
            engine: tokio::sync::OnceCell::new(),
        })
    }

    /// Output dimension, known without loading the model.
    pub fn dim(&self) -> usize {
        self.spec.dim
    }

    /// Canonical model name (the value stored in store meta).
    pub fn model_name(&self) -> &str {
        &self.spec.name
    }

    /// Load (downloading on first run) the model now instead of on the first
    /// embed call — lets a server pay the cold start at boot.
    pub async fn warm_up(&self) -> Result<()> {
        self.engine().await.map(|_| ())
    }

    async fn engine(&self) -> Result<Arc<Mutex<TextEmbedding>>> {
        self.engine
            .get_or_try_init(|| async {
                let spec = self.spec.clone();
                let dir = self.cache_dir.clone();
                tokio::task::spawn_blocking(move || load(&spec, dir))
                    .await
                    .context("local embedder load task panicked")?
            })
            .await
            .cloned()
    }
}

fn load(spec: &LocalModelSpec, cache_dir: PathBuf) -> Result<Arc<Mutex<TextEmbedding>>> {
    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("creating model cache dir {}", cache_dir.display()))?;
    // stdout is the MCP transport — progress/notices go to stderr only, and
    // fastembed's own progress bar stays off.
    tracing::info!(
        "loading local embedding model '{}' (cache: {}; first run downloads it)",
        spec.name,
        cache_dir.display()
    );
    let options = TextInitOptions::new(spec.model.clone())
        .with_cache_dir(cache_dir)
        .with_show_download_progress(false);
    let engine = TextEmbedding::try_new(options).map_err(|e| {
        anyhow!(
            "failed to load local embedding model '{}': {e} \
             (the first run needs network access to huggingface.co)",
            spec.name
        )
    })?;
    Ok(Arc::new(Mutex::new(engine)))
}

/// L2-normalise so the store's L2 distance ranks exactly like cosine.
fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn chat_unsupported<T>() -> Result<T> {
    Err(anyhow!(
        "LLM not configured: the local provider only computes embeddings"
    ))
}

#[async_trait::async_trait]
impl LlmClient for LocalEmbedder {
    async fn extract_facts(
        &self,
        _dialogue: &str,
        _valid_ccls: &[crate::domain::models::CclDefinition],
    ) -> Result<Vec<ExtractedFact>> {
        chat_unsupported()
    }

    async fn generate_ccl_description(&self, _ccl_name: &str, _context: &str) -> Result<String> {
        chat_unsupported()
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let engine = self.engine().await?;
        let text = text.to_string();
        let expected = self.spec.dim;
        let vector = tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
            let mut guard = engine
                .lock()
                .map_err(|_| anyhow!("local embedder mutex poisoned"))?;
            let mut out = guard
                .embed(vec![text], None)
                .map_err(|e| anyhow!("local embedding failed: {e}"))?;
            out.pop()
                .ok_or_else(|| anyhow!("local embedder returned no vector"))
        })
        .await
        .context("local embedding task panicked")??;
        if vector.len() != expected {
            return Err(anyhow!(
                "local embedder returned {} dims, expected {expected}",
                vector.len()
            ));
        }
        Ok(normalize(vector))
    }

    async fn compress_context(&self, _messages: &str) -> Result<String> {
        chat_unsupported()
    }

    /// Known from the model table — no model load, no network.
    async fn embedding_dim(&self) -> Result<usize> {
        Ok(self.spec.dim)
    }

    fn embedding_model_id(&self) -> String {
        format!("local:{}", self.spec.name)
    }

    fn chat_available(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_resolves_to_bge_small_384() {
        let spec = resolve_model(DEFAULT_LOCAL_MODEL).unwrap();
        assert_eq!(spec.model, EmbeddingModel::BGESmallENV15);
        assert_eq!(spec.dim, 384);
        assert_eq!(spec.name, "bge-small-en-v1.5");
    }

    #[test]
    fn aliases_are_case_insensitive_and_canonicalised() {
        let spec = resolve_model("  BGE-Small-EN-v1.5 ").unwrap();
        assert_eq!(spec.name, "bge-small-en-v1.5");
        // The enum name selects the same model and canonicalises to the alias,
        // so two spellings never look like a model change in store meta.
        let by_enum = resolve_model("BGESmallENV15").unwrap();
        assert_eq!(by_enum.name, "bge-small-en-v1.5");
    }

    #[test]
    fn every_alias_resolves_with_a_positive_dim() {
        for (alias, _) in ALIASES {
            let spec = resolve_model(alias).unwrap();
            assert!(spec.dim > 0, "{alias}");
            assert_eq!(&spec.name, alias);
        }
    }

    #[test]
    fn non_aliased_enum_name_canonicalises_to_lowercase() {
        let spec = resolve_model("SnowflakeArcticEmbedXS").unwrap();
        assert_eq!(spec.name, "snowflakearcticembedxs");
    }

    #[test]
    fn unknown_model_is_an_actionable_error() {
        let err = resolve_model("text-embedding-3-small")
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown local embedding model"), "{err}");
        assert!(err.contains(DEFAULT_LOCAL_MODEL), "{err}");
    }

    #[test]
    fn construction_and_dim_do_no_io() {
        // A cache dir that doesn't exist: new() + dim() must not touch it.
        let dir = std::env::temp_dir().join("neurolithe-never-created-xyz");
        let e = LocalEmbedder::new(DEFAULT_LOCAL_MODEL, dir.clone()).unwrap();
        assert_eq!(e.dim(), 384);
        assert!(!dir.exists());
    }

    #[test]
    fn normalize_yields_unit_length() {
        let v = normalize(vec![3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        assert_eq!(normalize(vec![0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[tokio::test]
    async fn chat_methods_report_llm_not_configured() {
        let e = LocalEmbedder::new(DEFAULT_LOCAL_MODEL, std::env::temp_dir()).unwrap();
        let err = e.compress_context("x").await.unwrap_err().to_string();
        assert!(err.starts_with("LLM not configured"), "{err}");
    }

    /// Downloads the real model (~130 MB) — run manually:
    /// `cargo test --lib local_embed -- --ignored`
    /// Honors `NEUROLITHE_TEST_MODEL_DIR` to reuse a cache between runs.
    #[tokio::test]
    #[ignore = "downloads the ONNX model from Hugging Face"]
    async fn real_model_embeds_and_ranks_semantically() {
        let dir = std::env::var("NEUROLITHE_TEST_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("neurolithe-test-models"));
        let e = LocalEmbedder::new(DEFAULT_LOCAL_MODEL, dir).unwrap();
        let q = e
            .embed_text("How do I renew my car insurance?")
            .await
            .unwrap();
        let near = e
            .embed_text("Vehicle insurance policy renewal steps")
            .await
            .unwrap();
        let far = e.embed_text("A recipe for sourdough bread").await.unwrap();
        assert_eq!(q.len(), 384);
        let norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "not normalised: {norm}");
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let (s_near, s_far) = (dot(&q, &near), dot(&q, &far));
        eprintln!("cos(near)={s_near:.3} cos(far)={s_far:.3}");
        assert!(s_near > s_far + 0.1);
    }
}
