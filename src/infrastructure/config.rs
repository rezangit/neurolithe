use serde::Deserialize;

/// Top-level runtime configuration for NeuroLithe V2.
///
/// V2 splits memory into two independent SQLite stores — a decaying short-term
/// store (`stm`) and a permanent long-term knowledge tree (`ltm`) — each with
/// its own path and vector dimension. The remaining sections wire the Kafka
/// feeder, the Pithos client, and the background schedulers. See
/// `IMPLEMENTATION-V2.md` slice 1 and `V2-DESIGN.md` §8.
#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub llm: LlmConfig,
    pub stm: StoreConfig,
    pub ltm: StoreConfig,
    pub kafka: KafkaConfig,
    pub pithos: PithosConfig,
    pub sweep: SweepConfig,
    pub metrics: MetricsConfig,
    pub feeder: FeederConfig,
    #[serde(default)]
    pub bus_query: BusQueryConfig,
    #[serde(default)]
    pub decay: DecayConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LlmConfig {
    pub provider: LlmProvider,
    pub model: String,
    pub embedding_model: String,
    pub base_url: Option<String>,
    /// Provider used for embeddings, independent of the chat `provider`.
    ///
    /// Claude has no embeddings endpoint, so when `provider` is `anthropic`
    /// this MUST point at a provider that does — e.g. `gemini` for Google
    /// `text-embedding-004`, or `custom` for a local Ollama/OpenAI-compatible
    /// endpoint. When unset it falls back to `provider` (single-provider mode,
    /// the historical behaviour).
    #[serde(default)]
    pub embedding_provider: Option<LlmProvider>,
    /// Base URL for the embedding provider (only used by `openai`/`custom`).
    /// Falls back to `base_url` when unset.
    #[serde(default)]
    pub embedding_base_url: Option<String>,
    /// GCP project for the `vertex` embedding provider (Vertex AI). Auth comes
    /// from the service-account key at `GOOGLE_APPLICATION_CREDENTIALS`.
    #[serde(default)]
    pub embedding_project: Option<String>,
    /// Vertex AI region for the `vertex` embedding provider, e.g. `us-central1`
    /// (drives data residency). `global` uses the unprefixed host. Defaults to
    /// `us-central1` when unset.
    #[serde(default)]
    pub embedding_location: Option<String>,
    /// Total per-request timeout (seconds) for LLM/embedding HTTP calls. A hung
    /// provider must not freeze the (serial) MCP loop forever (DEV-5).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
}

fn default_request_timeout_secs() -> u64 {
    120
}

impl LlmConfig {
    /// The provider actually used for embeddings: `embedding_provider` when set,
    /// otherwise the chat `provider`.
    pub fn effective_embedding_provider(&self) -> &LlmProvider {
        self.embedding_provider.as_ref().unwrap_or(&self.provider)
    }
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LlmProvider {
    Openai,
    Gemini,
    Anthropic,
    Custom,
    /// Google Vertex AI (service-account auth via `GOOGLE_APPLICATION_CREDENTIALS`).
    /// Used for embeddings, sharing Cadmus's Gemini/Vertex access.
    Vertex,
}

/// Configuration for one of the two SQLite memory stores (STM or LTM).
///
/// `vector_dimension` is locked at DB init — changing it on an existing file
/// breaks the `sqlite-vec` virtual table, so the DB must be rebuilt. The two
/// stores carry independent dimensions: STM and LTM can use different embedding
/// models (e.g. STM 1536, LTM 768 for local `nomic-embed-text`).
#[derive(Debug, Deserialize, Clone)]
pub struct StoreConfig {
    pub vector_dimension: usize,
    /// SQLite file path; `None` means an in-memory DB (used by tests).
    pub path: Option<String>,
}

/// Kafka connection settings for the feeder, command consumer, and metrics
/// publisher (rdkafka). Topics themselves are created out-of-band by
/// `../kafka/create-topics.sh` (slice 0); no auto-create.
#[derive(Debug, Deserialize, Clone)]
pub struct KafkaConfig {
    pub brokers: String,
    pub group_id: String,
}

/// Pithos archive-service connection — the feeder fetches `pt://` document text
/// over HTTP to distill meaning before writing to the stores. `token` is the
/// Pithos bearer token (read access); empty means unauthenticated (tests/local).
#[derive(Debug, Deserialize, Clone)]
pub struct PithosConfig {
    pub base_url: String,
    #[serde(default)]
    pub token: String,
}

/// How often the background STM decay sweep runs (`run_decay_sweep`). The sweep
/// decays each node by its *real* elapsed time (not a fixed pass), so a frequent
/// cadence is safe — it never over-decays durable facts. The default is a few
/// minutes so short-half-life `working` notes actually expire soon after they go
/// cold, rather than lingering until a daily pass.
#[derive(Debug, Deserialize, Clone)]
pub struct SweepConfig {
    pub interval_secs: u64,
}

/// How often the CT-scan snapshot is published to `memory.metrics` (slice 9).
#[derive(Debug, Deserialize, Clone)]
pub struct MetricsConfig {
    pub interval_secs: u64,
}

/// Whether the Kafka feeder (`document.completed` → dual-write) is active.
/// Lets the daemon run reset/introspection without consuming documents.
#[derive(Debug, Deserialize, Clone)]
pub struct FeederConfig {
    pub enabled: bool,
}

/// Whether the `memory.query` → `memory.result` request/reply loop is active —
/// the bus door that lets Metis read STM + LTM (design §§3–6). Independent of
/// the feeder so reads can be served without ingestion, and vice versa.
#[derive(Debug, Deserialize, Clone)]
pub struct BusQueryConfig {
    pub enabled: bool,
}

impl Default for BusQueryConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Per-layer STM decay half-lives (STM-WORKING-MEMORY slice 2). The `working`
/// layer (situational notes) is expressed in **minutes** because it is meant to
/// fade on a session timescale, while durable facts use the multi-day default.
#[derive(Debug, Deserialize, Clone)]
pub struct DecayConfig {
    /// Half-life (days) for every layer except `working`.
    pub default_half_life_days: f64,
    /// Half-life (minutes) for the `working` layer. Placeholder default 30 min —
    /// tunable against real sessions (STM-WORKING-MEMORY open item).
    pub working_half_life_minutes: f64,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            default_half_life_days: 7.0,
            working_half_life_minutes: 30.0,
        }
    }
}

impl DecayConfig {
    /// The `working` half-life converted to days (the unit the `DecayEngine`
    /// works in). 1440 minutes = 1 day.
    pub fn working_half_life_days(&self) -> f64 {
        self.working_half_life_minutes / 1440.0
    }
}

impl AppConfig {
    pub fn load() -> anyhow::Result<Self> {
        // Load .env file if it exists
        let _ = dotenvy::dotenv();

        let config = Self::build(std::path::Path::new("neurolithe.toml").exists(), None)?;
        config.validate()?;
        Ok(config)
    }

    /// Reject configurations that would panic or silently misbehave at runtime
    /// (DEV-12, ARC-23): a zero scheduler interval panics `tokio::time::interval`,
    /// a non-positive half-life makes decay NaN/inf, a zero vector dimension
    /// cannot back a sqlite-vec table, and Anthropic has no embeddings API.
    /// Collects every problem so one run reports them all.
    pub fn validate(&self) -> anyhow::Result<()> {
        let mut problems: Vec<String> = Vec::new();

        if self.sweep.interval_secs == 0 {
            problems.push("sweep.interval_secs must be > 0".into());
        }
        if self.metrics.interval_secs == 0 {
            problems.push("metrics.interval_secs must be > 0".into());
        }
        for (name, v) in [
            (
                "decay.default_half_life_days",
                self.decay.default_half_life_days,
            ),
            (
                "decay.working_half_life_minutes",
                self.decay.working_half_life_minutes,
            ),
        ] {
            if !(v.is_finite() && v > 0.0) {
                problems.push(format!("{name} must be a positive number (got {v})"));
            }
        }
        for (name, store) in [("stm", &self.stm), ("ltm", &self.ltm)] {
            if store.vector_dimension == 0 {
                problems.push(format!("{name}.vector_dimension must be > 0"));
            }
            if store.path.as_deref().is_some_and(|p| p.trim().is_empty()) {
                problems.push(format!("{name}.path must not be empty"));
            }
        }
        if self.llm.request_timeout_secs == 0 {
            problems.push("llm.request_timeout_secs must be > 0".into());
        }
        if self.llm.model.trim().is_empty() {
            problems.push("llm.model must not be empty".into());
        }
        if self.llm.embedding_model.trim().is_empty() {
            problems.push("llm.embedding_model must not be empty".into());
        }
        match self.llm.effective_embedding_provider() {
            LlmProvider::Anthropic => problems.push(
                "llm.embedding_provider resolves to 'anthropic', which has no embeddings API; \
                 set llm.embedding_provider to openai, gemini, vertex, or custom"
                    .into(),
            ),
            LlmProvider::Vertex
                if self
                    .llm
                    .embedding_project
                    .as_deref()
                    .is_none_or(|p| p.trim().is_empty()) =>
            {
                problems.push("llm.embedding_project is required for the vertex embedder".into())
            }
            _ => {}
        }

        if problems.is_empty() {
            Ok(())
        } else {
            anyhow::bail!("invalid configuration:\n  - {}", problems.join("\n  - "))
        }
    }

    /// Build the config from defaults → optional `neurolithe.toml` → env vars.
    ///
    /// `include_file` is split out so tests can exercise the pure
    /// defaults/env path without a stray `neurolithe.toml` in the CWD skewing
    /// the result.
    /// `env` replaces the process environment as the override source when
    /// given — tests inject variables this way instead of mutating global
    /// process state with `set_var`, which races other parallel tests (QA-14).
    fn build(
        include_file: bool,
        env: Option<std::collections::HashMap<String, String>>,
    ) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder()
            .set_default("llm.provider", "openai")?
            .set_default("llm.model", "gpt-4o-mini")?
            .set_default("llm.embedding_model", "text-embedding-3-small")?
            // STM keeps the historical default dimension (1536, OpenAI).
            .set_default("stm.vector_dimension", 1536)?
            .set_default("stm.path", "neurolithe-stm.sqlite")?
            // LTM defaults to 768 (local nomic-embed-text) per the offline goal;
            // the embedding provider stays configurable (decided at deploy time).
            .set_default("ltm.vector_dimension", 768)?
            .set_default("ltm.path", "neurolithe-ltm.sqlite")?
            .set_default("kafka.brokers", "localhost:9092")?
            .set_default("kafka.group_id", "neurolithe")?
            .set_default("pithos.base_url", "http://192.168.4.48:8080")?
            .set_default("pithos.token", "")?
            // Frequent decay sweep (real-elapsed decay makes this safe); short
            // enough that cold `working` notes expire within minutes.
            .set_default("sweep.interval_secs", 300_i64)?
            .set_default("metrics.interval_secs", 60_i64)?
            .set_default("feeder.enabled", true)?
            .set_default("bus_query.enabled", true)?
            // Per-layer decay: durable facts 7-day, working notes 30-minute.
            .set_default("decay.default_half_life_days", 7.0)?
            .set_default("decay.working_half_life_minutes", 30.0)?;

        // If neurolithe.toml exists, load it
        if include_file && std::path::Path::new("neurolithe.toml").exists() {
            builder = builder.add_source(config::File::with_name("neurolithe.toml"));
        }

        // Environment variables override file config (e.g. NEUROLITHE__LLM__PROVIDER=gemini)
        builder = builder.add_source(
            config::Environment::with_prefix("NEUROLITHE")
                .separator("__")
                .source(env.map(|m| m.into_iter().collect())),
        );

        let config = builder.build()?;
        let app_config: AppConfig = config.try_deserialize()?;

        Ok(app_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// An empty injected environment: the pure-defaults path, isolated from
    /// whatever `NEUROLITHE__*` vars the test process happens to carry.
    fn no_env() -> Option<HashMap<String, String>> {
        Some(HashMap::new())
    }

    /// Config parses with the V2 defaults (independent STM 1536 / LTM 768
    /// dimensions, Kafka/Pithos/scheduler sections) and the defaults validate.
    #[test]
    fn test_defaults() {
        let cfg = AppConfig::build(false, no_env()).expect("config should load from defaults");

        assert_eq!(cfg.stm.vector_dimension, 1536);
        assert_eq!(cfg.ltm.vector_dimension, 768);
        assert_eq!(cfg.stm.path.as_deref(), Some("neurolithe-stm.sqlite"));
        assert_eq!(cfg.ltm.path.as_deref(), Some("neurolithe-ltm.sqlite"));
        assert_eq!(cfg.kafka.brokers, "localhost:9092");
        assert_eq!(cfg.kafka.group_id, "neurolithe");
        assert_eq!(cfg.pithos.base_url, "http://192.168.4.48:8080");
        assert_eq!(cfg.sweep.interval_secs, 300);
        assert_eq!(cfg.metrics.interval_secs, 60);
        assert_eq!(cfg.llm.request_timeout_secs, 120);
        assert!(cfg.feeder.enabled);
        assert!(cfg.bus_query.enabled);
        // Per-layer decay defaults.
        assert_eq!(cfg.decay.default_half_life_days, 7.0);
        assert_eq!(cfg.decay.working_half_life_minutes, 30.0);
        assert!((cfg.decay.working_half_life_days() - 30.0 / 1440.0).abs() < 1e-12);
        // STM and LTM dimensions are independent — changing one never implies
        // the other.
        assert_ne!(cfg.stm.vector_dimension, cfg.ltm.vector_dimension);
        cfg.validate().expect("defaults must validate");
    }

    /// Env vars override a single store's dimension in isolation. The overrides
    /// are injected (no `set_var`), so this cannot race other tests (QA-14).
    #[test]
    fn test_env_override_is_injected_not_global() {
        let env: HashMap<String, String> = [
            ("NEUROLITHE__LTM__VECTOR_DIMENSION", "1024"),
            ("NEUROLITHE__FEEDER__ENABLED", "false"),
            ("NEUROLITHE__DECAY__WORKING_HALF_LIFE_MINUTES", "5"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

        let cfg = AppConfig::build(false, Some(env)).expect("config should load with overrides");

        // LTM override applied; STM untouched (dimensions are independent).
        assert_eq!(cfg.ltm.vector_dimension, 1024);
        assert_eq!(cfg.stm.vector_dimension, 1536);
        assert!(!cfg.feeder.enabled);
        assert_eq!(cfg.decay.working_half_life_minutes, 5.0);
        // The process environment was never touched.
        assert!(std::env::var("NEUROLITHE__LTM__VECTOR_DIMENSION").is_err());
    }

    fn valid() -> AppConfig {
        AppConfig::build(false, no_env()).unwrap()
    }

    /// DEV-12: a zero interval (panics `tokio::time::interval`) and a zero or
    /// non-finite half-life (NaN decay) are rejected up front, all reported.
    #[test]
    fn test_validate_rejects_zero_intervals_and_half_lives() {
        let mut cfg = valid();
        cfg.sweep.interval_secs = 0;
        cfg.metrics.interval_secs = 0;
        cfg.decay.default_half_life_days = 0.0;
        cfg.decay.working_half_life_minutes = f64::NAN;
        cfg.ltm.vector_dimension = 0;
        cfg.llm.request_timeout_secs = 0;
        let err = cfg.validate().unwrap_err().to_string();
        for needle in [
            "sweep.interval_secs",
            "metrics.interval_secs",
            "decay.default_half_life_days",
            "decay.working_half_life_minutes",
            "ltm.vector_dimension",
            "llm.request_timeout_secs",
        ] {
            assert!(err.contains(needle), "missing '{needle}' in: {err}");
        }
    }

    /// ARC-23: an embedder that cannot embed is a config error, not a runtime
    /// surprise on the first query.
    #[test]
    fn test_validate_rejects_unusable_embedders() {
        let mut cfg = valid();
        cfg.llm.provider = LlmProvider::Anthropic;
        cfg.llm.embedding_provider = None; // falls back to anthropic
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("anthropic")
        );

        let mut cfg = valid();
        cfg.llm.embedding_provider = Some(LlmProvider::Vertex);
        cfg.llm.embedding_project = None;
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("embedding_project")
        );
    }
}
