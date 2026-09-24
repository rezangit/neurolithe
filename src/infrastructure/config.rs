use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Name of the config file inside the home directory.
pub const CONFIG_FILE_NAME: &str = "neurolithe.toml";
/// Workspace used when nothing selects one.
pub const DEFAULT_WORKSPACE: &str = "default";

/// Top-level runtime configuration for NeuroLithe V2.
///
/// V2 splits memory into two independent SQLite stores — a decaying short-term
/// store (`stm`) and a permanent long-term knowledge tree (`ltm`) — each with
/// its own vector dimension. The remaining sections wire the optional Kafka
/// mode and the background schedulers.
#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    /// The resolved home directory (`--home` → `NEUROLITHE_HOME` →
    /// `~/.neurolithe`). Not a config key: set by [`AppConfig::load`].
    #[serde(skip)]
    pub home: PathBuf,
    /// The selected workspace (`--workspace` → `NEUROLITHE_WORKSPACE` → config
    /// `workspace` → `"default"`). Its stores live under
    /// `<home>/workspaces/<workspace>/`.
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[serde(default)]
    pub mcp: McpConfig,
    #[serde(default)]
    pub log: LogConfig,
    pub llm: LlmConfig,
    #[serde(default)]
    pub stm: StoreConfig,
    #[serde(default)]
    pub ltm: StoreConfig,
    pub kafka: KafkaConfig,
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
    /// Where the `local` embedder caches model files: `<home>/models`. Not a
    /// config key — set by the loader once the home directory is resolved.
    #[serde(skip)]
    pub models_dir: Option<PathBuf>,
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
    /// Used for embeddings.
    Vertex,
    /// Offline embeddings via fastembed-rs/ONNX (`local-embeddings` feature).
    /// Embeddings only — never a chat provider. The default embedder.
    Local,
    /// No chat LLM (the default). Memory is stored and searched, but nothing
    /// that needs a chat model runs: fact extraction, compression and
    /// summaries report "LLM not configured". Not valid as an embedder.
    None,
}

impl LlmProvider {
    /// Whether this provider can serve chat calls (extraction, compression).
    pub fn is_chat(&self) -> bool {
        !matches!(self, LlmProvider::Local | LlmProvider::None)
    }
}

/// Configuration for one of the two SQLite memory stores (STM or LTM).
///
/// There is no dimension setting: both stores take their vector dimension from
/// the configured embedder (recorded in each store's `meta`; a change requires
/// `neurolithe reembed`). A legacy `vector_dimension` key is ignored. There is
/// no path setting: store files always live in the selected workspace
/// directory (`<home>/workspaces/<name>/{stm,ltm}.sqlite`).
#[derive(Debug, Deserialize, Clone, Default)]
pub struct StoreConfig {
    /// `[[ltm.spine]]` — the curated top-level branches of the LTM tree, each
    /// `{path, description}` (`path` is `/`-separated under the root). Empty =
    /// the generic default spine. Only read for `[ltm]`; ignored under `[stm]`.
    #[serde(default)]
    pub spine: Vec<crate::domain::ltm::SpineSeed>,
}

impl StoreConfig {
    /// The configured spine, or the generic default when none is configured.
    pub fn spine_or_default(&self) -> Vec<crate::domain::ltm::SpineSeed> {
        if self.spine.is_empty() {
            crate::domain::ltm::default_spine()
        } else {
            self.spine.clone()
        }
    }
}

fn default_workspace() -> String {
    DEFAULT_WORKSPACE.to_string()
}

/// `[mcp]` — MCP-server behaviour.
#[derive(Debug, Deserialize, Clone)]
pub struct McpConfig {
    /// Whether the `workspace_switch` tool may change the active workspace
    /// mid-session. Off pins each MCP-server entry to its startup workspace.
    #[serde(default = "default_true")]
    pub allow_workspace_switch: bool,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            allow_workspace_switch: true,
        }
    }
}

fn default_true() -> bool {
    true
}

/// `[log]` — diagnostics. Logs always go to stderr (stdout is the MCP
/// transport). `RUST_LOG`, when set, overrides `level`.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct LogConfig {
    /// Filter directive, e.g. `"info"`, `"debug"`, `"neurolithe=debug,warn"`.
    /// Unset = `info`.
    #[serde(default)]
    pub level: Option<String>,
}

/// Where to load from: the CLI's `--home`, `--config` and `--workspace` (each
/// optional; `None` falls through to env, then the default).
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    pub home: Option<PathBuf>,
    pub config: Option<PathBuf>,
    pub workspace: Option<String>,
}

/// A non-empty environment value.
fn non_empty(lookup: &dyn Fn(&str) -> Option<String>, name: &str) -> Option<String> {
    lookup(name).filter(|v| !v.trim().is_empty())
}

/// Make `p` absolute without touching the filesystem. Only an explicitly
/// relative `--home`/`--config` is resolved against the CWD; nothing is ever
/// *searched for* there.
fn absolute(p: PathBuf) -> PathBuf {
    std::path::absolute(&p).unwrap_or(p)
}

/// The home directory: `--home` → `NEUROLITHE_HOME` → `$HOME/.neurolithe`
/// (`%USERPROFILE%\.neurolithe` on Windows). The CWD is never a fallback.
pub fn resolve_home(
    cli: Option<&Path>,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> anyhow::Result<PathBuf> {
    if let Some(p) = cli {
        return Ok(absolute(p.to_path_buf()));
    }
    if let Some(p) = non_empty(lookup, "NEUROLITHE_HOME") {
        return Ok(absolute(PathBuf::from(p)));
    }
    let user_home = non_empty(lookup, "HOME").or_else(|| non_empty(lookup, "USERPROFILE"));
    match user_home {
        Some(h) => Ok(PathBuf::from(h).join(".neurolithe")),
        None => anyhow::bail!(
            "cannot determine the NeuroLithe home directory: neither HOME nor USERPROFILE \
             is set; pass --home <dir> or set NEUROLITHE_HOME"
        ),
    }
}

/// The config file: `--config` → `NEUROLITHE_CONFIG` → `<home>/neurolithe.toml`.
pub fn resolve_config_file(
    cli: Option<&Path>,
    lookup: &dyn Fn(&str) -> Option<String>,
    home: &Path,
) -> PathBuf {
    if let Some(p) = cli {
        return absolute(p.to_path_buf());
    }
    if let Some(p) = non_empty(lookup, "NEUROLITHE_CONFIG") {
        return absolute(PathBuf::from(p));
    }
    home.join(CONFIG_FILE_NAME)
}

/// The workspace: `--workspace` → `NEUROLITHE_WORKSPACE` → the config file's
/// `workspace` → `"default"`. (Name validity is checked by the workspace layer.)
pub fn resolve_workspace(
    cli: Option<&str>,
    lookup: &dyn Fn(&str) -> Option<String>,
    from_config: Option<&str>,
) -> String {
    cli.map(str::to_string)
        .or_else(|| non_empty(lookup, "NEUROLITHE_WORKSPACE"))
        .or_else(|| from_config.map(str::to_string))
        .unwrap_or_else(default_workspace)
}

/// Create `dir` (and parents) if missing; a directory this process creates gets
/// mode 0700 on unix (it holds memory stores and secrets).
pub fn ensure_private_dir(dir: &Path) -> std::io::Result<()> {
    if dir.is_dir() {
        return Ok(());
    }
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir)
}

/// The `KEY=value` pairs of `<home>/.env` (empty if absent). The ONLY `.env`
/// NeuroLithe reads — never the CWD's or a parent's (2.1).
pub fn dotenv_pairs(home: &Path) -> anyhow::Result<Vec<(String, String)>> {
    let path = home.join(".env");
    if !path.is_file() {
        return Ok(Vec::new());
    }
    dotenvy::from_path_iter(&path)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?
        .map(|item| item.map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display())))
        .collect()
}

/// `[kafka]` — the optional Kafka mode (built with `--features kafka`): the
/// document feeder, the command and query consumers, and the metrics publisher.
/// Topics are not auto-created; create them on the broker first.
///
/// `Debug` is hand-written: it redacts secret-looking `[kafka.client]` values.
#[derive(Deserialize, Clone)]
pub struct KafkaConfig {
    pub brokers: String,
    /// Base consumer group. The command and query consumers use
    /// `<group_id>-cmd` and `<group_id>-query`.
    pub group_id: String,
    #[serde(default)]
    pub topics: KafkaTopics,
    /// `[kafka.client]` — extra librdkafka properties applied to **every**
    /// Kafka client (consumers and producers), e.g. for SASL/TLS:
    /// `"security.protocol" = "SASL_SSL"`, `"sasl.mechanism" = "SCRAM-SHA-512"`,
    /// `"sasl.username"`, `"sasl.password"`, `"ssl.ca.location"`. Values here
    /// cannot override `bootstrap.servers`, `group.id` or the offset settings
    /// the clients rely on.
    #[serde(default)]
    pub client: std::collections::BTreeMap<String, String>,
}

/// Whether a `[kafka.client]` key names a secret (its value must never be
/// printed): the key contains `password`, `secret`, `key` or `token`,
/// case-insensitively. Deliberately broad — over-redacting is harmless.
pub fn is_secret_client_key(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    ["password", "secret", "key", "token"]
        .iter()
        .any(|word| k.contains(word))
}

impl std::fmt::Debug for KafkaConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let client: std::collections::BTreeMap<&str, &str> = self
            .client
            .iter()
            .map(|(k, v)| {
                let shown = if is_secret_client_key(k) {
                    "<redacted>"
                } else {
                    v.as_str()
                };
                (k.as_str(), shown)
            })
            .collect();
        f.debug_struct("KafkaConfig")
            .field("brokers", &self.brokers)
            .field("group_id", &self.group_id)
            .field("topics", &self.topics)
            .field("client", &client)
            .finish()
    }
}

/// `[kafka.topics]` — topic names. Defaults keep the historical names.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(default)]
pub struct KafkaTopics {
    /// Document events to ingest (`{data_id, title?, text, tags?, ts?}`).
    pub documents: String,
    /// Write/reset commands (remember, forget, reset_soft, reset_hard).
    pub commands: String,
    /// Recall requests.
    pub queries: String,
    /// Replies to `queries`, keyed by correlation id.
    pub results: String,
    /// Periodic store metrics snapshot (compacted, single key).
    pub metrics: String,
    /// Dead letters: valid but failed/invalid messages.
    pub dlq: String,
    /// Un-parseable messages.
    pub parking: String,
}

impl Default for KafkaTopics {
    fn default() -> Self {
        Self {
            documents: "document.completed".into(),
            commands: "memory.command".into(),
            queries: "memory.query".into(),
            results: "memory.result".into(),
            metrics: "memory.metrics".into(),
            dlq: "dlq.memory".into(),
            parking: "parking.lot".into(),
        }
    }
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

/// How often the metrics snapshot is published to the metrics topic.
#[derive(Debug, Deserialize, Clone)]
pub struct MetricsConfig {
    pub interval_secs: u64,
}

/// Whether the Kafka feeder (documents topic → dual-write) is active.
/// Lets the daemon run reset/introspection without consuming documents.
#[derive(Debug, Deserialize, Clone)]
pub struct FeederConfig {
    pub enabled: bool,
}

/// Whether the Kafka query → result request/reply loop is active — the bus
/// door that lets other services read STM + LTM. Independent of
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
    /// Resolve home, apply `<home>/.env` to the process environment (variables
    /// already set win), then load `<config file>` + `NEUROLITHE__*` env. The
    /// CWD is never consulted. Creates `<home>` (0700) on first run.
    pub fn load(opts: &LoadOptions) -> anyhow::Result<Self> {
        let process_env = |k: &str| std::env::var(k).ok();
        let home = resolve_home(opts.home.as_deref(), &process_env)?;
        ensure_private_dir(&home)
            .map_err(|e| anyhow::anyhow!("creating home {}: {e}", home.display()))?;
        for (key, value) in dotenv_pairs(&home)? {
            if std::env::var_os(&key).is_none() {
                // SAFETY: called once at startup, before the runtime or any
                // other thread exists.
                unsafe { std::env::set_var(&key, value) };
            }
        }
        let opts = LoadOptions {
            home: Some(home),
            ..opts.clone()
        };
        Self::load_with(&opts, &process_env, None)
    }

    /// [`load`](Self::load) without touching the process: `lookup` supplies the
    /// plain env vars (`NEUROLITHE_HOME`, `NEUROLITHE_CONFIG`,
    /// `NEUROLITHE_WORKSPACE`, `HOME`) and `env` the `NEUROLITHE__*` overrides
    /// (`None` = the process environment). Does not read `.env`.
    pub fn load_with(
        opts: &LoadOptions,
        lookup: &dyn Fn(&str) -> Option<String>,
        env: Option<HashMap<String, String>>,
    ) -> anyhow::Result<Self> {
        let home = resolve_home(opts.home.as_deref(), lookup)?;
        let config_file = resolve_config_file(opts.config.as_deref(), lookup, &home);
        // A missing file is fine (defaults + env apply) — unless the caller
        // named it explicitly, in which case a typo must not be ignored.
        let explicit = opts.config.is_some() || non_empty(lookup, "NEUROLITHE_CONFIG").is_some();
        if explicit && !config_file.is_file() {
            anyhow::bail!("config file not found: {}", config_file.display());
        }
        let mut config = Self::build(config_file.is_file().then_some(&config_file), env)?;
        config.workspace = resolve_workspace(
            opts.workspace.as_deref(),
            lookup,
            Some(config.workspace.as_str()),
        );
        // The local embedder caches its model files under the home dir.
        config.llm.models_dir = Some(home.join("models"));
        config.home = home;
        config.validate()?;
        Ok(config)
    }

    /// The directory holding all workspaces: `<home>/workspaces`.
    pub fn workspaces_dir(&self) -> PathBuf {
        self.home.join("workspaces")
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
        if self.llm.request_timeout_secs == 0 {
            problems.push("llm.request_timeout_secs must be > 0".into());
        }
        if let Some(level) = &self.log.level
            && let Err(e) = crate::infrastructure::logging::parse_filter(level)
        {
            problems.push(format!("log.level: {e}"));
        }
        match self.llm.provider {
            LlmProvider::Local => problems.push(
                "llm.provider = 'local' only computes embeddings; set llm.provider to \
                 'none' (no chat LLM) or a chat provider (openai, anthropic, gemini, vertex, custom)"
                    .into(),
            ),
            LlmProvider::None => {}
            _ if self.llm.model.trim().is_empty() => {
                problems.push("llm.model must not be empty when a chat provider is set".into())
            }
            _ => {}
        }
        if self.llm.embedding_model.trim().is_empty() {
            problems.push("llm.embedding_model must not be empty".into());
        }
        match self.llm.effective_embedding_provider() {
            LlmProvider::Anthropic => problems.push(
                "llm.embedding_provider resolves to 'anthropic', which has no embeddings API; \
                 set llm.embedding_provider to local, openai, gemini, vertex, or custom"
                    .into(),
            ),
            LlmProvider::None => problems.push(
                "llm.embedding_provider must not be 'none': memory needs an embedder; \
                 use 'local' (offline, no API key) or a remote provider"
                    .into(),
            ),
            LlmProvider::Local if !cfg!(feature = "local-embeddings") => problems.push(
                "llm.embedding_provider = 'local' but this binary was built without the \
                 'local-embeddings' feature; rebuild with default features or pick a remote embedder"
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

    /// Build the config from defaults → optional config file → env vars.
    ///
    /// `env` replaces the process environment as the override source when
    /// given — tests inject variables this way instead of mutating global
    /// process state with `set_var`, which races other parallel tests (QA-14).
    fn build(file: Option<&PathBuf>, env: Option<HashMap<String, String>>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder()
            // Zero-key defaults: no chat LLM, offline local embeddings.
            .set_default("llm.provider", "none")?
            .set_default("llm.model", "")?
            .set_default("llm.embedding_provider", "local")?
            .set_default("llm.embedding_model", "bge-small-en-v1.5")?
            .set_default("kafka.brokers", "localhost:9092")?
            .set_default("kafka.group_id", "neurolithe")?
            // Frequent decay sweep (real-elapsed decay makes this safe); short
            // enough that cold `working` notes expire within minutes.
            .set_default("sweep.interval_secs", 300_i64)?
            .set_default("metrics.interval_secs", 60_i64)?
            .set_default("feeder.enabled", true)?
            .set_default("bus_query.enabled", true)?
            // Per-layer decay: durable facts 7-day, working notes 30-minute.
            .set_default("decay.default_half_life_days", 7.0)?
            .set_default("decay.working_half_life_minutes", 30.0)?;

        if let Some(path) = file {
            builder = builder
                .add_source(config::File::from(path.as_path()).format(config::FileFormat::Toml));
        }

        // Environment variables override file config (e.g. NEUROLITHE__LLM__PROVIDER=gemini)
        builder = builder.add_source(
            config::Environment::with_prefix("NEUROLITHE")
                .separator("__")
                .source(env.map(|m| m.into_iter().collect())),
        );

        let config = builder.build().map_err(|e| match file {
            Some(p) => anyhow::anyhow!("loading config {}: {e}", p.display()),
            None => anyhow::anyhow!("loading config: {e}"),
        })?;
        // Store paths were removed in 0.3 (stores live in the workspace dir);
        // say so instead of silently ignoring a stale setting.
        for key in ["stm.path", "ltm.path"] {
            if config.get_string(key).is_ok() {
                tracing::warn!(
                    "`{key}` is no longer supported and is ignored; \
                     stores live in <home>/workspaces/<name>/ (see `neurolithe workspace import`)"
                );
            }
        }
        // Store dimensions were removed in 0.3: both stores take theirs from
        // the embedder (recorded in store meta). Warn, don't fail.
        for key in ["stm.vector_dimension", "ltm.vector_dimension"] {
            if config.get::<i64>(key).is_ok() {
                tracing::warn!(
                    "`{key}` is no longer supported and is ignored; the vector dimension \
                     comes from the embedder (llm.embedding_model), and a model change \
                     needs `neurolithe reembed`"
                );
            }
        }
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

    /// Config parses with the defaults (zero-key: no chat LLM, local 384-d
    /// embeddings; Kafka/scheduler sections) and the defaults validate.
    #[test]
    fn test_defaults() {
        let cfg = AppConfig::build(None, no_env()).expect("config should load from defaults");

        assert_eq!(cfg.llm.provider, LlmProvider::None);
        assert_eq!(cfg.llm.embedding_provider, Some(LlmProvider::Local));
        assert_eq!(cfg.llm.embedding_model, "bge-small-en-v1.5");
        assert_eq!(cfg.workspace, DEFAULT_WORKSPACE);
        assert!(cfg.mcp.allow_workspace_switch);
        assert_eq!(cfg.kafka.brokers, "localhost:9092");
        assert_eq!(cfg.kafka.group_id, "neurolithe");
        assert_eq!(cfg.kafka.topics, KafkaTopics::default());
        assert!(cfg.kafka.client.is_empty());
        assert_eq!(
            cfg.ltm.spine_or_default(),
            crate::domain::ltm::default_spine()
        );
        assert_eq!(cfg.sweep.interval_secs, 300);
        assert_eq!(cfg.metrics.interval_secs, 60);
        assert_eq!(cfg.llm.request_timeout_secs, 120);
        assert!(cfg.feeder.enabled);
        assert!(cfg.bus_query.enabled);
        // Per-layer decay defaults.
        assert_eq!(cfg.decay.default_half_life_days, 7.0);
        assert_eq!(cfg.decay.working_half_life_minutes, 30.0);
        assert!((cfg.decay.working_half_life_days() - 30.0 / 1440.0).abs() < 1e-12);
        cfg.validate().expect("defaults must validate");
    }

    /// Env vars override single keys. The overrides are injected (no
    /// `set_var`), so this cannot race other tests (QA-14).
    #[test]
    fn test_env_override_is_injected_not_global() {
        let env: HashMap<String, String> = [
            ("NEUROLITHE__LLM__EMBEDDING_MODEL", "bge-base-en-v1.5"),
            ("NEUROLITHE__FEEDER__ENABLED", "false"),
            ("NEUROLITHE__DECAY__WORKING_HALF_LIFE_MINUTES", "5"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

        let cfg = AppConfig::build(None, Some(env)).expect("config should load with overrides");

        assert_eq!(cfg.llm.embedding_model, "bge-base-en-v1.5");
        assert!(!cfg.feeder.enabled);
        assert_eq!(cfg.decay.working_half_life_minutes, 5.0);
        // The process environment was never touched.
        assert!(std::env::var("NEUROLITHE__LLM__EMBEDDING_MODEL").is_err());
    }

    fn valid() -> AppConfig {
        AppConfig::build(None, no_env()).unwrap()
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
        cfg.llm.request_timeout_secs = 0;
        let err = cfg.validate().unwrap_err().to_string();
        for needle in [
            "sweep.interval_secs",
            "metrics.interval_secs",
            "decay.default_half_life_days",
            "decay.working_half_life_minutes",
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

        let mut cfg = valid();
        cfg.llm.embedding_provider = Some(LlmProvider::None);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("must not be 'none'"), "{err}");
    }

    /// The chat LLM is optional: `none` needs no model; `local` is not a chat
    /// provider; a real chat provider needs a model name.
    #[test]
    fn test_validate_chat_provider_rules() {
        let cfg = valid();
        assert_eq!(cfg.llm.provider, LlmProvider::None);
        assert!(cfg.llm.model.is_empty());
        cfg.validate().expect("no chat LLM is a valid setup");

        let mut cfg = valid();
        cfg.llm.provider = LlmProvider::Local;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("only computes embeddings"), "{err}");

        let mut cfg = valid();
        cfg.llm.provider = LlmProvider::Openai;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("llm.model must not be empty"), "{err}");
        cfg.llm.model = "gpt-4o-mini".into();
        cfg.validate().expect("openai chat + local embeddings");
    }

    fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    /// 2.1: home = `--home` → `NEUROLITHE_HOME` → `$HOME/.neurolithe` →
    /// `%USERPROFILE%\.neurolithe`; never the CWD.
    #[test]
    fn test_resolve_home_precedence() {
        let env = lookup(&[("NEUROLITHE_HOME", "/env/home"), ("HOME", "/users/me")]);
        assert_eq!(
            resolve_home(Some(Path::new("/cli/home")), &env).unwrap(),
            PathBuf::from("/cli/home")
        );
        assert_eq!(
            resolve_home(None, &env).unwrap(),
            PathBuf::from("/env/home")
        );

        let env = lookup(&[("HOME", "/users/me")]);
        assert_eq!(
            resolve_home(None, &env).unwrap(),
            PathBuf::from("/users/me/.neurolithe")
        );
        let env = lookup(&[("USERPROFILE", "/win/me")]);
        assert_eq!(
            resolve_home(None, &env).unwrap(),
            PathBuf::from("/win/me/.neurolithe")
        );
        // Nothing to go on: an error, not the CWD.
        assert!(resolve_home(None, &lookup(&[])).is_err());
        // A blank NEUROLITHE_HOME is ignored.
        let env = lookup(&[("NEUROLITHE_HOME", " "), ("HOME", "/users/me")]);
        assert_eq!(
            resolve_home(None, &env).unwrap(),
            PathBuf::from("/users/me/.neurolithe")
        );
    }

    /// 2.1: config = `--config` → `NEUROLITHE_CONFIG` → `<home>/neurolithe.toml`.
    #[test]
    fn test_resolve_config_file_precedence() {
        let home = Path::new("/h");
        let env = lookup(&[("NEUROLITHE_CONFIG", "/env/c.toml")]);
        assert_eq!(
            resolve_config_file(Some(Path::new("/cli/c.toml")), &env, home),
            PathBuf::from("/cli/c.toml")
        );
        assert_eq!(
            resolve_config_file(None, &env, home),
            PathBuf::from("/env/c.toml")
        );
        assert_eq!(
            resolve_config_file(None, &lookup(&[]), home),
            PathBuf::from("/h/neurolithe.toml")
        );
    }

    /// 2.2: workspace = `--workspace` → `NEUROLITHE_WORKSPACE` → config → default.
    #[test]
    fn test_resolve_workspace_precedence() {
        let env = lookup(&[("NEUROLITHE_WORKSPACE", "envws")]);
        assert_eq!(resolve_workspace(Some("cli"), &env, Some("cfg")), "cli");
        assert_eq!(resolve_workspace(None, &env, Some("cfg")), "envws");
        assert_eq!(resolve_workspace(None, &lookup(&[]), Some("cfg")), "cfg");
        assert_eq!(
            resolve_workspace(None, &lookup(&[]), None),
            DEFAULT_WORKSPACE
        );
    }

    /// 2.1: the config file is read from the home dir (not the CWD), a missing
    /// home config is fine, and the workspace resolves through the file.
    #[test]
    fn test_load_with_reads_config_from_home_only() {
        let home = tempfile::tempdir().unwrap();
        let opts = LoadOptions {
            home: Some(home.path().to_path_buf()),
            ..Default::default()
        };
        // No file: defaults (whatever neurolithe.toml the CWD may hold is ignored).
        let cfg = AppConfig::load_with(&opts, &lookup(&[]), no_env()).unwrap();
        assert_eq!(cfg.sweep.interval_secs, 300);
        assert_eq!(cfg.home, home.path());
        assert_eq!(cfg.workspaces_dir(), home.path().join("workspaces"));
        // The local embedder's model cache lives under home, never the CWD.
        assert_eq!(cfg.llm.models_dir, Some(home.path().join("models")));

        std::fs::write(
            home.path().join(CONFIG_FILE_NAME),
            "workspace = \"research\"\n[sweep]\ninterval_secs = 42\n[mcp]\nallow_workspace_switch = false\n",
        )
        .unwrap();
        let cfg = AppConfig::load_with(&opts, &lookup(&[]), no_env()).unwrap();
        assert_eq!(cfg.sweep.interval_secs, 42);
        assert_eq!(cfg.workspace, "research");
        assert!(!cfg.mcp.allow_workspace_switch);

        // CLI workspace beats the file.
        let cli = LoadOptions {
            workspace: Some("cli-ws".into()),
            ..opts.clone()
        };
        let cfg = AppConfig::load_with(&cli, &lookup(&[]), no_env()).unwrap();
        assert_eq!(cfg.workspace, "cli-ws");
    }

    /// An explicitly named config file that doesn't exist is an error (a typo
    /// must not silently fall back to defaults).
    #[test]
    fn test_explicit_missing_config_is_error() {
        let home = tempfile::tempdir().unwrap();
        let opts = LoadOptions {
            home: Some(home.path().to_path_buf()),
            config: Some(home.path().join("nope.toml")),
            ..Default::default()
        };
        let err = AppConfig::load_with(&opts, &lookup(&[]), no_env()).unwrap_err();
        assert!(err.to_string().contains("nope.toml"), "{err}");
    }

    /// 2.2/2.3: legacy `[stm].path` and `vector_dimension` no longer drive
    /// anything (ignored), and the config still loads.
    #[test]
    fn test_legacy_store_paths_ignored() {
        let home = tempfile::tempdir().unwrap();
        std::fs::write(
            home.path().join(CONFIG_FILE_NAME),
            "[stm]\npath = \"/somewhere/stm.sqlite\"\nvector_dimension = 8\n",
        )
        .unwrap();
        let opts = LoadOptions {
            home: Some(home.path().to_path_buf()),
            ..Default::default()
        };
        let cfg = AppConfig::load_with(&opts, &lookup(&[]), no_env()).unwrap();
        assert!(cfg.stm.spine.is_empty());

        // A stale dimension override from the environment is ignored too.
        let env: HashMap<String, String> = [(
            "NEUROLITHE__LTM__VECTOR_DIMENSION".to_string(),
            "1024".to_string(),
        )]
        .into();
        AppConfig::build(None, Some(env)).expect("stale env key must not fail the load");
    }

    /// 2.1: `.env` comes from `<home>/.env` only.
    #[test]
    fn test_dotenv_pairs_from_home() {
        let home = tempfile::tempdir().unwrap();
        assert!(dotenv_pairs(home.path()).unwrap().is_empty());
        std::fs::write(
            home.path().join(".env"),
            "OPENAI_API_KEY=sk-test\n# c\nX=1\n",
        )
        .unwrap();
        let pairs = dotenv_pairs(home.path()).unwrap();
        assert!(pairs.contains(&("OPENAI_API_KEY".into(), "sk-test".into())));
        assert!(pairs.contains(&("X".into(), "1".into())));
    }

    /// 2.1: directories we create are private (0700).
    #[cfg(unix)]
    #[test]
    fn test_ensure_private_dir_mode() {
        use std::os::unix::fs::PermissionsExt;
        let base = tempfile::tempdir().unwrap();
        let dir = base.path().join("a/b");
        ensure_private_dir(&dir).unwrap();
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
        ensure_private_dir(&dir).unwrap(); // idempotent
    }

    /// §6: `[kafka.topics]` overrides only the named topics (the rest keep
    /// their defaults); `[kafka.client]` keeps dotted librdkafka keys intact;
    /// `[[ltm.spine]]` replaces the default spine.
    #[test]
    fn test_kafka_topics_client_and_ltm_spine_from_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("neurolithe.toml");
        std::fs::write(
            &path,
            r#"
[kafka]
brokers = "broker:9093"
group_id = "nl"

[kafka.topics]
documents = "docs.in"

[kafka.client]
"security.protocol" = "SASL_SSL"
"sasl.mechanism" = "SCRAM-SHA-512"

[[ltm.spine]]
path = "work/projects"
description = "Project work"
"#,
        )
        .unwrap();
        let cfg = AppConfig::build(Some(&path), no_env()).unwrap();
        assert_eq!(cfg.kafka.topics.documents, "docs.in");
        assert_eq!(cfg.kafka.topics.commands, KafkaTopics::default().commands);
        assert_eq!(
            cfg.kafka
                .client
                .get("security.protocol")
                .map(String::as_str),
            Some("SASL_SSL")
        );
        assert_eq!(
            cfg.kafka.client.get("sasl.mechanism").map(String::as_str),
            Some("SCRAM-SHA-512")
        );
        assert_eq!(
            cfg.ltm.spine_or_default(),
            vec![crate::domain::ltm::SpineSeed::new(
                "work/projects",
                "Project work"
            )]
        );
        assert!(cfg.stm.spine.is_empty());
    }

    /// §6: a Kafka client secret can come from the environment
    /// (`NEUROLITHE__KAFKA__CLIENT__SASL_PASSWORD`); the key arrives underscored
    /// and the Kafka layer maps it to `sasl.password`.
    #[test]
    fn test_kafka_client_secret_from_env() {
        let env: HashMap<String, String> = [(
            "NEUROLITHE__KAFKA__CLIENT__SASL_PASSWORD".to_string(),
            "pw".to_string(),
        )]
        .into_iter()
        .collect();
        let cfg = AppConfig::build(None, Some(env)).unwrap();
        assert_eq!(
            cfg.kafka.client.get("sasl_password").map(String::as_str),
            Some("pw")
        );
    }

    /// P2R-12: `Debug` of the Kafka config never prints secret-looking
    /// `[kafka.client]` values (password/secret/key/token in the key name).
    #[test]
    fn test_kafka_config_debug_redacts_secrets() {
        let kafka = KafkaConfig {
            brokers: "b:9092".into(),
            group_id: "g".into(),
            topics: KafkaTopics::default(),
            client: [
                ("sasl.password", "hunter2-pw"),
                ("SASL_PASSWORD", "hunter3-pw"),
                ("sasl.oauthbearer.client.secret", "s3cr3t"),
                ("ssl.key.password", "kpw"),
                ("sasl.oauthbearer.token.endpoint.url", "https://idp/token"),
                ("sasl.username", "alice"),
                ("security.protocol", "SASL_SSL"),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        };
        let shown = format!("{kafka:?}");
        for secret in [
            "hunter2-pw",
            "hunter3-pw",
            "s3cr3t",
            "kpw",
            "https://idp/token",
        ] {
            assert!(!shown.contains(secret), "leaked {secret}: {shown}");
        }
        assert!(
            shown.contains("alice") && shown.contains("SASL_SSL"),
            "{shown}"
        );
        assert!(shown.contains("<redacted>"), "{shown}");
        // AppConfig's derived Debug goes through the same impl.
        let mut cfg = AppConfig::build(None, no_env()).unwrap();
        cfg.kafka = kafka;
        assert!(!format!("{cfg:?}").contains("hunter2-pw"));
    }
}
