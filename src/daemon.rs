//! Daemon / composition roots.
//!
//! Two entry points share the same store + service wiring:
//! - [`run_mcp`] — the **standalone** MCP server over stdio (default build): just
//!   the embedded stores + recall + CT-scan tools. No Kafka, no network broker.
//! - [`run`] (feature `kafka`) — the full **daemon**: MCP + the Kafka
//!   feeder, `memory.command` consumer, `memory.query` door, and decay/metrics
//!   schedulers, all on a single-threaded `LocalSet` (the SQLite-backed services
//!   are `!Sync`; the loops are `!Send` and run cooperatively).

use crate::application::app::NeurolitheApp;
use crate::application::documents::DocumentService;
use crate::application::introspection::IntrospectionService;
use crate::application::ltm_retrieval::LtmRetrieval;
use crate::application::query_service::QueryService;
use crate::application::retrieval::RetrievalService;
use crate::application::scheduler::{PeriodicTask, SweepTask, run_periodic};
use crate::application::workspace::{
    WorkspaceHost, WorkspaceInfo, WorkspaceManager, WorkspaceServices, validate_name,
};
use crate::domain::ltm::LtmRepository;
use crate::domain::ports::{EmbeddingIdentity, LlmClient, MemoryRepository, embedding_identity};
use crate::domain::thresholds::Thresholds;
use crate::infrastructure::config::AppConfig;
use crate::infrastructure::database::{
    LTM_FILE, MemoryStores, STM_FILE, delete_workspace_dir, export_workspace, list_workspace_dirs,
    open_workspace_stores, store_bytes,
};
use crate::infrastructure::llm::create_llm_client;
use crate::infrastructure::ltm_repository::SqliteLtmRepository;
use crate::infrastructure::repository::SqliteMemoryRepository;
use crate::interfaces::mcp_server::McpServer;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

/// Build the LLM client from config + the process environment, logging any
/// startup warnings (e.g. a missing API key). A missing key does not abort
/// startup: LLM-backed tools then fail with "LLM not configured: set …" while
/// introspection keeps working (no `dummy_key` is ever sent — QA-10).
pub fn build_llm(config: &AppConfig) -> Arc<dyn LlmClient> {
    let setup = create_llm_client(&config.llm, &|name| std::env::var(name).ok());
    for warning in &setup.warnings {
        tracing::warn!("{warning}");
    }
    setup.client
}

/// Opened repositories for one workspace. Each repository carries the
/// workspace lease (the shared lock that keeps other processes from
/// deleting/re-embedding it), so the workspace stays marked "open" for exactly
/// as long as anything — a service, an in-flight sweep, the daemon — still
/// holds one of its connections.
pub struct WorkspaceRepos {
    pub stm: Arc<dyn MemoryRepository>,
    pub ltm: Arc<dyn LtmRepository>,
    /// The process's effective distance thresholds (resolved once).
    pub thresholds: Thresholds,
}

/// Log the notes (migration backups, warnings) from opening a workspace.
fn log_open_notes(notes: &[String]) {
    for note in notes {
        tracing::info!("{note}");
    }
}

/// The embedder's identity (model id + output dimension), which every store is
/// checked against. Probing may load a local model or call a remote embedder.
pub async fn resolve_identity(llm: &dyn LlmClient) -> Result<EmbeddingIdentity> {
    embedding_identity(llm)
        .await
        .context("determining the embedder's model and dimension (needed to open the stores)")
}

/// Embed a workspace's spine concepts (best-effort; logs and continues).
async fn embed_spine_now(
    name: &str,
    ltm: &Arc<dyn LtmRepository>,
    llm: &Arc<dyn LlmClient>,
    dim: usize,
) {
    match crate::application::ltm_placement::embed_spine_concepts(ltm, llm, dim).await {
        Ok(0) => {}
        Ok(n) => tracing::info!("embedded {n} spine concept(s) in workspace {name:?}"),
        Err(e) => {
            tracing::warn!("spine embedding for {name:?} skipped (retried next open): {e:#}")
        }
    }
}

/// Workspace storage over the SQLite layout `<home>/workspaces/<name>/`, and
/// the per-workspace service wiring. The composition root's implementation of
/// the [`WorkspaceHost`] port.
pub struct SqliteWorkspaceHost {
    config: AppConfig,
    llm: Arc<dyn LlmClient>,
    /// Resolved once, on first open.
    identity: tokio::sync::OnceCell<EmbeddingIdentity>,
    /// Resolved once from the identity + config (logged when resolved).
    thresholds: tokio::sync::OnceCell<Thresholds>,
}

impl SqliteWorkspaceHost {
    pub fn new(config: AppConfig, llm: Arc<dyn LlmClient>) -> Self {
        Self {
            config,
            llm,
            identity: tokio::sync::OnceCell::new(),
            thresholds: tokio::sync::OnceCell::new(),
        }
    }

    fn dir(&self, name: &str) -> PathBuf {
        self.config.workspaces_dir().join(name)
    }

    /// The embedder identity (cached after the first probe).
    pub async fn identity(&self) -> Result<EmbeddingIdentity> {
        self.identity
            .get_or_try_init(|| resolve_identity(self.llm.as_ref()))
            .await
            .cloned()
    }

    /// The effective distance thresholds for this process: config values,
    /// else the embedding model's defaults, else the generic fallback (which
    /// logs one warning suggesting calibration). Resolved and logged once.
    pub async fn thresholds(&self) -> Result<Thresholds> {
        self.thresholds
            .get_or_try_init(|| async {
                let identity = self.identity().await?;
                let (thresholds, warning) =
                    Thresholds::resolve(&identity.model, self.config.threshold_overrides())?;
                if let Some(warning) = warning {
                    tracing::warn!("{warning}");
                }
                tracing::info!(
                    "distance thresholds for {}: placement {} ({:?}), assimilation {} ({:?}), accommodation {} ({:?})",
                    thresholds.embedding_model,
                    thresholds.placement_max_distance.value,
                    thresholds.placement_max_distance.source,
                    thresholds.assimilation.value,
                    thresholds.assimilation.source,
                    thresholds.accommodation.value,
                    thresholds.accommodation.source,
                );
                Ok::<_, anyhow::Error>(thresholds)
            })
            .await
            .cloned()
    }

    /// Open (creating on demand) a workspace's stores — migrating them and
    /// checking the embedder identity — and seed its spine from config.
    pub async fn open_repos(&self, name: &str) -> Result<WorkspaceRepos> {
        validate_name(name)?;
        let identity = self.identity().await?;
        let thresholds = self.thresholds().await?;
        let (MemoryStores { stm, ltm, lease }, notes) =
            open_workspace_stores(&self.dir(name), &identity)?;
        log_open_notes(&notes);
        let lease: Arc<dyn std::any::Any + Send + Sync> = Arc::new(lease);
        let stm: Arc<dyn MemoryRepository> =
            Arc::new(SqliteMemoryRepository::new(stm).with_lease(lease.clone()));
        let ltm: Arc<dyn LtmRepository> = Arc::new(SqliteLtmRepository::new(ltm).with_lease(lease));
        ltm.seed_spine_from(&self.config.ltm.spine_or_default())?;
        Ok(WorkspaceRepos {
            stm,
            ltm,
            thresholds,
        })
    }

    /// Give the spine placement vectors so documents file under a concept
    /// instead of the inbox, and recall has concepts to land on. Best-effort:
    /// an embedder outage must not block opening (it's retried next open).
    pub async fn embed_spine(&self, name: &str, repos: &WorkspaceRepos) {
        match self.identity().await {
            Ok(identity) => embed_spine_now(name, &repos.ltm, &self.llm, identity.dim).await,
            Err(e) => tracing::warn!("spine embedding skipped: {e:#}"),
        }
    }

    /// The MCP-facing services over one workspace's repositories.
    pub fn services(&self, repos: &WorkspaceRepos) -> WorkspaceServices {
        let app = Arc::new(NeurolitheApp::new(
            repos.stm.clone(),
            self.llm.clone(),
            self.config.decay.default_half_life_days,
            self.config.decay.working_half_life_days(),
            &repos.thresholds,
        ));
        self.services_with_app(repos, app)
    }

    /// Like [`services`](Self::services) but sharing an existing app (the
    /// daemon's sweep task and MCP use the same one).
    pub fn services_with_app(
        &self,
        repos: &WorkspaceRepos,
        app: Arc<NeurolitheApp>,
    ) -> WorkspaceServices {
        WorkspaceServices {
            app,
            introspection: Arc::new(IntrospectionService::new(
                repos.stm.clone(),
                repos.ltm.clone(),
                repos.thresholds.clone(),
            )),
            // Same recall service as the bus door: STM hybrid search +
            // reference-returning LTM recall.
            query: QueryService::new(
                RetrievalService::new(self.llm.clone(), repos.stm.clone()),
                LtmRetrieval::new(repos.ltm.clone()),
                self.llm.clone(),
            ),
            documents: DocumentService::new(repos.ltm.clone(), self.llm.clone(), &repos.thresholds),
        }
    }
}

#[async_trait::async_trait(?Send)]
impl WorkspaceHost for SqliteWorkspaceHost {
    async fn open(&self, name: &str) -> Result<WorkspaceServices> {
        let repos = self.open_repos(name).await?;
        // Per-workspace startup work, so it also runs on a switch. In MCP mode
        // this runs inside the concurrent startup (see `serve_mcp`), so it
        // delays tool calls, never `initialize`/`tools/list`.
        self.embed_spine(name, &repos).await;
        Ok(self.services(&repos))
    }

    fn exists(&self, name: &str) -> bool {
        self.dir(name).is_dir()
    }

    fn list(&self) -> Result<Vec<WorkspaceInfo>> {
        Ok(
            list_workspace_dirs(&self.config.workspaces_dir(), |n| validate_name(n).is_ok())?
                .into_iter()
                .map(|(name, dir)| WorkspaceInfo {
                    path: dir.display().to_string(),
                    stm_bytes: store_bytes(&dir.join(STM_FILE)),
                    ltm_bytes: store_bytes(&dir.join(LTM_FILE)),
                    active: false,
                    name,
                })
                .collect(),
        )
    }

    fn create(&self, name: &str) -> Result<bool> {
        if self.exists(name) {
            return Ok(false);
        }
        // Just the directory: the stores are created (at the embedder's
        // dimension) on first open.
        create_workspace_dir(&self.dir(name))?;
        Ok(true)
    }

    fn delete(&self, name: &str) -> Result<()> {
        // Refused while the workspace is open anywhere (another process, or
        // this daemon's own workspace after its MCP session switched away).
        delete_workspace_dir(&self.dir(name), name)
    }

    fn export(&self, name: &str) -> Result<serde_json::Value> {
        export_workspace(&self.dir(name), name)
    }
}

/// Run ONLY the MCP server over stdio — no feeder or consumers.
///
/// The standalone mode (`neurolithe mcp`), and also how a transient client
/// session (e.g. Claude via `docker exec -i`) shares the live stores with a
/// running daemon: reads (recall + CT-scan) run concurrently under WAL; the
/// occasional write waits on `busy_timeout`. Serves the selected workspace
/// (created on demand) until stdin closes (the normal shutdown) or SIGINT/
/// SIGTERM arrives; either way the stores' WAL is then checkpointed.
pub async fn run_mcp(config: AppConfig) -> Result<()> {
    let llm = build_llm(&config);
    serve_mcp(
        config,
        llm,
        tokio::io::BufReader::new(tokio::io::stdin()),
        tokio::io::stdout(),
    )
    .await
}

/// The MCP server loop plus its workspace startup and background work.
///
/// The server starts answering at once: `initialize`, `tools/list` and `ping`
/// never wait on the embedder probe, store migrations, or model loading (MCP
/// clients time out slow starters). Workspace startup runs concurrently; tool
/// calls wait for it. If startup fails (e.g. the stores were embedded by a
/// different model), the error is returned and the process exits non-zero.
/// Once started, the decay sweep runs alongside.
pub async fn serve_mcp<R, W>(
    config: AppConfig,
    llm: Arc<dyn LlmClient>,
    reader: R,
    writer: W,
) -> Result<()>
where
    R: tokio::io::AsyncBufRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    // Cheap checks fail fast, before serving (a bad name must exit non-zero
    // even if the client disconnects immediately).
    validate_name(&config.workspace)?;
    let server = McpServer::pending();
    let host = SqliteWorkspaceHost::new(config.clone(), llm);
    let startup = async {
        let manager = WorkspaceManager::start(
            Box::new(host),
            &config.workspace,
            config.mcp.allow_workspace_switch,
        )
        .await?;
        Ok::<_, anyhow::Error>(Rc::new(manager))
    };

    let serve = server.serve(reader, writer);
    tokio::pin!(serve);
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);
    let workspaces = tokio::select! {
        // stdin closed before startup finished: nothing left to do.
        served = &mut serve => return served,
        signal = &mut shutdown => {
            tracing::info!("received {signal} during startup; shutting down");
            return Ok(());
        }
        started = startup => started?,
    };
    server.set_workspaces(workspaces.clone());

    // Decay sweep + command-ledger pruning, on whichever workspace is active
    // at each tick (so it follows `workspace_switch`).
    let sweep: Arc<dyn PeriodicTask> = Arc::new(ActiveWorkspaceSweep {
        workspaces: workspaces.clone(),
    });
    let sweeping = run_periodic(
        sweep,
        std::time::Duration::from_secs(config.sweep.interval_secs),
    );
    tracing::info!(
        "serving MCP over stdio (home: {}, workspace: {})",
        config.home.display(),
        config.workspace
    );
    let served = tokio::select! {
        served = serve => {
            tracing::info!("stdin closed; shutting down");
            served
        }
        signal = shutdown => {
            tracing::info!("received {signal}; shutting down");
            Ok(())
        }
        () = sweeping => Ok(()),
    };
    // The loops above are dropped (cancelled) here, so the store connections
    // are idle: fold the WAL back into the active workspace's stores.
    checkpoint_on_shutdown(&config, &[workspaces.current_name()]);
    served
}

/// Resolve on SIGINT (Ctrl-C) or, on Unix, SIGTERM (`docker stop`, a
/// supervisor, an MCP client ending the session). Returns the signal's name.
pub async fn shutdown_signal() -> &'static str {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => "SIGINT",
                    _ = term.recv() => "SIGTERM",
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                "SIGINT"
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        "Ctrl-C"
    }
}

/// Final shutdown step: checkpoint + truncate the WAL of each named workspace
/// (deduplicated), so the stopped stores are self-contained `.sqlite` files.
fn checkpoint_on_shutdown(config: &AppConfig, workspaces: &[String]) {
    let mut done: Vec<&String> = Vec::new();
    for name in workspaces {
        if done.contains(&name) {
            continue;
        }
        done.push(name);
        let dir = config.workspaces_dir().join(name);
        if crate::infrastructure::database::checkpoint_workspace(&dir) {
            tracing::debug!("workspace {name:?}: stores checkpointed");
        }
    }
}

/// The STM maintenance sweep applied to the *active* workspace each tick.
struct ActiveWorkspaceSweep {
    workspaces: Rc<WorkspaceManager>,
}

#[async_trait::async_trait(?Send)]
impl PeriodicTask for ActiveWorkspaceSweep {
    async fn run_once(&self) {
        let app = self.workspaces.current().services.app.clone();
        SweepTask::new(app).run_once().await;
    }
}

/// Create an (empty) workspace directory with private permissions.
pub fn create_workspace_dir(dir: &Path) -> Result<()> {
    crate::infrastructure::config::ensure_private_dir(dir)
        .with_context(|| format!("creating workspace dir {}", dir.display()))
}

/// The full Kafka daemon (feeder + consumers + schedulers). Only built
/// with the `kafka` feature; the standalone MCP build omits it (and rdkafka).
#[cfg(feature = "kafka")]
pub use full::run;

#[cfg(feature = "kafka")]
mod full {
    use super::*;
    use crate::application::ingestion::IngestionService;
    use crate::application::monitoring::{FeederStats, MonitoringService};
    use crate::application::reset_service::ResetService;
    use crate::application::workspace::OpenWorkspace;
    use crate::application::write_service::WriteService;
    use crate::domain::models::WORKSPACE_TENANT as FEEDER_TENANT;
    use crate::infrastructure::metrics_publisher::MetricsPublisher;
    use crate::interfaces::command_consumer::{
        CommandConsumer, ConsumerRewind, FeederRewind, NoopRewind, NoopSpineEmbedder, SpineEmbedder,
    };
    use crate::interfaces::kafka_feeder::KafkaFeeder;
    use crate::interfaces::query_consumer::QueryConsumer;
    use async_trait::async_trait;
    use std::time::Duration;

    /// Periodic metrics snapshot -> `memory.metrics`. A composition-root task
    /// tying the monitoring snapshot to the publisher + live feeder stats.
    struct MetricsTask {
        monitoring: MonitoringService,
        publisher: Arc<MetricsPublisher>,
        stats: Arc<FeederStats>,
    }

    #[async_trait(?Send)]
    impl PeriodicTask for MetricsTask {
        async fn run_once(&self) {
            // Session count is wired when the SessionManager exposes it; 0 for now.
            let runtime = self.stats.to_runtime(0);
            match self.monitoring.snapshot(&runtime) {
                Ok(metrics) => {
                    if let Err(e) = self.publisher.publish(&metrics).await {
                        tracing::warn!("metrics publish failed: {e}");
                    }
                }
                Err(e) => tracing::warn!("metrics snapshot failed: {e}"),
            }
        }
    }

    /// Re-embeds the curated spine after a hard reset (bridges the interface-layer
    /// `SpineEmbedder` port to the application + infra it needs).
    struct DaemonSpineEmbedder {
        ltm: Arc<dyn LtmRepository>,
        llm: Arc<dyn LlmClient>,
        dim: usize,
    }

    #[async_trait(?Send)]
    impl SpineEmbedder for DaemonSpineEmbedder {
        async fn embed_spine(&self) -> Result<()> {
            let n = crate::application::ltm_placement::embed_spine_concepts(
                &self.ltm, &self.llm, self.dim,
            )
            .await?;
            if n > 0 {
                tracing::info!("re-embedded {n} spine concept(s) after hard reset");
            }
            Ok(())
        }
    }

    /// How long the Kafka loops get to finish their in-flight message, commit
    /// and flush after a shutdown signal. Keep below the supervisor's grace
    /// period (Compose `stop_grace_period`; Docker's default is 10s).
    const DRAIN_TIMEOUT: Duration = Duration::from_secs(8);

    /// Assemble and run the daemon until SIGINT/SIGTERM, then shut down
    /// gracefully: stop the MCP session and schedulers, let each Kafka loop
    /// finish its in-flight message and commit offsets synchronously, flush the
    /// producers, and checkpoint the stores' WAL.
    pub async fn run(config: AppConfig) -> Result<()> {
        // --- external adapters ---
        let llm = build_llm(&config);

        // --- stores: the selected workspace (created on demand). Both stores
        // use the embedder's dimension, so the feeder's one embedding fits both.
        let host = SqliteWorkspaceHost::new(config.clone(), llm.clone());
        let dim = host.identity().await?.dim;
        let repos = host.open_repos(&config.workspace).await?;
        let stm_repo = repos.stm.clone();
        let ltm_repo = repos.ltm.clone();
        host.embed_spine(&config.workspace, &repos).await;

        // --- services ---
        let app = Arc::new(NeurolitheApp::new(
            stm_repo.clone(),
            llm.clone(),
            config.decay.default_half_life_days,
            config.decay.working_half_life_days(),
            &repos.thresholds,
        ));
        let ingestion = Arc::new(IngestionService::new(
            stm_repo.clone(),
            ltm_repo.clone(),
            llm.clone(),
            dim,
            FEEDER_TENANT,
            &repos.thresholds,
        ));

        // Inbox gardener: re-home inbox documents that now match a concept using
        // stored embeddings — no replay, no LLM. Runs once at startup; idempotent.
        if config.feeder.enabled {
            match ingestion.garden_inbox() {
                Ok(0) => {}
                Ok(n) => tracing::info!("inbox gardener re-homed {n} document(s) under concepts"),
                Err(e) => tracing::warn!("inbox gardener failed: {e}"),
            }
        }

        // Hard reset is disabled unless NEUROLITHE_RESET_TOKEN is ≥16 chars.
        let reset = Arc::new(
            ResetService::from_env(stm_repo.clone(), ltm_repo.clone())
                .with_spine(config.ltm.spine_or_default()),
        );
        let write = Arc::new(WriteService::new(
            stm_repo.clone(),
            ltm_repo.clone(),
            ingestion.clone(),
            llm.clone(),
            dim,
            &repos.thresholds,
        ));
        let feeder_stats = Arc::new(FeederStats::default());

        // --- feeder (optional) + rewind handle for hard reset ---
        let feeder = if config.feeder.enabled {
            Some(Arc::new(KafkaFeeder::new(
                &config.kafka,
                ingestion.clone(),
                feeder_stats.clone(),
            )?))
        } else {
            None
        };
        let rewind: Arc<dyn FeederRewind> = match &feeder {
            Some(f) => Arc::new(ConsumerRewind::new(f.consumer())),
            None => Arc::new(NoopRewind),
        };

        // After a hard reset the spine is re-seeded but un-embedded — re-embed it
        // so the replay files documents under concepts, not the inbox.
        let spine_embedder: Arc<dyn SpineEmbedder> = if config.feeder.enabled {
            Arc::new(DaemonSpineEmbedder {
                ltm: ltm_repo.clone(),
                llm: llm.clone(),
                dim,
            })
        } else {
            Arc::new(NoopSpineEmbedder)
        };

        // --- command consumer (reset), distinct group id ---
        let command = Arc::new(CommandConsumer::new(
            &config.kafka,
            reset,
            write,
            rewind,
            spine_embedder,
        )?);

        // --- query consumer (memory.query -> memory.result), distinct group id ---
        let query = if config.bus_query.enabled {
            let query_service = Arc::new(QueryService::new(
                RetrievalService::new(llm.clone(), stm_repo.clone()),
                LtmRetrieval::new(ltm_repo.clone()),
                llm.clone(),
            ));
            Some(Arc::new(QueryConsumer::new(&config.kafka, query_service)?))
        } else {
            None
        };

        // --- schedulers ---
        let sweep_task: Arc<dyn PeriodicTask> = Arc::new(SweepTask::new(app.clone()));
        let metrics_publisher = Arc::new(MetricsPublisher::new(&config.kafka)?);
        let metrics_task: Arc<dyn PeriodicTask> = Arc::new(MetricsTask {
            monitoring: MonitoringService::new(stm_repo.clone(), ltm_repo.clone()),
            publisher: metrics_publisher.clone(),
            stats: feeder_stats.clone(),
        });
        let sweep_interval = Duration::from_secs(config.sweep.interval_secs);
        let metrics_interval = Duration::from_secs(config.metrics.interval_secs);

        // --- MCP server (stdio) over the daemon's workspace, sharing its stores
        // and app. A `workspace_switch` only moves this MCP session; the Kafka
        // loops stay on the startup workspace.
        let services = host.services_with_app(&repos, app.clone());
        let workspaces = Rc::new(WorkspaceManager::with_active(
            Box::new(host),
            OpenWorkspace {
                name: config.workspace.clone(),
                services,
            },
            config.mcp.allow_workspace_switch,
        ));
        let server = Arc::new(McpServer::new(workspaces.clone()));

        // --- spawn all loops on the LocalSet ---
        let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
        let mcp_task = {
            let s = server.clone();
            tokio::task::spawn_local(async move {
                if let Err(e) = s.run_stdio().await {
                    tracing::error!("MCP stdio server stopped: {e}");
                }
            })
        };
        let mut kafka_loops = Vec::new();
        if let Some(f) = feeder.clone() {
            let stop = stop_rx.clone();
            kafka_loops.push(tokio::task::spawn_local(async move {
                if let Err(e) = f.run(stop).await {
                    tracing::error!("feeder stopped: {e}");
                }
            }));
        }
        {
            let c = command.clone();
            let stop = stop_rx.clone();
            kafka_loops.push(tokio::task::spawn_local(async move {
                if let Err(e) = c.run(stop).await {
                    tracing::error!("command consumer stopped: {e}");
                }
            }));
        }
        if let Some(q) = query.clone() {
            let stop = stop_rx.clone();
            kafka_loops.push(tokio::task::spawn_local(async move {
                if let Err(e) = q.run(stop).await {
                    tracing::error!("query consumer stopped: {e}");
                }
            }));
        }
        let schedulers = [
            tokio::task::spawn_local(run_periodic(sweep_task, sweep_interval)),
            tokio::task::spawn_local(run_periodic(metrics_task, metrics_interval)),
        ];

        tracing::info!(
            "daemon running (workspace: {}, feeder: {}, query door: {})",
            config.workspace,
            config.feeder.enabled,
            config.bus_query.enabled
        );
        let signal = shutdown_signal().await;
        tracing::info!("received {signal}; shutting down");

        // 1. No new work: end the MCP session and the schedulers.
        let mcp_workspace = workspaces.current_name();
        for task in std::iter::once(mcp_task).chain(schedulers) {
            task.abort();
            let _ = task.await;
        }
        // 2. Kafka loops finish their in-flight message, commit, flush.
        let _ = stop_tx.send(true);
        let deadline = tokio::time::Instant::now() + DRAIN_TIMEOUT;
        for task in kafka_loops {
            if tokio::time::timeout_at(deadline, task).await.is_err() {
                tracing::warn!("a Kafka loop did not stop within {DRAIN_TIMEOUT:?}; abandoning it");
            }
        }
        // 3. Deliver the last metrics snapshot.
        metrics_publisher.flush();
        // 4. Fold the WAL into the stores (the daemon's workspace, plus the
        // one the MCP session had switched to, if different).
        checkpoint_on_shutdown(&config, &[config.workspace.clone(), mcp_workspace]);
        tracing::info!("shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::models::CclDefinition;
    use crate::domain::ports::ExtractedFact;
    use crate::infrastructure::config::LoadOptions;
    use crate::infrastructure::database::init_db;

    /// Offline 4-d embedder.
    struct StubLlm;

    #[async_trait::async_trait]
    impl LlmClient for StubLlm {
        async fn extract_facts(
            &self,
            _d: &str,
            _c: &[CclDefinition],
        ) -> Result<Vec<ExtractedFact>> {
            Ok(Vec::new())
        }
        async fn generate_ccl_description(&self, _n: &str, _c: &str) -> Result<String> {
            Ok(String::new())
        }
        async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            let l = text.len() as f32;
            Ok(vec![1.0, l.sin(), l.cos(), 0.5])
        }
        async fn compress_context(&self, m: &str) -> Result<String> {
            Ok(m.to_string())
        }
    }

    fn config_in(home: &Path) -> AppConfig {
        let opts = LoadOptions {
            home: Some(home.to_path_buf()),
            ..Default::default()
        };
        AppConfig::load_with(&opts, &|_| None, Some(Default::default())).unwrap()
    }

    async fn start(home: &Path) -> (Rc<WorkspaceManager>, AppConfig) {
        let config = config_in(home);
        let host = SqliteWorkspaceHost::new(config.clone(), Arc::new(StubLlm));
        let manager = WorkspaceManager::start(Box::new(host), &config.workspace, true)
            .await
            .unwrap();
        (Rc::new(manager), config)
    }

    /// Thresholds are resolved once from config + the embedder's model and
    /// reach the services: a configured placement distance is reported (as
    /// `config`) by `placement_debug`, and the unset values come from the
    /// fallback (the stub embedder's model id is unknown).
    #[tokio::test]
    async fn test_thresholds_resolved_from_config_reach_services() {
        use crate::domain::thresholds::{FALLBACK, ThresholdSource};
        let home = tempfile::tempdir().unwrap();
        let mut config = config_in(home.path());
        config.ltm.placement_max_distance = Some(0.42);
        let host = SqliteWorkspaceHost::new(config.clone(), Arc::new(StubLlm));

        let repos = host.open_repos("default").await.unwrap();
        let t = &repos.thresholds;
        assert_eq!(t.placement_max_distance.value, 0.42);
        assert_eq!(t.placement_max_distance.source, ThresholdSource::Config);
        assert_eq!(t.assimilation.value, FALLBACK.assimilation);
        assert_eq!(t.assimilation.source, ThresholdSource::Fallback);
        assert_eq!(host.thresholds().await.unwrap(), *t, "resolved once");

        let services = host.services(&repos);
        let debug = services.introspection.placement_debug(5).unwrap();
        assert_eq!(debug.thresholds, *t);
        let json = serde_json::to_value(&debug).unwrap();
        assert_eq!(
            json["thresholds"]["placement_max_distance"]["source"],
            "config"
        );
        assert_eq!(json["thresholds"]["assimilation"]["source"], "fallback");
        assert!(json["probes"].is_array());
    }

    /// An invalid combination (config assimilation above the model/fallback
    /// accommodation) fails at startup with a clear message.
    #[tokio::test]
    async fn test_invalid_resolved_thresholds_fail_at_open() {
        let home = tempfile::tempdir().unwrap();
        let mut config = config_in(home.path());
        config.stm.assimilation_threshold = Some(5.0);
        let host = SqliteWorkspaceHost::new(config, Arc::new(StubLlm));
        let err = match host.open_repos("default").await {
            Ok(_) => panic!("must fail"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("must be smaller"), "{err}");
    }

    fn scalar(path: &Path, sql: &str) -> i64 {
        init_db(Some(&path))
            .unwrap()
            .query_row(sql, [], |r| r.get(0))
            .unwrap()
    }

    /// 2.5: in MCP mode the spine is embedded when a workspace is opened —
    /// at startup and again for a workspace reached by `workspace_switch` —
    /// so placement and recall have concept vectors (it used to happen only
    /// in the Kafka daemon).
    #[tokio::test]
    async fn test_spine_embedded_on_open_and_switch() {
        let home = tempfile::tempdir().unwrap();
        let (ws, config) = start(home.path()).await;
        let ltm_of = |name: &str| config.workspaces_dir().join(name).join(LTM_FILE);

        let concepts = |name: &str| {
            scalar(
                &ltm_of(name),
                "SELECT COUNT(*) FROM tree_nodes WHERE kind IN ('spine','grown')",
            )
        };
        let vectors = |name: &str| scalar(&ltm_of(name), "SELECT COUNT(*) FROM vec_ltm");
        assert!(concepts("default") > 0);
        assert_eq!(
            vectors("default"),
            concepts("default"),
            "every concept embedded"
        );

        ws.create("work").unwrap();
        ws.switch("work").await.unwrap();
        assert!(vectors("work") > 0);
        assert_eq!(vectors("work"), concepts("work"));
    }

    /// 2.5: the MCP-mode maintenance sweep runs on whichever workspace is
    /// active — after a switch the old workspace is no longer swept and the
    /// new one is.
    #[tokio::test]
    async fn test_maintenance_sweep_follows_active_workspace() {
        let home = tempfile::tempdir().unwrap();
        let (ws, config) = start(home.path()).await;
        ws.create("work").unwrap();
        let work_stm = config.workspaces_dir().join("work").join(STM_FILE);
        // Materialize "work" (switch opens it), then return to "default".
        ws.switch("work").await.unwrap();
        ws.switch("default").await.unwrap();

        // A long-untouched fact in "work": any sweep of "work" archives it.
        init_db(Some(&work_stm))
            .unwrap()
            .execute(
                "INSERT INTO nodes (tenant_id, payload, relevance_score, last_accessed_at, last_decayed_at)
                 VALUES ('default', '{\"fact\":\"stale\"}', 0.15,
                         datetime('now','-100 days'), datetime('now','-100 days'))",
                [],
            )
            .unwrap();
        let archived = || {
            scalar(
                &work_stm,
                "SELECT COUNT(*) FROM nodes WHERE status = 'archived'",
            )
        };

        let sweep = ActiveWorkspaceSweep {
            workspaces: ws.clone(),
        };
        sweep.run_once().await; // active = default
        assert_eq!(archived(), 0, "an inactive workspace is not swept");

        ws.switch("work").await.unwrap();
        sweep.run_once().await; // active = work
        assert_eq!(archived(), 1, "the sweep followed the switch");
    }

    /// A bad workspace name fails before serving, even if stdin closes at once.
    #[tokio::test]
    async fn test_invalid_workspace_fails_before_serving() {
        let home = tempfile::tempdir().unwrap();
        let mut config = config_in(home.path());
        config.workspace = "../evil".into();
        let empty: &[u8] = b"";
        let err = serve_mcp(
            config,
            Arc::new(StubLlm),
            tokio::io::BufReader::new(empty),
            tokio::io::sink(),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("invalid workspace name"), "{err}");
    }

    /// An embedder that takes `delay` per call (a cold local model, or a slow
    /// remote API).
    struct SlowLlm {
        delay: std::time::Duration,
    }

    #[async_trait::async_trait]
    impl LlmClient for SlowLlm {
        async fn extract_facts(
            &self,
            _d: &str,
            _c: &[CclDefinition],
        ) -> Result<Vec<ExtractedFact>> {
            Ok(Vec::new())
        }
        async fn generate_ccl_description(&self, _n: &str, _c: &str) -> Result<String> {
            Ok(String::new())
        }
        async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            tokio::time::sleep(self.delay).await;
            StubLlm.embed_text(text).await
        }
        async fn compress_context(&self, m: &str) -> Result<String> {
            Ok(m.to_string())
        }
    }

    /// First-run UX: `initialize` and `tools/list` are answered promptly even
    /// while the embedder is slow (startup probes it and embeds the spine),
    /// and a tool call made during startup still succeeds once it's ready.
    #[tokio::test]
    async fn test_initialize_does_not_wait_for_slow_embedder() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::time::{Duration, timeout};

        let home = tempfile::tempdir().unwrap();
        let config = config_in(home.path());
        let llm: Arc<dyn LlmClient> = Arc::new(SlowLlm {
            delay: Duration::from_secs(2),
        });
        let (client, server_io) = tokio::io::duplex(1 << 16);
        let (srv_r, srv_w) = tokio::io::split(server_io);
        let (cli_r, mut cli_w) = tokio::io::split(client);
        let mut replies = BufReader::new(cli_r).lines();

        let server = serve_mcp(config, llm, BufReader::new(srv_r), srv_w);
        let client = async {
            let prompt = Duration::from_millis(500);
            cli_w
                .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\"}}\n")
                .await
                .unwrap();
            let line = timeout(prompt, replies.next_line())
                .await
                .expect("initialize must not wait on the embedder")
                .unwrap()
                .unwrap();
            let v: serde_json::Value = serde_json::from_str(&line).unwrap();
            assert_eq!(v["result"]["serverInfo"]["name"], "NeuroLithe", "{v}");

            cli_w
                .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}\n")
                .await
                .unwrap();
            let line = timeout(prompt, replies.next_line())
                .await
                .expect("tools/list must not wait on the embedder")
                .unwrap()
                .unwrap();
            assert!(line.contains("workspace_current"), "{line}");

            // A tool call waits for startup, then works.
            cli_w
                .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"workspace_current\",\"arguments\":{}}}\n")
                .await
                .unwrap();
            let line = timeout(Duration::from_secs(30), replies.next_line())
                .await
                .expect("tool call completes after startup")
                .unwrap()
                .unwrap();
            assert!(line.contains("\\\"name\\\":\\\"default\\\""), "{line}");
        };
        tokio::select! {
            r = server => panic!("server ended early: {r:?}"),
            () = client => {}
        }
    }

    /// P2R-3: a daemon keeps its own workspace open (leased) for its whole run,
    /// so its MCP session can't delete it after switching away — nor can any
    /// other process — while a workspace nobody has open can be deleted.
    #[tokio::test]
    async fn test_daemon_workspace_cannot_be_deleted_via_switched_session() {
        let home = tempfile::tempdir().unwrap();
        let config = config_in(home.path());
        // The daemon's own open of "default" (kept for the whole run).
        let daemon_host = SqliteWorkspaceHost::new(config.clone(), Arc::new(StubLlm));
        let daemon_repos = daemon_host.open_repos("default").await.unwrap();

        // Its MCP session starts on "default", then switches away.
        let (ws, _config) = start(home.path()).await;
        ws.create("work").unwrap();
        ws.switch("work").await.unwrap();
        let err = ws
            .delete("default", Some("default"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("in use"), "{err}");
        assert!(config.workspaces_dir().join("default").exists());

        // The session's own lease on "default" was released by the switch, so
        // once the daemon lets go, the delete goes through.
        drop(daemon_repos);
        ws.delete("default", Some("default")).unwrap();
        assert!(!config.workspaces_dir().join("default").exists());
    }

    /// N3: the lease lives exactly as long as the connections. Something still
    /// holding the old workspace's app after a switch (an in-flight sweep,
    /// say) keeps it marked open, so it can't be deleted under that holder;
    /// once the last holder drops, it can.
    #[tokio::test]
    async fn test_lease_lives_as_long_as_any_store_holder() {
        let home = tempfile::tempdir().unwrap();
        let (ws, config) = start(home.path()).await;
        let in_flight = ws.current().services.app.clone(); // e.g. a sweep mid-run
        ws.create("work").unwrap();
        ws.switch("work").await.unwrap();

        let err = ws
            .delete("default", Some("default"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("in use"), "{err}");
        assert!(config.workspaces_dir().join("default").exists());

        drop(in_flight);
        ws.delete("default", Some("default")).unwrap();
    }
}
