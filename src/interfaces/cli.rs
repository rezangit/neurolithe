//! Command-line interface.
//!
//! ```text
//! neurolithe [--home DIR] [--config FILE] [--workspace NAME] <COMMAND>
//!   mcp                                   serve MCP over stdio (standalone default)
//!   daemon                                Kafka daemon (builds with --features kafka)
//!   init                                  create the home dir + config, print an MCP snippet
//!   workspace list
//!   workspace create <name>
//!   workspace delete <name> --yes
//!   workspace export <name> [--out FILE]
//!   workspace backup <name> [--out DIR]
//!   workspace import <name> --stm FILE --ltm FILE
//!   reembed                               re-embed the selected workspace with the current embedder
//! ```
//! The global flags may appear before or after the subcommand.

use crate::application::workspace::validate_name;
use crate::domain::ports::EmbeddingIdentity;
use crate::infrastructure::config::{
    AppConfig, CONFIG_FILE_NAME, LoadOptions, ensure_private_dir, resolve_config_file, resolve_home,
};
use crate::infrastructure::database::{
    LTM_FILE, STM_FILE, backup_workspace, delete_workspace_dir, export_workspace,
    import_store_files, list_workspace_dirs, lock_workspace_exclusive, migrate_workspace_stores,
    open_workspace_stores, quick_check, store_bytes, stored_embedding_model,
};
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::{Path, PathBuf};

/// The commented default config `init` writes (the repo's example file).
const EXAMPLE_CONFIG: &str = include_str!("../../neurolithe.example.toml");

#[derive(Debug, Parser)]
#[command(
    name = "neurolithe",
    version,
    about = "Embedded contextual memory for AI agents, served over MCP."
)]
pub struct Cli {
    /// Home directory holding config, .env and workspaces
    /// [default: $NEUROLITHE_HOME, else ~/.neurolithe]
    #[arg(long, global = true, value_name = "DIR")]
    pub home: Option<PathBuf>,

    /// Config file [default: $NEUROLITHE_CONFIG, else <home>/neurolithe.toml]
    #[arg(long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Workspace to use [default: $NEUROLITHE_WORKSPACE, else config
    /// `workspace`, else "default"]
    #[arg(long, global = true, value_name = "NAME")]
    pub workspace: Option<String>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Serve the MCP tools over stdio (the standalone mode)
    Mcp,
    /// Run the Kafka daemon: MCP + feeder + bus memory API + schedulers
    #[cfg(feature = "kafka")]
    Daemon,
    /// Create the home directory and a default config, fetch the local
    /// embedding model, and print an MCP client snippet
    Init {
        /// Don't download the local embedding model now (it is fetched on
        /// first use instead)
        #[arg(long)]
        no_model_download: bool,
    },
    /// Manage workspaces (separate memories)
    #[command(subcommand)]
    Workspace(WorkspaceCommand),
    /// Re-embed the selected workspace (--workspace) with the configured
    /// embedder, after changing embedding model. Backs up both stores first.
    Reembed,
}

#[derive(Debug, Subcommand)]
pub enum WorkspaceCommand {
    /// List workspaces and their store sizes
    List,
    /// Create an empty workspace
    Create { name: String },
    /// Permanently delete a workspace and all of its memory
    Delete {
        name: String,
        /// Confirm the deletion (required)
        #[arg(long)]
        yes: bool,
    },
    /// Write a JSON dump of a workspace's STM facts and LTM leaves
    Export {
        name: String,
        /// Output file [default: stdout]
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
    },
    /// Back up a workspace's stores (timestamped VACUUM INTO copies)
    Backup {
        name: String,
        /// Output directory [default: <home>/backups]
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
    },
    /// Import existing store files into a new workspace (then migrate them).
    /// The source stores must not be in use: stop any process (daemon, MCP
    /// server) using them first — a changing source is refused.
    Import {
        name: String,
        /// Legacy STM store file
        #[arg(long, value_name = "FILE")]
        stm: PathBuf,
        /// Legacy LTM store file
        #[arg(long, value_name = "FILE")]
        ltm: PathBuf,
    },
}

impl Cli {
    pub fn load_options(&self) -> LoadOptions {
        LoadOptions {
            home: self.home.clone(),
            config: self.config.clone(),
            workspace: self.workspace.clone(),
        }
    }
}

/// Parse the process arguments and run. Returns the process exit code.
pub fn main_entry() -> Result<i32> {
    let cli = Cli::parse();
    // Before the config loads, so its warnings are logged (stderr only).
    crate::infrastructure::logging::init();
    let opts = cli.load_options();
    match cli.command {
        Some(Command::Mcp) => {
            let config = load_config(&opts)?;
            run_local(crate::daemon::run_mcp(config))?;
            Ok(0)
        }
        #[cfg(feature = "kafka")]
        Some(Command::Daemon) => run_daemon(&opts),
        Some(Command::Init { no_model_download }) => {
            let exe = std::env::current_exe()
                .and_then(|p| p.canonicalize())
                .context("locating the neurolithe binary")?;
            let config = init(
                &opts,
                &|k| std::env::var(k).ok(),
                &exe,
                &mut std::io::stdout(),
            )?;
            // Pay the one-time model download now, not on the first MCP start.
            if !no_model_download {
                prefetch_local_model(&config)?;
            }
            Ok(0)
        }
        Some(Command::Workspace(WorkspaceCommand::Import { name, stm, ltm })) => {
            let config = load_config(&opts)?;
            let identity = run_local(async {
                let llm = crate::daemon::build_llm(&config);
                crate::daemon::resolve_identity(llm.as_ref()).await
            })?;
            import_workspace(
                &config,
                &name,
                &stm,
                &ltm,
                &identity,
                &mut std::io::stdout(),
            )?;
            Ok(0)
        }
        Some(Command::Workspace(cmd)) => {
            let config = load_config(&opts)?;
            workspace_command(&config, cmd, &mut std::io::stdout())?;
            Ok(0)
        }
        Some(Command::Reembed) => {
            let config = load_config(&opts)?;
            run_local(async {
                let llm = crate::daemon::build_llm(&config);
                reembed_command(&config, llm, &mut std::io::stdout()).await
            })?;
            Ok(0)
        }
        None => no_command(&opts),
    }
}

/// With the `kafka` feature, a bare `neurolithe` starts the daemon (the
/// historical Docker entry point).
#[cfg(feature = "kafka")]
fn no_command(opts: &LoadOptions) -> Result<i32> {
    run_daemon(opts)
}

#[cfg(not(feature = "kafka"))]
fn no_command(_opts: &LoadOptions) -> Result<i32> {
    use clap::CommandFactory;
    eprintln!(
        "This is a standalone NeuroLithe build (no Kafka daemon).\n\
         Start the MCP server with:  neurolithe mcp\n"
    );
    let _ = Cli::command().print_help();
    Ok(2)
}

#[cfg(feature = "kafka")]
fn run_daemon(opts: &LoadOptions) -> Result<i32> {
    let config = load_config(opts)?;
    run_local(crate::daemon::run(config))?;
    Ok(0)
}

/// Load the config and apply its `[log] level` (unless `RUST_LOG` is set).
fn load_config(opts: &LoadOptions) -> Result<AppConfig> {
    let config = AppConfig::load(opts)?;
    crate::infrastructure::logging::apply_config_level(config.log.level.as_deref());
    Ok(config)
}

/// Drive a future on a single-threaded runtime + `LocalSet` (the SQLite-backed
/// services are `!Sync`; the loops are `!Send` and run cooperatively).
fn run_local<T>(fut: impl std::future::Future<Output = Result<T>>) -> Result<T> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let out = tokio::task::LocalSet::new().block_on(&rt, fut);
    // Don't wait on blocking-pool threads: after a shutdown signal, tokio's
    // stdin reader may still be parked in a blocking read that would otherwise
    // hold up the exit until the client writes or closes stdin.
    rt.shutdown_background();
    out
}

/// Write `contents` to a new or truncated file with mode 0600 (unix).
fn write_private_file(path: &Path, contents: &[u8]) -> Result<()> {
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts
        .open(path)
        .with_context(|| format!("writing {}", path.display()))?;
    // `mode` only applies on create; an overwritten file keeps its old mode
    // unless we reset it (P2R-9).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(contents)?;
    Ok(())
}

/// `neurolithe init`: create `<home>` (0700), `<home>/workspaces`, and the
/// selected workspace; write the commented default config (0600) unless one
/// exists; print an MCP client snippet that uses the absolute binary path.
pub fn init(
    opts: &LoadOptions,
    lookup: &dyn Fn(&str) -> Option<String>,
    exe: &Path,
    out: &mut dyn Write,
) -> Result<AppConfig> {
    let home = resolve_home(opts.home.as_deref(), lookup)?;
    ensure_private_dir(&home).with_context(|| format!("creating {}", home.display()))?;
    let config_file = resolve_config_file(opts.config.as_deref(), lookup, &home);
    if config_file.exists() {
        writeln!(
            out,
            "Config:    {} (kept, already exists)",
            config_file.display()
        )?;
    } else {
        if let Some(parent) = config_file.parent() {
            ensure_private_dir(parent)?;
        }
        write_private_file(&config_file, EXAMPLE_CONFIG.as_bytes())?;
        writeln!(out, "Config:    {} (created)", config_file.display())?;
    }

    // Load what we just wrote (validates it) to resolve the workspace.
    let resolved = LoadOptions {
        home: Some(home.clone()),
        config: Some(config_file.clone()),
        workspace: opts.workspace.clone(),
    };
    let config = AppConfig::load_with(&resolved, lookup, None)?;
    validate_name(&config.workspace)?;
    let ws_dir = config.workspaces_dir().join(&config.workspace);
    // The stores themselves are created (at the embedder's dimension) on first
    // `neurolithe mcp`; init doesn't load or call the embedder.
    crate::daemon::create_workspace_dir(&ws_dir)?;
    writeln!(out, "Home:      {}", home.display())?;
    writeln!(
        out,
        "Workspace: {} ({})",
        config.workspace,
        ws_dir.display()
    )?;

    // The MCP client launches the binary without our shell env, so pin every
    // non-default location explicitly.
    let mut args: Vec<String> = vec!["mcp".into()];
    if opts.home.is_some() || lookup("NEUROLITHE_HOME").is_some_and(|v| !v.trim().is_empty()) {
        args.extend(["--home".into(), home.display().to_string()]);
    }
    let default_config = home.join(CONFIG_FILE_NAME);
    if config_file != default_config {
        args.extend(["--config".into(), config_file.display().to_string()]);
    }
    if opts.workspace.is_some() {
        args.extend(["--workspace".into(), config.workspace.clone()]);
    }
    let snippet = serde_json::json!({
        "mcpServers": {
            "neurolithe": {
                "command": exe.display().to_string(),
                "args": args,
            }
        }
    });
    writeln!(
        out,
        "\nAdd this to your MCP client config (API keys go in {}):\n\n{}",
        home.join(".env").display(),
        serde_json::to_string_pretty(&snippet)?
    )?;
    Ok(config)
}

/// `init`'s last step: download/load the local embedding model now, so the
/// first `neurolithe mcp` start doesn't pay the (~100+ MB) download while an
/// MCP client waits. Progress goes to stderr (stdout carries the snippet). A
/// failure (e.g. offline) is a warning, not an init failure: the model is
/// fetched on first use instead.
pub fn prefetch_local_model(config: &AppConfig) -> Result<()> {
    use crate::infrastructure::config::LlmProvider;
    if !matches!(
        config.llm.effective_embedding_provider(),
        LlmProvider::Local
    ) {
        return Ok(());
    }
    #[cfg(feature = "local-embeddings")]
    {
        use crate::infrastructure::local_embed::LocalEmbedder;
        let dir = config
            .llm
            .models_dir
            .clone()
            .context("model cache dir not set")?;
        let embedder = LocalEmbedder::new(&config.llm.embedding_model, dir.clone())?;
        eprintln!(
            "Fetching local embedding model '{}' into {} (downloaded once; may take a minute)…",
            embedder.model_name(),
            dir.display()
        );
        let started = std::time::Instant::now();
        let result = run_local(async {
            let warm = embedder.warm_up();
            tokio::pin!(warm);
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(5));
            tick.tick().await;
            loop {
                tokio::select! {
                    r = &mut warm => break r,
                    _ = tick.tick() => eprintln!(
                        "  … still fetching ({}s)", started.elapsed().as_secs()
                    ),
                }
            }
        });
        match result {
            Ok(()) => eprintln!(
                "Model ready ({:.1}s). `neurolithe mcp` will start without downloading.",
                started.elapsed().as_secs_f64()
            ),
            Err(e) => eprintln!(
                "Warning: could not fetch the embedding model now ({e:#}).\n\
                 It will be downloaded on first use; re-run `neurolithe init` when online."
            ),
        }
    }
    #[cfg(not(feature = "local-embeddings"))]
    eprintln!(
        "Warning: llm.embedding_provider = \"local\" but this build has no local-embeddings \
         feature; pick a remote embedder in the config."
    );
    Ok(())
}

/// Human-readable byte count.
fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = n as f64;
    let mut unit = 0;
    while v >= 1024.0 && unit < UNITS.len() - 1 {
        v /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[unit])
    }
}

/// An existing workspace's directory, or an error naming it.
fn existing_workspace(config: &AppConfig, name: &str) -> Result<PathBuf> {
    validate_name(name)?;
    let dir = config.workspaces_dir().join(name);
    if !dir.is_dir() {
        bail!(
            "workspace {name:?} does not exist (looked in {})",
            dir.display()
        );
    }
    Ok(dir)
}

/// `neurolithe workspace …`.
pub fn workspace_command(
    config: &AppConfig,
    cmd: WorkspaceCommand,
    out: &mut dyn Write,
) -> Result<()> {
    match cmd {
        WorkspaceCommand::List => {
            let all = list_workspace_dirs(&config.workspaces_dir(), |n| validate_name(n).is_ok())?;
            if all.is_empty() {
                writeln!(
                    out,
                    "No workspaces yet in {}",
                    config.workspaces_dir().display()
                )?;
            }
            for (name, dir) in all {
                let marker = if name == config.workspace { "*" } else { " " };
                writeln!(
                    out,
                    "{marker} {name:<24} stm {:>10}   ltm {:>10}",
                    human_bytes(store_bytes(&dir.join(STM_FILE))),
                    human_bytes(store_bytes(&dir.join(LTM_FILE))),
                )?;
            }
        }
        WorkspaceCommand::Create { name } => {
            validate_name(&name)?;
            let dir = config.workspaces_dir().join(&name);
            if dir.exists() {
                bail!("workspace {name:?} already exists");
            }
            // Stores are created at the embedder's dimension on first open.
            crate::daemon::create_workspace_dir(&dir)?;
            writeln!(out, "Created workspace {name:?} at {}", dir.display())?;
        }
        WorkspaceCommand::Delete { name, yes } => {
            let dir = existing_workspace(config, &name)?;
            if !yes {
                bail!(
                    "refusing to delete workspace {name:?} without --yes (this permanently \
                     removes {})",
                    dir.display()
                );
            }
            // Refused while any process has the workspace open (P2R-3).
            delete_workspace_dir(&dir, &name)?;
            writeln!(out, "Deleted workspace {name:?}")?;
        }
        WorkspaceCommand::Export { name, out: file } => {
            let dir = existing_workspace(config, &name)?;
            let dump = serde_json::to_string_pretty(&export_workspace(&dir, &name)?)?;
            match file {
                Some(path) => {
                    write_private_file(&path, dump.as_bytes())?;
                    writeln!(out, "Exported workspace {name:?} to {}", path.display())?;
                }
                None => writeln!(out, "{dump}")?,
            }
        }
        WorkspaceCommand::Backup { name, out: out_dir } => {
            let dir = existing_workspace(config, &name)?;
            let out_dir = out_dir.unwrap_or_else(|| config.home.join("backups"));
            for path in backup_workspace(&dir, &name, &out_dir)? {
                writeln!(out, "Wrote {}", path.display())?;
            }
        }
        WorkspaceCommand::Import { name, .. } => {
            bail!(
                "importing workspace {name:?} needs the embedder; use `neurolithe workspace import`"
            )
        }
    }
    Ok(())
}

/// `neurolithe reembed`: re-embed the selected workspace with `embedder`
/// (backing up both stores first) and record the new identity in its meta.
/// No other process may have the workspace open.
pub async fn reembed_command(
    config: &AppConfig,
    embedder: std::sync::Arc<dyn crate::domain::ports::LlmClient>,
    out: &mut dyn Write,
) -> Result<()> {
    let dir = existing_workspace(config, &config.workspace)?;
    // Hold the workspace exclusively: refused while any process has it open
    // (P2R-3), and nobody can open it mid-reembed.
    let _lock = lock_workspace_exclusive(&dir, &config.workspace)?;
    let report = crate::application::reembed::reembed_workspace_dir(&dir, embedder).await?;
    writeln!(
        out,
        "Re-embedded workspace {:?}: {} STM fact(s), {} LTM concept(s), {} LTM leaf/leaves",
        config.workspace, report.stm_nodes, report.ltm_concepts, report.ltm_leaves
    )?;
    for backup in &report.backups {
        writeln!(out, "Backup: {}", backup.display())?;
    }
    Ok(())
}

/// `neurolithe workspace import`: copy legacy store files into a new workspace
/// (VACUUM INTO) and migrate them. If they were embedded by a different model,
/// say how to fix it (`reembed`) instead of failing the import.
pub fn import_workspace(
    config: &AppConfig,
    name: &str,
    stm: &Path,
    ltm: &Path,
    identity: &EmbeddingIdentity,
    out: &mut dyn Write,
) -> Result<()> {
    validate_name(name)?;
    let dest = config.workspaces_dir().join(name);
    ensure_private_dir(&config.workspaces_dir())?;
    writeln!(
        out,
        "Note: the source stores must not be in use during import; stop any process using them first."
    )?;
    import_store_files(stm, ltm, &dest)?;
    // Which model embedded the legacy stores, if they recorded it.
    let old_model = stored_embedding_model(&dest.join(STM_FILE)).ok().flatten();
    let notes = match migrate_workspace_stores(&dest, identity.dim) {
        Ok((stores, notes)) => {
            drop(stores);
            notes
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&dest);
            return Err(e.context(format!("migrating imported workspace {name:?}")));
        }
    };
    // The copies must be sound before anyone serves them (N2).
    for file in [STM_FILE, LTM_FILE] {
        if let Err(e) = quick_check(&dest.join(file)) {
            let _ = std::fs::remove_dir_all(&dest);
            return Err(e.context(format!(
                "imported workspace {name:?} is damaged; nothing was imported"
            )));
        }
    }
    for note in &notes {
        writeln!(out, "{note}")?;
    }
    writeln!(
        out,
        "Imported {} + {} into workspace {name:?} ({})",
        stm.display(),
        ltm.display(),
        dest.display()
    )?;
    // Open as a server would (records/checks the embedder identity) and show
    // every note from that too — e.g. that a store predating embedding
    // metadata was assumed to come from the current model (P2R-2).
    match open_workspace_stores(&dest, identity) {
        Ok((stores, notes)) => {
            drop(stores);
            for note in &notes {
                writeln!(out, "{note}")?;
            }
            if old_model.as_deref() != Some(identity.model.as_str()) {
                writeln!(
                    out,
                    "Hint: the imported stores were embedded by {}; if that is not '{}', \
                     run: neurolithe reembed --workspace {name}",
                    old_model
                        .as_deref()
                        .map_or("an unrecorded model".to_string(), |m| format!("'{m}'")),
                    identity.model
                )?;
            }
        }
        Err(e) => writeln!(
            out,
            "Note: the imported stores don't match the current embedder ({e:#}).\n\
             Run: neurolithe reembed --workspace {name}"
        )?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("neurolithe").chain(args.iter().copied())).unwrap()
    }

    fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    fn config_in(home: &Path) -> AppConfig {
        let env: HashMap<String, String> = [
            ("NEUROLITHE__STM__VECTOR_DIMENSION", "4"),
            ("NEUROLITHE__LTM__VECTOR_DIMENSION", "4"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let opts = LoadOptions {
            home: Some(home.to_path_buf()),
            ..Default::default()
        };
        AppConfig::load_with(&opts, &|_| None, Some(env)).unwrap()
    }

    fn run(config: &AppConfig, cmd: WorkspaceCommand) -> Result<String> {
        let mut out = Vec::new();
        workspace_command(config, cmd, &mut out)?;
        Ok(String::from_utf8(out).unwrap())
    }

    /// The global flags work before or after the subcommand.
    #[test]
    fn test_parse_global_flags_anywhere() {
        let cli = parse(&["--home", "/h", "mcp", "--workspace", "work"]);
        assert!(matches!(cli.command, Some(Command::Mcp)));
        assert_eq!(cli.home.as_deref(), Some(Path::new("/h")));
        assert_eq!(cli.workspace.as_deref(), Some("work"));

        let cli = parse(&["workspace", "delete", "old", "--yes", "--config", "/c.toml"]);
        assert_eq!(cli.config.as_deref(), Some(Path::new("/c.toml")));
        assert!(matches!(
            cli.command,
            Some(Command::Workspace(WorkspaceCommand::Delete { ref name, yes: true })) if name == "old"
        ));

        let cli = parse(&["workspace", "import", "legacy", "--stm", "a", "--ltm", "b"]);
        assert!(matches!(
            cli.command,
            Some(Command::Workspace(WorkspaceCommand::Import { .. }))
        ));
        assert!(Cli::try_parse_from(["neurolithe", "workspace", "import", "x"]).is_err());
        assert!(parse(&[]).command.is_none());
    }

    /// 2.1: init creates a private home + config from the example, a
    /// workspace, and prints a snippet with the absolute binary path; a second
    /// run keeps the existing config.
    #[test]
    fn test_init_creates_home_config_and_snippet() {
        let base = tempfile::tempdir().unwrap();
        let home = base.path().join("nl-home");
        let opts = LoadOptions {
            home: Some(home.clone()),
            workspace: Some("research".into()),
            ..Default::default()
        };
        let exe = Path::new("/opt/bin/neurolithe");
        let mut out = Vec::new();
        init(&opts, &lookup(&[]), exe, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();

        let config_file = home.join(CONFIG_FILE_NAME);
        assert_eq!(
            std::fs::read_to_string(&config_file).unwrap(),
            EXAMPLE_CONFIG
        );
        assert!(home.join("workspaces/research").is_dir());
        let json_start = text.find('{').unwrap();
        let snippet: serde_json::Value = serde_json::from_str(&text[json_start..]).unwrap();
        let server = &snippet["mcpServers"]["neurolithe"];
        assert_eq!(server["command"], "/opt/bin/neurolithe");
        assert_eq!(
            server["args"],
            serde_json::json!([
                "mcp",
                "--home",
                home.display().to_string(),
                "--workspace",
                "research"
            ])
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&home), 0o700);
            assert_eq!(mode(&config_file), 0o600);
        }

        // Idempotent: an edited config is never overwritten.
        std::fs::write(&config_file, "workspace = \"research\"\n").unwrap();
        let mut out = Vec::new();
        init(&opts, &lookup(&[]), exe, &mut out).unwrap();
        assert!(String::from_utf8(out).unwrap().contains("kept"));
        assert_eq!(
            std::fs::read_to_string(&config_file).unwrap(),
            "workspace = \"research\"\n"
        );
    }

    /// With the default home (from $HOME), the snippet needs no --home.
    #[test]
    fn test_init_default_home_snippet_is_minimal() {
        let base = tempfile::tempdir().unwrap();
        let env = lookup(&[("HOME", base.path().to_str().unwrap())]);
        let mut out = Vec::new();
        init(
            &LoadOptions::default(),
            &env,
            Path::new("/bin/nl"),
            &mut out,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        let snippet: serde_json::Value =
            serde_json::from_str(&text[text.find('{').unwrap()..]).unwrap();
        assert_eq!(
            snippet["mcpServers"]["neurolithe"]["args"],
            serde_json::json!(["mcp"])
        );
        assert!(base.path().join(".neurolithe/neurolithe.toml").is_file());
        assert!(base.path().join(".neurolithe/workspaces/default").is_dir());
    }

    /// 2.2 CLI: create / list / export / backup / delete round trip.
    #[test]
    fn test_workspace_cli_lifecycle() {
        let home = tempfile::tempdir().unwrap();
        let config = config_in(home.path());

        assert!(
            run(&config, WorkspaceCommand::List)
                .unwrap()
                .contains("No workspaces")
        );
        run(
            &config,
            WorkspaceCommand::Create {
                name: "alpha".into(),
            },
        )
        .unwrap();
        assert!(
            run(
                &config,
                WorkspaceCommand::Create {
                    name: "alpha".into()
                }
            )
            .is_err()
        );
        assert!(
            run(
                &config,
                WorkspaceCommand::Create {
                    name: "Bad Name".into()
                }
            )
            .is_err()
        );
        let listing = run(&config, WorkspaceCommand::List).unwrap();
        assert!(listing.contains("alpha"), "{listing}");

        let dump_file = home.path().join("alpha.json");
        run(
            &config,
            WorkspaceCommand::Export {
                name: "alpha".into(),
                out: Some(dump_file.clone()),
            },
        )
        .unwrap();
        let dump: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&dump_file).unwrap()).unwrap();
        assert_eq!(dump["workspace"], "alpha");

        // A just-created workspace has no stores yet: nothing to back up.
        let backup_cmd = || WorkspaceCommand::Backup {
            name: "alpha".into(),
            out: None,
        };
        let err = run(&config, backup_cmd()).unwrap_err();
        assert!(err.to_string().contains("no store files"), "{err}");
        // Once opened (as `neurolithe mcp` would), both stores are backed up.
        drop(open_workspace_stores(&config.workspaces_dir().join("alpha"), &ident("m")).unwrap());
        let backup = run(&config, backup_cmd()).unwrap();
        assert_eq!(backup.matches("Wrote ").count(), 2, "{backup}");
        assert!(home.path().join("backups").is_dir());

        // Delete needs --yes.
        let err = run(
            &config,
            WorkspaceCommand::Delete {
                name: "alpha".into(),
                yes: false,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("--yes"));
        assert!(config.workspaces_dir().join("alpha").is_dir());
        run(
            &config,
            WorkspaceCommand::Delete {
                name: "alpha".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(!config.workspaces_dir().join("alpha").exists());
        assert!(
            run(
                &config,
                WorkspaceCommand::Export {
                    name: "alpha".into(),
                    out: None
                }
            )
            .is_err()
        );
    }

    fn ident(model: &str) -> EmbeddingIdentity {
        EmbeddingIdentity {
            provider: "test".into(),
            model: model.into(),
            dim: 4,
        }
    }

    /// 2.2: `workspace import` brings legacy store files into a new workspace
    /// (migrated), refuses to clobber an existing one, and points at `reembed`
    /// when the stores were embedded by another model.
    #[test]
    fn test_workspace_import_legacy_files() {
        let home = tempfile::tempdir().unwrap();
        let config = config_in(home.path());
        // A "legacy" pair: stores created outside the workspace layout, by an
        // older embedder.
        let legacy = home.path().join("legacy");
        {
            let (stores, _) = open_workspace_stores(&legacy, &ident("old-model")).unwrap();
            stores
                .stm
                .execute(
                    "INSERT INTO nodes (tenant_id, payload) VALUES ('legacy-tenant', '{\"fact\":\"old fact\"}')",
                    [],
                )
                .unwrap();
        }
        let import = |name: &str, identity: &EmbeddingIdentity| -> Result<String> {
            let mut out = Vec::new();
            import_workspace(
                &config,
                name,
                &legacy.join(STM_FILE),
                &legacy.join(LTM_FILE),
                identity,
                &mut out,
            )?;
            Ok(String::from_utf8(out).unwrap())
        };

        let text = import("same", &ident("old-model")).unwrap();
        assert!(!text.contains("reembed"), "{text}");
        let dump = export_workspace(&config.workspaces_dir().join("same"), "same").unwrap();
        assert_eq!(dump["stm_facts"][0]["payload"]["fact"], "old fact");

        let text = import("newer", &ident("new-model")).unwrap();
        assert!(
            text.contains("neurolithe reembed --workspace newer"),
            "{text}"
        );

        assert!(
            import("same", &ident("old-model")).is_err(),
            "must not overwrite"
        );
        assert!(import("../evil", &ident("old-model")).is_err());
    }

    /// An offline embedder of a given width; `model` distinguishes identities.
    struct DimStub {
        dim: usize,
        model: &'static str,
    }

    #[async_trait::async_trait]
    impl crate::domain::ports::LlmClient for DimStub {
        async fn extract_facts(
            &self,
            _d: &str,
            _c: &[crate::domain::models::CclDefinition],
        ) -> Result<Vec<crate::domain::ports::ExtractedFact>> {
            Ok(Vec::new())
        }
        async fn generate_ccl_description(&self, _n: &str, _c: &str) -> Result<String> {
            Ok(String::new())
        }
        async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            let mut v = vec![0.1_f32; self.dim];
            v[0] = text.len() as f32;
            Ok(v)
        }
        async fn compress_context(&self, m: &str) -> Result<String> {
            Ok(m.to_string())
        }
        fn embedding_model_id(&self) -> String {
            format!("stub:{}", self.model)
        }
    }

    /// 2.4: `neurolithe reembed` moves a workspace built by one embedder to
    /// another: afterwards it opens under the new identity (it was refused
    /// before), and the report counts the re-embedded facts.
    #[tokio::test]
    async fn test_reembed_command_switches_workspace_to_new_embedder() {
        let home = tempfile::tempdir().unwrap();
        let config = config_in(home.path());
        let dir = config.workspaces_dir().join(&config.workspace);
        let old: std::sync::Arc<dyn crate::domain::ports::LlmClient> =
            std::sync::Arc::new(DimStub {
                dim: 4,
                model: "old",
            });
        let new: std::sync::Arc<dyn crate::domain::ports::LlmClient> =
            std::sync::Arc::new(DimStub {
                dim: 6,
                model: "new",
            });
        let old_id = crate::domain::ports::embedding_identity(old.as_ref())
            .await
            .unwrap();
        let new_id = crate::domain::ports::embedding_identity(new.as_ref())
            .await
            .unwrap();
        {
            let (stores, _) = open_workspace_stores(&dir, &old_id).unwrap();
            let repo = crate::infrastructure::repository::SqliteMemoryRepository::new(stores.stm);
            use crate::domain::ports::MemoryRepository;
            let node = crate::domain::models::MemoryNode {
                id: None,
                tenant_id: crate::domain::models::TenantId("default".into()),
                source_episode_id: None,
                payload: serde_json::json!({ "fact": "re-embed me" }),
                status: "active".into(),
                ccl: "reality".into(),
                is_explicit: true,
                support_count: 1,
                relevance_score: 1.0,
                context_key: None,
            };
            repo.store_node(&node, &old.embed_text("re-embed me").await.unwrap())
                .unwrap();
        }
        assert!(
            open_workspace_stores(&dir, &new_id).is_err(),
            "precondition: the new embedder is refused before reembed"
        );

        let mut out = Vec::new();
        reembed_command(&config, new, &mut out).await.unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("1 STM fact(s)"), "{text}");
        assert!(text.contains("Backup: "), "{text}");
        assert!(
            open_workspace_stores(&dir, &new_id).is_ok(),
            "opens after reembed"
        );

        // A workspace that doesn't exist is an error, not a silent create.
        let mut other = config.clone();
        other.workspace = "missing".into();
        let old_again: std::sync::Arc<dyn crate::domain::ports::LlmClient> =
            std::sync::Arc::new(DimStub {
                dim: 4,
                model: "old",
            });
        assert!(
            reembed_command(&other, old_again, &mut Vec::new())
                .await
                .is_err()
        );
    }

    #[test]
    fn test_parse_init_flags() {
        assert!(matches!(
            parse(&["init"]).command,
            Some(Command::Init {
                no_model_download: false
            })
        ));
        assert!(matches!(
            parse(&["init", "--no-model-download", "--home", "/h"]).command,
            Some(Command::Init {
                no_model_download: true
            })
        ));
    }

    #[test]
    fn test_parse_reembed() {
        let cli = parse(&["reembed", "--workspace", "work"]);
        assert!(matches!(cli.command, Some(Command::Reembed)));
        assert_eq!(cli.workspace.as_deref(), Some("work"));
    }

    /// Prefetch only applies to the local embedder: with a remote one it's a
    /// no-op (no network, no model dir needed).
    #[test]
    fn test_prefetch_skips_remote_embedder() {
        let home = tempfile::tempdir().unwrap();
        let mut config = config_in(home.path());
        config.llm.embedding_provider = Some(crate::infrastructure::config::LlmProvider::Openai);
        config.llm.models_dir = None;
        prefetch_local_model(&config).unwrap();
        assert!(!home.path().join("models").exists());
    }

    /// P2R-2: importing a store that predates embedding metadata prints the
    /// "assuming … came from" note from the identity check, plus a hint to
    /// reembed (the old model is unknown).
    #[test]
    fn test_import_prints_all_notes_and_reembed_hint() {
        use crate::domain::ports::MemoryRepository;
        let home = tempfile::tempdir().unwrap();
        let config = config_in(home.path());
        // A v0 store: legacy schema, one vector, no meta.
        let legacy = home.path().join("legacy");
        std::fs::create_dir_all(&legacy).unwrap();
        {
            let stm =
                crate::infrastructure::database::init_db(Some(&legacy.join(STM_FILE))).unwrap();
            crate::infrastructure::schema::init_schema(&stm, 4).unwrap();
            let repo = crate::infrastructure::repository::SqliteMemoryRepository::new(stm);
            let node = crate::domain::models::MemoryNode {
                id: None,
                tenant_id: crate::domain::models::TenantId("legacy-era".into()),
                source_episode_id: None,
                payload: serde_json::json!({ "fact": "legacy" }),
                status: "active".into(),
                ccl: "reality".into(),
                is_explicit: true,
                support_count: 1,
                relevance_score: 1.0,
                context_key: None,
            };
            repo.store_node(&node, &[0.5, 0.5, 0.5, 0.5]).unwrap();
            let ltm =
                crate::infrastructure::database::init_db(Some(&legacy.join(LTM_FILE))).unwrap();
            crate::infrastructure::schema::init_ltm_schema(&ltm, 4).unwrap();
        }
        let mut out = Vec::new();
        import_workspace(
            &config,
            "old",
            &legacy.join(STM_FILE),
            &legacy.join(LTM_FILE),
            &ident("current-model"),
            &mut out,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("predates embedding metadata"), "{text}");
        assert!(
            text.contains("neurolithe reembed --workspace old"),
            "{text}"
        );
    }

    /// P2R-9: `workspace export --out` over an existing, world-readable file
    /// leaves it owner-only.
    #[cfg(unix)]
    #[test]
    fn test_export_overwrite_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let home = tempfile::tempdir().unwrap();
        let config = config_in(home.path());
        run(&config, WorkspaceCommand::Create { name: "e".into() }).unwrap();
        let file = home.path().join("dump.json");
        std::fs::write(&file, "old").unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();
        run(
            &config,
            WorkspaceCommand::Export {
                name: "e".into(),
                out: Some(file.clone()),
            },
        )
        .unwrap();
        let mode = std::fs::metadata(&file).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    /// P2R-3: the CLI refuses to delete or reembed a workspace that a running
    /// process has open.
    #[tokio::test]
    async fn test_cli_delete_and_reembed_refused_while_open() {
        let home = tempfile::tempdir().unwrap();
        let config = config_in(home.path());
        let dir = config.workspaces_dir().join(&config.workspace);
        let held = open_workspace_stores(&dir, &ident("m")).unwrap(); // "a server"

        let err = run(
            &config,
            WorkspaceCommand::Delete {
                name: config.workspace.clone(),
                yes: true,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("in use"), "{err}");
        assert!(dir.exists());

        let embedder: std::sync::Arc<dyn crate::domain::ports::LlmClient> =
            std::sync::Arc::new(DimStub { dim: 4, model: "m" });
        let err = reembed_command(&config, embedder, &mut Vec::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("in use"), "{err}");

        drop(held);
        run(
            &config,
            WorkspaceCommand::Delete {
                name: config.workspace.clone(),
                yes: true,
            },
        )
        .unwrap();
    }

    /// N2: an import whose copy fails the integrity check is refused and
    /// leaves no workspace behind.
    #[test]
    fn test_import_of_corrupt_store_fails_and_cleans_up() {
        let home = tempfile::tempdir().unwrap();
        let config = config_in(home.path());
        let legacy = home.path().join("legacy");
        {
            let (stores, _) = open_workspace_stores(&legacy, &ident("m")).unwrap();
            stores
                .stm
                .execute_batch(
                    "PRAGMA journal_mode=DELETE; CREATE TABLE filler(x TEXT);
                     WITH RECURSIVE c(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM c WHERE i < 3000)
                     INSERT INTO filler SELECT printf('%0100d', i) FROM c;",
                )
                .unwrap();
        }
        let stm = legacy.join(STM_FILE);
        let mut bytes = std::fs::read(&stm).unwrap();
        let start = bytes.len() - 4096 * 3; // a filler page near the end
        for b in &mut bytes[start..start + 4096] {
            *b = 0xAB;
        }
        std::fs::write(&stm, bytes).unwrap();

        let mut out = Vec::new();
        let err = import_workspace(
            &config,
            "broken",
            &stm,
            &legacy.join(LTM_FILE),
            &ident("m"),
            &mut out,
        )
        .unwrap_err();
        assert!(!config.workspaces_dir().join("broken").exists(), "{err:#}");
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("must not be in use during import")
        );
    }
}
