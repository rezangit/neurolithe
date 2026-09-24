// Kept crate-wide on purpose: the SQLite repositories own a `rusqlite::Connection`
// (Send, !Sync) and are shared as `Arc<dyn …>` on a single-threaded LocalSet.
// Removing this flags ~20 sites across the app/infra layers; the real fix (DB
// actor / spawn_blocking, dropping the unsound `unsafe impl Send/Sync` on
// NeurolitheApp) is the Phase 3 storage-runtime redesign (PLAN 3.3, ARC-5).
#![allow(clippy::arc_with_non_send_sync)]
pub mod application;
pub mod daemon;
pub mod domain;
pub mod infrastructure;
pub mod interfaces;

use crate::infrastructure::config::AppConfig;

fn main() -> anyhow::Result<()> {
    let config = AppConfig::load()?;

    // `neurolithe mcp` serves the MCP tools over stdio — the standalone default.
    // With the `kafka` feature, any other invocation runs the full daemon
    // (feeder + consumers + schedulers). A standalone build has no daemon.
    let mcp_only = std::env::args().nth(1).as_deref() == Some("mcp");

    // Standalone build: the daemon does not exist — only the MCP server.
    #[cfg(not(feature = "kafka"))]
    if !mcp_only {
        eprintln!(
            "This is a standalone NeuroLithe build (no Kafka daemon).\n\
             Start the MCP server with:  neurolithe mcp"
        );
        std::process::exit(2);
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = tokio::task::LocalSet::new();
    if mcp_only {
        local.block_on(&rt, crate::daemon::run_mcp(config))
    } else {
        // Full daemon on a single-threaded runtime + LocalSet (the SQLite-backed
        // services are !Sync; the loops are !Send and run cooperatively).
        #[cfg(feature = "kafka")]
        {
            local.block_on(&rt, crate::daemon::run(config))
        }
        #[cfg(not(feature = "kafka"))]
        {
            unreachable!("standalone non-mcp invocation exits above")
        }
    }
}
