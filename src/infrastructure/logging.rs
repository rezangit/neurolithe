//! Diagnostics: a `tracing` subscriber that writes to **stderr only**.
//!
//! stdout is the MCP transport (newline-delimited JSON-RPC), so nothing but
//! protocol frames may ever reach it; user-facing CLI output (`init`,
//! `workspace list`, …) is written to stdout explicitly and is not logging.
//!
//! The level comes from `RUST_LOG` when set (full `EnvFilter` syntax, e.g.
//! `neurolithe=debug,rdkafka=warn`), else from config `[log] level`, else
//! `info`. The subscriber is installed before the config loads (so config
//! warnings are captured) and the filter is swapped once the config is known.

use std::io::IsTerminal;
use std::sync::OnceLock;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry, fmt, reload};

/// The level used when neither `RUST_LOG` nor `[log] level` is set.
pub const DEFAULT_LEVEL: &str = "info";

static RELOAD: OnceLock<reload::Handle<EnvFilter, Registry>> = OnceLock::new();

/// `RUST_LOG`, when set to a non-empty value.
fn rust_log() -> Option<String> {
    std::env::var("RUST_LOG")
        .ok()
        .filter(|v| !v.trim().is_empty())
}

/// Parse a filter directive string (`info`, `neurolithe=debug,warn`, …).
pub fn parse_filter(directives: &str) -> Result<EnvFilter, String> {
    EnvFilter::try_new(directives).map_err(|e| format!("invalid log filter {directives:?}: {e}"))
}

/// Install the global stderr subscriber. Idempotent: later calls (and a
/// subscriber already set by a test harness) are ignored.
pub fn init() {
    let filter = rust_log()
        .and_then(|v| parse_filter(&v).ok())
        .unwrap_or_else(|| EnvFilter::new(DEFAULT_LEVEL));
    let (filter, handle) = reload::Layer::new(filter);
    let layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal());
    if tracing_subscriber::registry()
        .with(filter)
        .with(layer)
        .try_init()
        .is_ok()
    {
        let _ = RELOAD.set(handle);
    }
    if let Some(v) = rust_log()
        && let Err(e) = parse_filter(&v)
    {
        tracing::warn!("ignoring RUST_LOG: {e}; using {DEFAULT_LEVEL}");
    }
}

/// Re-apply the filter once the config is loaded: `RUST_LOG` (which may have
/// just come from `<home>/.env`) wins, else config `[log] level`, else the
/// startup filter stays.
pub fn apply_config_level(level: Option<&str>) {
    let (directives, source) = match rust_log() {
        Some(v) => (v, "RUST_LOG"),
        None => match level.filter(|l| !l.trim().is_empty()) {
            Some(l) => (l.to_string(), "[log] level"),
            None => return,
        },
    };
    let Some(handle) = RELOAD.get() else {
        return;
    };
    match parse_filter(&directives) {
        Ok(filter) => {
            if let Err(e) = handle.reload(filter) {
                tracing::warn!("could not apply {source} {directives:?}: {e}");
            }
        }
        Err(e) => tracing::warn!("ignoring {source}: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_filter() {
        assert!(parse_filter("debug").is_ok());
        assert!(parse_filter("neurolithe=debug,rdkafka=warn").is_ok());
        assert!(parse_filter("neurolithe=notalevel").is_err());
    }
}
