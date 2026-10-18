//! Diagnostics: a `tracing` subscriber that writes to **stderr only**.
//!
//! stdout is the MCP transport (newline-delimited JSON-RPC), so nothing but
//! protocol frames may ever reach it; user-facing CLI output (`init`,
//! `workspace list`, …) is written to stdout explicitly and is not logging.
//!
//! The level comes from `RUST_LOG` when set (full `EnvFilter` syntax, e.g.
//! `neurolithe=debug,rdkafka=warn`), else from config `[log] level`, else
//! `info`; the latter two also quiet the MCP SDK (`rmcp=warn`) unless they
//! name `rmcp` themselves. The subscriber is installed before the config
//! loads (so config warnings are captured) and the filter is swapped once the
//! config is known.

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

/// Added to the config/default level: the MCP SDK logs every session's
/// lifecycle (and, at debug, whole messages, i.e. memory content) under
/// `rmcp`. Its warnings and errors still show.
const QUIET_SDK: &str = "rmcp=warn";

/// The directives to apply: `RUST_LOG` verbatim when set; else config
/// `[log] level`, else [`DEFAULT_LEVEL`], plus [`QUIET_SDK`] when that only
/// lowers rmcp's level: the level's global part is `info`, `debug` or
/// `trace`, and it says nothing about `rmcp` itself.
pub fn directives(rust_log: Option<&str>, config_level: Option<&str>) -> String {
    if let Some(v) = rust_log {
        return v.to_string();
    }
    let level = config_level.unwrap_or(DEFAULT_LEVEL);
    let parts: Vec<&str> = level.split(',').map(str::trim).collect();
    let names_rmcp = parts.iter().any(|d| d.starts_with("rmcp"));
    // The bare directive (no `target=`) sets everyone else's level; the
    // last one wins. Without one, unmatched targets are already quiet.
    let global = parts
        .iter()
        .rev()
        .find(|d| !d.is_empty() && !d.contains('=') && !d.contains('['))
        .map(|d| d.to_ascii_lowercase());
    let chatty = matches!(global.as_deref(), Some("info" | "debug" | "trace"));
    if chatty && !names_rmcp {
        format!("{level},{QUIET_SDK}")
    } else {
        level.to_string()
    }
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
        .unwrap_or_else(|| EnvFilter::new(directives(None, None)));
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
    let (filter_spec, source) = match rust_log() {
        Some(v) => (v, "RUST_LOG"),
        None => match level.filter(|l| !l.trim().is_empty()) {
            Some(l) => (directives(None, Some(l)), "[log] level"),
            None => return,
        },
    };
    let Some(handle) = RELOAD.get() else {
        return;
    };
    match parse_filter(&filter_spec) {
        Ok(filter) => {
            if let Err(e) = handle.reload(filter) {
                tracing::warn!("could not apply {source} {filter_spec:?}: {e}");
            }
        }
        Err(e) => tracing::warn!("ignoring {source}: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Which of an rmcp and a neurolithe INFO event pass `directives`.
    fn passes(directives: &str) -> (bool, bool) {
        let (rmcp_info, ours_info, _) = observe(directives);
        (rmcp_info, ours_info)
    }

    /// Whether an rmcp WARN event passes `directives`.
    fn rmcp_warn_passes(directives: &str) -> bool {
        observe(directives).2
    }

    /// (rmcp INFO, neurolithe INFO, rmcp WARN) passed `directives`.
    fn observe(directives: &str) -> (bool, bool, bool) {
        use std::sync::{Arc, Mutex};
        let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
        let sink = buf.clone();
        let subscriber = tracing_subscriber::registry()
            .with(parse_filter(directives).unwrap())
            .with(
                fmt::layer()
                    .with_ansi(false)
                    .with_writer(move || SinkWriter(sink.clone())),
            );
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target: "rmcp::service", "rmcp chatter");
            tracing::info!(target: "neurolithe::daemon", "our news");
            tracing::warn!(target: "rmcp::service", "rmcp warning");
        });
        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        (
            out.contains("rmcp chatter"),
            out.contains("our news"),
            out.contains("rmcp warning"),
        )
    }

    /// W1R-2: the quieting only ever lowers rmcp's level. A quieter global
    /// level (error, off, or one that sets no global level) stays as is,
    /// instead of `rmcp=warn` raising rmcp above it.
    #[test]
    fn test_rmcp_quieting_never_raises_its_level() {
        for level in ["error", "off", "neurolithe=debug", "neurolithe=debug,error"] {
            assert!(
                !rmcp_warn_passes(&directives(None, Some(level))),
                "{level}: rmcp warnings must stay hidden"
            );
        }
        for level in ["info", "debug", "trace", "neurolithe=debug,info", "warn"] {
            assert!(
                rmcp_warn_passes(&directives(None, Some(level))),
                "{level}: rmcp warnings still show"
            );
        }
    }

    struct SinkWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for SinkWriter {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// rmcp's per-session INFO chatter is quiet by default and under a
    /// config level, while our own INFO logs still show; an explicit
    /// RUST_LOG is applied verbatim (it can turn rmcp back on).
    #[test]
    fn test_rmcp_quiet_unless_rust_log_says_otherwise() {
        assert_eq!(passes(&directives(None, None)), (false, true));
        assert_eq!(passes(&directives(None, Some("info"))), (false, true));
        assert_eq!(passes(&directives(None, Some("debug"))), (false, true));
        // A config level that names rmcp is respected.
        assert_eq!(
            passes(&directives(None, Some("info,rmcp=info"))),
            (true, true)
        );
        assert_eq!(passes(&directives(Some("info"), None)), (true, true));
        assert_eq!(
            passes(&directives(Some("rmcp=info,warn"), Some("info"))),
            (true, false)
        );
    }

    #[test]
    fn test_parse_filter() {
        assert!(parse_filter("debug").is_ok());
        assert!(parse_filter("neurolithe=debug,rdkafka=warn").is_ok());
        assert!(parse_filter("neurolithe=notalevel").is_err());
    }
}
