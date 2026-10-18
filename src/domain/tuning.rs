//! Fixed tuning knobs of the memory engine, in one place.
//!
//! These shape behaviour (how fast notes fade out of recall, how much session
//! history stays raw, how many hits a recall returns) but are not exposed in
//! the config file. The configurable ones (half-lives, distance thresholds,
//! intervals) live in `AppConfig`. Storage mechanics, such as KNN over-fetch,
//! stay next to the SQL that needs them.

/// STM relevance below which a decayed node is archived: it leaves recall and
/// the vector index. Nodes start at 1.0, so an untouched node is archived after
/// ~3.3 half-lives.
pub const ARCHIVE_THRESHOLD: f64 = 0.1;

/// Number of hits an STM recall returns when the caller names no breadth.
pub const STM_RECALL_K: usize = 10;

/// Number of related facts a session pulls from STM as context for a
/// dialogue turn.
pub const SESSION_CONTEXT_K: usize = 5;

/// Estimated session size (tokens) at which older messages are compressed
/// into the running summary.
pub const SESSION_TOKEN_THRESHOLD: usize = 4000;

/// Newest messages a session keeps verbatim when it compresses or trims.
pub const SESSION_KEEP_RECENT: usize = 10;

/// Rough characters per token for session size estimates (GPT tokenizer
/// heuristic).
pub const CHARS_PER_TOKEN: usize = 4;
