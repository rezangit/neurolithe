//! Input limits for the MCP surface (SEC-12). The schemas advertise them and
//! the argument parser enforces them.

/// Largest JSON-RPC line accepted from stdin. Bigger lines are drained and
/// rejected without being buffered whole. Deliberately well above the
/// per-field caps below, so ordinary oversized arguments get a precise
/// `isError` naming the field rather than a transport-level rejection.
pub const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;
/// Largest `new_message` accepted by `push_dialogue` (bytes).
pub const MAX_MESSAGE_BYTES: usize = 64 * 1024;
/// Largest `fact_text` accepted by `store_memory` (bytes).
pub const MAX_FACT_BYTES: usize = 16 * 1024;
/// Largest document accepted by `remember_document` (bytes).
pub const MAX_DOCUMENT_BYTES: usize = 512 * 1024;
/// Largest search query (bytes).
pub const MAX_QUERY_BYTES: usize = 4 * 1024;
/// Largest identifier-like string (session, ccl, data_id, workspace, …).
pub const MAX_ID_BYTES: usize = 256;
/// Most tags on one fact or document.
pub const MAX_TAGS: usize = 64;
/// Most layers in a `ccl_filter`.
pub const MAX_CCL_FILTER: usize = 32;

/// Default recall breadth for the query tools when the caller omits `k`.
pub const DEFAULT_K: usize = 10;
pub const MAX_K: usize = 100;
pub const MAX_LIST_LIMIT: usize = 500;
pub const MAX_OFFSET: usize = 1_000_000;
pub const MAX_SAMPLE: usize = 500;
pub const MAX_DEPTH: usize = 10;
pub const MAX_CHILD_LIMIT: usize = 500;
pub const MAX_SUMMARY_CHARS: usize = 100_000;
