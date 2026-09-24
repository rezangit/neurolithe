use crate::application::introspection::LeafPage;
use crate::application::query_service::{QueryRequest, QueryScope};
use crate::application::workspace::WorkspaceManager;
use crate::domain::models::{TimeFilter, WORKSPACE_TENANT};
use crate::interfaces::bus_query::flatten_recall;
use crate::interfaces::mcp_types::{JsonRpcRequest, JsonRpcResponse, McpToolResult};
use serde::Serialize;
use serde_json::{Value, json};
use std::rc::Rc;
use tokio::io::{self, AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};

// ---------------------------------------------------------------------------
// Protocol + input limits
// ---------------------------------------------------------------------------

/// MCP protocol revisions this server can speak (newest first). The tool
/// surface is plain text content, valid under all of them. `initialize` echoes
/// the client's requested version when it is listed here, else offers the
/// newest (per the spec's version negotiation).
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Largest JSON-RPC line accepted from stdin. Bigger lines are drained and
/// rejected without being buffered whole (SEC-12). Deliberately well above the
/// per-field caps below, so ordinary oversized arguments get a precise
/// `isError` naming the field rather than a transport-level rejection.
pub const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;
/// How much of an oversized line is kept to recover its request `id`.
const ID_PROBE_BYTES: usize = 1024;
/// Largest `new_message` accepted by `push_dialogue` (bytes).
const MAX_MESSAGE_BYTES: usize = 64 * 1024;
/// Largest `fact_text` accepted by `store_memory` (bytes).
const MAX_FACT_BYTES: usize = 16 * 1024;
/// Largest document accepted by `remember_document` (bytes).
const MAX_DOCUMENT_BYTES: usize = 512 * 1024;
/// Largest search query (bytes).
const MAX_QUERY_BYTES: usize = 4 * 1024;
/// Largest identifier-like string (session, ccl, dataId, workspace, …).
const MAX_ID_BYTES: usize = 256;

/// Default recall breadth for the MCP query tools when the caller omits it.
const DEFAULT_K: usize = 10;
const MAX_K: usize = 100;
const MAX_LIST_LIMIT: usize = 500;
const MAX_OFFSET: usize = 1_000_000;
const MAX_SAMPLE: usize = 500;
const MAX_DEPTH: usize = 10;
const MAX_CHILD_LIMIT: usize = 500;
const MAX_SUMMARY_CHARS: usize = 100_000;

/// Render an introspection result as an MCP tool result (JSON text on success).
fn introspect_result<T: Serialize>(result: anyhow::Result<T>) -> McpToolResult {
    match result {
        Ok(value) => {
            McpToolResult::ok(serde_json::to_string(&value).unwrap_or_else(|_| "null".into()))
        }
        Err(e) => McpToolResult::err(format!("introspection failed: {e}")),
    }
}

/// Render a workspace-operation result: JSON on success, the (already
/// user-facing) error text as `isError` on failure.
fn json_result<T: Serialize>(result: anyhow::Result<T>) -> McpToolResult {
    match result {
        Ok(value) => {
            McpToolResult::ok(serde_json::to_string(&value).unwrap_or_else(|_| "null".into()))
        }
        Err(e) => McpToolResult::err(format!("{e}")),
    }
}

// ---------------------------------------------------------------------------
// Argument parsing — every problem becomes a clear `isError` tool result
// ---------------------------------------------------------------------------

/// A caller mistake, reported back as an `isError` tool result.
type ArgResult<T> = Result<T, String>;

/// Tool arguments. A missing `arguments` is treated as `{}`; a non-object is a
/// caller error.
struct Args<'a>(&'a serde_json::Map<String, Value>);

impl<'a> Args<'a> {
    fn get(&self, name: &str) -> Option<&'a Value> {
        self.0.get(name).filter(|v| !v.is_null())
    }

    /// An optional string, type-checked and size-capped. Blank → `None`.
    fn opt_str(&self, name: &str, max_bytes: usize) -> ArgResult<Option<&'a str>> {
        match self.get(name) {
            None => Ok(None),
            Some(Value::String(s)) => {
                if s.len() > max_bytes {
                    Err(format!(
                        "'{name}' is too large ({} bytes; max {max_bytes})",
                        s.len()
                    ))
                } else if s.trim().is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(s.as_str()))
                }
            }
            Some(_) => Err(format!("'{name}' must be a string")),
        }
    }

    /// A required, non-blank, size-capped string.
    fn req_str(&self, name: &str, max_bytes: usize) -> ArgResult<&'a str> {
        self.opt_str(name, max_bytes)?
            .ok_or_else(|| format!("'{name}' is required and must be a non-empty string"))
    }

    /// An optional non-negative integer clamped to `[min, max]`. Negative or
    /// fractional values are rejected — never cast (a negative `k` used to
    /// wrap to "no limit", SEC-12).
    fn count(&self, name: &str, default: usize, min: usize, max: usize) -> ArgResult<usize> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => match v.as_u64() {
                Some(n) => Ok(usize::try_from(n).unwrap_or(max).clamp(min, max)),
                None => Err(format!("'{name}' must be a non-negative integer")),
            },
        }
    }

    /// An optional count with no default (`None` when absent).
    fn opt_count(&self, name: &str, min: usize, max: usize) -> ArgResult<Option<usize>> {
        match self.get(name) {
            None => Ok(None),
            Some(_) => self.count(name, 0, min, max).map(Some),
        }
    }

    /// A required integer id (may be any i64).
    fn req_id(&self, name: &str) -> ArgResult<i64> {
        self.get(name)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| format!("'{name}' is required and must be an integer"))
    }

    /// An optional array of strings (each size-capped).
    fn str_list(&self, name: &str, max_items: usize) -> ArgResult<Vec<String>> {
        match self.get(name) {
            None => Ok(Vec::new()),
            Some(Value::Array(items)) => {
                if items.len() > max_items {
                    return Err(format!("'{name}' has too many items (max {max_items})"));
                }
                items
                    .iter()
                    .map(|v| match v.as_str() {
                        Some(s) if s.len() <= MAX_ID_BYTES => Ok(s.to_string()),
                        Some(_) => Err(format!("an item of '{name}' is too long")),
                        None => Err(format!("'{name}' must be an array of strings")),
                    })
                    .collect()
            }
            Some(_) => Err(format!("'{name}' must be an array of strings")),
        }
    }

    /// `time_filter: {after?, before?}` with `YYYY-MM-DD[...]` dates. Malformed
    /// input is an error, not silently ignored (QA-9).
    fn time_filter(&self) -> ArgResult<TimeFilter> {
        let Some(v) = self.get("time_filter") else {
            return Ok(TimeFilter::default());
        };
        let Some(obj) = v.as_object() else {
            return Err("'time_filter' must be an object {after?, before?}".into());
        };
        let date = |key: &str| -> ArgResult<Option<String>> {
            match obj.get(key).filter(|v| !v.is_null()) {
                None => Ok(None),
                Some(Value::String(s)) if is_iso_date_prefix(s) => Ok(Some(s.clone())),
                Some(_) => Err(format!(
                    "'time_filter.{key}' must be a date string like 2026-01-31"
                )),
            }
        };
        Ok(TimeFilter {
            after: date("after")?,
            before: date("before")?,
        })
    }
}

/// `YYYY-MM-DD`, optionally followed by a time part (≤ 40 chars total).
fn is_iso_date_prefix(s: &str) -> bool {
    let b = s.as_bytes();
    s.len() <= 40
        && b.len() >= 10
        && b[..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b'-'
        && b[8..10].iter().all(u8::is_ascii_digit)
}

// ---------------------------------------------------------------------------
// Bounded line reader (SEC-12)
// ---------------------------------------------------------------------------

enum Line {
    Text(String),
    /// Over the cap; carries the line's first [`ID_PROBE_BYTES`] bytes so the
    /// error reply can still be correlated to the request id.
    TooLong(Vec<u8>),
    Eof,
}

/// Best-effort recovery of a JSON-RPC `id` (number or string) from the start
/// of a truncated request, so the client gets an error it can match instead of
/// waiting for a reply that never comes. `Null` when not found.
///
/// Only a **top-level** `"id"` key counts: the scan tracks string state and
/// object/array depth, so a nested key (e.g. `params.arguments.id`, which comes
/// first when `params` precedes `id`) is never mistaken for the request id
/// (REV-5). Once the scan reaches the truncation point the answer is `Null`.
fn probe_request_id(head: &[u8]) -> Value {
    let text = String::from_utf8_lossy(head);
    let bytes = text.as_bytes();
    let mut depth: usize = 0;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                // Read the whole string (honouring escapes).
                let start = i + 1;
                let mut j = start;
                while j < bytes.len() && bytes[j] != b'"' {
                    j += if bytes[j] == b'\\' { 2 } else { 1 };
                }
                if j >= bytes.len() {
                    return Value::Null; // truncated inside a string
                }
                let is_top_level_id_key = depth == 1 && &text[start..j] == "id" && {
                    let after = text[j + 1..].trim_start();
                    after.starts_with(':')
                };
                if is_top_level_id_key {
                    let value = text[j + 1..].trim_start()[1..].trim_start();
                    return parse_id_value(value);
                }
                i = j + 1;
            }
            b'{' | b'[' => {
                depth += 1;
                i += 1;
            }
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            _ => i += 1,
        }
    }
    Value::Null
}

/// Parse a JSON-RPC id value (string without escapes, or integer) at the start
/// of `value`; `Null` for anything else (including a truncated value).
fn parse_id_value(value: &str) -> Value {
    if let Some(stripped) = value.strip_prefix('"') {
        return match stripped.find('"') {
            Some(end) if !stripped[..end].contains('\\') => Value::String(stripped[..end].into()),
            _ => Value::Null,
        };
    }
    let num: String = value
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    let terminated = value[num.len()..].trim_start().starts_with([',', '}']);
    if !terminated {
        return Value::Null; // truncated mid-number: don't guess
    }
    num.parse::<i64>().map(Value::from).unwrap_or(Value::Null)
}

/// Read one `\n`-terminated line of at most `max` bytes. An oversized line is
/// consumed and discarded chunk by chunk (never buffered whole) and reported as
/// [`Line::TooLong`], so one huge message can't exhaust memory.
async fn read_bounded_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    max: usize,
) -> io::Result<Line> {
    let mut buf: Vec<u8> = Vec::new();
    let mut head: Vec<u8> = Vec::new();
    let mut overflow = false;
    let mut saw_any = false;
    loop {
        let (consumed, done) = {
            let available = reader.fill_buf().await?;
            if available.is_empty() {
                break; // EOF
            }
            saw_any = true;
            let newline = available.iter().position(|&b| b == b'\n');
            let chunk = &available[..newline.unwrap_or(available.len())];
            if !overflow {
                if buf.len() + chunk.len() > max {
                    overflow = true;
                    buf.extend_from_slice(
                        &chunk[..ID_PROBE_BYTES.saturating_sub(buf.len()).min(chunk.len())],
                    );
                    buf.truncate(ID_PROBE_BYTES);
                    head = std::mem::take(&mut buf);
                } else {
                    buf.extend_from_slice(chunk);
                }
            }
            match newline {
                Some(i) => (i + 1, true),
                None => (available.len(), false),
            }
        };
        reader.consume(consumed);
        if done {
            break;
        }
    }
    if !saw_any {
        return Ok(Line::Eof);
    }
    if overflow {
        return Ok(Line::TooLong(head));
    }
    if buf.last() == Some(&b'\r') {
        buf.pop();
    }
    Ok(Line::Text(String::from_utf8_lossy(&buf).into_owned()))
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

pub struct McpServer {
    /// The active workspace and its services, once startup has opened it.
    /// Every tool call resolves the services through it, so `workspace_switch`
    /// takes effect immediately. Recall runs on the same `QueryService` the
    /// `memory.query` bus door uses, so the two doors can't drift.
    ///
    /// `None` while the workspace is still starting (embedder probe, store
    /// migrations): `initialize` / `tools/list` / `ping` are answered anyway —
    /// an MCP client must never time out waiting on a model load — and tool
    /// calls wait for [`set_workspaces`](Self::set_workspaces).
    workspaces: std::cell::RefCell<Option<Rc<WorkspaceManager>>>,
    ready: tokio::sync::Notify,
}

impl McpServer {
    /// A server over an already-open workspace.
    pub fn new(workspaces: Rc<WorkspaceManager>) -> Self {
        let server = Self::pending();
        server.set_workspaces(workspaces);
        server
    }

    /// A server whose workspace is still starting; see [`set_workspaces`](Self::set_workspaces).
    pub fn pending() -> Self {
        Self {
            workspaces: std::cell::RefCell::new(None),
            ready: tokio::sync::Notify::new(),
        }
    }

    /// Hand over the opened workspace; releases any waiting tool calls.
    pub fn set_workspaces(&self, workspaces: Rc<WorkspaceManager>) {
        *self.workspaces.borrow_mut() = Some(workspaces);
        self.ready.notify_waiters();
    }

    /// The workspace manager, waiting for startup if needed.
    async fn workspaces(&self) -> Rc<WorkspaceManager> {
        loop {
            // Register before checking so a concurrent set can't be missed.
            let notified = self.ready.notified();
            if let Some(ws) = self.workspaces.borrow().clone() {
                return ws;
            }
            notified.await;
        }
    }

    /// Build a [`QueryRequest`] from MCP tool arguments, applying the shared
    /// defaults (`reality` layer, breadth [`DEFAULT_K`]). Isolation comes from
    /// the workspace, so the tenant is always [`WORKSPACE_TENANT`]. `scope` is fixed per tool. Keeps both query tools on one
    /// parse path so their defaults never diverge. `query` is required: an
    /// empty query used to reach sqlite-vec and leak its SQL error (QA-9).
    fn query_request(&self, args: &Args, scope: QueryScope) -> ArgResult<QueryRequest> {
        Ok(QueryRequest {
            scope,
            tenant: WORKSPACE_TENANT.to_string(),
            query: args.req_str("query", MAX_QUERY_BYTES)?.to_string(),
            k: args.count("k", DEFAULT_K, 1, MAX_K)?,
            time_filter: args.time_filter()?,
            ccl: args.str_list("ccl_filter", 32)?,
            context_key: None,
        })
    }

    /// Serve MCP over the process's stdin/stdout until stdin closes.
    pub async fn run_stdio(&self) -> anyhow::Result<()> {
        self.serve(BufReader::new(io::stdin()), io::stdout()).await
    }

    /// Serve newline-delimited JSON-RPC from `reader` to `writer` until EOF.
    /// Generic so tests can drive the real loop in memory.
    pub async fn serve<R, W>(&self, mut reader: R, mut writer: W) -> anyhow::Result<()>
    where
        R: AsyncBufRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        loop {
            let line = match read_bounded_line(&mut reader, MAX_LINE_BYTES).await? {
                Line::Eof => break,
                Line::TooLong(head) => {
                    let resp = JsonRpcResponse::error(
                        probe_request_id(&head),
                        -32600,
                        format!("Request too large (max {MAX_LINE_BYTES} bytes per line)"),
                    );
                    write_response(&mut writer, &resp).await?;
                    continue;
                }
                Line::Text(line) => line,
            };
            if line.trim().is_empty() {
                continue;
            }

            let response = match serde_json::from_str::<JsonRpcRequest>(&line) {
                // Notifications (no id) get no response.
                Ok(req) if req.id.is_none() => continue,
                Ok(req) => self.handle_request(req).await,
                Err(e) => JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {e}")),
            };
            write_response(&mut writer, &response).await?;
        }

        Ok(())
    }

    async fn handle_request(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id.clone().unwrap_or(Value::Null);

        match req.method.as_str() {
            "tools/call" => {
                let Some(tool_name) = req.params.get("name").and_then(|n| n.as_str()) else {
                    return JsonRpcResponse::error(id, -32602, "tools/call requires a 'name'");
                };
                // An unknown tool is a protocol error (invalid params), not a
                // tool result — per the MCP spec (REV-6).
                if !TOOL_NAMES.contains(&tool_name) {
                    return JsonRpcResponse::error(
                        id,
                        -32602,
                        format!("Unknown tool: {tool_name}"),
                    );
                }
                let empty = serde_json::Map::new();
                let result = match req.params.get("arguments") {
                    None | Some(Value::Null) => self.call_tool(tool_name, &Args(&empty)).await,
                    Some(Value::Object(map)) => self.call_tool(tool_name, &Args(map)).await,
                    Some(_) => Err("'arguments' must be an object".to_string()),
                };
                let result = result.unwrap_or_else(McpToolResult::err);
                match serde_json::to_value(result) {
                    Ok(v) => JsonRpcResponse::success(id, v),
                    Err(e) => JsonRpcResponse::error(id, -32603, format!("Internal error: {e}")),
                }
            }
            "initialize" => {
                let requested = req
                    .params
                    .get("protocolVersion")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let version = SUPPORTED_PROTOCOL_VERSIONS
                    .iter()
                    .find(|v| **v == requested)
                    .copied()
                    .unwrap_or(SUPPORTED_PROTOCOL_VERSIONS[0]);
                JsonRpcResponse::success(
                    id,
                    json!({
                        "protocolVersion": version,
                        // The tool list is static: we never emit
                        // notifications/tools/list_changed, so don't advertise it.
                        "capabilities": { "tools": { "listChanged": false } },
                        "serverInfo": {
                            "name": "NeuroLithe",
                            "version": env!("CARGO_PKG_VERSION")
                        }
                    }),
                )
            }
            "ping" => JsonRpcResponse::success(id, json!({})),
            "tools/list" => JsonRpcResponse::success(id, tools_list()),
            other => JsonRpcResponse::error(id, -32601, format!("Method not found: {other}")),
        }
    }

    /// Dispatch one tool. `Err` is a caller mistake (bad/missing argument),
    /// rendered as an `isError` result; operational failures are rendered by
    /// each arm.
    async fn call_tool(&self, tool: &str, args: &Args<'_>) -> ArgResult<McpToolResult> {
        // Resolve the active workspace once per call (a switch mid-call can't
        // split one operation across two workspaces).
        let manager = self.workspaces().await;
        let ws = manager.current();
        let svc = &ws.services;
        Ok(match tool {
            "store_memory" => {
                // Explicit fact storage (bypasses the Sleep pipeline).
                let fact_text = args.req_str("fact_text", MAX_FACT_BYTES)?;
                let tags = args.str_list("tags", 64)?;
                let ccl = args.opt_str("ccl", MAX_ID_BYTES)?.unwrap_or("reality");
                match svc
                    .app
                    .store_explicit_fact(WORKSPACE_TENANT, fact_text, &tags, ccl)
                    .await
                {
                    Ok(_) => McpToolResult::ok("Memory fact explicitly stored."),
                    Err(e) => McpToolResult::err(format!("Failed to store memory: {e}")),
                }
            }
            "push_dialogue" => {
                // Flow 1: push dialogue to STM, compress, return optimized context.
                let session_id = args.req_str("session_id", MAX_ID_BYTES)?;
                let new_message = args.req_str("new_message", MAX_MESSAGE_BYTES)?;
                let ccl = args.opt_str("ccl", MAX_ID_BYTES)?.unwrap_or("reality");
                match svc
                    .app
                    .push_dialogue(WORKSPACE_TENANT, session_id, new_message, ccl)
                    .await
                {
                    Ok(context_window) => McpToolResult::ok(
                        serde_json::to_string(&context_window).unwrap_or_else(|_| "{}".into()),
                    ),
                    Err(e) => McpToolResult::err(format!("Failed to process dialogue: {e}")),
                }
            }
            "query_memory" => {
                // STM recall over the shared QueryService (same path as the bus door).
                let req = self.query_request(args, QueryScope::Stm)?;
                match svc.query.execute(&req).await {
                    Ok(outcome) => McpToolResult::ok(
                        serde_json::to_string(&outcome.stm).unwrap_or_else(|_| "[]".into()),
                    ),
                    Err(e) => McpToolResult::err(format!("Query failed: {e}")),
                }
            }
            "recall_ltm" => {
                // Reference-returning search of the permanent archive: locate
                // the nearest concept/document and surface its `dataId` +
                // provenance so the caller can fetch the original.
                let req = self.query_request(args, QueryScope::Ltm)?;
                match svc.query.execute(&req).await {
                    Ok(outcome) => {
                        let entries: Vec<_> = outcome.ltm.iter().flat_map(flatten_recall).collect();
                        McpToolResult::ok(
                            serde_json::to_string(&entries).unwrap_or_else(|_| "[]".into()),
                        )
                    }
                    Err(e) => McpToolResult::err(format!("LTM recall failed: {e}")),
                }
            }
            "remember_document" => {
                // File a document/note into the permanent LTM tree (upsert by
                // data_id). Summarized when a chat LLM is configured.
                let doc = crate::application::documents::RememberDocument {
                    title: args
                        .opt_str("title", MAX_QUERY_BYTES)?
                        .unwrap_or_default()
                        .to_string(),
                    text: args.req_str("text", MAX_DOCUMENT_BYTES)?.to_string(),
                    data_id: args.opt_str("data_id", MAX_ID_BYTES)?.map(str::to_string),
                    tags: args.str_list("tags", 64)?,
                };
                json_result(svc.documents.remember(doc).await)
            }
            // --- workspaces (physically separate memories) ---
            "workspace_current" => json_result(manager.current_info()),
            "workspace_list" => json_result(manager.list()),
            "workspace_create" => {
                let name = args.req_str("name", MAX_ID_BYTES)?;
                json_result(manager.create(name))
            }
            "workspace_switch" => {
                let name = args.req_str("name", MAX_ID_BYTES)?;
                // Drop our handle on the old workspace before its stores close.
                drop(ws);
                json_result(manager.switch(name).await)
            }
            "workspace_export" => {
                let name = args.opt_str("name", MAX_ID_BYTES)?;
                json_result(manager.export(name))
            }
            "workspace_delete" => {
                // Destructive: `confirm` must repeat the name; the active
                // workspace can't be deleted.
                let name = args.req_str("name", MAX_ID_BYTES)?;
                let confirm = args.opt_str("confirm", MAX_ID_BYTES)?;
                match manager.delete(name, confirm) {
                    Ok(()) => McpToolResult::ok(format!("Deleted workspace {name:?}.")),
                    Err(e) => McpToolResult::err(format!("{e}")),
                }
            }
            // --- read-only introspection (CT scan) ---
            "memory_stats" => introspect_result(svc.introspection.memory_stats()),
            "health" => introspect_result(svc.introspection.health()),
            "placement_debug" => {
                let sample = args.count("sample", 30, 1, MAX_SAMPLE)?;
                introspect_result(svc.introspection.placement_debug(sample))
            }
            "stm_list" => {
                let limit = args.count("limit", 20, 1, MAX_LIST_LIMIT)?;
                let offset = args.count("offset", 0, 0, MAX_OFFSET)?;
                let status = args.opt_str("status", MAX_ID_BYTES)?;
                if let Some(s) = status
                    && s != "active"
                    && s != "archived"
                {
                    return Err("'status' must be 'active' or 'archived'".into());
                }
                let contains = args.opt_str("contains", MAX_QUERY_BYTES)?;
                introspect_result(svc.introspection.stm_list(limit, offset, status, contains))
            }
            "ltm_map" => {
                let depth = args.count("depth", 3, 1, MAX_DEPTH)?;
                introspect_result(svc.introspection.ltm_map(depth))
            }
            "inspect_node" => {
                let node_id = args.req_id("id")?;
                let page = LeafPage {
                    child_limit: args.opt_count("child_limit", 1, MAX_CHILD_LIMIT)?,
                    child_offset: args.count("child_offset", 0, 0, MAX_OFFSET)?,
                    summary_max_chars: args.opt_count("summary_max_chars", 0, MAX_SUMMARY_CHARS)?,
                };
                introspect_result(svc.introspection.inspect_node(node_id, page))
            }
            "subtree" => {
                let node_id = args.req_id("node")?;
                let depth = args.count("depth", 2, 1, MAX_DEPTH)?;
                introspect_result(svc.introspection.subtree(node_id, depth))
            }
            "trace_dataId" => {
                let data_id = args.req_str("dataId", MAX_ID_BYTES)?;
                introspect_result(svc.introspection.trace_data_id(data_id))
            }
            other => return Err(format!("Unknown tool: {other}")),
        })
    }
}

async fn write_response<W: AsyncWrite + Unpin>(
    writer: &mut W,
    response: &JsonRpcResponse,
) -> anyhow::Result<()> {
    let text = serde_json::to_string(response)?;
    writer.write_all(text.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tool catalogue
// ---------------------------------------------------------------------------

/// Every tool `call_tool` dispatches — kept in lockstep with [`tools_list`]
/// (a test asserts the two agree). Names outside this list are rejected with
/// JSON-RPC -32602 before dispatch.
const TOOL_NAMES: &[&str] = &[
    "push_dialogue",
    "store_memory",
    "query_memory",
    "recall_ltm",
    "remember_document",
    "workspace_current",
    "workspace_list",
    "workspace_create",
    "workspace_switch",
    "workspace_export",
    "workspace_delete",
    "memory_stats",
    "health",
    "placement_debug",
    "stm_list",
    "ltm_map",
    "inspect_node",
    "subtree",
    "trace_dataId",
];

fn name_prop() -> Value {
    json!({
        "type": "string",
        "pattern": "^[a-z0-9][a-z0-9_-]{0,63}$",
        "description": "Workspace name: 1-64 chars of a-z, 0-9, '_' or '-', starting with a letter or digit."
    })
}

fn ccl_prop() -> Value {
    json!({
        "type": "string",
        "description": "Cognitive context layer. Defaults to 'reality'."
    })
}

fn time_filter_prop() -> Value {
    json!({
        "type": "object",
        "description": "Optional temporal boundaries",
        "properties": {
            "after": { "type": "string", "description": "Only return memories after this date (YYYY-MM-DD)" },
            "before": { "type": "string", "description": "Only return memories before this date (YYYY-MM-DD)" }
        }
    })
}

fn k_prop(default: usize) -> Value {
    json!({
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_K,
        "description": format!("Max results to return (1-{MAX_K}). Defaults to {default}.")
    })
}

fn tools_list() -> Value {
    let read_only = json!({ "readOnlyHint": true });
    json!({
        "tools": [
            {
                "name": "push_dialogue",
                "description": "Push the latest conversation turn to Short-Term Memory. The service extracts facts from it (via the configured LLM) and returns the optimized context window.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": { "type": "string", "description": "The session ID for the conversation" },
                        "new_message": { "type": "string", "maxLength": MAX_MESSAGE_BYTES, "description": "The new dialogue message to process" },
                        "ccl": ccl_prop()
                    },
                    "required": ["session_id", "new_message"]
                }
            },
            {
                "name": "store_memory",
                "description": "Explicitly store a crucial fact immediately, bypassing the background extraction pipeline.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "fact_text": { "type": "string", "maxLength": MAX_FACT_BYTES, "description": "The factual statement to store" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags for categorizing the fact" },
                        "ccl": ccl_prop()
                    },
                    "required": ["fact_text"]
                }
            },
            {
                "name": "query_memory",
                "description": "Search SHORT-TERM working memory (recent/decaying facts) for relevant context. Hybrid keyword + semantic search; returns token-optimized facts with 1-hop connections and temporal bounds. For the permanent archive of documents, use recall_ltm.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "maxLength": MAX_QUERY_BYTES, "description": "The query to search for in memory" },
                        "k": k_prop(DEFAULT_K),
                        "time_filter": time_filter_prop(),
                        "ccl_filter": { "type": "array", "items": { "type": "string" }, "description": "Cognitive layers to search. Defaults to ['reality']." }
                    },
                    "required": ["query"]
                },
                "annotations": read_only
            },
            {
                "name": "recall_ltm",
                "description": "Search the PERMANENT long-term archive (all ingested documents) by meaning. Reference-returning: each hit carries the document's dataId + provenance + ancestor concepts, so you can fetch the original. This is the primary tool for finding a scanned document, receipt, letter, or report.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "maxLength": MAX_QUERY_BYTES, "description": "What to look for in the archive (a phrase, topic, merchant, or document description)." },
                        "k": k_prop(DEFAULT_K)
                    },
                    "required": ["query"]
                },
                "annotations": read_only
            },
            {
                "name": "remember_document",
                "description": "File a document or note into the PERMANENT long-term archive: it is summarized (if a chat LLM is configured), embedded, and placed under the best-matching concept. Returns {data_id, leaf_id, concept_path, updated, summarized}. Passing an existing data_id replaces that document (upsert).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string", "description": "Human-facing title. Defaults to the first line of text." },
                        "text": { "type": "string", "maxLength": MAX_DOCUMENT_BYTES, "description": "The document text." },
                        "data_id": { "type": "string", "description": "Stable id for upserts. Omitted: a new id is minted." },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Optional tags." }
                    },
                    "required": ["text"]
                },
                // Writes to memory; only idempotent when a data_id is given
                // (without one each call files a new document).
                "annotations": { "readOnlyHint": false, "destructiveHint": false, "idempotentHint": false }
            },
            {
                "name": "workspace_current",
                "description": "The active workspace: {name, path, stm_bytes, ltm_bytes}. Each workspace is a completely separate memory (its own STM + LTM stores).",
                "inputSchema": { "type": "object", "properties": {}, "required": [] },
                "annotations": read_only
            },
            {
                "name": "workspace_list",
                "description": "All workspaces with their store sizes; the active one has active=true.",
                "inputSchema": { "type": "object", "properties": {}, "required": [] },
                "annotations": read_only
            },
            {
                "name": "workspace_create",
                "description": "Create a new, empty workspace (a separate memory). Does not switch to it.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": name_prop()
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "workspace_switch",
                "description": "Make another existing workspace active. Its stores are opened and the session buffers start empty; all memory tools then read and write that workspace. May be disabled by the server config.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": name_prop()
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "workspace_export",
                "description": "JSON dump of a workspace's STM facts and LTM document leaves (read-only). Defaults to the active workspace.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": name_prop()
                    },
                    "required": []
                },
                "annotations": read_only
            },
            {
                "name": "workspace_delete",
                "description": "DESTRUCTIVE: permanently delete a workspace and all of its memory. Requires confirm equal to name. The active workspace cannot be deleted.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": name_prop(),
                        "confirm": { "type": "string", "description": "Must equal name, to confirm the deletion." }
                    },
                    "required": ["name", "confirm"]
                },
                "annotations": { "destructiveHint": true, "idempotentHint": true }
            },
            {
                "name": "memory_stats",
                "description": "CT scan: full metrics snapshot of both memory stores (STM counts/decay histogram, LTM tree size/depth/inbox, DB sizes).",
                "inputSchema": { "type": "object", "properties": {}, "required": [] },
                "annotations": read_only
            },
            {
                "name": "health",
                "description": "Compact health summary: STM/LTM counts, orphan leaves, DB sizes, feeder lag (feeder_lag = -1 means unknown on this on-demand path; live lag is on the memory.metrics stream).",
                "inputSchema": { "type": "object", "properties": {}, "required": [] },
                "annotations": read_only
            },
            {
                "name": "placement_debug",
                "description": "Placement calibration: for a sample of document leaves, the distance to their nearest concept (threshold-free). Used to tune the placement threshold to real embedding distances.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sample": { "type": "integer", "minimum": 1, "maximum": MAX_SAMPLE, "description": "Number of leaves to probe. Defaults to 30." }
                    },
                    "required": []
                },
                "annotations": read_only
            },
            {
                "name": "stm_list",
                "description": "List STM working-memory facts (most-relevant first) with score, status, and age. Supports pagination and a keyword filter so you can find facts without pulling the whole store.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": { "type": "integer", "minimum": 1, "maximum": MAX_LIST_LIMIT, "description": "Max facts to return. Defaults to 20." },
                        "offset": { "type": "integer", "minimum": 0, "description": "How many facts to skip (pagination). Defaults to 0." },
                        "status": { "type": "string", "enum": ["active", "archived"], "description": "Optional filter: 'active' or 'archived'." },
                        "contains": { "type": "string", "description": "Optional case-insensitive substring the fact text must contain." }
                    },
                    "required": []
                },
                "annotations": read_only
            },
            {
                "name": "ltm_map",
                "description": "The top N concept layers of the long-term knowledge tree (a compact table of contents).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "depth": { "type": "integer", "minimum": 1, "maximum": MAX_DEPTH, "description": "Number of layers from the roots. Defaults to 3." }
                    },
                    "required": []
                },
                "annotations": read_only
            },
            {
                "name": "inspect_node",
                "description": "Inspect one LTM node: its summary, parents, children, and document leaves (dataIds + provenance). Children/leaves are paged (child_limit/child_offset) and summaries capped (summary_max_chars); child_count/leaf_count report the totals.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "integer", "description": "The LTM node id." },
                        "child_limit": { "type": "integer", "minimum": 1, "maximum": MAX_CHILD_LIMIT, "description": "Max children/leaves to return. Defaults to 50." },
                        "child_offset": { "type": "integer", "minimum": 0, "description": "How many children/leaves to skip (pagination). Defaults to 0." },
                        "summary_max_chars": { "type": "integer", "minimum": 0, "description": "Cap each summary to this many chars (0 = full). Defaults to 200." }
                    },
                    "required": ["id"]
                },
                "annotations": read_only
            },
            {
                "name": "subtree",
                "description": "A branch of the LTM tree from a node down to a given depth (concepts only).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "node": { "type": "integer", "description": "The LTM node id to start from." },
                        "depth": { "type": "integer", "minimum": 1, "maximum": MAX_DEPTH, "description": "Number of layers. Defaults to 2." }
                    },
                    "required": ["node"]
                },
                "annotations": read_only
            },
            {
                "name": "trace_dataId",
                "description": "Locate a document by dataId across the brain: its LTM leaf + ancestor branch, and how many STM facts carry it.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataId": { "type": "string", "description": "The document's dataId (archive reference)." }
                    },
                    "required": ["dataId"]
                },
                "annotations": read_only
            }
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::SqliteWorkspaceHost;
    use crate::domain::models::CclDefinition;
    use crate::domain::ports::{ExtractedFact, LlmClient};
    use crate::infrastructure::config::{AppConfig, LoadOptions};
    use std::collections::HashMap;
    use std::sync::Arc;

    const DIM: usize = 4;

    /// Deterministic offline LLM: embeds by text length, extracts nothing.
    struct StubLlm;

    #[async_trait::async_trait]
    impl LlmClient for StubLlm {
        async fn extract_facts(
            &self,
            _dialogue: &str,
            _valid_ccls: &[CclDefinition],
        ) -> anyhow::Result<Vec<ExtractedFact>> {
            Ok(Vec::new())
        }
        async fn generate_ccl_description(&self, _n: &str, _c: &str) -> anyhow::Result<String> {
            Ok("layer".into())
        }
        async fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            let l = text.len() as f32;
            Ok(vec![1.0, l.sin(), l.cos(), 0.5])
        }
        async fn compress_context(&self, m: &str) -> anyhow::Result<String> {
            Ok(m.chars().take(20).collect())
        }
    }

    struct Harness {
        server: McpServer,
        _home: tempfile::TempDir,
    }

    /// A throwaway home with tiny embedding dimensions.
    fn test_config(home: &std::path::Path, allow_switch: bool) -> AppConfig {
        let env: HashMap<String, String> = [
            ("NEUROLITHE__STM__VECTOR_DIMENSION", DIM.to_string()),
            ("NEUROLITHE__LTM__VECTOR_DIMENSION", DIM.to_string()),
            (
                "NEUROLITHE__MCP__ALLOW_WORKSPACE_SWITCH",
                allow_switch.to_string(),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let opts = LoadOptions {
            home: Some(home.to_path_buf()),
            ..Default::default()
        };
        AppConfig::load_with(&opts, &|_| None, Some(env)).unwrap()
    }

    /// The real server over the real SQLite workspace layout in a temp home,
    /// with an offline stub LLM. Starts on workspace "default".
    async fn harness_with(allow_switch: bool) -> Harness {
        let home = tempfile::tempdir().unwrap();
        let config = test_config(home.path(), allow_switch);
        let host = SqliteWorkspaceHost::new(config.clone(), Arc::new(StubLlm));
        let workspaces = WorkspaceManager::start(Box::new(host), &config.workspace, allow_switch)
            .await
            .unwrap();
        Harness {
            server: McpServer::new(Rc::new(workspaces)),
            _home: home,
        }
    }

    async fn harness() -> Harness {
        harness_with(true).await
    }

    async fn rpc(h: &Harness, method: &str, params: Value) -> Value {
        let req: JsonRpcRequest = serde_json::from_value(json!({
            "jsonrpc": "2.0", "id": 1, "method": method, "params": params
        }))
        .unwrap();
        serde_json::to_value(h.server.handle_request(req).await).unwrap()
    }

    async fn call(h: &Harness, tool: &str, args: Value) -> Value {
        rpc(h, "tools/call", json!({ "name": tool, "arguments": args })).await["result"].clone()
    }

    fn is_error(result: &Value) -> bool {
        result["isError"] == json!(true)
    }

    fn text(result: &Value) -> String {
        result["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string()
    }

    /// Facts in the ACTIVE workspace's STM.
    fn stm_count(h: &Harness) -> usize {
        h.server
            .workspaces
            .borrow()
            .clone()
            .expect("workspace ready")
            .current()
            .services
            .introspection
            .stm_list(1000, 0, None, None)
            .unwrap()
            .len()
    }

    /// QA-11: initialize echoes a supported client version (else offers the
    /// newest) and no longer advertises list-change notifications it never sends.
    #[tokio::test]
    async fn test_initialize_negotiates_version_and_no_list_changed() {
        let h = harness().await;
        let r = rpc(&h, "initialize", json!({ "protocolVersion": "2025-03-26" })).await;
        assert_eq!(r["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(r["result"]["capabilities"]["tools"]["listChanged"], false);

        let r = rpc(&h, "initialize", json!({ "protocolVersion": "1999-01-01" })).await;
        assert_eq!(
            r["result"]["protocolVersion"],
            SUPPORTED_PROTOCOL_VERSIONS[0]
        );
    }

    /// QA-11: `ping` answers with an empty result (it used to be -32601).
    #[tokio::test]
    async fn test_ping() {
        let h = harness().await;
        let r = rpc(&h, "ping", json!({})).await;
        assert_eq!(r["result"], json!({}));
        assert!(r.get("error").is_none());
    }

    /// DEV-1 / QA-9: a missing or blank fact is an `isError` result and stores
    /// nothing (it used to store an empty fact and report success).
    #[tokio::test]
    async fn test_store_memory_requires_fact_text() {
        let h = harness().await;
        for args in [
            json!({}),
            json!({ "fact_text": "   " }),
            json!({ "fact_text": 7 }),
        ] {
            let r = call(&h, "store_memory", args.clone()).await;
            assert!(is_error(&r), "{args} should be rejected: {r}");
            assert!(text(&r).contains("fact_text"));
        }
        assert_eq!(stm_count(&h), 0, "nothing may be stored");

        let r = call(
            &h,
            "store_memory",
            json!({ "fact_text": "The sky is blue" }),
        )
        .await;
        assert!(!is_error(&r), "{r}");
        assert_eq!(stm_count(&h), 1);
    }

    /// QA-9: an empty query is a clear argument error, not a leaked vec0/SQL error.
    #[tokio::test]
    async fn test_query_memory_requires_query() {
        let h = harness().await;
        for tool in ["query_memory", "recall_ltm"] {
            let r = call(&h, tool, json!({ "query": "" })).await;
            assert!(is_error(&r), "{tool}: {r}");
            let msg = text(&r);
            assert!(msg.contains("'query' is required"), "{tool}: {msg}");
            assert!(!msg.to_lowercase().contains("sql"), "{tool}: {msg}");
        }
    }

    /// SEC-12 / QA-9: negative or non-integer counts are rejected; huge ones
    /// are clamped instead of being cast to "no limit"; bad dates are errors.
    #[tokio::test]
    async fn test_counts_rejected_or_clamped_and_dates_validated() {
        let h = harness().await;
        let r = call(&h, "query_memory", json!({ "query": "x", "k": -5 })).await;
        assert!(is_error(&r) && text(&r).contains("'k'"), "{r}");
        let r = call(&h, "stm_list", json!({ "limit": 2.5 })).await;
        assert!(is_error(&r) && text(&r).contains("'limit'"), "{r}");
        let r = call(
            &h,
            "query_memory",
            json!({ "query": "x", "time_filter": { "after": "yesterday" } }),
        )
        .await;
        assert!(
            is_error(&r) && text(&r).contains("time_filter.after"),
            "{r}"
        );

        let empty = serde_json::Map::new();
        let mut m = serde_json::Map::new();
        m.insert("k".into(), json!(u64::MAX));
        assert_eq!(Args(&m).count("k", DEFAULT_K, 1, MAX_K), Ok(MAX_K));
        m.insert("k".into(), json!(0));
        assert_eq!(Args(&m).count("k", DEFAULT_K, 1, MAX_K), Ok(1));
        assert_eq!(Args(&empty).count("k", DEFAULT_K, 1, MAX_K), Ok(DEFAULT_K));

        let r = call(&h, "stm_list", json!({ "limit": 1_000_000 })).await;
        assert!(!is_error(&r), "huge limit is clamped, not an error: {r}");
    }

    /// SEC-12: oversized text arguments are refused before touching the LLM/DB.
    #[tokio::test]
    async fn test_oversized_fact_rejected() {
        let h = harness().await;
        let big = "a".repeat(MAX_FACT_BYTES + 1);
        let r = call(&h, "store_memory", json!({ "fact_text": big })).await;
        assert!(is_error(&r) && text(&r).contains("too large"), "{r}");
        assert_eq!(stm_count(&h), 0);
    }

    /// 2.2: workspaces are completely separate memories. A fact stored in
    /// "default" is invisible after switching to "work", and visible again
    /// after switching back.
    #[tokio::test]
    async fn test_workspace_switch_isolates_memory() {
        let h = harness().await;
        let r = call(&h, "workspace_current", json!({})).await;
        let cur: Value = serde_json::from_str(&text(&r)).unwrap();
        assert_eq!(cur["name"], "default", "{r}");

        call(
            &h,
            "store_memory",
            json!({ "fact_text": "lives in default" }),
        )
        .await;
        assert_eq!(stm_count(&h), 1);

        let r = call(&h, "workspace_create", json!({ "name": "work" })).await;
        assert!(!is_error(&r), "{r}");
        let r = call(&h, "workspace_switch", json!({ "name": "work" })).await;
        assert!(!is_error(&r), "{r}");
        assert_eq!(stm_count(&h), 0, "a new workspace shares nothing");
        let r = call(&h, "query_memory", json!({ "query": "lives in default" })).await;
        assert_eq!(text(&r), "[]", "no cross-workspace recall: {r}");

        call(&h, "workspace_switch", json!({ "name": "default" })).await;
        assert_eq!(stm_count(&h), 1);

        let r = call(&h, "workspace_list", json!({})).await;
        let list: Value = serde_json::from_str(&text(&r)).unwrap();
        let names: Vec<&str> = list
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["default", "work"]);
        assert_eq!(list[0]["active"], true);
        assert_eq!(list[1]["active"], false);
    }

    /// 2.2: switching to a missing or badly named workspace is refused, as is
    /// creating one twice.
    #[tokio::test]
    async fn test_workspace_switch_and_create_validation() {
        let h = harness().await;
        let r = call(&h, "workspace_switch", json!({ "name": "nope" })).await;
        assert!(is_error(&r) && text(&r).contains("does not exist"), "{r}");
        let r = call(&h, "workspace_create", json!({ "name": "../escape" })).await;
        assert!(
            is_error(&r) && text(&r).contains("invalid workspace name"),
            "{r}"
        );
        let r = call(&h, "workspace_create", json!({ "name": "default" })).await;
        assert!(is_error(&r) && text(&r).contains("already exists"), "{r}");
        let r = call(&h, "workspace_switch", json!({})).await;
        assert!(is_error(&r) && text(&r).contains("'name'"), "{r}");
    }

    /// 2.2: `[mcp] allow_workspace_switch = false` pins the server.
    #[tokio::test]
    async fn test_workspace_switch_can_be_disabled() {
        let h = harness_with(false).await;
        let r = call(&h, "workspace_switch", json!({ "name": "default" })).await;
        assert!(is_error(&r) && text(&r).contains("disabled"), "{r}");
    }

    /// P2R-4: a pinned session (switching disabled) can't reach any other
    /// workspace — no create, delete, or export of another — but can still
    /// export its own.
    #[tokio::test]
    async fn test_pinned_session_cannot_touch_other_workspaces() {
        let h = harness_with(false).await;
        // Another workspace exists on disk (made by some other session/CLI).
        let home = h._home.path().to_path_buf();
        std::fs::create_dir_all(home.join("workspaces/other")).unwrap();

        let r = call(&h, "workspace_create", json!({ "name": "new" })).await;
        assert!(is_error(&r) && text(&r).contains("pinned"), "{r}");
        let r = call(
            &h,
            "workspace_delete",
            json!({ "name": "other", "confirm": "other" }),
        )
        .await;
        assert!(is_error(&r) && text(&r).contains("pinned"), "{r}");
        assert!(home.join("workspaces/other").exists());
        let r = call(&h, "workspace_export", json!({ "name": "other" })).await;
        assert!(is_error(&r) && text(&r).contains("pinned"), "{r}");

        let r = call(&h, "workspace_export", json!({})).await;
        assert!(!is_error(&r), "own workspace exports: {r}");
        let r = call(&h, "workspace_export", json!({ "name": "default" })).await;
        assert!(!is_error(&r), "{r}");
    }

    /// 2.2 / SEC-07: delete needs confirm == name, and the active workspace
    /// can't be deleted.
    #[tokio::test]
    async fn test_workspace_delete_rules() {
        let h = harness().await;
        call(&h, "workspace_create", json!({ "name": "scratch" })).await;

        let r = call(&h, "workspace_delete", json!({ "name": "scratch" })).await;
        assert!(is_error(&r) && text(&r).contains("confirm"), "{r}");
        let r = call(
            &h,
            "workspace_delete",
            json!({ "name": "scratch", "confirm": "other" }),
        )
        .await;
        assert!(is_error(&r), "{r}");
        let r = call(
            &h,
            "workspace_delete",
            json!({ "name": "default", "confirm": "default" }),
        )
        .await;
        assert!(is_error(&r) && text(&r).contains("active"), "{r}");

        let r = call(
            &h,
            "workspace_delete",
            json!({ "name": "scratch", "confirm": "scratch" }),
        )
        .await;
        assert!(!is_error(&r), "{r}");
        let r = call(&h, "workspace_list", json!({})).await;
        assert!(!text(&r).contains("scratch"), "{r}");
    }

    /// 2.2: export dumps the active (or named) workspace's facts.
    #[tokio::test]
    async fn test_workspace_export() {
        let h = harness().await;
        call(
            &h,
            "store_memory",
            json!({ "fact_text": "exportable fact" }),
        )
        .await;
        let r = call(&h, "workspace_export", json!({})).await;
        assert!(!is_error(&r), "{r}");
        let dump: Value = serde_json::from_str(&text(&r)).unwrap();
        assert_eq!(dump["workspace"], "default");
        assert_eq!(dump["stm_facts"][0]["payload"]["fact"], "exportable fact");
        let r = call(&h, "workspace_export", json!({ "name": "missing" })).await;
        assert!(is_error(&r), "{r}");
    }

    /// 2.5: remember_document files into LTM (upsert by data_id) and needs text.
    #[tokio::test]
    async fn test_remember_document_upserts() {
        let h = harness().await;
        let r = call(&h, "remember_document", json!({ "title": "t" })).await;
        assert!(is_error(&r) && text(&r).contains("'text'"), "{r}");

        let args =
            json!({ "title": "Lease", "text": "Apartment lease for 2026", "data_id": "doc_lease" });
        let r = call(&h, "remember_document", args.clone()).await;
        assert!(!is_error(&r), "{r}");
        let first: Value = serde_json::from_str(&text(&r)).unwrap();
        assert_eq!(first["data_id"], "doc_lease");
        assert_eq!(first["updated"], false);

        let r = call(&h, "remember_document", args).await;
        let second: Value = serde_json::from_str(&text(&r)).unwrap();
        assert_eq!(second["updated"], true, "same data_id replaces: {r}");
    }

    /// 2.2: `tenant_id` is gone from every tool schema, and the old tenant
    /// tools no longer exist.
    #[tokio::test]
    async fn test_no_tenant_surface() {
        let h = harness().await;
        let list = rpc(&h, "tools/list", json!({})).await;
        let rendered = list.to_string();
        assert!(!rendered.contains("tenant"), "tenant leaked into schemas");
        for gone in ["delete_tenant", "export_tenant"] {
            let r = rpc(&h, "tools/call", json!({ "name": gone, "arguments": {} })).await;
            assert_eq!(r["error"]["code"], -32602, "{gone}: {r}");
        }
    }

    /// REV-6: an unknown tool is a JSON-RPC invalid-params error (-32602)
    /// naming the tool, not an `isError` tool result.
    #[tokio::test]
    async fn test_unknown_tool_is_protocol_error() {
        let h = harness().await;
        let r = rpc(&h, "tools/call", json!({ "name": "nope", "arguments": {} })).await;
        assert_eq!(r["error"]["code"], -32602, "{r}");
        assert!(r["error"]["message"].as_str().unwrap().contains("nope"));
        assert!(r.get("result").is_none());
    }

    /// DEV-15: the schemas state the real tenant default, declare the params
    /// the dispatcher accepts, and list exactly the dispatchable tools.
    #[tokio::test]
    async fn test_tool_schemas_match_dispatcher() {
        let h = harness().await;
        let list = rpc(&h, "tools/list", json!({})).await["result"]["tools"].clone();
        let tools = list.as_array().unwrap();

        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(
            names, TOOL_NAMES,
            "tools/list and the dispatcher must agree"
        );

        let rendered = list.to_string();
        assert!(
            !rendered.contains("Defaults to 'default'"),
            "a schema still claims the tenant default is 'default'"
        );

        let props = |name: &str| {
            tools.iter().find(|t| t["name"] == name).unwrap()["inputSchema"]["properties"].clone()
        };
        for (tool, param) in [
            ("push_dialogue", "ccl"),
            ("store_memory", "ccl"),
            ("query_memory", "ccl_filter"),
            ("recall_ltm", "k"),
            ("workspace_delete", "confirm"),
        ] {
            assert!(
                props(tool).get(param).is_some(),
                "{tool} must declare '{param}'"
            );
        }

        // Every listed tool dispatches (arg errors are fine; "Unknown" is not).
        for name in TOOL_NAMES {
            let r = rpc(&h, "tools/call", json!({ "name": name, "arguments": {} })).await;
            assert!(r.get("error").is_none(), "{name} not dispatched: {r}");
        }
    }

    /// SEC-12: a line over the cap is rejected (-32600) without killing the
    /// loop; the next request is still served.
    #[tokio::test]
    async fn test_oversized_line_rejected_and_loop_continues() {
        let h = harness().await;
        let mut input = Vec::new();
        input.extend_from_slice(
            br#"{"id":41,"jsonrpc":"2.0","method":"tools/call","params":{"x":""#,
        );
        input.extend(std::iter::repeat_n(b'x', MAX_LINE_BYTES + 10));
        input.extend_from_slice(b"\"}}\n");
        input.extend_from_slice(br#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#);
        input.push(b'\n');

        let mut out: Vec<u8> = Vec::new();
        h.server
            .serve(BufReader::new(&input[..]), &mut out)
            .await
            .unwrap();
        let lines: Vec<Value> = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[0]["error"]["code"], -32600);
        assert_eq!(
            lines[0]["id"], 41,
            "the oversized request's id is recovered"
        );
        assert_eq!(lines[1]["id"], 2);
        assert_eq!(lines[1]["result"], json!({}));
    }

    /// REV-5: only a top-level `id` is recovered. With `params` before `id`, a
    /// nested `arguments.id` comes first in the byte stream and must be skipped;
    /// if the real id was truncated away, the answer is null (never a wrong id).
    #[test]
    fn test_probe_request_id_ignores_nested_ids() {
        let nested_first =
            br#"{"jsonrpc":"2.0","params":{"name":"inspect_node","arguments":{"id":999}},"id":5}"#;
        assert_eq!(probe_request_id(nested_first), json!(5));

        let truncated = br#"{"jsonrpc":"2.0","params":{"arguments":{"id":999,"text":"xxxxx"#;
        assert_eq!(probe_request_id(truncated), Value::Null);

        // A string containing `"id":` (escaped quotes) is not a key.
        let in_string = br#"{"method":"x "id": 3","id":8}"#;
        assert_eq!(probe_request_id(in_string), json!(8));
    }

    #[test]
    fn test_probe_request_id() {
        assert_eq!(probe_request_id(br#"{"jsonrpc":"2.0","id": 7,"#), json!(7));
        assert_eq!(probe_request_id(br#"{"id":"abc","method""#), json!("abc"));
        assert_eq!(probe_request_id(b"xxxxxxxx"), Value::Null);
    }

    /// Notifications (no id) get no reply; a final line without `\n` is served.
    #[tokio::test]
    async fn test_notifications_silent_and_unterminated_last_line() {
        let h = harness().await;
        let input = b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}";
        let mut out: Vec<u8> = Vec::new();
        h.server
            .serve(BufReader::new(&input[..]), &mut out)
            .await
            .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(text.lines().count(), 1, "{text}");
        assert!(text.contains("\"id\":9"));
    }
}
