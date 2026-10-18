//! The MCP server, on the official Rust SDK (`rmcp`).
//!
//! - [`registry`] — the single tool registry (names, descriptions, schemas,
//!   annotations, `inspect` views); `tools/list` and the generated docs come
//!   from it.
//! - [`tools`] — argument parsing + dispatch into the active workspace.
//! - [`transport`] — bounded newline-delimited JSON-RPC over stdio.
//! - [`docs`] — the Markdown tool reference (`neurolithe tools`).
//!
//! rmcp negotiates the protocol version (every revision it knows, including
//! the `initialize`-less 2026-07-28 lifecycle) and answers `initialize`,
//! `ping` and `tools/list` without touching the stores, so a client never
//! times out waiting on a model load. Tool calls wait for the workspace.

pub mod args;
pub mod docs;
pub mod limits;
pub mod registry;
pub mod tools;
pub mod transport;

use crate::application::workspace::WorkspaceManager;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
    ListToolsResult, PaginatedRequestParams, ProtocolVersion, ServerCapabilities, ServerConfig,
    Tool, ToolsCapability,
};
use rmcp::service::{RequestContext, RoleServer, ServerInitializeError, serve_server};
use rmcp::{ErrorData as McpError, ServerHandler};
use serde_json::Value;
use std::cell::RefCell;
use std::rc::Rc;
use tokio::io::{AsyncBufRead, AsyncWrite};

use args::Args;
use tools::Outcome;
use transport::LineTransport;

/// Short guidance for the agent, sent in the `initialize` result.
const INSTRUCTIONS: &str = "NeuroLithe is your memory. Short-term memory: push_dialogue \
    (each conversation turn), store_memory (one explicit fact), query_memory (search recent \
    facts). Long-term archive: remember_document (file a document or note), recall_ltm (find \
    one). Each workspace is a separate memory; workspace_* tools manage them. inspect is a \
    read-only look inside the stores, for debugging.";

/// The MCP server. A cheap handle: clones share the same state.
#[derive(Clone)]
pub struct McpServer {
    inner: Rc<Inner>,
}

struct Inner {
    /// The active workspace and its services, once startup has opened it.
    /// Every tool call resolves the services through it, so `workspace_switch`
    /// takes effect immediately. `None` while the workspace is still
    /// starting (embedder probe, store migrations); tool calls wait for
    /// [`McpServer::set_workspaces`].
    workspaces: RefCell<Option<Rc<WorkspaceManager>>>,
    ready: tokio::sync::Notify,
    /// `[mcp] allow_workspace_switch`: decides which tools are listed.
    allow_switch: bool,
    /// Tool calls run one at a time, as before the SDK migration: the
    /// SQLite-backed services were written for that, and some flows
    /// (remember_document's upsert, a switch) aren't safe to interleave.
    one_at_a_time: tokio::sync::Mutex<()>,
    /// Tool calls accepted and not yet finished (including those queued on
    /// `one_at_a_time`); see [`McpServer::drain_calls`].
    in_flight: std::cell::Cell<usize>,
    calls_done: tokio::sync::Notify,
    /// Set once shutdown has begun: no new tool call starts.
    closing: std::cell::Cell<bool>,
    /// Replies the transport still owes the client.
    owed: std::sync::Arc<transport::Owed>,
}

/// Counts one accepted tool call for as long as it runs.
struct InFlight<'a>(&'a Inner);

impl<'a> InFlight<'a> {
    fn enter(inner: &'a Inner) -> Self {
        inner.in_flight.set(inner.in_flight.get() + 1);
        Self(inner)
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        let left = self.0.in_flight.get() - 1;
        self.0.in_flight.set(left);
        if left == 0 {
            self.0.calls_done.notify_waiters();
        }
    }
}

/// The reply to a call that arrives (or reaches its turn) after shutdown began.
const SHUTTING_DOWN: &str = "NeuroLithe is shutting down; the call was not run.";

impl McpServer {
    /// A server over an already-open workspace.
    pub fn new(workspaces: Rc<WorkspaceManager>, allow_switch: bool) -> Self {
        let server = Self::pending(allow_switch);
        server.set_workspaces(workspaces);
        server
    }

    /// A server whose workspace is still starting; see [`set_workspaces`](Self::set_workspaces).
    pub fn pending(allow_switch: bool) -> Self {
        Self {
            inner: Rc::new(Inner {
                workspaces: RefCell::new(None),
                ready: tokio::sync::Notify::new(),
                allow_switch,
                one_at_a_time: tokio::sync::Mutex::new(()),
                in_flight: std::cell::Cell::new(0),
                calls_done: tokio::sync::Notify::new(),
                closing: std::cell::Cell::new(false),
                owed: std::sync::Arc::default(),
            }),
        }
    }

    /// Hand over the opened workspace; releases any waiting tool calls.
    pub fn set_workspaces(&self, workspaces: Rc<WorkspaceManager>) {
        *self.inner.workspaces.borrow_mut() = Some(workspaces);
        self.inner.ready.notify_waiters();
    }

    /// The workspace manager, waiting for startup if needed.
    async fn workspaces(&self) -> Rc<WorkspaceManager> {
        loop {
            // Register before checking so a concurrent set can't be missed.
            let notified = self.inner.ready.notified();
            if let Some(ws) = self.inner.workspaces.borrow().clone() {
                return ws;
            }
            notified.await;
        }
    }

    /// Serve MCP over the process's stdin/stdout until stdin closes.
    pub async fn run_stdio(&self) -> anyhow::Result<()> {
        self.serve(
            tokio::io::BufReader::new(tokio::io::stdin()),
            tokio::io::stdout(),
        )
        .await
    }

    /// Serve newline-delimited JSON-RPC from `reader` to `writer` until EOF.
    /// Generic so tests can drive the real server in memory. Must run inside
    /// a `LocalSet` (request handlers are spawned locally).
    pub async fn serve<R, W>(&self, reader: R, writer: W) -> anyhow::Result<()>
    where
        R: AsyncBufRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let transport = LineTransport::new(reader, writer).with_owed(self.inner.owed.clone());
        let running = match serve_server(self.clone(), transport).await {
            Ok(running) => running,
            // The client went away before (or instead of) starting a session.
            Err(ServerInitializeError::ConnectionClosed(_)) => return Ok(()),
            Err(e) => return Err(anyhow::anyhow!("MCP session failed to start: {e}")),
        };
        let reason = running.waiting().await?;
        tracing::debug!("MCP session ended: {reason:?}");
        Ok(())
    }

    /// Begin shutdown: refuse new tool calls, and wait (up to `budget`) for
    /// the running ones to finish and for every owed reply (including the
    /// "shutting down" ones) to be written, so the work is complete and the
    /// client told before the stores are checkpointed. `false` when the
    /// budget ran out first.
    pub async fn drain_calls(&self, budget: std::time::Duration) -> bool {
        self.inner.closing.set(true);
        let drained = tokio::time::timeout(budget, async {
            loop {
                let done = self.inner.calls_done.notified();
                tokio::pin!(done);
                done.as_mut().enable();
                if self.inner.in_flight.get() == 0 {
                    break;
                }
                done.await;
            }
            self.inner.owed.all_settled().await;
        })
        .await
        .is_ok();
        if !drained {
            tracing::warn!(
                "after {budget:?}: {} tool call(s) still running, {} repl(ies) unsent; \
                 shutting down anyway",
                self.inner.in_flight.get(),
                self.inner.owed.count()
            );
        }
        drained
    }

    /// Run one tool call; protocol-level problems (an unknown tool) are `Err`.
    pub async fn call(
        &self,
        name: &str,
        arguments: Option<&serde_json::Map<String, Value>>,
    ) -> Result<Outcome, McpError> {
        // An unknown tool is a protocol error (invalid params), not a tool
        // result — per the MCP spec (REV-6).
        if registry::find(name).is_none() {
            return Err(McpError::invalid_params(
                format!("Unknown tool: {name}"),
                None,
            ));
        }
        if self.inner.closing.get() {
            return Ok(Outcome::Error(SHUTTING_DOWN.into()));
        }
        let _in_flight = InFlight::enter(&self.inner);
        let manager = self.workspaces().await;
        let _turn = self.inner.one_at_a_time.lock().await;
        // Shutdown may have begun while this call waited for its turn.
        if self.inner.closing.get() {
            return Ok(Outcome::Error(SHUTTING_DOWN.into()));
        }
        let empty = serde_json::Map::new();
        let args = Args(arguments.unwrap_or(&empty));
        Ok(tools::call(&manager, name, &args)
            .await
            .unwrap_or_else(Outcome::Error))
    }
}

/// Render an [`Outcome`] as a `tools/call` result. JSON payloads are sent as
/// text (every client reads that) and, to peers on 2025-06-18 or later, also
/// as `structuredContent` — an object, so a top-level array is wrapped as
/// `{"items": [...]}`.
pub fn render(outcome: Outcome, structured: bool) -> CallToolResult {
    match outcome {
        Outcome::Json(value) => {
            let text = value.to_string();
            let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
            if structured {
                result.structured_content = match value {
                    Value::Object(_) => Some(value),
                    Value::Array(_) => Some(serde_json::json!({ "items": value })),
                    _ => None,
                };
            }
            result
        }
        Outcome::Text(text) => CallToolResult::success(vec![ContentBlock::text(text)]),
        Outcome::Error(text) => CallToolResult::error(vec![ContentBlock::text(text)]),
    }
}

/// Whether a protocol revision has `structuredContent` (2025-06-18+).
fn has_structured_content(version: Option<&ProtocolVersion>) -> bool {
    version.is_some_and(|v| v.as_str() >= ProtocolVersion::V_2025_06_18.as_str())
}

impl ServerHandler for McpServer {
    fn get_info(&self) -> ServerConfig {
        let mut tools = ToolsCapability::default();
        // The list is fixed for the session; we never send list_changed.
        tools.list_changed = Some(false);
        let mut capabilities = ServerCapabilities::default();
        capabilities.tools = Some(tools);
        ServerConfig::new(capabilities)
            .with_server_info(Implementation::new("NeuroLithe", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(
            registry::listed(self.inner.allow_switch)
                .map(registry::ToolDef::to_tool)
                .collect(),
        ))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        registry::listed(self.inner.allow_switch)
            .find(|t| t.name == name)
            .map(registry::ToolDef::to_tool)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let structured = has_structured_content(context.protocol_version().as_ref());
        let outcome = self.call(&request.name, request.arguments.as_ref()).await?;
        Ok(render(outcome, structured).into())
    }
}

#[cfg(test)]
mod tests;
