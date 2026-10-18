//! Tool dispatch: argument parsing plus the call into the active workspace's
//! services. Transport-free, so it is unit-tested directly.

use crate::application::introspection::LeafPage;
use crate::application::ltm_retrieval::RECALL_MAX_K;
use crate::application::query_service::{QueryRequest, QueryScope};
use crate::application::workspace::WorkspaceManager;
use crate::domain::models::WORKSPACE_TENANT;
use crate::interfaces::bus_query::flatten_recall;
use serde::Serialize;
use serde_json::Value;

use super::args::{ArgResult, Args};
use super::limits::*;
use super::registry::{self, INSPECT_VIEWS};

/// What a tool call produced.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Success with a JSON payload.
    Json(Value),
    /// Success with a plain-text message.
    Text(String),
    /// A tool execution error (`isError: true`), already user-facing.
    Error(String),
}

impl Outcome {
    fn json<T: Serialize>(result: anyhow::Result<T>, context: &str) -> Self {
        match result {
            Ok(value) => match serde_json::to_value(value) {
                Ok(v) => Outcome::Json(v),
                Err(e) => Outcome::Error(format!("{context}: could not encode the result: {e}")),
            },
            Err(e) if context.is_empty() => Outcome::Error(format!("{e}")),
            Err(e) => Outcome::Error(format!("{context}: {e}")),
        }
    }
}

/// Run one known tool against the active workspace. `Err` is a caller
/// mistake (bad or missing argument); operational failures come back as
/// [`Outcome::Error`].
pub async fn call(manager: &WorkspaceManager, tool: &str, args: &Args<'_>) -> ArgResult<Outcome> {
    // Resolve the active workspace once per call (a switch mid-call can't
    // split one operation across two workspaces).
    let ws = manager.current();
    let svc = &ws.services;
    Ok(match tool {
        "store_memory" => {
            // Explicit fact storage (bypasses the Sleep pipeline).
            let fact_text = args.req_str("fact_text", MAX_FACT_BYTES)?;
            let tags = args.str_list("tags", MAX_TAGS)?;
            let ccl = args.opt_str("ccl", MAX_ID_BYTES)?.unwrap_or("reality");
            match svc
                .app
                .store_explicit_fact(WORKSPACE_TENANT, fact_text, &tags, ccl)
                .await
            {
                Ok(()) => Outcome::Text("Memory fact explicitly stored.".into()),
                Err(e) => Outcome::Error(format!("Failed to store memory: {e}")),
            }
        }
        "push_dialogue" => {
            // Push dialogue to STM, compress, return the optimized context.
            let session_id = args.req_str("session_id", MAX_ID_BYTES)?;
            let new_message = args.req_str("new_message", MAX_MESSAGE_BYTES)?;
            let ccl = args.opt_str("ccl", MAX_ID_BYTES)?.unwrap_or("reality");
            Outcome::json(
                svc.app
                    .push_dialogue(WORKSPACE_TENANT, session_id, new_message, ccl)
                    .await,
                "Failed to process dialogue",
            )
        }
        "query_memory" => {
            // STM recall over the shared QueryService (same path as the bus door).
            let req = query_request(args, QueryScope::Stm)?;
            Outcome::json(svc.query.execute(&req).await.map(|o| o.stm), "Query failed")
        }
        "recall_ltm" => {
            // Reference-returning search of the permanent archive.
            let req = query_request(args, QueryScope::Ltm)?;
            Outcome::json(
                svc.query
                    .execute(&req)
                    .await
                    .map(|o| o.ltm.iter().flat_map(flatten_recall).collect::<Vec<_>>()),
                "LTM recall failed",
            )
        }
        "remember_document" => {
            // File a document/note into the LTM tree (upsert by data_id).
            let doc = crate::application::documents::RememberDocument {
                title: args
                    .opt_str("title", MAX_QUERY_BYTES)?
                    .unwrap_or_default()
                    .to_string(),
                text: args.req_str("text", MAX_DOCUMENT_BYTES)?.to_string(),
                data_id: args.opt_str("data_id", MAX_ID_BYTES)?.map(str::to_string),
                tags: args.str_list("tags", MAX_TAGS)?,
            };
            Outcome::json(svc.documents.remember(doc).await, "")
        }
        // --- workspaces (physically separate memories) ---
        "workspace_current" => Outcome::json(manager.current_info(), ""),
        "workspace_list" => Outcome::json(manager.list(), ""),
        "workspace_create" => {
            let name = args.req_str("name", MAX_ID_BYTES)?;
            Outcome::json(manager.create(name), "")
        }
        "workspace_switch" => {
            let name = args.req_str("name", MAX_ID_BYTES)?;
            // Drop our handle on the old workspace before its stores close.
            drop(ws);
            Outcome::json(manager.switch(name).await, "")
        }
        "workspace_export" => {
            let name = args.opt_str("name", MAX_ID_BYTES)?;
            Outcome::json(manager.export(name), "")
        }
        "workspace_delete" => {
            // Destructive: `confirm` must repeat the name; the active
            // workspace can't be deleted.
            let name = args.req_str("name", MAX_ID_BYTES)?;
            let confirm = args.opt_str("confirm", MAX_ID_BYTES)?;
            match manager.delete(name, confirm) {
                Ok(()) => Outcome::Text(format!("Deleted workspace {name:?}.")),
                Err(e) => Outcome::Error(format!("{e}")),
            }
        }
        "inspect" => inspect(svc, args)?,
        other => return Err(format!("Unknown tool: {other}")),
    })
}

/// A [`QueryRequest`] from tool arguments, with the shared defaults
/// (`reality` layer, breadth [`DEFAULT_K`]). Isolation comes from the
/// workspace, so the tenant is always [`WORKSPACE_TENANT`]. `query` is
/// required: an empty query used to reach sqlite-vec and leak its SQL error
/// (QA-9).
fn query_request(args: &Args, scope: QueryScope) -> ArgResult<QueryRequest> {
    // LTM recall returns at most RECALL_MAX_K hits; say so in the schema
    // rather than silently returning fewer.
    let max_k = match scope {
        QueryScope::Ltm => RECALL_MAX_K,
        _ => MAX_K,
    };
    Ok(QueryRequest {
        scope,
        tenant: WORKSPACE_TENANT.to_string(),
        query: args.req_str("query", MAX_QUERY_BYTES)?.to_string(),
        k: args.count("k", DEFAULT_K, 1, max_k)?,
        time_filter: args.time_filter()?,
        ccl: args.str_list("ccl_filter", MAX_CCL_FILTER)?,
        context_key: None,
    })
}

/// The read-only `inspect` tool: one view of the active workspace.
fn inspect(
    svc: &crate::application::workspace::WorkspaceServices,
    args: &Args,
) -> ArgResult<Outcome> {
    let names = || {
        INSPECT_VIEWS
            .iter()
            .map(|v| v.name)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let view_name = args
        .opt_str("view", MAX_ID_BYTES)?
        .ok_or_else(|| format!("'view' is required: one of {}", names()))?;
    let view = registry::find_view(view_name)
        .ok_or_else(|| format!("unknown view {view_name:?}: expected one of {}", names()))?;
    // An argument of another view is almost certainly a mix-up (e.g. `node`
    // for subtree's `id`): say so instead of silently ignoring it. Unknown
    // keys are ignored, as on every other tool.
    for arg in args.present() {
        if arg != "view"
            && !view.params.contains(&arg)
            && INSPECT_VIEWS.iter().any(|v| v.params.contains(&arg))
        {
            return Err(format!(
                "'{arg}' does not apply to view {:?} (it takes: {})",
                view.name,
                if view.params.is_empty() {
                    "no arguments".to_string()
                } else {
                    view.params.join(", ")
                }
            ));
        }
    }
    let intro = &svc.introspection;
    const CTX: &str = "inspect failed";
    Ok(match view.name {
        "stats" => Outcome::json(intro.memory_stats(), CTX),
        "health" => Outcome::json(intro.health(), CTX),
        "stm" => {
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
            Outcome::json(intro.stm_list(limit, offset, status, contains), CTX)
        }
        "ltm_map" => {
            let depth = args.count("depth", 3, 1, MAX_DEPTH)?;
            Outcome::json(intro.ltm_map(depth), CTX)
        }
        "node" => {
            let id = args.req_id("id")?;
            let page = LeafPage {
                child_limit: args.opt_count("child_limit", 1, MAX_CHILD_LIMIT)?,
                child_offset: args.count("child_offset", 0, 0, MAX_OFFSET)?,
                summary_max_chars: args.opt_count("summary_max_chars", 0, MAX_SUMMARY_CHARS)?,
            };
            match intro.inspect_node(id, page) {
                Ok(None) => Outcome::Error(format!("no LTM node with id {id}")),
                other => Outcome::json(other, CTX),
            }
        }
        "subtree" => {
            let id = args.req_id("id")?;
            let depth = args.count("depth", 2, 1, MAX_DEPTH)?;
            match intro.subtree(id, depth) {
                Ok(None) => Outcome::Error(format!("no LTM node with id {id}")),
                other => Outcome::json(other, CTX),
            }
        }
        "trace" => {
            let data_id = args.req_str("data_id", MAX_ID_BYTES)?;
            Outcome::json(intro.trace_data_id(data_id), CTX)
        }
        "placement" => {
            let sample = args.count("sample", 30, 1, MAX_SAMPLE)?;
            Outcome::json(intro.placement_debug(sample), CTX)
        }
        other => return Err(format!("unknown view {other:?}")),
    })
}
