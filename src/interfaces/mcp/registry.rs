//! The tool registry: the single source of every MCP tool's name, title,
//! description, input schema and annotations, and of the `inspect` views.
//!
//! `tools/list`, the dispatcher's name check, the `neurolithe tools` command
//! and the generated reference page (`docs/src/mcp/tools.md`) are all built
//! from these tables, so they can't drift apart.

use rmcp::model::{JsonObject, Tool, ToolAnnotations};
use serde_json::{Value, json};

use super::limits::*;
use crate::application::ltm_retrieval::RECALL_MAX_K;

/// Which part of the docs a tool belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Memory,
    Workspace,
    Inspect,
}

impl Group {
    pub const ALL: [Group; 3] = [Group::Memory, Group::Workspace, Group::Inspect];

    pub fn heading(self) -> &'static str {
        match self {
            Group::Memory => "Memory tools",
            Group::Workspace => "Workspace tools",
            Group::Inspect => "Inspection",
        }
    }
}

/// MCP tool annotations (hints for the client; never relied on for safety).
#[derive(Debug, Clone, Copy)]
pub struct Hints {
    pub read_only: bool,
    /// Only meaningful when `read_only` is false.
    pub destructive: bool,
    pub idempotent: bool,
}

const READ_ONLY: Hints = Hints {
    read_only: true,
    destructive: false,
    idempotent: true,
};

const ADDITIVE: Hints = Hints {
    read_only: false,
    destructive: false,
    idempotent: false,
};

/// One MCP tool.
pub struct ToolDef {
    pub name: &'static str,
    pub title: &'static str,
    pub group: Group,
    pub description: &'static str,
    /// JSON Schema of the arguments (always an object schema).
    pub input_schema: fn() -> Value,
    pub hints: Hints,
    /// Listed only when the server may reach other workspaces
    /// (`[mcp] allow_workspace_switch = true`). A pinned server hides it from
    /// `tools/list`; calling it anyway still gets a clear `isError` result.
    pub needs_switching: bool,
}

/// One `inspect` view.
pub struct ViewDef {
    pub name: &'static str,
    pub description: &'static str,
    /// The `inspect` arguments this view reads (besides `view`).
    pub params: &'static [&'static str],
}

/// The tool surface, in `tools/list` order.
pub const TOOLS: &[ToolDef] = &[
    ToolDef {
        name: "push_dialogue",
        title: "Push a dialogue turn",
        group: Group::Memory,
        description: "Push the latest conversation turn to short-term memory. The turn is archived, \
            facts are extracted from it (when a chat LLM is configured), and the optimized context \
            window is returned: {summary, recent_messages, relevant_facts, learning_error?, \
            warnings?}. Extraction problems are reported in the result, not as a failure; don't \
            retry because of them (the turn is already archived).",
        input_schema: push_dialogue_schema,
        hints: ADDITIVE,
        needs_switching: false,
    },
    ToolDef {
        name: "store_memory",
        title: "Store a fact",
        group: Group::Memory,
        description: "Store one crucial fact in short-term memory immediately, bypassing the \
            extraction pipeline.",
        input_schema: store_memory_schema,
        hints: ADDITIVE,
        needs_switching: false,
    },
    ToolDef {
        name: "query_memory",
        title: "Search short-term memory",
        group: Group::Memory,
        description: "Search SHORT-TERM working memory (recent, decaying facts) for relevant \
            context. Hybrid keyword + semantic search; returns facts (best first) with 1-hop \
            connections and temporal bounds. Reading a fact reinforces it (its relevance is \
            restored and its decay clock restarts). For the permanent document archive, use \
            recall_ltm.",
        input_schema: query_memory_schema,
        // Not read-only: recall reinforces the facts it returns. That write
        // is harmless and repeating it changes nothing further.
        hints: Hints {
            read_only: false,
            destructive: false,
            idempotent: true,
        },
        needs_switching: false,
    },
    ToolDef {
        name: "recall_ltm",
        title: "Search the long-term archive",
        group: Group::Memory,
        description: "Search the PERMANENT long-term archive (all filed documents and notes) by \
            meaning. Reference-returning: each hit carries the document's dataId, provenance and \
            ancestor concepts, so you can fetch the original. The primary tool for finding a \
            document, receipt, letter, or report.",
        input_schema: recall_ltm_schema,
        hints: READ_ONLY,
        needs_switching: false,
    },
    ToolDef {
        name: "remember_document",
        title: "File a document",
        group: Group::Memory,
        description: "File a document or note into the PERMANENT long-term archive: it is \
            summarized (if a chat LLM is configured), embedded, and placed under the best-matching \
            concept. Returns {data_id, leaf_id, concept_path, updated, summarized}. Passing the \
            data_id of an existing document replaces that document (upsert).",
        input_schema: remember_document_schema,
        // Without a data_id every call files a new document, so not idempotent.
        // A replacement happens only for a data_id the caller names explicitly.
        hints: ADDITIVE,
        needs_switching: false,
    },
    ToolDef {
        name: "workspace_current",
        title: "Active workspace",
        group: Group::Workspace,
        description: "The active workspace: {name, path, stm_bytes, ltm_bytes, active}. Each \
            workspace is a completely separate memory (its own STM and LTM stores); every memory \
            tool reads and writes the active one.",
        input_schema: empty_schema,
        hints: READ_ONLY,
        needs_switching: false,
    },
    ToolDef {
        name: "workspace_list",
        title: "List workspaces",
        group: Group::Workspace,
        description: "All workspaces with their store sizes, sorted by name; the active one has \
            active=true.",
        input_schema: empty_schema,
        hints: READ_ONLY,
        needs_switching: false,
    },
    ToolDef {
        name: "workspace_create",
        title: "Create a workspace",
        group: Group::Workspace,
        description: "Create a new, empty workspace (a separate memory) and return its info. \
            Does not switch to it.",
        input_schema: name_required_schema,
        hints: Hints {
            read_only: false,
            destructive: false,
            idempotent: true,
        },
        needs_switching: true,
    },
    ToolDef {
        name: "workspace_switch",
        title: "Switch workspace",
        group: Group::Workspace,
        description: "Make another existing workspace active and return its info. Its stores \
            are opened and the session buffers start empty; all memory tools then read and write \
            that workspace.",
        input_schema: name_required_schema,
        hints: Hints {
            read_only: false,
            destructive: false,
            idempotent: true,
        },
        needs_switching: true,
    },
    ToolDef {
        name: "workspace_export",
        title: "Export a workspace",
        group: Group::Workspace,
        description: "JSON dump of a workspace's STM facts and LTM document leaves: \
            {workspace, stm_facts, ltm_leaves}. Defaults to the active workspace.",
        input_schema: name_optional_schema,
        hints: READ_ONLY,
        needs_switching: false,
    },
    ToolDef {
        name: "workspace_delete",
        title: "Delete a workspace",
        group: Group::Workspace,
        description: "DESTRUCTIVE: permanently delete a workspace and all of its memory. \
            Requires confirm equal to name. The active workspace cannot be deleted; switch away \
            first.",
        input_schema: workspace_delete_schema,
        hints: Hints {
            read_only: false,
            destructive: true,
            idempotent: true,
        },
        needs_switching: true,
    },
    ToolDef {
        name: "inspect",
        title: "Inspect memory",
        group: Group::Inspect,
        description: "Read-only look inside the active workspace's memory (a \"CT scan\"); never \
            calls an LLM. Pick what to see with `view`: stats (full metrics of both stores), \
            health (compact summary), stm (list STM facts), ltm_map (top layers of the LTM \
            concept tree), node (one LTM node with its children and documents), subtree (a branch \
            of the LTM tree), trace (where a document lives, by data_id), placement (distance \
            calibration for the placement thresholds). Arguments that belong to another view are \
            rejected.",
        input_schema: inspect_schema,
        hints: READ_ONLY,
        needs_switching: false,
    },
];

/// The `inspect` views, in schema-enum order. Each view's `params` is the
/// exact set of arguments it reads; the dispatcher rejects the others.
pub const INSPECT_VIEWS: &[ViewDef] = &[
    ViewDef {
        name: "stats",
        description: "Full metrics snapshot of both stores: STM active/archived counts, average \
            relevance, decay histogram; LTM tree nodes, leaves, edges, inbox size, orphan leaves, \
            max depth; store sizes.",
        params: &[],
    },
    ViewDef {
        name: "health",
        description: "Compact summary: STM/LTM counts, orphan leaves, store sizes, feeder_lag \
            (-1 = unknown here; Kafka mode publishes live lag on its metrics topic).",
        params: &[],
    },
    ViewDef {
        name: "stm",
        description: "STM facts, most relevant first, with score, status, support count, layer, \
            last access and data_id. Paged, with optional status and substring filters.",
        params: &["limit", "offset", "status", "contains"],
    },
    ViewDef {
        name: "ltm_map",
        description: "The top concept layers of the LTM tree (a table of contents).",
        params: &["depth"],
    },
    ViewDef {
        name: "node",
        description: "One LTM node: summary, parents, children and document leaves (paged), plus \
            child_count / leaf_count totals.",
        params: &["id", "child_limit", "child_offset", "summary_max_chars"],
    },
    ViewDef {
        name: "subtree",
        description: "A branch of concepts below an LTM node.",
        params: &["id", "depth"],
    },
    ViewDef {
        name: "trace",
        description: "Where a document lives: its LTM leaf and ancestor branch, and how many STM \
            facts carry its data_id.",
        params: &["data_id"],
    },
    ViewDef {
        name: "placement",
        description: "{thresholds, probes}: the effective distance thresholds with their source \
            (config / model_default / fallback), and for a sample of document leaves the distance \
            to their nearest concept. Used to tune the placement thresholds.",
        params: &["sample"],
    },
];

/// Look a tool up by name.
pub fn find(name: &str) -> Option<&'static ToolDef> {
    TOOLS.iter().find(|t| t.name == name)
}

/// Look an `inspect` view up by name.
pub fn find_view(name: &str) -> Option<&'static ViewDef> {
    INSPECT_VIEWS.iter().find(|v| v.name == name)
}

/// The tools a server lists: all of them, or — when it's pinned to its
/// workspace — only those that can't reach another workspace.
pub fn listed(allow_switch: bool) -> impl Iterator<Item = &'static ToolDef> {
    TOOLS
        .iter()
        .filter(move |t| allow_switch || !t.needs_switching)
}

impl ToolDef {
    /// The `tools/list` entry.
    pub fn to_tool(&self) -> Tool {
        let schema: JsonObject = match (self.input_schema)() {
            Value::Object(map) => map,
            other => panic!("{}: input schema is not an object: {other}", self.name),
        };
        let mut annotations = ToolAnnotations::with_title(self.title)
            .read_only(self.hints.read_only)
            .idempotent(self.hints.idempotent)
            // Everything stays inside the local stores.
            .open_world(false);
        if !self.hints.read_only {
            annotations = annotations.destructive(self.hints.destructive);
        }
        Tool::new(self.name, self.description, schema)
            .with_title(self.title)
            .with_annotations(annotations)
    }
}

// ---------------------------------------------------------------------------
// Input schemas
// ---------------------------------------------------------------------------

/// The workspace-name rule, as a schema `pattern` (kept in step with
/// `validate_name` by a test).
pub const WORKSPACE_NAME_PATTERN: &str = "^[a-z0-9][a-z0-9_-]{0,63}$";

fn name_prop() -> Value {
    json!({
        "type": "string",
        "pattern": WORKSPACE_NAME_PATTERN,
        "description": "Workspace name: 1-64 chars of a-z, 0-9, '_' or '-', starting with a letter or digit."
    })
}

fn ccl_prop() -> Value {
    json!({
        "type": "string",
        "maxLength": MAX_ID_BYTES,
        "description": "Cognitive context layer. Defaults to 'reality'."
    })
}

fn tags_prop(description: &str) -> Value {
    json!({
        "type": "array",
        "items": { "type": "string", "maxLength": MAX_ID_BYTES },
        "maxItems": MAX_TAGS,
        "description": description
    })
}

fn k_prop(max: usize) -> Value {
    json!({
        "type": "integer",
        "minimum": 1,
        "maximum": max,
        "description": format!("Max results to return; larger values are clamped. Defaults to {DEFAULT_K}.")
    })
}

fn object(properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required
    })
}

fn empty_schema() -> Value {
    object(json!({}), &[])
}

fn push_dialogue_schema() -> Value {
    object(
        json!({
            "session_id": { "type": "string", "maxLength": MAX_ID_BYTES, "description": "The conversation's session id." },
            "new_message": { "type": "string", "maxLength": MAX_MESSAGE_BYTES, "description": "The new dialogue turn." },
            "ccl": ccl_prop()
        }),
        &["session_id", "new_message"],
    )
}

fn store_memory_schema() -> Value {
    object(
        json!({
            "fact_text": { "type": "string", "maxLength": MAX_FACT_BYTES, "description": "The factual statement to store." },
            "tags": tags_prop("Tags for categorizing the fact."),
            "ccl": ccl_prop()
        }),
        &["fact_text"],
    )
}

fn query_memory_schema() -> Value {
    object(
        json!({
            "query": { "type": "string", "maxLength": MAX_QUERY_BYTES, "description": "What to look for." },
            "k": k_prop(MAX_K),
            "time_filter": {
                "type": "object",
                "description": "Optional temporal bounds.",
                "properties": {
                    "after": { "type": "string", "description": "Only facts after this date (YYYY-MM-DD)." },
                    "before": { "type": "string", "description": "Only facts before this date (YYYY-MM-DD)." }
                }
            },
            "ccl_filter": {
                "type": "array",
                "items": { "type": "string", "maxLength": MAX_ID_BYTES },
                "maxItems": MAX_CCL_FILTER,
                "description": "Cognitive layers to search. Defaults to ['reality']."
            }
        }),
        &["query"],
    )
}

fn recall_ltm_schema() -> Value {
    object(
        json!({
            "query": { "type": "string", "maxLength": MAX_QUERY_BYTES, "description": "What to look for in the archive: a phrase, topic, merchant, or document description." },
            "k": k_prop(RECALL_MAX_K)
        }),
        &["query"],
    )
}

fn remember_document_schema() -> Value {
    object(
        json!({
            "title": { "type": "string", "maxLength": MAX_QUERY_BYTES, "description": "Human-facing title. Defaults to the first line of text." },
            "text": { "type": "string", "maxLength": MAX_DOCUMENT_BYTES, "description": "The document text." },
            "data_id": { "type": "string", "maxLength": MAX_ID_BYTES, "description": "Stable id for upserts. Omitted: a new id is minted." },
            "tags": tags_prop("Optional tags.")
        }),
        &["text"],
    )
}

fn name_required_schema() -> Value {
    object(json!({ "name": name_prop() }), &["name"])
}

fn name_optional_schema() -> Value {
    object(json!({ "name": name_prop() }), &[])
}

fn workspace_delete_schema() -> Value {
    object(
        json!({
            "name": name_prop(),
            "confirm": { "type": "string", "description": "Must equal name, to confirm the deletion." }
        }),
        &["name", "confirm"],
    )
}

/// Which views use a given `inspect` argument, for its description.
fn used_by(param: &str) -> String {
    let views: Vec<&str> = INSPECT_VIEWS
        .iter()
        .filter(|v| v.params.contains(&param))
        .map(|v| v.name)
        .collect();
    format!("[{}]", views.join(", "))
}

fn inspect_schema() -> Value {
    let views: Vec<&str> = INSPECT_VIEWS.iter().map(|v| v.name).collect();
    object(
        json!({
            "view": {
                "type": "string",
                "enum": views,
                "description": "What to inspect."
            },
            "limit": {
                "type": "integer", "minimum": 1, "maximum": MAX_LIST_LIMIT,
                "description": format!("{} Max facts to return. Defaults to 20.", used_by("limit"))
            },
            "offset": {
                "type": "integer", "minimum": 0,
                "description": format!("{} Facts to skip (pagination). Defaults to 0.", used_by("offset"))
            },
            "status": {
                "type": "string", "enum": ["active", "archived"],
                "description": format!("{} Only facts with this status.", used_by("status"))
            },
            "contains": {
                "type": "string", "maxLength": MAX_QUERY_BYTES,
                "description": format!("{} Case-insensitive substring the fact text must contain.", used_by("contains"))
            },
            "depth": {
                "type": "integer", "minimum": 1, "maximum": MAX_DEPTH,
                "description": format!("{} Number of layers. Defaults to 3 for ltm_map, 2 for subtree.", used_by("depth"))
            },
            "id": {
                "type": "integer",
                "description": format!("{} The LTM node id (required by these views).", used_by("id"))
            },
            "child_limit": {
                "type": "integer", "minimum": 1, "maximum": MAX_CHILD_LIMIT,
                "description": format!("{} Max children/leaves to return. Defaults to 50.", used_by("child_limit"))
            },
            "child_offset": {
                "type": "integer", "minimum": 0,
                "description": format!("{} Children/leaves to skip (pagination). Defaults to 0.", used_by("child_offset"))
            },
            "summary_max_chars": {
                "type": "integer", "minimum": 0, "maximum": MAX_SUMMARY_CHARS,
                "description": format!("{} Cap each summary to this many chars (0 = full). Defaults to 200.", used_by("summary_max_chars"))
            },
            "data_id": {
                "type": "string", "maxLength": MAX_ID_BYTES,
                "description": format!("{} The document's data_id (required by this view).", used_by("data_id"))
            },
            "sample": {
                "type": "integer", "minimum": 1, "maximum": MAX_SAMPLE,
                "description": format!("{} Number of document leaves to probe. Defaults to 30.", used_by("sample"))
            }
        }),
        &["view"],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn test_names_unique_and_schemas_well_formed() {
        let mut seen = BTreeSet::new();
        for tool in TOOLS {
            assert!(seen.insert(tool.name), "duplicate tool {}", tool.name);
            let schema = (tool.input_schema)();
            assert_eq!(schema["type"], "object", "{}", tool.name);
            let props = schema["properties"].as_object().unwrap();
            for req in schema["required"].as_array().unwrap() {
                assert!(
                    props.contains_key(req.as_str().unwrap()),
                    "{}: required {req} not declared",
                    tool.name
                );
            }
            // Converts without panicking.
            let _ = tool.to_tool();
        }
    }

    /// Every argument an `inspect` view reads is declared in the schema, and
    /// every declared argument (besides `view`) is read by some view.
    #[test]
    fn test_inspect_views_match_schema() {
        let schema = inspect_schema();
        let props: BTreeSet<&str> = schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        let used: BTreeSet<&str> = INSPECT_VIEWS
            .iter()
            .flat_map(|v| v.params.iter().copied())
            .chain(["view"])
            .collect();
        assert_eq!(props, used);
        let enum_names: Vec<&str> = schema["properties"]["view"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let view_names: Vec<&str> = INSPECT_VIEWS.iter().map(|v| v.name).collect();
        assert_eq!(enum_names, view_names);
    }

    /// The schema pattern and the server-side name rule agree.
    #[test]
    fn test_workspace_name_pattern_matches_validator() {
        use crate::application::workspace::validate_name;
        for (name, ok) in [
            ("default", true),
            ("a", true),
            ("work_2-x", true),
            ("-lead", false),
            ("Upper", false),
            ("../x", false),
            ("", false),
        ] {
            assert_eq!(validate_name(name).is_ok(), ok, "{name:?}");
        }
        let long_ok = "a".repeat(64);
        let long_bad = "a".repeat(65);
        assert!(validate_name(&long_ok).is_ok());
        assert!(validate_name(&long_bad).is_err());
        assert_eq!(WORKSPACE_NAME_PATTERN, "^[a-z0-9][a-z0-9_-]{0,63}$");
    }

    /// W1R-3: query_memory reinforces what it reads, so it isn't read-only;
    /// the reinforcement is harmless and repeatable. recall_ltm only reads.
    #[test]
    fn test_query_memory_is_not_read_only() {
        let q = find("query_memory").unwrap();
        assert!(!q.hints.read_only && !q.hints.destructive && q.hints.idempotent);
        assert!(q.description.contains("reinforces"));
        assert!(find("recall_ltm").unwrap().hints.read_only);
    }

    /// Pinned servers list no tool that could reach another workspace.
    #[test]
    fn test_pinned_listing_hides_cross_workspace_tools() {
        let pinned: Vec<&str> = listed(false).map(|t| t.name).collect();
        for hidden in ["workspace_switch", "workspace_create", "workspace_delete"] {
            assert!(!pinned.contains(&hidden), "{hidden} listed while pinned");
        }
        for kept in [
            "workspace_current",
            "workspace_list",
            "workspace_export",
            "inspect",
        ] {
            assert!(pinned.contains(&kept), "{kept} missing while pinned");
        }
        assert_eq!(listed(true).count(), TOOLS.len());
    }
}
