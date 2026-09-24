//! MCP / JSON-RPC protocol conformance of `neurolithe mcp` over STDIO, including
//! golden-JSON assertions on the tool-result shape.
mod common;

use common::Harness;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::time::Duration;

fn keys(v: &Value) -> BTreeSet<String> {
    v.as_object()
        .unwrap_or_else(|| panic!("not an object: {v}"))
        .keys()
        .cloned()
        .collect()
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn initialize_reports_server_info_and_tools_capability() {
    let h = Harness::start();
    let init = &h.init;
    assert!(
        init["protocolVersion"].is_string(),
        "protocolVersion missing: {init}"
    );
    assert_eq!(init["serverInfo"]["name"], "NeuroLithe");
    assert_eq!(init["serverInfo"]["version"], env!("CARGO_PKG_VERSION"));
    assert!(
        init["capabilities"]["tools"].is_object(),
        "tools capability missing: {init}"
    );
}

/// QA-11: the server never emits `notifications/tools/list_changed`, so it must
/// not advertise `listChanged: true`.
#[test]
fn initialize_does_not_advertise_list_changed() {
    let h = Harness::start();
    let list_changed = &h.init["capabilities"]["tools"]["listChanged"];
    assert_ne!(
        list_changed,
        &json!(true),
        "QA-11: advertises listChanged but never sends it"
    );
}

/// QA-11: `ping` is a required MCP method; the reply is an empty result.
#[test]
fn ping_returns_empty_result() {
    let mut h = Harness::start();
    let resp = h.server.request("ping", json!({}));
    assert!(resp.get("error").is_none(), "QA-11: ping failed: {resp}");
    assert_eq!(resp["result"], json!({}), "QA-11: ping result: {resp}");
}

/// The advertised tool surface. Update this list deliberately when tools change.
const EXPECTED_TOOLS: &[&str] = &[
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

#[test]
fn tools_list_matches_expected_surface_with_valid_schemas() {
    let mut h = Harness::start();
    let resp = h.server.request("tools/list", json!({}));
    let tools = resp["result"]["tools"]
        .as_array()
        .unwrap_or_else(|| panic!("no tools array: {resp}"))
        .clone();

    let names: BTreeSet<String> = tools
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, set(EXPECTED_TOOLS), "tool surface changed");
    assert_eq!(names.len(), tools.len(), "duplicate tool names");

    for tool in &tools {
        let name = tool["name"].as_str().unwrap();
        assert!(
            tool["description"]
                .as_str()
                .is_some_and(|d| !d.trim().is_empty()),
            "{name}: empty description"
        );
        let schema = &tool["inputSchema"];
        assert_eq!(schema["type"], "object", "{name}: inputSchema.type");
        let props = schema["properties"]
            .as_object()
            .unwrap_or_else(|| panic!("{name}: properties missing"));
        for req in schema["required"].as_array().cloned().unwrap_or_default() {
            let req = req.as_str().unwrap();
            assert!(
                props.contains_key(req),
                "{name}: required '{req}' not in properties"
            );
        }
    }
}

/// QA-4 golden shape (success): `{"content":[{"type":"text","text":…}],"isError":false}`.
#[test]
fn tool_success_result_has_mcp_golden_shape() {
    let mut h = Harness::start();
    let res = h.call("health", json!({}));
    assert_eq!(
        keys(&res.raw),
        set(&["content", "isError"]),
        "QA-4: result keys: {}",
        res.raw
    );
    assert_eq!(res.raw["isError"], json!(false), "QA-4");
    let content = res.raw["content"].as_array().expect("content array");
    assert_eq!(content.len(), 1);
    assert_eq!(keys(&content[0]), set(&["type", "text"]));
    assert_eq!(content[0]["type"], "text");
    assert!(content[0]["text"].is_string());
    // The payload is JSON text.
    assert!(res.json().is_object());
}

/// QA-4 golden shape (tool execution error): a real tool given bad arguments
/// answers with a *result* `{"content":[{"type":"text",…}],"isError":true}`,
/// camelCase, not a JSON-RPC error.
#[test]
fn tool_error_result_has_mcp_golden_shape() {
    let mut h = Harness::start();
    let resp = h.server.request(
        "tools/call",
        json!({"name": "store_memory", "arguments": {}}),
    );
    assert!(resp.get("error").is_none(), "expected a result: {resp}");
    let raw = &resp["result"];
    assert_eq!(
        keys(raw),
        set(&["content", "isError"]),
        "QA-4: result keys: {raw}"
    );
    assert_eq!(raw["isError"], json!(true), "QA-4");
    let content = raw["content"].as_array().expect("content array");
    assert_eq!(content.len(), 1);
    assert_eq!(keys(&content[0]), set(&["type", "text"]));
    assert_eq!(content[0]["type"], "text");
    assert!(
        content[0]["text"]
            .as_str()
            .is_some_and(|t| t.contains("fact_text")),
        "error should name the bad argument: {raw}"
    );
}

/// REV-6: an unknown tool name is a protocol error (JSON-RPC -32602, no
/// `result`), not a tool execution error.
#[test]
fn unknown_tool_is_json_rpc_invalid_params() {
    let mut h = Harness::start();
    let resp = h.server.request(
        "tools/call",
        json!({"name": "no_such_tool", "arguments": {}}),
    );
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("result").is_none(), "REV-6: {resp}");
    assert_eq!(resp["error"]["code"], -32602, "REV-6: {resp}");
    assert!(
        resp["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("no_such_tool")),
        "REV-6: error should name the tool: {resp}"
    );
    // Still serving.
    h.call("health", json!({})).assert_ok();
}

#[test]
fn unknown_method_is_json_rpc_method_not_found() {
    let mut h = Harness::start();
    let resp = h.server.request("no/such_method", json!({}));
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["error"]["code"], -32601, "{resp}");
    assert!(resp.get("result").is_none());
}

#[test]
fn malformed_line_yields_parse_error_and_server_keeps_serving() {
    let mut h = Harness::start();
    h.server.send_raw("this is not json");
    let resp = h
        .server
        .read_message(common::RESPONSE_TIMEOUT)
        .expect("parse error response");
    assert_eq!(resp["error"]["code"], -32700, "{resp}");
    assert_eq!(resp["id"], Value::Null);
    // Still alive.
    h.call("health", json!({})).assert_ok();
}

#[test]
fn notifications_get_no_response() {
    let mut h = Harness::start();
    h.server.notify("notifications/initialized", json!({}));
    h.server
        .notify("notifications/cancelled", json!({"requestId": 99}));
    // The next message on stdout must be the reply to this request, not a
    // reply to either notification (request() would skip it, so read raw).
    h.server
        .send_raw(r#"{"jsonrpc":"2.0","id":"after-notify","method":"tools/list"}"#);
    let resp = h
        .server
        .read_message(common::RESPONSE_TIMEOUT)
        .expect("response");
    assert_eq!(resp["id"], "after-notify", "unexpected message: {resp}");
}

#[test]
fn string_ids_are_echoed() {
    let mut h = Harness::start();
    h.server
        .send_raw(r#"{"jsonrpc":"2.0","id":"abc-1","method":"tools/list"}"#);
    let resp = h
        .server
        .read_message(common::RESPONSE_TIMEOUT)
        .expect("response");
    assert_eq!(resp["id"], "abc-1");
}

#[test]
fn server_exits_cleanly_on_stdin_eof() {
    let h = Harness::start();
    let (code, stderr) = h.server.shutdown();
    assert_eq!(code, Some(0), "exit code on EOF; stderr:\n{stderr}");
}

#[test]
fn introspection_answers_quickly_without_calling_the_llm() {
    let mut h = Harness::start();
    // Startup embeds the LTM spine (§5); wait for that to settle first.
    let mut last = usize::MAX;
    common::eventually(Duration::from_secs(10), || {
        let now = h.llm.requests().len();
        let settled = now == last;
        last = now;
        std::thread::sleep(Duration::from_millis(200));
        settled
    });
    let before = h.llm.requests().len();
    let started = std::time::Instant::now();
    for tool in [
        "health",
        "memory_stats",
        "stm_list",
        "ltm_map",
        "workspace_current",
        "workspace_list",
    ] {
        h.call(tool, json!({})).assert_ok();
    }
    assert!(started.elapsed() < Duration::from_secs(10));
    assert_eq!(
        h.llm.requests().len(),
        before,
        "introspection must not call the LLM"
    );
}

/// Phase 2 (§2): tenancy is gone from the MCP surface. No schema accepts
/// `tenant_id`, and the tenant tools no longer exist (JSON-RPC -32602).
#[test]
fn tenant_surface_is_removed() {
    let mut h = Harness::start();
    let resp = h.server.request("tools/list", json!({}));
    for tool in resp["result"]["tools"].as_array().unwrap() {
        assert!(
            tool["inputSchema"]["properties"].get("tenant_id").is_none(),
            "{} still declares tenant_id",
            tool["name"]
        );
    }
    for gone in ["delete_tenant", "export_tenant"] {
        let resp = h.server.request(
            "tools/call",
            json!({"name": gone, "arguments": {"tenant_id": "x", "confirm": "x"}}),
        );
        assert_eq!(resp["error"]["code"], -32602, "{gone}: {resp}");
    }
}

/// Phase 2 (§2): `workspace_delete` is annotated destructive, and every
/// workspace-name argument carries the name regex as a schema `pattern`.
#[test]
fn workspace_tool_schemas_are_annotated() {
    let mut h = Harness::start();
    let resp = h.server.request("tools/list", json!({}));
    let tools = resp["result"]["tools"].as_array().unwrap().clone();
    let tool = |name: &str| {
        tools
            .iter()
            .find(|t| t["name"] == name)
            .unwrap_or_else(|| panic!("{name} missing"))
            .clone()
    };
    let hints = |name: &str| tool(name)["annotations"].clone();
    assert_eq!(hints("workspace_delete")["destructiveHint"], json!(true));
    assert_eq!(hints("workspace_delete")["idempotentHint"], json!(true));
    for name in [
        "query_memory",
        "recall_ltm",
        "workspace_current",
        "workspace_list",
        "workspace_export",
        "memory_stats",
        "health",
        "placement_debug",
        "stm_list",
        "ltm_map",
        "inspect_node",
        "subtree",
        "trace_dataId",
    ] {
        assert_eq!(
            hints(name)["readOnlyHint"],
            json!(true),
            "{name}: {}",
            hints(name)
        );
    }
    for name in [
        "store_memory",
        "push_dialogue",
        "remember_document",
        "workspace_delete",
    ] {
        assert_ne!(
            hints(name)["readOnlyHint"],
            json!(true),
            "{name} marked read-only"
        );
    }
    assert_eq!(hints("remember_document")["destructiveHint"], json!(false));
    for name in [
        "workspace_create",
        "workspace_switch",
        "workspace_delete",
        "workspace_export",
    ] {
        let pattern = &tool(name)["inputSchema"]["properties"]["name"]["pattern"];
        assert_eq!(
            pattern,
            &json!("^[a-z0-9][a-z0-9_-]{0,63}$"),
            "{name}: name pattern"
        );
    }
    let required = tool("workspace_delete")["inputSchema"]["required"].clone();
    assert!(
        required.as_array().unwrap().iter().any(|r| r == "confirm"),
        "workspace_delete must require confirm: {required}"
    );
}
