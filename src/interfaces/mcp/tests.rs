use super::*;
use crate::daemon::SqliteWorkspaceHost;
use crate::domain::models::CclDefinition;
use crate::domain::ports::{ExtractedFact, LlmClient};
use crate::infrastructure::config::{AppConfig, LoadOptions};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

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
    home: tempfile::TempDir,
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

/// [`StubLlm`] whose embeddings take `delay` (a slow model).
struct SlowLlm {
    delay: std::time::Duration,
}

#[async_trait::async_trait]
impl LlmClient for SlowLlm {
    async fn extract_facts(
        &self,
        d: &str,
        c: &[CclDefinition],
    ) -> anyhow::Result<Vec<ExtractedFact>> {
        StubLlm.extract_facts(d, c).await
    }
    async fn generate_ccl_description(&self, n: &str, c: &str) -> anyhow::Result<String> {
        StubLlm.generate_ccl_description(n, c).await
    }
    async fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        tokio::time::sleep(self.delay).await;
        StubLlm.embed_text(text).await
    }
    async fn compress_context(&self, m: &str) -> anyhow::Result<String> {
        StubLlm.compress_context(m).await
    }
}

/// The real server over the real SQLite workspace layout in a temp home,
/// with an offline stub LLM. Starts on workspace "default".
async fn harness_with(allow_switch: bool) -> Harness {
    harness_llm(allow_switch, Arc::new(StubLlm)).await
}

async fn harness_llm(allow_switch: bool, llm: Arc<dyn LlmClient>) -> Harness {
    let home = tempfile::tempdir().unwrap();
    let config = test_config(home.path(), allow_switch);
    let host = SqliteWorkspaceHost::new(config.clone(), llm);
    let workspaces = WorkspaceManager::start(Box::new(host), &config.workspace, allow_switch)
        .await
        .unwrap();
    Harness {
        server: McpServer::new(Rc::new(workspaces), allow_switch),
        home,
    }
}

async fn harness() -> Harness {
    harness_with(true).await
}

/// One tool call through the dispatcher, rendered as the wire result
/// (structured content on).
async fn call(h: &Harness, tool: &str, args: Value) -> Value {
    let map = args.as_object().cloned().unwrap_or_default();
    match h.server.call(tool, Some(&map)).await {
        Ok(outcome) => serde_json::to_value(render(outcome, true)).unwrap(),
        Err(e) => json!({ "protocol_error": { "code": e.code.0, "message": e.message } }),
    }
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
        .inner
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

/// Run `f` inside a `LocalSet` (rmcp spawns request handlers locally).
async fn local<F: std::future::Future>(f: F) -> F::Output {
    tokio::task::LocalSet::new().run_until(f).await
}

// ---------------------------------------------------------------------------
// Protocol, end to end over an in-memory stdio pair
// ---------------------------------------------------------------------------

/// A JSON-RPC client over an in-memory pipe to a served [`McpServer`].
struct Wire {
    tx: tokio::io::WriteHalf<tokio::io::DuplexStream>,
    rx: tokio::io::Lines<BufReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>>,
}

impl Wire {
    fn start(server: &McpServer) -> Self {
        let (client, server_io) = tokio::io::duplex(1 << 16);
        let (srv_r, srv_w) = tokio::io::split(server_io);
        let (cli_r, cli_w) = tokio::io::split(client);
        let s = server.clone();
        tokio::task::spawn_local(async move { s.serve(BufReader::new(srv_r), srv_w).await });
        Self {
            tx: cli_w,
            rx: BufReader::new(cli_r).lines(),
        }
    }

    async fn send(&mut self, line: &str) {
        self.tx.write_all(line.as_bytes()).await.unwrap();
        self.tx.write_all(b"\n").await.unwrap();
    }

    async fn recv(&mut self) -> Value {
        let line = tokio::time::timeout(std::time::Duration::from_secs(10), self.rx.next_line())
            .await
            .expect("reply in time")
            .unwrap()
            .expect("a reply line");
        serde_json::from_str(&line).unwrap()
    }

    async fn request(&mut self, id: i64, method: &str, params: Value) -> Value {
        self.send(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        )
        .await;
        self.recv().await
    }

    async fn initialize(&mut self, version: &str) -> Value {
        let r = self
            .request(
                0,
                "initialize",
                json!({
                    "protocolVersion": version,
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"}
                }),
            )
            .await;
        self.send(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
            .await;
        r
    }
}

/// QA-11 / RES-F2: initialize echoes a supported client version, falls back
/// to the newest handshake revision for an unknown one, and doesn't
/// advertise list-change notifications it never sends.
#[tokio::test]
async fn test_initialize_negotiates_version_and_no_list_changed() {
    local(async {
        let h = harness().await;
        for (asked, agreed) in [
            ("2025-03-26", "2025-03-26"),
            ("2025-06-18", "2025-06-18"),
            ("2025-11-25", "2025-11-25"),
            ("1999-01-01", "2025-11-25"),
        ] {
            let mut wire = Wire::start(&h.server);
            let r = wire.initialize(asked).await;
            assert_eq!(r["result"]["protocolVersion"], agreed, "{asked}: {r}");
            assert_eq!(r["result"]["capabilities"]["tools"]["listChanged"], false);
            assert_eq!(r["result"]["serverInfo"]["name"], "NeuroLithe");
            assert!(r["result"]["instructions"].is_string());
        }
    })
    .await;
}

/// QA-11: `ping` answers with an empty result; unknown methods are -32601;
/// unknown tools -32602 naming the tool (REV-6); a tool error is an
/// `isError` result, not a JSON-RPC error.
#[tokio::test]
async fn test_protocol_errors_and_ping() {
    local(async {
        let h = harness().await;
        let mut wire = Wire::start(&h.server);
        wire.initialize("2025-06-18").await;
        let r = wire.request(1, "ping", json!({})).await;
        assert_eq!(r["result"], json!({}), "{r}");

        let r = wire.request(2, "no/such_method", json!({})).await;
        assert_eq!(r["error"]["code"], -32601, "{r}");

        let r = wire
            .request(3, "tools/call", json!({"name": "nope", "arguments": {}}))
            .await;
        assert_eq!(r["error"]["code"], -32602, "{r}");
        assert!(r["error"]["message"].as_str().unwrap().contains("nope"));

        let r = wire
            .request(
                4,
                "tools/call",
                json!({"name": "store_memory", "arguments": {}}),
            )
            .await;
        assert_eq!(r["result"]["isError"], true, "{r}");
        assert!(text(&r["result"]).contains("fact_text"), "{r}");
    })
    .await;
}

/// SEC-12: a line over the cap is rejected (-32600, id recovered) without
/// killing the session; the next request is still served.
#[tokio::test]
async fn test_oversized_line_rejected_and_session_continues() {
    local(async {
        let h = harness().await;
        let mut wire = Wire::start(&h.server);
        wire.initialize("2025-06-18").await;
        let mut big = br#"{"id":41,"jsonrpc":"2.0","method":"tools/call","params":{"x":""#.to_vec();
        big.extend(std::iter::repeat_n(b'x', limits::MAX_LINE_BYTES + 10));
        big.extend_from_slice(b"\"}}");
        // Write while the server drains, or the pipe fills up.
        let Wire { tx, rx } = &mut wire;
        let writer = async {
            tx.write_all(&big).await.unwrap();
            tx.write_all(b"\n").await.unwrap();
            tx.write_all(b"{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n")
                .await
                .unwrap();
        };
        let mut replies = Vec::new();
        let reader = async {
            for _ in 0..2 {
                let line = wire_line(rx).await;
                replies.push(line);
            }
        };
        tokio::join!(writer, reader);
        assert_eq!(replies[0]["error"]["code"], -32600, "{replies:?}");
        assert_eq!(
            replies[0]["id"], 41,
            "the oversized request's id is recovered"
        );
        assert_eq!(replies[1]["id"], 2);
        assert_eq!(replies[1]["result"], json!({}));
    })
    .await;
}

async fn wire_line(
    rx: &mut tokio::io::Lines<BufReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>>,
) -> Value {
    let line = tokio::time::timeout(std::time::Duration::from_secs(30), rx.next_line())
        .await
        .expect("reply in time")
        .unwrap()
        .expect("a reply line");
    serde_json::from_str(&line).unwrap()
}

/// Notifications get no reply; a malformed line gets -32700 and the session
/// continues; stdin EOF ends `serve` cleanly.
#[tokio::test]
async fn test_notifications_silent_parse_error_and_clean_eof() {
    local(async {
        let h = harness().await;
        let (client, server_io) = tokio::io::duplex(1 << 16);
        let (srv_r, srv_w) = tokio::io::split(server_io);
        let (cli_r, mut cli_w) = tokio::io::split(client);
        let s = h.server.clone();
        let served =
            tokio::task::spawn_local(async move { s.serve(BufReader::new(srv_r), srv_w).await });
        let mut rx = BufReader::new(cli_r).lines();
        cli_w
            .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"t\",\"version\":\"0\"}}}\n")
            .await
            .unwrap();
        assert_eq!(wire_line(&mut rx).await["id"], 1);
        cli_w
            .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\nnot json\n{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}")
            .await
            .unwrap();
        let r = wire_line(&mut rx).await;
        assert_eq!(r["error"]["code"], -32700, "{r}");
        assert_eq!(r["id"], Value::Null);
        // The unterminated last line is served once stdin closes.
        cli_w.shutdown().await.unwrap();
        let r = wire_line(&mut rx).await;
        assert_eq!(r["id"], 9, "{r}");
        let result = tokio::time::timeout(std::time::Duration::from_secs(10), served)
            .await
            .expect("serve ends on EOF")
            .unwrap();
        assert!(result.is_ok(), "{result:?}");
    })
    .await;
}

/// W1R-4b: a request before `initialize` (without the 2026-07-28 inline
/// `_meta`) is told to initialize first; ping still works, and the session
/// then starts normally.
#[tokio::test]
async fn test_request_before_initialize_says_initialize_first() {
    local(async {
        let h = harness().await;
        let mut wire = Wire::start(&h.server);
        let r = wire.request(1, "tools/list", json!({})).await;
        assert_eq!(r["error"]["code"], -32602, "{r}");
        let msg = r["error"]["message"].as_str().unwrap();
        assert!(msg.contains("send initialize first"), "{msg}");
        let r = wire.request(2, "ping", json!({})).await;
        assert_eq!(r["result"], json!({}), "{r}");
        let r = wire.initialize("2025-06-18").await;
        assert_eq!(r["result"]["protocolVersion"], "2025-06-18", "{r}");
        let r = wire.request(3, "tools/list", json!({})).await;
        assert!(r["result"]["tools"].is_array(), "{r}");
    })
    .await;
}

/// EOF before any request is a normal, clean end.
#[tokio::test]
async fn test_eof_before_initialize_is_clean() {
    local(async {
        let h = harness().await;
        let empty: &[u8] = b"";
        h.server
            .serve(BufReader::new(empty), tokio::io::sink())
            .await
            .unwrap();
    })
    .await;
}

/// 3.1: tools/list comes from the registry, with annotations; a pinned
/// server doesn't list the tools that could reach another workspace.
#[tokio::test]
async fn test_tools_list_from_registry_and_pinned_hides_cross_workspace_tools() {
    local(async {
        let h = harness().await;
        let mut wire = Wire::start(&h.server);
        wire.initialize("2025-06-18").await;
        let r = wire.request(1, "tools/list", json!({})).await;
        let tools = r["result"]["tools"].as_array().unwrap().clone();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        let expected: Vec<&str> = registry::TOOLS.iter().map(|t| t.name).collect();
        assert_eq!(names, expected);
        let inspect = tools.iter().find(|t| t["name"] == "inspect").unwrap();
        assert_eq!(inspect["annotations"]["readOnlyHint"], true, "{inspect}");
        let delete = tools
            .iter()
            .find(|t| t["name"] == "workspace_delete")
            .unwrap();
        assert_eq!(delete["annotations"]["destructiveHint"], true, "{delete}");
        assert!(
            !r.to_string().contains("tenant"),
            "tenant leaked into schemas"
        );

        let pinned = harness_with(false).await;
        let mut wire = Wire::start(&pinned.server);
        wire.initialize("2025-06-18").await;
        let r = wire.request(1, "tools/list", json!({})).await;
        let listed: Vec<String> = r["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        for hidden in ["workspace_switch", "workspace_create", "workspace_delete"] {
            assert!(!listed.iter().any(|n| n == hidden), "{hidden}: {listed:?}");
        }
        assert!(listed.iter().any(|n| n == "workspace_current"));
    })
    .await;
}

/// Structured output: JSON results carry `structuredContent` (arrays wrapped
/// as `{items}`) for 2025-06-18+ peers and not for older ones.
#[tokio::test]
async fn test_structured_content_follows_protocol_version() {
    local(async {
        let h = harness().await;
        for (version, expect) in [("2025-06-18", true), ("2025-03-26", false)] {
            let mut wire = Wire::start(&h.server);
            wire.initialize(version).await;
            let r = wire
                .request(
                    1,
                    "tools/call",
                    json!({"name": "workspace_current", "arguments": {}}),
                )
                .await;
            let res = &r["result"];
            assert_eq!(res["isError"], false, "{r}");
            let parsed: Value = serde_json::from_str(&text(res)).unwrap();
            if expect {
                assert_eq!(res["structuredContent"], parsed, "{version}: {r}");
            } else {
                assert!(res.get("structuredContent").is_none(), "{version}: {r}");
            }
            let r = wire
                .request(
                    2,
                    "tools/call",
                    json!({"name": "workspace_list", "arguments": {}}),
                )
                .await;
            if expect {
                let items = &r["result"]["structuredContent"]["items"];
                assert!(items.is_array(), "{r}");
            }
        }
    })
    .await;
}

// ---------------------------------------------------------------------------
// Tools (through the dispatcher)
// ---------------------------------------------------------------------------

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
    let r = call(&h, "inspect", json!({ "view": "stm", "limit": 2.5 })).await;
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
    let r = call(&h, "inspect", json!({ "view": "stm", "limit": 1_000_000 })).await;
    assert!(!is_error(&r), "huge limit is clamped, not an error: {r}");
}

/// SEC-12: oversized text arguments are refused before touching the LLM/DB.
#[tokio::test]
async fn test_oversized_fact_rejected() {
    let h = harness().await;
    let big = "a".repeat(limits::MAX_FACT_BYTES + 1);
    let r = call(&h, "store_memory", json!({ "fact_text": big })).await;
    assert!(is_error(&r) && text(&r).contains("too large"), "{r}");
    assert_eq!(stm_count(&h), 0);
}

/// 3.2: `inspect` serves every view; `view` is required and checked; an
/// argument of another view is rejected; a missing node is an error, not
/// `null`.
#[tokio::test]
async fn test_inspect_views() {
    let h = harness().await;
    call(&h, "store_memory", json!({ "fact_text": "Metro line 4" })).await;

    let r = call(&h, "inspect", json!({})).await;
    assert!(
        is_error(&r) && text(&r).contains("'view' is required"),
        "{r}"
    );
    let r = call(&h, "inspect", json!({ "view": "nope" })).await;
    assert!(is_error(&r) && text(&r).contains("unknown view"), "{r}");
    let r = call(&h, "inspect", json!({ "view": "subtree", "node": 1 })).await;
    // `node` isn't an inspect argument at all: ignored, so `id` is missing.
    assert!(is_error(&r) && text(&r).contains("'id'"), "{r}");
    let r = call(&h, "inspect", json!({ "view": "ltm_map", "sample": 3 })).await;
    assert!(
        is_error(&r) && text(&r).contains("'sample' does not apply"),
        "{r}"
    );

    let r = call(&h, "inspect", json!({ "view": "stats" })).await;
    assert!(!is_error(&r) && r["structuredContent"].is_object(), "{r}");
    let r = call(&h, "inspect", json!({ "view": "health" })).await;
    assert!(!is_error(&r), "{r}");
    let r = call(&h, "inspect", json!({ "view": "stm", "contains": "metro" })).await;
    let facts: Value = serde_json::from_str(&text(&r)).unwrap();
    assert_eq!(facts.as_array().unwrap().len(), 1, "{r}");
    let r = call(&h, "inspect", json!({ "view": "ltm_map", "depth": 2 })).await;
    let map: Value = serde_json::from_str(&text(&r)).unwrap();
    let root = map[0]["id"].as_i64().expect("a spine root");
    let r = call(&h, "inspect", json!({ "view": "node", "id": root })).await;
    assert!(!is_error(&r), "{r}");
    let r = call(
        &h,
        "inspect",
        json!({ "view": "subtree", "id": root, "depth": 1 }),
    )
    .await;
    assert!(!is_error(&r), "{r}");
    let r = call(&h, "inspect", json!({ "view": "node", "id": 999_999 })).await;
    assert!(is_error(&r) && text(&r).contains("999999"), "{r}");
    let r = call(&h, "inspect", json!({ "view": "subtree", "id": 999_999 })).await;
    assert!(is_error(&r), "{r}");
    let r = call(
        &h,
        "inspect",
        json!({ "view": "trace", "data_id": "doc_x" }),
    )
    .await;
    assert!(!is_error(&r), "{r}");
    let r = call(&h, "inspect", json!({ "view": "trace" })).await;
    assert!(is_error(&r) && text(&r).contains("'data_id'"), "{r}");
    let r = call(&h, "inspect", json!({ "view": "placement", "sample": 3 })).await;
    assert!(!is_error(&r), "{r}");
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

/// 2.2: `[mcp] allow_workspace_switch = false` pins the server; the hidden
/// tool still explains itself when called.
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
    let home = h.home.path().to_path_buf();
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

/// 2.2 / 3.2: the old tenant tools and the folded-in inspection tools are
/// unknown tools now (-32602).
#[tokio::test]
async fn test_removed_tools_are_unknown() {
    let h = harness().await;
    for gone in [
        "delete_tenant",
        "export_tenant",
        "memory_stats",
        "health",
        "stm_list",
        "ltm_map",
        "inspect_node",
        "subtree",
        "trace_dataId",
        "placement_debug",
    ] {
        let r = call(&h, gone, json!({})).await;
        assert_eq!(r["protocol_error"]["code"], -32602, "{gone}: {r}");
    }
}

/// Every registered tool dispatches (argument errors are fine; "Unknown" is
/// not).
#[tokio::test]
async fn test_every_registered_tool_dispatches() {
    let h = harness().await;
    for tool in registry::TOOLS {
        let r = call(&h, tool.name, json!({})).await;
        assert!(r.get("protocol_error").is_none(), "{}: {r}", tool.name);
        assert!(!text(&r).contains("Unknown tool"), "{}: {r}", tool.name);
    }
}

/// W1R-1 (signals): `drain_calls` stops new calls and waits for the running
/// one, so its work completes before the stores are checkpointed; past the
/// budget it gives up and says so.
#[tokio::test]
async fn test_shutdown_waits_for_in_flight_calls_and_refuses_new_ones() {
    local(async {
        let slow = harness_llm(
            true,
            Arc::new(SlowLlm {
                delay: std::time::Duration::from_millis(400),
            }),
        )
        .await;
        let server = slow.server.clone();
        let running = tokio::task::spawn_local(async move {
            let args = json!({ "fact_text": "finished before exit" });
            server.call("store_memory", args.as_object()).await
        });
        tokio::task::yield_now().await;
        assert!(
            slow.server
                .drain_calls(std::time::Duration::from_secs(10))
                .await,
            "drained within the budget"
        );
        assert!(
            running.is_finished(),
            "shutdown returned before the call ended"
        );
        assert!(!is_error(
            &serde_json::to_value(render(running.await.unwrap().unwrap(), true)).unwrap()
        ));
        assert_eq!(stm_count(&slow), 1, "the call's work completed");

        let r = call(&slow, "store_memory", json!({ "fact_text": "too late" })).await;
        assert!(is_error(&r) && text(&r).contains("shutting down"), "{r}");
        assert_eq!(stm_count(&slow), 1);

        // A budget shorter than the running call: gives up, reports it.
        let slower = harness_llm(
            true,
            Arc::new(SlowLlm {
                delay: std::time::Duration::from_millis(600),
            }),
        )
        .await;
        let server = slower.server.clone();
        let _running = tokio::task::spawn_local(async move {
            let args = json!({ "fact_text": "slow" });
            server.call("store_memory", args.as_object()).await
        });
        tokio::task::yield_now().await;
        assert!(
            !slower
                .server
                .drain_calls(std::time::Duration::from_millis(50))
                .await
        );
    })
    .await;
}
