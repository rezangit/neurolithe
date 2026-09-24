//! Shared black-box harness for the MCP STDIO integration tests.
//!
//! * [`FakeLlm`] — an in-process, OpenAI-compatible HTTP server (plain tokio
//!   `TcpListener`, no extra deps) serving `/v1/embeddings` and
//!   `/v1/chat/completions` with deterministic outputs. Every request is
//!   recorded so tests can assert on what the binary sent.
//! * [`McpProcess`] — spawns the real `neurolithe mcp` binary with a clean
//!   environment, its CWD set to a tempdir holding a generated
//!   `neurolithe.toml`, and speaks newline-delimited JSON-RPC over its stdio.
//! * [`Harness`] — the two wired together plus an initialized session.
#![allow(dead_code)]

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Embedding dimension used by the fake LLM and both generated stores.
pub const DIM: usize = 64;
/// The API key the harness hands the binary; the fake LLM records what it gets.
pub const API_KEY: &str = "test-key";
/// Upper bound for any single response from the server.
pub const RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Fake OpenAI-compatible LLM
// ---------------------------------------------------------------------------

/// One HTTP request the fake LLM received.
#[derive(Clone, Debug)]
pub struct RecordedRequest {
    pub path: String,
    pub authorization: Option<String>,
    pub body: Value,
}

pub struct FakeLlm {
    pub port: u16,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
}

impl FakeLlm {
    /// Bind 127.0.0.1:0 and serve forever on a background thread with its own
    /// single-threaded tokio runtime (so tests themselves stay plain `#[test]`).
    pub fn start() -> Self {
        let std_listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind fake LLM");
        std_listener.set_nonblocking(true).unwrap();
        let port = std_listener.local_addr().unwrap().port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::from_std(std_listener).unwrap();
                loop {
                    let Ok((sock, _)) = listener.accept().await else {
                        continue;
                    };
                    let recorded = recorded.clone();
                    tokio::spawn(async move {
                        let _ = serve_connection(sock, recorded).await;
                    });
                }
            });
        });
        Self { port, requests }
    }

    /// OpenAI-style base URL (`…/v1`) to put in the generated config.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }

    pub fn count(&self, path_suffix: &str) -> usize {
        self.requests()
            .iter()
            .filter(|r| r.path.ends_with(path_suffix))
            .count()
    }
}

/// Serve HTTP/1.1 requests (keep-alive) on one connection until EOF.
async fn serve_connection(
    mut sock: tokio::net::TcpStream,
    recorded: Arc<Mutex<Vec<RecordedRequest>>>,
) -> std::io::Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        // Headers.
        let header_end = loop {
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break pos;
            }
            let n = sock.read(&mut chunk).await?;
            if n == 0 {
                return Ok(());
            }
            buf.extend_from_slice(&chunk[..n]);
        };
        let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
        let mut lines = head.split("\r\n");
        let path = lines
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .unwrap_or("")
            .to_string();
        let mut content_length = 0usize;
        let mut authorization = None;
        for line in lines {
            if let Some((k, v)) = line.split_once(':') {
                match k.trim().to_ascii_lowercase().as_str() {
                    "content-length" => content_length = v.trim().parse().unwrap_or(0),
                    "authorization" => authorization = Some(v.trim().to_string()),
                    _ => {}
                }
            }
        }
        // Body.
        let body_start = header_end + 4;
        while buf.len() < body_start + content_length {
            let n = sock.read(&mut chunk).await?;
            if n == 0 {
                return Ok(());
            }
            buf.extend_from_slice(&chunk[..n]);
        }
        let body: Value = serde_json::from_slice(&buf[body_start..body_start + content_length])
            .unwrap_or(Value::Null);
        buf.drain(..body_start + content_length);

        recorded.lock().unwrap().push(RecordedRequest {
            path: path.clone(),
            authorization,
            body: body.clone(),
        });

        let (status, payload) = fake_response(&path, &body);
        let bytes = serde_json::to_vec(&payload).unwrap();
        let head = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            bytes.len()
        );
        sock.write_all(head.as_bytes()).await?;
        sock.write_all(&bytes).await?;
        sock.flush().await?;
    }
}

fn fake_response(path: &str, body: &Value) -> (&'static str, Value) {
    if path.ends_with("/embeddings") {
        let inputs: Vec<String> = match &body["input"] {
            Value::Array(items) => items
                .iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect(),
            v => vec![v.as_str().unwrap_or_default().to_string()],
        };
        let data: Vec<Value> = inputs
            .iter()
            .enumerate()
            .map(
                |(i, t)| json!({"object": "embedding", "index": i, "embedding": fake_embedding(t)}),
            )
            .collect();
        return ("200 OK", json!({"object": "list", "data": data}));
    }
    if path.ends_with("/chat/completions") {
        let messages = body["messages"].as_array().cloned().unwrap_or_default();
        let system = messages
            .first()
            .and_then(|m| m["content"].as_str())
            .unwrap_or_default();
        let user = messages
            .last()
            .and_then(|m| m["content"].as_str())
            .unwrap_or_default();
        let content = if system.contains("Extract independent factual statements") {
            if user.contains(MALFORMED_EXTRACTION) {
                // Not JSON at all: makes fact extraction fail.
                "this is {not valid json".to_string()
            } else {
                fake_extraction(user).to_string()
            }
        } else if system.contains("memory layer") || user.contains("memory layer") {
            "A test memory layer.".to_string()
        } else if system.contains("Compress") {
            format!("SUMMARY of {} chars", user.len())
        } else {
            "ok".to_string()
        };
        return (
            "200 OK",
            json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}),
        );
    }
    (
        "404 Not Found",
        json!({"error": {"message": "unknown path"}}),
    )
}

/// The fact text the fake LLM "extracts" from a dialogue turn.
pub fn extracted_fact_text(dialogue: &str) -> String {
    format!("Extracted: {}", dialogue.trim())
}

/// Dialogue marker: the fake LLM answers the extraction request for a turn
/// containing it with malformed (non-JSON) content.
pub const MALFORMED_EXTRACTION: &str = "[malformed-extraction]";

/// Fake fact extraction: one fact per dialogue turn (or N facts for a
/// `[many-facts:N]` marker). Every `[rel:NAME]` marker
/// in the dialogue becomes a `RELATED_TO` relationship to entity `NAME`, which
/// makes the sleep pipeline create an entity node + an edge.
pub fn fake_extraction(dialogue: &str) -> Value {
    if dialogue.trim().is_empty() {
        return json!({"facts": []});
    }
    // `[many-facts:N]` → N distinct facts (exercises the per-extraction cap).
    if let Some(n) = dialogue
        .split("[many-facts:")
        .nth(1)
        .and_then(|rest| rest.split_once(']'))
        .and_then(|(n, _)| n.trim().parse::<usize>().ok())
    {
        let facts: Vec<Value> = (0..n)
            .map(|i| json!({"fact": format!("Bulk fact {i}"), "ccl": "reality", "tags": []}))
            .collect();
        return json!({"facts": facts});
    }
    let relationships: Vec<Value> = dialogue
        .split("[rel:")
        .skip(1)
        .filter_map(|rest| {
            rest.split_once(']')
                .map(|(name, _)| name.trim().to_string())
        })
        .filter(|name| !name.is_empty())
        .map(|name| {
            json!({"target_entity": name, "relation": "RELATED_TO", "ccl": "reality",
                   "valid_from": null, "valid_until": null})
        })
        .collect();
    json!({"facts": [{
        "fact": extracted_fact_text(dialogue),
        "ccl": "reality",
        "tags": ["test"],
        "relationships": relationships
    }]})
}

/// Deterministic, unit-norm pseudo-random embedding seeded by the text (FNV-1a
/// hash, then xorshift). Distinct texts are near-orthogonal, so the conflict resolver
/// never merges unrelated facts; identical texts embed identically.
pub fn fake_embedding(text: &str) -> Vec<f32> {
    let mut seed: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        seed ^= b as u64;
        seed = seed.wrapping_mul(0x0100_0000_01b3);
    }
    let mut state = seed | 1;
    let mut v: Vec<f32> = (0..DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        })
        .collect();
    let norm = v
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

// ---------------------------------------------------------------------------
// Generated config
// ---------------------------------------------------------------------------

/// Write a minimal, generic `neurolithe.toml` into `dir`: OpenAI-compatible
/// provider pointed at the fake LLM, both stores inside `dir` at [`DIM`].
pub fn write_config(dir: &Path, llm_base_url: &str) {
    let stm = dir.join("stm.sqlite");
    let ltm = dir.join("ltm.sqlite");
    let toml = format!(
        r#"[llm]
provider = "openai"
model = "fake-chat"
embedding_model = "fake-embed"
base_url = "{llm_base_url}"

[stm]
path = "{stm}"
vector_dimension = {DIM}

[ltm]
path = "{ltm}"
vector_dimension = {DIM}
"#,
        stm = stm.display(),
        ltm = ltm.display(),
    );
    std::fs::write(dir.join("neurolithe.toml"), toml).expect("write config");
}

// ---------------------------------------------------------------------------
// The MCP server process
// ---------------------------------------------------------------------------

/// A tool-call result, decoded.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// The raw JSON-RPC `result` object.
    pub raw: Value,
    pub is_error: bool,
    /// Concatenated text of all `text` content items.
    pub text: String,
}

impl ToolResult {
    /// Parse the text payload as JSON (panics with context if it is not).
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.text)
            .unwrap_or_else(|e| panic!("tool text is not JSON ({e}): {}", self.text))
    }

    #[track_caller]
    pub fn assert_ok(&self) -> &Self {
        assert!(!self.is_error, "expected success, got error: {}", self.text);
        self
    }
}

pub struct McpProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    stderr: Arc<Mutex<String>>,
    next_id: i64,
}

impl McpProcess {
    /// Spawn `neurolithe mcp` in `dir` with a cleared environment plus `envs`.
    pub fn spawn(dir: &Path, envs: &[(&str, &str)]) -> Self {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_neurolithe"));
        cmd.arg("mcp")
            .current_dir(dir)
            .env_clear()
            .env("HOME", dir)
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, v) in envs {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().expect("spawn neurolithe binary");

        let stdout = child.stdout.take().unwrap();
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        let stderr = Arc::new(Mutex::new(String::new()));
        let sink = stderr.clone();
        let mut err_pipe = child.stderr.take().unwrap();
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            while let Ok(n) = err_pipe.read(&mut buf) {
                if n == 0 {
                    break;
                }
                sink.lock()
                    .unwrap()
                    .push_str(&String::from_utf8_lossy(&buf[..n]));
            }
        });

        Self {
            stdin: child.stdin.take(),
            child,
            lines,
            stderr,
            next_id: 1,
        }
    }

    /// Spawn with the standard test API key in both env vars the binary reads.
    pub fn spawn_with_key(dir: &Path) -> Self {
        Self::spawn(
            dir,
            &[("OPENAI_API_KEY", API_KEY), ("NEUROLITHE_API_KEY", API_KEY)],
        )
    }

    pub fn stderr(&self) -> String {
        self.stderr.lock().unwrap().clone()
    }

    /// Write one raw line to the server's stdin.
    pub fn send_raw(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("stdin already closed");
        let _ = writeln!(stdin, "{line}");
        let _ = stdin.flush();
    }

    /// Next JSON line from stdout, or `None` on timeout / EOF.
    pub fn read_message(&mut self, timeout: Duration) -> Option<Value> {
        match self.lines.recv_timeout(timeout) {
            Ok(line) => Some(
                serde_json::from_str(&line)
                    .unwrap_or_else(|e| panic!("server wrote non-JSON line ({e}): {line}")),
            ),
            Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => None,
        }
    }

    /// Send a JSON-RPC notification (no id, no reply expected).
    pub fn notify(&mut self, method: &str, params: Value) {
        let msg = json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.send_raw(&msg.to_string());
    }

    /// Send a request and return the full JSON-RPC response with the same id.
    #[track_caller]
    pub fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.send_raw(&msg.to_string());
        let deadline = Instant::now() + RESPONSE_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let Some(resp) = self.read_message(remaining) else {
                panic!(
                    "no response to {method} (id {id}) within {RESPONSE_TIMEOUT:?}; stderr:\n{}",
                    self.stderr()
                );
            };
            // Skip server-initiated notifications / unrelated messages.
            if resp.get("id") == Some(&json!(id)) {
                return resp;
            }
        }
    }

    /// `initialize` + `notifications/initialized`; returns the init result.
    #[track_caller]
    pub fn initialize(&mut self) -> Value {
        let resp = self.request(
            "initialize",
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "neurolithe-tests", "version": "0"}
            }),
        );
        self.notify("notifications/initialized", json!({}));
        resp["result"].clone()
    }

    /// `tools/call` → decoded [`ToolResult`]. Panics on a JSON-RPC-level error.
    #[track_caller]
    pub fn call_tool(&mut self, name: &str, arguments: Value) -> ToolResult {
        let resp = self.request("tools/call", json!({"name": name, "arguments": arguments}));
        let raw = resp
            .get("result")
            .cloned()
            .unwrap_or_else(|| panic!("tools/call {name} returned no result: {resp}"));
        // Read `isError` (MCP spec) but tolerate the legacy `is_error` spelling
        // so behavioural tests fail only for their own bug; the camelCase key
        // itself is pinned by the golden-shape tests (QA-4).
        let is_error = raw
            .get("isError")
            .or_else(|| raw.get("is_error"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let text = raw["content"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(|c| c["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();
        ToolResult {
            raw,
            is_error,
            text,
        }
    }

    /// Close stdin and wait for exit; returns (exit code, stderr).
    pub fn shutdown(mut self) -> (Option<i32>, String) {
        drop(self.stdin.take());
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                // Give the stderr reader a moment to drain.
                std::thread::sleep(Duration::from_millis(50));
                return (status.code(), self.stderr());
            }
            if Instant::now() > deadline {
                let _ = self.child.kill();
                panic!("server did not exit after stdin EOF");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Wait (bounded) for the process to exit on its own, e.g. a startup failure.
    pub fn wait_exit(&mut self, timeout: Duration) -> Option<i32> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Ok(Some(status)) = self.child.try_wait() {
                std::thread::sleep(Duration::from_millis(50));
                return status.code();
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        None
    }
}

impl Drop for McpProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------------------
// Harness = fake LLM + tempdir config + initialized server
// ---------------------------------------------------------------------------

pub struct Harness {
    pub llm: FakeLlm,
    pub dir: tempfile::TempDir,
    pub server: McpProcess,
    pub init: Value,
}

impl Harness {
    #[track_caller]
    pub fn start() -> Self {
        let llm = FakeLlm::start();
        let dir = tempfile::tempdir().expect("tempdir");
        write_config(dir.path(), &llm.base_url());
        let mut server = McpProcess::spawn_with_key(dir.path());
        let init = server.initialize();
        Self {
            llm,
            dir,
            server,
            init,
        }
    }

    pub fn path(&self) -> PathBuf {
        self.dir.path().to_path_buf()
    }

    #[track_caller]
    pub fn call(&mut self, name: &str, arguments: Value) -> ToolResult {
        self.server.call_tool(name, arguments)
    }

    /// Facts visible via `stm_list` (fact texts only).
    #[track_caller]
    pub fn stm_facts(&mut self) -> Vec<String> {
        let res = self.call("stm_list", json!({"limit": 1000}));
        res.assert_ok();
        res.json()
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .filter_map(|f| f["fact"].as_str().map(str::to_string))
            .collect()
    }

    /// Fact texts in `export_tenant` for `tenant`.
    #[track_caller]
    pub fn exported_facts(&mut self, tenant: &str) -> Vec<String> {
        let res = self.call("export_tenant", json!({"tenant_id": tenant}));
        res.assert_ok();
        res.json()["extracted_facts"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .filter_map(|f| f["fact"].as_str().map(str::to_string))
            .collect()
    }
}

/// Poll `cond` until true or `timeout`; returns the final value.
pub fn eventually(timeout: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if cond() {
            return true;
        }
        if Instant::now() > deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}
