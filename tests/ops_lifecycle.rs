//! Ops lifecycle of `neurolithe mcp` (PLAN 2.8, OPS-7/OPS-14):
//!
//! - **stdout purity:** with logging at the most verbose level, stdout carries
//!   nothing but JSON-RPC 2.0 frames — every diagnostic goes to stderr.
//! - **graceful shutdown:** stdin EOF (the normal MCP shutdown) and SIGTERM both
//!   exit 0 and checkpoint the stores (the `-wal` file is truncated).
//!
//! Spawns the binary directly (not via `McpProcess`) so the test sees every
//! stdout byte, including anything written after the last reply.
mod common;

use common::{FakeLlm, Home, bin};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const REPLY_TIMEOUT: Duration = Duration::from_secs(60);
const EXIT_TIMEOUT: Duration = Duration::from_secs(15);

/// A raw `neurolithe mcp` process: every stdout line and all of stderr kept.
struct Raw {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    stdout_log: Arc<Mutex<Vec<String>>>,
    stderr: Arc<Mutex<String>>,
    next_id: i64,
}

impl Raw {
    fn spawn(home: &Home, extra_env: &[(&str, &str)]) -> Self {
        let mut cmd = Command::new(bin());
        cmd.arg("mcp")
            .current_dir(home.cwd.path())
            .env_clear()
            .env("HOME", home.cwd.path())
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, v) in home.env() {
            cmd.env(k, v);
        }
        for (k, v) in extra_env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().expect("spawn neurolithe mcp");

        let stdout_log = Arc::new(Mutex::new(Vec::new()));
        let (tx, lines) = mpsc::channel();
        let out = child.stdout.take().unwrap();
        let log = stdout_log.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines() {
                let Ok(line) = line else { break };
                log.lock().unwrap().push(line.clone());
                let _ = tx.send(line);
            }
        });
        let stderr = Arc::new(Mutex::new(String::new()));
        let sink = stderr.clone();
        let mut err = child.stderr.take().unwrap();
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            while let Ok(n) = err.read(&mut buf) {
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
            stdout_log,
            stderr,
            next_id: 1,
        }
    }

    fn stderr(&self) -> String {
        self.stderr.lock().unwrap().clone()
    }

    fn send(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("stdin open");
        writeln!(stdin, "{line}").unwrap();
        stdin.flush().unwrap();
    }

    /// Send a request and wait for the line that answers it. Any stdout line
    /// that is not valid JSON fails the test here, with stderr for context.
    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        );
        let deadline = Instant::now() + REPLY_TIMEOUT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let line = self.lines.recv_timeout(left).unwrap_or_else(|_| {
                panic!(
                    "no reply to {method} within {REPLY_TIMEOUT:?}; stderr:\n{}",
                    self.stderr()
                )
            });
            let msg: Value = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("non-JSON on stdout ({e}): {line:?}"));
            if msg["id"] == json!(id) {
                return msg;
            }
        }
    }

    fn initialize(&mut self) {
        let resp = self.request(
            "initialize",
            json!({"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "ops-test", "version": "0"}}),
        );
        assert!(resp.get("result").is_some(), "initialize failed: {resp}");
        self.send(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
    }

    fn call(&mut self, name: &str, arguments: Value) -> Value {
        let resp = self.request("tools/call", json!({"name": name, "arguments": arguments}));
        assert!(
            resp["result"]["isError"] != json!(true) && resp.get("error").is_none(),
            "{name} failed: {resp}\nstderr:\n{}",
            self.stderr()
        );
        resp
    }

    /// Wait for exit; returns the exit code (None = killed by a signal).
    fn wait_exit(&mut self) -> Option<i32> {
        let deadline = Instant::now() + EXIT_TIMEOUT;
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                // Let the reader threads drain the pipes.
                std::thread::sleep(Duration::from_millis(100));
                return status.code();
            }
            if Instant::now() > deadline {
                let _ = self.child.kill();
                panic!(
                    "server did not exit within {EXIT_TIMEOUT:?}; stderr:\n{}",
                    self.stderr()
                );
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Every stdout line the process ever wrote must be a JSON-RPC 2.0 frame.
    fn assert_stdout_pure(&self) {
        let lines = self.stdout_log.lock().unwrap().clone();
        assert!(!lines.is_empty(), "no stdout at all");
        for line in &lines {
            let msg: Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("non-JSON-RPC on stdout ({e}): {line:?}"));
            assert_eq!(msg["jsonrpc"], "2.0", "not a JSON-RPC 2.0 frame: {line}");
            assert!(
                msg.get("result").is_some()
                    || msg.get("error").is_some()
                    || msg.get("method").is_some(),
                "not a JSON-RPC response/notification: {line}"
            );
        }
    }
}

impl Drop for Raw {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Some STM writes, so the WAL has frames to checkpoint.
fn write_some_memory(p: &mut Raw) {
    for fact in ["Ada likes tea", "Ada lives in Paris", "Ada's cat is Pixel"] {
        p.call("store_memory", json!({"fact_text": fact}));
    }
}

/// Open (and keep open) an idle connection on each store of `dir`.
///
/// P2R-8: SQLite deletes the WAL when the *last* connection closes, so an
/// empty/missing `-wal` after exit proves nothing on its own. While these
/// connections stay open the server's close cannot remove the WAL; only the
/// explicit `wal_checkpoint(TRUNCATE)` at shutdown can empty it. The WAL must
/// hold frames before shutdown (asserted here) for the check to mean anything.
fn hold_stores_open(dir: &Path) -> Vec<rusqlite::Connection> {
    ["stm.sqlite", "ltm.sqlite"]
        .iter()
        .map(|store| {
            let conn = rusqlite::Connection::open(dir.join(store)).unwrap();
            // Touch the schema so the connection has attached to the WAL index,
            // then end the read (no snapshot held, so TRUNCATE can proceed).
            let _: i64 = conn
                .query_row("SELECT count(*) FROM sqlite_master", [], |r| r.get(0))
                .unwrap();
            conn
        })
        .collect()
}

/// The STM `-wal` holds un-checkpointed frames (precondition for P2R-8).
fn assert_wal_has_frames(dir: &Path) {
    let wal = dir.join("stm.sqlite-wal");
    let len = std::fs::metadata(&wal).map(|m| m.len()).unwrap_or(0);
    assert!(
        len > 0,
        "precondition: {} is empty before shutdown",
        wal.display()
    );
}

/// `<store>-wal` is gone or empty (checkpointed with TRUNCATE). Call while
/// [`hold_stores_open`] connections are still alive.
fn assert_wal_truncated(dir: &Path) {
    for store in ["stm.sqlite", "ltm.sqlite"] {
        assert!(
            dir.join(store).is_file(),
            "{store} missing in {}",
            dir.display()
        );
        let wal = dir.join(format!("{store}-wal"));
        let len = std::fs::metadata(&wal).map(|m| m.len()).unwrap_or(0);
        assert_eq!(len, 0, "{} not checkpointed ({len} bytes)", wal.display());
    }
}

/// OPS-7: with every log level enabled (`RUST_LOG=trace`, which also turns on
/// the HTTP client's and our own debug logging), stdout stays pure JSON-RPC.
#[test]
fn stdout_is_pure_jsonrpc_with_trace_logging() {
    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    let mut p = Raw::spawn(&home, &[("RUST_LOG", "trace")]);
    p.initialize();
    p.request("tools/list", json!({}));
    p.request("ping", json!({}));
    write_some_memory(&mut p);
    p.call("query_memory", json!({"query": "where does Ada live"}));
    p.call(
        "push_dialogue",
        json!({"new_message": "I moved to Lyon last week", "session_id": "s1"}),
    );
    p.call("memory_stats", json!({}));
    p.call("workspace_list", json!({}));
    // Error paths log too: a parse error and an unknown method.
    p.send("this is not json");
    p.request("no/such/method", json!({}));
    drop(p.stdin.take());
    assert_eq!(p.wait_exit(), Some(0), "stderr:\n{}", p.stderr());

    p.assert_stdout_pure();
    let stderr = p.stderr();
    assert!(
        stderr.contains("TRACE") || stderr.contains("DEBUG"),
        "trace logging was not active (nothing verbose on stderr):\n{stderr}"
    );
}

/// OPS-14: stdin EOF is the normal MCP shutdown: exit 0, WAL checkpointed.
#[test]
fn stdin_eof_exits_cleanly_and_checkpoints() {
    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    let mut p = Raw::spawn(&home, &[("RUST_LOG", "debug")]);
    p.initialize();
    write_some_memory(&mut p);
    let dir = home.workspace_dir("default");
    assert_wal_has_frames(&dir);
    let readers = hold_stores_open(&dir);
    drop(p.stdin.take());
    assert_eq!(p.wait_exit(), Some(0), "stderr:\n{}", p.stderr());
    assert!(
        p.stderr().contains("stdin closed"),
        "stderr:\n{}",
        p.stderr()
    );
    // P2R-8: proves the explicit shutdown checkpoint ran.
    assert!(
        p.stderr().contains("stores checkpointed"),
        "no checkpoint log line:\n{}",
        p.stderr()
    );
    assert_wal_truncated(&dir);
    drop(readers);
    p.assert_stdout_pure();
}

/// OPS-14: SIGTERM (supervisor / `docker stop`) while stdin is still open:
/// exit 0 promptly, WAL checkpointed, no stray stdout.
#[cfg(unix)]
#[test]
fn sigterm_exits_cleanly_and_checkpoints() {
    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    let mut p = Raw::spawn(&home, &[("RUST_LOG", "debug")]);
    p.initialize();
    write_some_memory(&mut p);
    let dir = home.workspace_dir("default");
    assert_wal_has_frames(&dir);
    let readers = hold_stores_open(&dir);
    let status = Command::new("kill")
        .args(["-TERM", &p.child.id().to_string()])
        .status()
        .expect("run kill");
    assert!(status.success());
    // stdin stays open: the signal alone must end the process.
    assert_eq!(p.wait_exit(), Some(0), "stderr:\n{}", p.stderr());
    assert!(p.stderr().contains("SIGTERM"), "stderr:\n{}", p.stderr());
    assert!(
        p.stderr().contains("stores checkpointed"),
        "no checkpoint log line:\n{}",
        p.stderr()
    );
    assert_wal_truncated(&dir);
    drop(readers);
    p.assert_stdout_pure();
}
