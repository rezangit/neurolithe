//! W1R-1: a session never ends with work half done. Tool calls still running
//! when the client closes stdin, calls queued during a slow startup, and
//! calls in flight when SIGTERM arrives all finish, and their responses go
//! out, before the process exits.
mod common;

use common::{FakeLlm, Harness, Home, SLOW_MARKER};
use serde_json::{Value, json};
use std::time::Duration;

/// Every message the server writes until it closes stdout.
fn drain(server: &mut common::McpProcess) -> Vec<Value> {
    let mut out = Vec::new();
    while let Some(m) = server.read_message(Duration::from_secs(40)) {
        out.push(m);
    }
    out
}

fn response(messages: &[Value], id: Value) -> Value {
    messages
        .iter()
        .find(|m| m["id"] == id)
        .cloned()
        .unwrap_or_else(|| panic!("no response to id {id}: {messages:?}"))
}

/// The STM fact texts of `home`'s default workspace, read by a fresh server.
fn facts(home: &Home) -> Vec<String> {
    let mut s = home.spawn_mcp(&[], &[]);
    s.initialize();
    let res = s.call_tool("inspect", json!({"view": "stm", "limit": 500}));
    res.json()
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|f| f["fact"].as_str().map(str::to_string))
        .collect()
}

fn store_call(id: i64, fact: &str) -> String {
    json!({"jsonrpc": "2.0", "id": id, "method": "tools/call",
           "params": {"name": "store_memory", "arguments": {"fact_text": fact}}})
    .to_string()
}

/// A slow call (its embedding takes longer than rmcp's own 5 s drain) is
/// still answered and its fact stored after the client closes stdin.
#[test]
fn eof_waits_for_the_in_flight_call() {
    let mut h = Harness::start();
    let fact = format!("{SLOW_MARKER} Lena keeps bees");
    h.server.send_raw(&store_call(100, &fact));
    std::thread::sleep(Duration::from_millis(300));
    h.server.close_stdin();

    let messages = drain(&mut h.server);
    let resp = response(&messages, json!(100));
    assert_eq!(resp["result"]["isError"], false, "{resp}");
    assert_eq!(
        h.server.wait_exit(Duration::from_secs(20)),
        Some(0),
        "stderr:\n{}",
        h.server.stderr()
    );
    assert!(facts(&h.home).contains(&fact), "the fact was not stored");
}

/// Requests piped in while startup is still slow (the embedder probe and the
/// spine take longer than 5 s), then EOF: every queued call still runs and
/// is answered.
#[test]
fn startup_then_eof_runs_the_queued_calls() {
    let llm = FakeLlm::start();
    llm.set_embed_delay(Duration::from_secs(3));
    let home = Home::new(&llm.base_url());
    let mut server = home.spawn_mcp(&[], &[]);
    server.send_raw(
        &json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "neurolithe-tests", "version": "0"}}})
        .to_string(),
    );
    server.notify("notifications/initialized", json!({}));
    server.send_raw(&store_call(2, "Queued: Omar plays oud"));
    server.send_raw(
        &json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "workspace_current", "arguments": {}}})
        .to_string(),
    );
    server.close_stdin();

    let messages = drain(&mut server);
    assert!(response(&messages, json!(1))["result"].is_object());
    for id in [2, 3] {
        let resp = response(&messages, json!(id));
        assert_eq!(resp["result"]["isError"], false, "{resp}");
    }
    assert_eq!(
        server.wait_exit(Duration::from_secs(20)),
        Some(0),
        "stderr:\n{}",
        server.stderr()
    );
    llm.set_embed_delay(Duration::ZERO);
    assert!(facts(&home).contains(&"Queued: Omar plays oud".to_string()));
}

/// SIGTERM while a call is running: the call finishes (and is answered)
/// before the stores are checkpointed and the process exits.
#[cfg(unix)]
#[test]
fn sigterm_waits_for_the_in_flight_call() {
    let mut h = Harness::start();
    let fact = format!("{SLOW_MARKER} Ravi restores clocks");
    h.server.send_raw(&store_call(100, &fact));
    std::thread::sleep(Duration::from_millis(300));
    h.server.signal("TERM");

    let messages = drain(&mut h.server);
    let resp = response(&messages, json!(100));
    assert_eq!(resp["result"]["isError"], false, "{resp}");
    assert_eq!(
        h.server.wait_exit(Duration::from_secs(20)),
        Some(0),
        "stderr:\n{}",
        h.server.stderr()
    );
    assert!(h.server.stderr().contains("SIGTERM"));
    assert!(facts(&h.home).contains(&fact), "the fact was not stored");
}

/// W1R-5: a call the client cancelled gets no reply (rmcp drops it), so it
/// must not hold the session open at EOF: the process exits promptly, not
/// after the 300 s safety cap.
#[test]
fn eof_after_a_cancelled_call_exits_promptly() {
    let mut h = Harness::start();
    h.server
        .send_raw(&store_call(200, &format!("{SLOW_MARKER} cancelled")));
    std::thread::sleep(Duration::from_millis(300));
    h.server.notify(
        "notifications/cancelled",
        json!({"requestId": 200, "reason": "user pressed Esc"}),
    );
    std::thread::sleep(Duration::from_millis(100));
    h.server.close_stdin();
    let started = std::time::Instant::now();
    let _ = drain(&mut h.server);
    let code = h.server.wait_exit(Duration::from_secs(60));
    let took = started.elapsed();
    assert_eq!(code, Some(0), "stderr:\n{}", h.server.stderr());
    assert!(
        took < Duration::from_secs(12),
        "EOF held {took:?} by a cancelled call; stderr:\n{}",
        h.server.stderr()
    );
}

/// W1R-6: SIGTERM with call A running and call B queued behind it: both
/// get their reply (A its result, B "shutting down") before the exit.
#[cfg(unix)]
#[test]
fn sigterm_answers_the_running_and_the_queued_call() {
    let mut h = Harness::start();
    h.server
        .send_raw(&store_call(100, &format!("{SLOW_MARKER} running")));
    h.server.send_raw(&store_call(101, "queued behind it"));
    std::thread::sleep(Duration::from_millis(300));
    h.server.signal("TERM");

    let messages = drain(&mut h.server);
    let a = response(&messages, json!(100));
    assert_eq!(a["result"]["isError"], false, "{a}");
    let b = response(&messages, json!(101));
    assert_eq!(b["result"]["isError"], true, "{b}");
    assert!(b.to_string().contains("shutting down"), "{b}");
    assert_eq!(
        h.server.wait_exit(Duration::from_secs(20)),
        Some(0),
        "stderr:\n{}",
        h.server.stderr()
    );
}

/// W1R-8: a request reusing the id of one still in flight is refused
/// (-32600) instead of being silently dropped and holding EOF open.
#[test]
fn duplicate_in_flight_id_is_refused_and_eof_stays_prompt() {
    let mut h = Harness::start();
    h.server
        .send_raw(&store_call(300, &format!("{SLOW_MARKER} first")));
    h.server.send_raw(&store_call(300, "reuses the id"));
    std::thread::sleep(Duration::from_millis(300));
    h.server.close_stdin();
    let started = std::time::Instant::now();
    let messages = drain(&mut h.server);
    let code = h.server.wait_exit(Duration::from_secs(60));
    let took = started.elapsed();
    assert_eq!(code, Some(0), "stderr:\n{}", h.server.stderr());
    assert!(took < Duration::from_secs(12), "EOF held {took:?}");

    let for_300: Vec<&Value> = messages.iter().filter(|m| m["id"] == json!(300)).collect();
    assert_eq!(for_300.len(), 2, "{messages:?}");
    assert!(
        for_300.iter().any(|m| m["error"]["code"] == json!(-32600)),
        "the duplicate is refused: {for_300:?}"
    );
    assert!(
        for_300
            .iter()
            .any(|m| m["result"]["isError"] == json!(false)),
        "the first is answered: {for_300:?}"
    );
}
