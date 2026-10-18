//! Newline-delimited JSON-RPC over a byte stream (stdio), as an rmcp
//! [`Transport`].
//!
//! rmcp ships an async read/write transport, but it buffers a whole line
//! however long it is and silently drops unparsable input. This one keeps
//! our stdio contract:
//! - a line longer than [`MAX_LINE_BYTES`] is drained without being
//!   buffered whole and answered with `-32600`, its `id` recovered from the
//!   line's head when possible (SEC-12);
//! - a line that isn't JSON is answered with `-32700` (id `null`);
//! - well-formed JSON that isn't a valid message is answered with `-32600`
//!   (or `-32602` when it names a method), echoing its `id`;
//! - a final line without `\n` is still served;
//! - stray notifications before the first request are ignored (rmcp would
//!   otherwise end the session), and a request before `initialize` is told
//!   to initialize first;
//! - a request reusing the id of one still in flight is refused (`-32600`):
//!   rmcp would answer only one of them;
//!
//! - at stdin EOF, every request already handed to rmcp is answered before
//!   `receive` reports the end (W1R-1). rmcp would otherwise end the session
//!   after a fixed 5 s drain, dropping the replies of slower calls and
//!   abandoning their work — the old hand-rolled loop never did that.
//!
//! Reads are cancellation-safe: rmcp polls [`Transport::receive`] inside a
//! `select!`, so all partial-line state lives in the struct, not the future.

use rmcp::RoleServer;
use rmcp::model::{
    CallToolRequestParams, ClientRequest, InitializeRequestParams, PaginatedRequestParams,
    RequestId,
};
use rmcp::service::{RxJsonRpcMessage, TxJsonRpcMessage};
use rmcp::transport::Transport;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::Mutex;
use tokio::sync::Notify;

use super::limits::MAX_LINE_BYTES;

/// How much of an oversized line is kept to recover its request `id`.
const ID_PROBE_BYTES: usize = 1024;

/// Upper bound on waiting, at EOF, for the replies still owed. Generous: a
/// tool call may wait on a slow model or remote LLM. Past it the session
/// ends anyway (with a warning) rather than hang forever.
const EOF_DRAIN_TIMEOUT: Duration = Duration::from_secs(300);

/// The `_meta` key a 2026-07-28 (initialize-less) request carries.
const INLINE_VERSION_KEY: &str = "io.modelcontextprotocol/protocolVersion";

/// Requests handed to rmcp whose reply hasn't been written yet, counted per
/// id (defensive: a reused in-flight id is refused before it gets here).
/// Shared with [`super::McpServer`], whose shutdown waits for it to empty.
#[derive(Default)]
pub(crate) struct Owed {
    ids: std::sync::Mutex<HashMap<String, usize>>,
    settled: Notify,
}

impl Owed {
    fn key(id: &RequestId) -> String {
        serde_json::to_string(id).unwrap_or_default()
    }

    fn add(&self, id: &RequestId) {
        *self.ids.lock().unwrap().entry(Self::key(id)).or_default() += 1;
    }

    /// One request with this id is answered (or will never be).
    fn settle(&self, id: &RequestId) {
        let mut ids = self.ids.lock().unwrap();
        let key = Self::key(id);
        let Some(n) = ids.get_mut(&key) else {
            return;
        };
        *n -= 1;
        if *n == 0 {
            ids.remove(&key);
        }
        if ids.is_empty() {
            self.settled.notify_waiters();
        }
    }

    fn contains(&self, id: &RequestId) -> bool {
        self.ids.lock().unwrap().contains_key(&Self::key(id))
    }

    pub(crate) fn count(&self) -> usize {
        self.ids.lock().unwrap().values().sum()
    }

    /// Wait until every owed reply has been written.
    pub(crate) async fn all_settled(&self) {
        loop {
            let settled = self.settled.notified();
            tokio::pin!(settled);
            settled.as_mut().enable();
            if self.count() == 0 {
                return;
            }
            settled.await;
        }
    }
}

pub struct LineTransport<R, W> {
    reader: R,
    writer: Arc<Mutex<W>>,
    max_line: usize,
    /// Bytes of the line being read (or its head, once it overflowed).
    buf: Vec<u8>,
    overflow: bool,
    /// A reply the transport itself owes (to a bad line), being written by
    /// a detached task so a cancelled `receive` can't cut it in half.
    pending_reply: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    /// Set once a request has arrived; notifications before that are dropped.
    seen_request: bool,
    /// Set by a well-formed `initialize` (or an inline-lifecycle request).
    session_started: bool,
    eof: bool,
    owed: Arc<Owed>,
    eof_drain: Duration,
}

enum Line {
    Text(Vec<u8>),
    TooLong(Vec<u8>),
    Eof,
}

impl<R, W> LineTransport<R, W>
where
    R: AsyncBufRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    pub fn new(reader: R, writer: W) -> Self {
        Self::with_max_line(reader, writer, MAX_LINE_BYTES)
    }

    pub fn with_max_line(reader: R, writer: W, max_line: usize) -> Self {
        Self {
            reader,
            writer: Arc::new(Mutex::new(writer)),
            max_line,
            buf: Vec::new(),
            overflow: false,
            pending_reply: None,
            seen_request: false,
            session_started: false,
            eof: false,
            owed: Arc::default(),
            eof_drain: EOF_DRAIN_TIMEOUT,
        }
    }

    /// Track owed replies in `owed` (shared with the server's shutdown).
    pub(crate) fn with_owed(mut self, owed: Arc<Owed>) -> Self {
        self.owed = owed;
        self
    }

    /// Bound the wait for owed replies at EOF (tests).
    #[cfg(test)]
    fn with_eof_drain(mut self, eof_drain: Duration) -> Self {
        self.eof_drain = eof_drain;
        self
    }

    /// Read one `\n`-terminated line of at most `max_line` bytes. An
    /// oversized line is consumed chunk by chunk and never buffered whole.
    async fn next_line(&mut self) -> std::io::Result<Line> {
        if self.eof {
            return Ok(Line::Eof);
        }
        loop {
            let (consumed, done) = {
                let available = self.reader.fill_buf().await?;
                if available.is_empty() {
                    self.eof = true;
                    break;
                }
                let newline = available.iter().position(|&b| b == b'\n');
                let chunk = &available[..newline.unwrap_or(available.len())];
                if !self.overflow {
                    if self.buf.len() + chunk.len() > self.max_line {
                        self.overflow = true;
                        let room = ID_PROBE_BYTES.saturating_sub(self.buf.len());
                        self.buf.extend_from_slice(&chunk[..room.min(chunk.len())]);
                        self.buf.truncate(ID_PROBE_BYTES);
                    } else {
                        self.buf.extend_from_slice(chunk);
                    }
                }
                match newline {
                    Some(i) => (i + 1, true),
                    None => (available.len(), false),
                }
            };
            // Nothing is awaited between fill_buf and consume, so the bytes
            // accounted for above are exactly the ones consumed.
            self.reader.consume(consumed);
            if done {
                break;
            }
        }
        let line = std::mem::take(&mut self.buf);
        let overflow = std::mem::replace(&mut self.overflow, false);
        if overflow {
            return Ok(Line::TooLong(line));
        }
        if line.is_empty() && self.eof {
            return Ok(Line::Eof);
        }
        Ok(Line::Text(line))
    }

    /// Queue a reply to a bad line; it's written before anything else is read.
    fn reply(&mut self, message: Value) {
        let writer = self.writer.clone();
        self.pending_reply = Some(tokio::spawn(async move {
            write_line(&writer, &message.to_string()).await
        }));
    }

    /// Decide what to do with one complete line.
    fn handle_line(&mut self, line: &[u8]) -> Option<RxJsonRpcMessage<RoleServer>> {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let line = line.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(line);
        if line.iter().all(u8::is_ascii_whitespace) {
            return None;
        }
        let value: Value = match serde_json::from_slice(line) {
            Ok(v) => v,
            Err(e) => {
                self.reply(error(Value::Null, -32700, format!("Parse error: {e}")));
                return None;
            }
        };
        if !value.is_object() {
            // e.g. a JSON-RPC batch, which MCP no longer allows.
            self.reply(error(
                Value::Null,
                -32600,
                "Invalid request: expected a JSON object".into(),
            ));
            return None;
        }
        let id = value.get("id").cloned();
        let method = value
            .get("method")
            .and_then(Value::as_str)
            .map(str::to_owned);
        // rmcp drops the reply to a cancelled request, so it is no longer
        // owed (W1R-5). Settling an id that was already answered is a no-op.
        if method.as_deref() == Some("notifications/cancelled")
            && let Ok(cancelled) =
                serde_json::from_value::<RequestId>(value["params"]["requestId"].clone())
        {
            self.owed.settle(&cancelled);
        }
        // Before any session exists, rmcp answers a request with "request
        // _meta is missing … protocolVersion" (the 2026-07-28 inline
        // lifecycle). Most clients just forgot `initialize`: say that.
        if !self.session_started
            && let (Some(id), Some(method)) = (&id, method.as_deref())
        {
            let inline = value["params"]["_meta"][INLINE_VERSION_KEY].is_string();
            match method {
                "initialize" | "ping" | "server/discover" => {}
                _ if inline => self.session_started = true,
                _ => {
                    self.reply(error(
                        id.clone(),
                        -32602,
                        "Session not initialized: send initialize first (protocol 2026-07-28 \
                         clients: include the request _meta fields instead)"
                            .into(),
                    ));
                    return None;
                }
            }
        }
        match serde_json::from_value::<RxJsonRpcMessage<RoleServer>>(value) {
            Ok(msg) => {
                // rmcp falls back to a "custom" request when a standard
                // method's params don't parse, and would then answer
                // "method not found" (or, before `initialize`, ask for
                // 2026-07-28 `_meta`). Say what's actually wrong.
                if let RxJsonRpcMessage::<RoleServer>::Request(req) = &msg
                    && let ClientRequest::CustomRequest(custom) = &req.request
                    && let Some(problem) = standard_params_error(&custom.method, &custom.params)
                {
                    let id = serde_json::to_value(&req.id).unwrap_or(Value::Null);
                    self.reply(error(id, -32602, format!("Invalid params: {problem}")));
                    return None;
                }
                let is_request = matches!(msg, RxJsonRpcMessage::<RoleServer>::Request(_));
                if let RxJsonRpcMessage::<RoleServer>::Request(req) = &msg {
                    // rmcp tracks in-flight requests by id and answers only
                    // one per id (W1R-8): refuse a reuse up front.
                    if self.owed.contains(&req.id) {
                        let id = serde_json::to_value(&req.id).unwrap_or(Value::Null);
                        self.reply(error(
                            id,
                            -32600,
                            "Invalid request: duplicate request id (a request with this id \
                             is still in flight)"
                                .into(),
                        ));
                        return None;
                    }
                    self.owed.add(&req.id);
                    if matches!(req.request, ClientRequest::InitializeRequest(_)) {
                        self.session_started = true;
                    }
                }
                if !is_request && !self.seen_request {
                    tracing::debug!("ignoring a non-request message before the first request");
                    return None;
                }
                self.seen_request |= is_request;
                Some(msg)
            }
            // A notification (no id) never gets a reply, even when malformed.
            Err(e) => {
                if let Some(id) = id {
                    let (code, kind) = match method {
                        Some(_) => (-32602, "Invalid params"),
                        None => (-32600, "Invalid request"),
                    };
                    self.reply(error(id, code, format!("{kind}: {e}")));
                } else {
                    tracing::debug!("ignoring a malformed message without an id: {e}");
                }
                None
            }
        }
    }
}

impl<R, W> Transport<RoleServer> for LineTransport<R, W>
where
    R: AsyncBufRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    type Error = std::io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleServer>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let writer = self.writer.clone();
        let owed = self.owed.clone();
        let answers = match &item {
            TxJsonRpcMessage::<RoleServer>::Response(r) => Some(r.id.clone()),
            TxJsonRpcMessage::<RoleServer>::Error(e) => e.id.clone(),
            _ => None,
        };
        async move {
            let text = serde_json::to_string(&item);
            let written = match text {
                Ok(text) => write_line(&writer, &text).await,
                Err(e) => Err(e.into()),
            };
            // Settled even when the write failed: the client is gone then.
            if let Some(id) = answers {
                owed.settle(&id);
            }
            written
        }
    }

    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleServer>> {
        loop {
            if let Some(pending) = self.pending_reply.as_mut() {
                let written = pending.await;
                self.pending_reply = None;
                if !matches!(written, Ok(Ok(()))) {
                    return None; // the client is gone
                }
            }
            match self.next_line().await {
                Ok(Line::Eof) => {
                    // Let rmcp finish and answer what it was given first.
                    if tokio::time::timeout(self.eof_drain, self.owed.all_settled())
                        .await
                        .is_err()
                    {
                        tracing::warn!(
                            "stdin closed; gave up waiting for {} unanswered request(s) after {:?}",
                            self.owed.count(),
                            self.eof_drain
                        );
                    }
                    return None;
                }
                Ok(Line::TooLong(head)) => {
                    let max = self.max_line;
                    self.reply(error(
                        probe_request_id(&head),
                        -32600,
                        format!("Request too large (max {max} bytes per line)"),
                    ));
                }
                Ok(Line::Text(line)) => {
                    if let Some(msg) = self.handle_line(&line) {
                        return Some(msg);
                    }
                }
                Err(e) => {
                    tracing::error!("reading stdin failed: {e}");
                    return None;
                }
            }
        }
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        if let Some(pending) = self.pending_reply.take() {
            let _ = pending.await;
        }
        self.writer.lock().await.flush().await
    }
}

/// Why a standard MCP method's params failed to parse; `None` for
/// non-standard methods (those are rmcp's "method not found").
fn standard_params_error(method: &str, params: &Option<Value>) -> Option<String> {
    let params = params.clone().unwrap_or(Value::Null);
    let parse_err = match method {
        "initialize" => serde_json::from_value::<InitializeRequestParams>(params).err(),
        "tools/call" => serde_json::from_value::<CallToolRequestParams>(params).err(),
        "tools/list" | "ping" => {
            serde_json::from_value::<Option<PaginatedRequestParams>>(params).err()
        }
        _ => return None,
    };
    Some(parse_err.map_or_else(|| "malformed params".to_string(), |e| e.to_string()))
}

async fn write_line<W: AsyncWrite + Unpin>(writer: &Mutex<W>, text: &str) -> std::io::Result<()> {
    let mut w = writer.lock().await;
    w.write_all(text.as_bytes()).await?;
    w.write_all(b"\n").await?;
    w.flush().await
}

fn error(id: Value, code: i32, message: String) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// Best-effort recovery of a JSON-RPC `id` (number or string) from the start
/// of a truncated request, so the client gets an error it can match instead of
/// waiting for a reply that never comes. `Null` when not found.
///
/// Only a **top-level** `"id"` key counts: the scan tracks string state and
/// object/array depth, so a nested key (e.g. `params.arguments.id`, which comes
/// first when `params` precedes `id`) is never mistaken for the request id
/// (REV-5). Once the scan reaches the truncation point the answer is `Null`.
pub(crate) fn probe_request_id(head: &[u8]) -> Value {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// REV-5: only a top-level `id` is recovered. With `params` before `id`, a
    /// nested `arguments.id` comes first in the byte stream and must be skipped;
    /// if the real id was truncated away, the answer is null (never a wrong id).
    #[test]
    fn test_probe_request_id_ignores_nested_ids() {
        let nested_first =
            br#"{"jsonrpc":"2.0","params":{"name":"inspect","arguments":{"id":999}},"id":5}"#;
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

    type TestTransport =
        LineTransport<tokio::io::BufReader<std::io::Cursor<Vec<u8>>>, tokio::io::DuplexStream>;

    /// A transport over in-memory input; returns it and the output buffer.
    fn transport(input: &[u8], max_line: usize) -> (TestTransport, tokio::io::DuplexStream) {
        let (out_w, out_r) = tokio::io::duplex(1 << 20);
        let reader = tokio::io::BufReader::with_capacity(16, std::io::Cursor::new(input.to_vec()));
        let t = LineTransport::with_max_line(reader, out_w, max_line)
            .with_eof_drain(Duration::from_millis(50));
        (t, out_r)
    }

    async fn written(mut out: tokio::io::DuplexStream, t: impl Sized) -> Vec<Value> {
        use tokio::io::AsyncReadExt;
        drop(t);
        let mut s = String::new();
        out.read_to_string(&mut s).await.unwrap();
        s.lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    /// SEC-12: a line over the cap is drained (in small chunks — the reader's
    /// buffer is 16 bytes) and answered with -32600 carrying the recovered id;
    /// the next request is still read.
    #[tokio::test]
    async fn test_oversized_line_rejected_and_next_line_read() {
        let mut input =
            br#"{"id":41,"jsonrpc":"2.0","method":"tools/call","params":{"x":""#.to_vec();
        input.extend(std::iter::repeat_n(b'x', 5000));
        input.extend_from_slice(b"\"}}\n");
        input.extend_from_slice(br#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#);
        input.push(b'\n');
        let (mut t, out) = transport(&input, 1000);
        let msg = t.receive().await.expect("the ping");
        assert!(matches!(msg, RxJsonRpcMessage::<RoleServer>::Request(_)));
        assert!(t.receive().await.is_none());
        t.close().await.unwrap();
        let lines = written(out, t).await;
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0]["error"]["code"], -32600);
        assert_eq!(
            lines[0]["id"], 41,
            "the oversized request's id is recovered"
        );
    }

    /// Bad JSON → -32700 with a null id; a malformed request → an error that
    /// echoes its id; a malformed notification → nothing. A final line without
    /// `\n` is still delivered, and notifications before the first request
    /// are dropped.
    #[tokio::test]
    async fn test_bad_lines_answered_and_unterminated_last_line_served() {
        let input = b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n\
            not json\n\
            \n\
            {\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{}}\n\
            {\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\"}}\n\
            {\"jsonrpc\":\"2.0\",\"method\":\"tools/call\",\"params\":{}}\n\
            [1,2]\n\
            {\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}";
        let (mut t, out) = transport(input, 4096);
        let msg = t.receive().await.expect("the ping");
        let RxJsonRpcMessage::<RoleServer>::Request(req) = msg else {
            panic!("expected a request");
        };
        assert_eq!(serde_json::to_value(&req.id).unwrap(), json!(9));
        assert!(t.receive().await.is_none());
        t.close().await.unwrap();
        let lines = written(out, t).await;
        let codes: Vec<(Value, Value)> = lines
            .iter()
            .map(|l| (l["id"].clone(), l["error"]["code"].clone()))
            .collect();
        assert_eq!(
            codes,
            vec![
                (Value::Null, json!(-32700)),
                (json!(3), json!(-32602)),
                (json!(4), json!(-32602)),
                (Value::Null, json!(-32600)),
            ],
        );
    }

    /// W1R-1: at EOF, `receive` reports the end only once every request it
    /// handed out has been answered, so rmcp can't cut a slow call off.
    #[tokio::test]
    async fn test_eof_waits_until_owed_replies_are_written() {
        use rmcp::model::{ServerJsonRpcMessage, ServerResult};
        let input = b"{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n";
        let (t, out) = transport(input, 4096);
        let mut t = t.with_eof_drain(Duration::from_secs(30));
        let RxJsonRpcMessage::<RoleServer>::Request(req) = t.receive().await.unwrap() else {
            panic!("expected a request");
        };
        // EOF is next, but the ping is still owed: receive must wait.
        let early = tokio::time::timeout(Duration::from_millis(200), t.receive()).await;
        assert!(early.is_err(), "EOF reported with a reply still owed");
        let reply = ServerJsonRpcMessage::response(ServerResult::empty(()), req.id);
        t.send(reply).await.unwrap();
        let end = tokio::time::timeout(Duration::from_secs(5), t.receive()).await;
        assert!(matches!(end, Ok(None)), "EOF once the reply is out");
        t.close().await.unwrap();
        let lines = written(out, t).await;
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0]["id"], 7);
    }

    /// W1R-5: a request the client cancelled is no longer owed (rmcp drops
    /// its reply), so EOF isn't held up by it.
    #[tokio::test]
    async fn test_cancelled_request_is_not_owed() {
        let input = b"{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n\
            {\"jsonrpc\":\"2.0\",\"method\":\"notifications/cancelled\",\"params\":{\"requestId\":7}}\n";
        let (t, _out) = transport(input, 4096);
        let mut t = t.with_eof_drain(Duration::from_secs(30));
        assert!(t.receive().await.is_some(), "the ping");
        assert!(t.receive().await.is_some(), "the cancellation is passed on");
        let end = tokio::time::timeout(Duration::from_secs(2), t.receive()).await;
        assert!(
            matches!(end, Ok(None)),
            "EOF not held by a cancelled request"
        );
    }

    /// W1R-8: rmcp can answer only one in-flight request per id, so a second
    /// request reusing a still-owed id is refused here (-32600) — never passed
    /// on, never owed — and can't hold EOF.
    #[tokio::test]
    async fn test_duplicate_in_flight_id_is_refused() {
        use rmcp::model::{ServerJsonRpcMessage, ServerResult};
        let input = b"{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n\
            {\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n";
        let (t, out) = transport(input, 4096);
        let mut t = t.with_eof_drain(Duration::from_secs(30));
        let RxJsonRpcMessage::<RoleServer>::Request(req) = t.receive().await.unwrap() else {
            panic!("expected a request");
        };
        // The duplicate isn't handed out; EOF waits only for the first.
        let early = tokio::time::timeout(Duration::from_millis(200), t.receive()).await;
        assert!(early.is_err(), "the first ping is still owed");
        t.send(ServerJsonRpcMessage::response(
            ServerResult::empty(()),
            req.id,
        ))
        .await
        .unwrap();
        let end = tokio::time::timeout(Duration::from_secs(2), t.receive()).await;
        assert!(
            matches!(end, Ok(None)),
            "EOF once the one owed reply is out"
        );
        t.close().await.unwrap();
        let lines = written(out, t).await;
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[0]["id"], 7);
        assert_eq!(lines[0]["error"]["code"], -32600, "{lines:?}");
        assert!(
            lines[0]["error"]["message"]
                .as_str()
                .unwrap()
                .contains("duplicate")
        );
        assert_eq!(lines[1]["id"], 7);
        assert!(lines[1].get("result").is_some(), "{lines:?}");

        // Once answered, the id may be reused.
        let input = b"{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n";
        let (t, _out) = transport(input, 4096);
        let mut t = t.with_eof_drain(Duration::from_secs(30));
        let RxJsonRpcMessage::<RoleServer>::Request(req) = t.receive().await.unwrap() else {
            panic!("expected a request");
        };
        t.send(ServerJsonRpcMessage::response(
            ServerResult::empty(()),
            req.id.clone(),
        ))
        .await
        .unwrap();
        assert_eq!(t.owed.count(), 0);
    }
}
