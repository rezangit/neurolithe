//! Pithos archive client — read-only HTTP access to artifact text.
//!
//! Dereferences a `pt://archive/<dataId>/<artifactType>` pointer (from a
//! `document.completed` event) to the page text the feeder distills. A 404/gone
//! artifact yields [`FetchOutcome::Missing`] so the feeder skips gracefully
//! (ADR-0004). NeuroLithe never writes — Pithos stays the source of truth.

use crate::domain::ports::{ArtifactStore, FetchOutcome};
use crate::infrastructure::llm::build_http_client;
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use std::time::Duration;

/// Total per-request budget for a Pithos fetch. A hung archive must not stall
/// the (single-threaded) feeder forever (SEC-11 / REV-3).
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

pub struct PithosClient {
    base_url: String,
    /// Pithos bearer token (read access). Empty = send no auth header.
    token: String,
    http: reqwest::Client,
}

impl PithosClient {
    pub fn new(base_url: impl Into<String>, token: impl Into<String>) -> Self {
        Self::with_timeout(base_url, token, DEFAULT_REQUEST_TIMEOUT)
    }

    /// Like [`new`](Self::new) with an explicit total request timeout. Uses the
    /// shared client builder, so connect + total timeouts always apply.
    pub fn with_timeout(
        base_url: impl Into<String>,
        token: impl Into<String>,
        request_timeout: Duration,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            token: token.into(),
            http: build_http_client(request_timeout),
        }
    }
}

/// Map a `pt://` logical URI to an HTTP URL under the configured base.
///
/// `pt://archive/<dataId>/<artifactType>` -> `<base_url>/archive/<dataId>/<artifactType>`
/// — matches the path the sibling consumers (Aristotle/Cadmus) use. Requests
/// carry a Pithos bearer token (read access) when configured.
fn pt_uri_to_url(base_url: &str, uri: &str) -> String {
    let path = uri.strip_prefix("pt://").unwrap_or(uri);
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

#[async_trait]
impl ArtifactStore for PithosClient {
    async fn fetch_text(&self, uri: &str) -> Result<FetchOutcome> {
        let url = pt_uri_to_url(&self.base_url, uri);
        let mut request = self.http.get(&url);
        if !self.token.is_empty() {
            request = request.bearer_auth(&self.token);
        }
        let resp = request
            .send()
            .await
            .with_context(|| format!("Pithos GET {url}"))?;

        let status = resp.status();
        if status.is_success() {
            let text = resp
                .text()
                .await
                .with_context(|| format!("reading Pithos body for {url}"))?;
            Ok(FetchOutcome::Found(text))
        } else if status == reqwest::StatusCode::NOT_FOUND || status == reqwest::StatusCode::GONE {
            // Expected: the artifact is gone. Skip, don't fail.
            Ok(FetchOutcome::Missing)
        } else {
            // 5xx / unexpected — transient; let the caller retry (ADR-0004).
            Err(anyhow!("Pithos GET {url} failed with status {status}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn test_pt_uri_to_url_maps_scheme() {
        assert_eq!(
            pt_uri_to_url("http://host:8080", "pt://archive/doc_1/text"),
            "http://host:8080/archive/doc_1/text"
        );
        // Trailing slash on base + a non-pt path are both normalized.
        assert_eq!(
            pt_uri_to_url("http://host:8080/", "/archive/doc_1/text"),
            "http://host:8080/archive/doc_1/text"
        );
    }

    /// Serve a single canned HTTP response on a loopback port; return the port.
    async fn serve_once(status_line: &'static str, body: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await; // consume the request
                let resp = format!(
                    "{status_line}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.flush().await;
            }
        });
        port
    }

    #[tokio::test]
    async fn test_fetch_found_returns_body() {
        let port = serve_once("HTTP/1.1 200 OK", "the page text").await;
        let client = PithosClient::new(format!("http://127.0.0.1:{port}"), "");

        let outcome = client.fetch_text("pt://archive/doc_1/text").await.unwrap();
        assert_eq!(outcome, FetchOutcome::Found("the page text".into()));
    }

    #[tokio::test]
    async fn test_fetch_missing_is_skipped_not_error() {
        let port = serve_once("HTTP/1.1 404 Not Found", "").await;
        let client = PithosClient::new(format!("http://127.0.0.1:{port}"), "");

        // A missing artifact is a typed skip, never an Err.
        let outcome = client.fetch_text("pt://archive/gone/text").await.unwrap();
        assert_eq!(outcome, FetchOutcome::Missing);
    }

    /// A configured token is sent as a Bearer Authorization header.
    #[tokio::test]
    async fn test_sends_bearer_token() {
        use tokio::sync::oneshot;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = vec![0u8; 2048];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let _ = sock
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nhi",
                    )
                    .await;
                let _ = tx.send(req);
            }
        });

        let client = PithosClient::new(format!("http://127.0.0.1:{port}"), "secret-tok");
        client.fetch_text("pt://archive/d/text").await.unwrap();

        let req = rx.await.unwrap().to_lowercase();
        assert!(
            req.contains("authorization: bearer secret-tok"),
            "request was: {req}"
        );
    }

    /// REV-3 / SEC-11: an archive that accepts the connection but never
    /// answers must time out instead of hanging the feeder forever (the old
    /// `reqwest::Client::new()` had no timeout at all).
    #[tokio::test]
    async fn test_hung_archive_times_out() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let _hold = tokio::spawn(async move {
            let (_sock, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(30)).await;
        });

        let client = PithosClient::with_timeout(
            format!("http://127.0.0.1:{port}"),
            "",
            Duration::from_millis(300),
        );
        let started = std::time::Instant::now();
        let res = tokio::time::timeout(
            Duration::from_secs(5),
            client.fetch_text("pt://archive/d/text"),
        )
        .await
        .expect("fetch must time out on its own, not hang");
        assert!(res.is_err(), "a hung archive is a (retryable) error");
        assert!(started.elapsed() < Duration::from_secs(3));
    }
}
