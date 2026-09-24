use crate::domain::models::{Episode, MemoryResult, SessionId, TenantId, TimeFilter};
use crate::domain::ports::{LlmClient, MemoryRepository};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The optimized context window returned by push_dialogue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindow {
    /// Dense summary of older messages that were compressed
    pub summary: Option<String>,
    /// The most recent raw messages still in the buffer
    pub recent_messages: Vec<String>,
    /// Relevant facts from the knowledge graph
    pub relevant_facts: Vec<MemoryResult>,
    /// Set when the message was archived but learning from it (fact
    /// extraction) failed. The dialogue itself is safely stored, so the call
    /// still succeeds — retrying would archive the message twice (REV-1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learning_error: Option<String>,
    /// Non-fatal problems while building the window after the message was
    /// archived (e.g. compression or fact recall unavailable because no LLM /
    /// embedder is configured). The window is returned regardless.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// At most this many session buffers are kept in memory; beyond it the least
/// recently used session is evicted (REV-2 / SEC-12). The raw dialogue is always
/// archived as episodes, so eviction only drops the in-memory rolling window.
pub const MAX_SESSIONS: usize = 256;

/// A session untouched for this long is evicted on the next push (REV-2).
pub const SESSION_IDLE_TTL: Duration = Duration::from_secs(6 * 60 * 60);

/// Per-session buffer entry
#[derive(Debug, Clone)]
struct SessionBuffer {
    messages: Vec<String>,
    summary: Option<String>,
    token_count: usize,
    last_used: Instant,
}

impl SessionBuffer {
    fn new(now: Instant) -> Self {
        Self {
            messages: Vec::new(),
            summary: None,
            token_count: 0,
            last_used: now,
        }
    }
}

/// Buffers are keyed by tenant **and** session, so two tenants using the same
/// session id (e.g. the default one) never share a rolling window.
fn session_key(tenant_id: &TenantId, session_id: &SessionId) -> String {
    format!("{}\u{1f}{}", tenant_id.0, session_id.0)
}

/// Drop sessions idle for longer than `ttl`, then the least recently used ones
/// until at most `max` remain. `keep` (the session being pushed to) is never
/// evicted. Returns how many sessions were dropped.
fn evict_sessions(
    sessions: &mut HashMap<String, SessionBuffer>,
    now: Instant,
    ttl: Duration,
    max: usize,
    keep: &str,
) -> usize {
    let before = sessions.len();
    sessions.retain(|k, b| k == keep || now.saturating_duration_since(b.last_used) <= ttl);
    while sessions.len() > max {
        let Some(oldest) = sessions
            .iter()
            .filter(|(k, _)| k.as_str() != keep)
            .min_by_key(|(_, b)| b.last_used)
            .map(|(k, _)| k.clone())
        else {
            break;
        };
        sessions.remove(&oldest);
    }
    before - sessions.len()
}

/// Manages per-session message buffers with token counting and context compression.
/// Implements blueprint section 2.2 (Short-Term Memory / Context Compressor).
pub struct SessionManager {
    memory_repo: Arc<dyn MemoryRepository>,
    llm_client: Arc<dyn LlmClient>,
    /// Per-session buffers, keyed by [`session_key`], bounded by
    /// [`MAX_SESSIONS`] and [`SESSION_IDLE_TTL`].
    sessions: Mutex<HashMap<String, SessionBuffer>>,
    /// Max token count before triggering compression
    token_threshold: usize,
    /// Number of recent messages to keep raw after compression
    keep_recent: usize,
}

impl SessionManager {
    pub fn new(
        memory_repo: Arc<dyn MemoryRepository>,
        llm_client: Arc<dyn LlmClient>,
        token_threshold: usize,
        keep_recent: usize,
    ) -> Self {
        Self {
            memory_repo,
            llm_client,
            sessions: Mutex::new(HashMap::new()),
            token_threshold,
            keep_recent,
        }
    }

    /// Rough token estimation: ~4 characters per token (GPT tokenizer heuristic)
    fn estimate_tokens(text: &str) -> usize {
        text.len() / 4
    }

    /// Number of session buffers currently held in memory.
    pub fn session_count(&self) -> usize {
        self.lock_sessions().len()
    }

    fn lock_sessions(&self) -> std::sync::MutexGuard<'_, HashMap<String, SessionBuffer>> {
        // A poisoned lock only means another push panicked mid-update; the map
        // itself is still structurally valid, so keep serving.
        self.sessions.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Push a new message to the session buffer.
    ///
    /// Returns the optimized context window
    /// ([Dense Summary] + [Most Recent Raw Messages] + [Relevant Graph Facts])
    /// together with the id of the episode the raw message was archived as, so
    /// the caller can attribute extracted facts to it.
    ///
    /// Only a failure to **archive** the message is an error. Once it is
    /// stored, compression or recall problems (e.g. no LLM configured) degrade
    /// the window and are reported in [`ContextWindow::warnings`] instead —
    /// failing the call would invite a retry that archives the message twice.
    pub async fn push_dialogue(
        &self,
        tenant_id: &TenantId,
        session_id: &SessionId,
        new_message: &str,
        ccl: &str,
    ) -> Result<(ContextWindow, i64)> {
        let message_tokens = Self::estimate_tokens(new_message);
        let key = session_key(tenant_id, session_id);
        let mut warnings = Vec::new();

        // 1. Archive the raw dialogue as an episode (Ground-Truth Preservation)
        let episode = Episode {
            id: None,
            tenant_id: tenant_id.clone(),
            session_id: session_id.clone(),
            raw_dialogue: new_message.to_string(),
            ccl: ccl.to_string(),
            created_at: None,
        };
        let episode_id = self.memory_repo.store_episode(&episode)?;

        // 2. Add to session buffer (evicting idle / least-recently-used sessions)
        let needs_compression = {
            let now = Instant::now();
            let mut sessions = self.lock_sessions();
            evict_sessions(&mut sessions, now, SESSION_IDLE_TTL, MAX_SESSIONS, &key);
            let buffer = sessions
                .entry(key.clone())
                .or_insert_with(|| SessionBuffer::new(now));
            buffer.last_used = now;
            buffer.messages.push(new_message.to_string());
            buffer.token_count += message_tokens;
            buffer.token_count > self.token_threshold
        };

        // 3. Compress if buffer exceeds threshold (best-effort: on failure the
        // buffer simply stays uncompressed).
        if needs_compression && let Err(e) = self.compress_buffer(&key).await {
            eprintln!("[neurolithe] context compression failed (episode {episode_id}): {e:#}");
            warnings.push(format!("context compression failed: {e:#}"));
        }

        // 4. Get the current optimized state
        let (summary, recent_messages) = {
            let sessions = self.lock_sessions();
            match sessions.get(&key) {
                Some(buffer) => (buffer.summary.clone(), buffer.messages.clone()),
                None => (None, vec![new_message.to_string()]),
            }
        };

        // 5. Retrieve relevant graph facts for the latest message (best-effort).
        let relevant_facts = match self.relevant_facts(tenant_id, new_message, ccl).await {
            Ok(facts) => facts,
            Err(e) => {
                eprintln!("[neurolithe] fact recall failed (episode {episode_id}): {e:#}");
                warnings.push(format!("fact recall failed: {e:#}"));
                Vec::new()
            }
        };

        Ok((
            ContextWindow {
                summary,
                recent_messages,
                relevant_facts,
                learning_error: None,
                warnings,
            },
            episode_id,
        ))
    }

    async fn relevant_facts(
        &self,
        tenant_id: &TenantId,
        message: &str,
        ccl: &str,
    ) -> Result<Vec<MemoryResult>> {
        let embedding = self.llm_client.embed_text(message).await?;
        self.memory_repo.query_with_graph(
            message,
            &embedding,
            tenant_id,
            &TimeFilter::default(),
            &[ccl.to_string()],
            5,
        )
    }

    /// Compress the oldest messages in a session buffer into a dense summary.
    /// A no-op if the session was evicted meanwhile.
    async fn compress_buffer(&self, key: &str) -> Result<()> {
        let (messages_to_compress, existing_summary) = {
            let sessions = self.lock_sessions();
            let Some(buffer) = sessions.get(key) else {
                return Ok(());
            };
            if buffer.messages.len() <= self.keep_recent {
                return Ok(());
            }
            let compress_count = buffer.messages.len() - self.keep_recent;
            (
                buffer.messages[..compress_count].to_vec(),
                buffer.summary.clone(),
            )
        };

        let text_to_compress = match existing_summary {
            Some(existing) => format!(
                "Previous summary: {}\n\nNew messages:\n{}",
                existing,
                messages_to_compress.join("\n")
            ),
            None => messages_to_compress.join("\n"),
        };

        // Call LLM to compress
        let new_summary = self.llm_client.compress_context(&text_to_compress).await?;

        // Update the buffer: remove exactly the compressed messages, update summary
        let mut sessions = self.lock_sessions();
        let Some(buffer) = sessions.get_mut(key) else {
            return Ok(());
        };
        let drained = messages_to_compress.len().min(buffer.messages.len());
        buffer.messages.drain(0..drained);
        buffer.summary = Some(new_summary);
        buffer.token_count = buffer
            .messages
            .iter()
            .map(|m| Self::estimate_tokens(m))
            .sum::<usize>()
            + buffer
                .summary
                .as_deref()
                .map(Self::estimate_tokens)
                .unwrap_or(0);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_estimation() {
        assert_eq!(SessionManager::estimate_tokens("hello world"), 2); // 11 chars / 4 ≈ 2
        assert_eq!(SessionManager::estimate_tokens(""), 0);
    }

    #[test]
    fn test_context_window_serialization() {
        let ctx = ContextWindow {
            summary: Some("User discussed Rust programming.".into()),
            recent_messages: vec!["What about borrowing?".into()],
            relevant_facts: vec![],
            learning_error: None,
            warnings: vec![],
        };
        let json = serde_json::to_string(&ctx).unwrap();
        assert!(json.contains("borrowing"));
        // The optional diagnostics are omitted when empty.
        assert!(!json.contains("learning_error"));
        assert!(!json.contains("warnings"));
    }

    fn buf(now: Instant, age_secs: u64) -> SessionBuffer {
        SessionBuffer::new(now - Duration::from_secs(age_secs))
    }

    /// REV-2: sessions idle past the TTL are evicted; the one being pushed to
    /// is always kept.
    #[test]
    fn evict_drops_idle_sessions() {
        let now = Instant::now() + Duration::from_secs(10_000);
        let ttl = Duration::from_secs(100);
        let mut m = HashMap::new();
        m.insert("fresh".to_string(), buf(now, 10));
        m.insert("stale".to_string(), buf(now, 500));
        m.insert("keep".to_string(), buf(now, 500));

        let dropped = evict_sessions(&mut m, now, ttl, 10, "keep");

        assert_eq!(dropped, 1);
        assert!(m.contains_key("fresh"));
        assert!(
            m.contains_key("keep"),
            "the active session is never evicted"
        );
        assert!(!m.contains_key("stale"));
    }

    /// REV-2: beyond the cap, the least recently used sessions go first.
    #[test]
    fn evict_enforces_cap_in_lru_order() {
        let now = Instant::now() + Duration::from_secs(10_000);
        let mut m = HashMap::new();
        for (k, age) in [("a", 50), ("b", 40), ("c", 30), ("d", 20), ("new", 0)] {
            m.insert(k.to_string(), buf(now, age));
        }

        evict_sessions(&mut m, now, Duration::from_secs(3600), 3, "new");

        let mut left: Vec<_> = m.keys().cloned().collect();
        left.sort();
        assert_eq!(left, vec!["c", "d", "new"]);
    }

    use crate::domain::models::CclDefinition;
    use crate::domain::ports::ExtractedFact;
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::repository::SqliteMemoryRepository;
    use crate::infrastructure::schema::init_schema;

    /// Embeds fine, but compression is unavailable.
    struct NoCompress;
    #[async_trait::async_trait]
    impl LlmClient for NoCompress {
        async fn extract_facts(
            &self,
            _d: &str,
            _c: &[CclDefinition],
        ) -> Result<Vec<ExtractedFact>> {
            Ok(vec![])
        }
        async fn generate_ccl_description(&self, _n: &str, _c: &str) -> Result<String> {
            Ok(String::new())
        }
        async fn embed_text(&self, _t: &str) -> Result<Vec<f32>> {
            Ok(vec![0.5, 0.1, 0.0, 0.0])
        }
        async fn compress_context(&self, _m: &str) -> Result<String> {
            anyhow::bail!("compressor down")
        }
    }

    fn manager(token_threshold: usize, keep_recent: usize) -> SessionManager {
        let conn = init_db(None as Option<&String>).unwrap();
        init_schema(&conn, 4).unwrap();
        SessionManager::new(
            Arc::new(SqliteMemoryRepository::new(conn)),
            Arc::new(NoCompress),
            token_threshold,
            keep_recent,
        )
    }

    /// REV-1: a compression failure after archiving degrades to an
    /// uncompressed window with a warning — it does not fail the push.
    #[tokio::test]
    async fn compression_failure_is_a_warning_not_an_error() {
        let sm = manager(0, 1);
        let (t, s) = (TenantId("t".into()), SessionId("s".into()));
        sm.push_dialogue(&t, &s, "first message", "reality")
            .await
            .unwrap();
        let (ctx, _) = sm
            .push_dialogue(&t, &s, "second message", "reality")
            .await
            .expect("archived push must succeed");
        assert_eq!(ctx.recent_messages.len(), 2, "buffer kept uncompressed");
        assert!(ctx.warnings.iter().any(|w| w.contains("compressor down")));
    }

    /// Two tenants using the same session id get separate buffers.
    #[tokio::test]
    async fn sessions_are_isolated_per_tenant() {
        let sm = manager(1_000_000, 10);
        let s = SessionId("default".into());
        sm.push_dialogue(&TenantId("a".into()), &s, "secret of a", "reality")
            .await
            .unwrap();
        let (ctx, _) = sm
            .push_dialogue(&TenantId("b".into()), &s, "hello from b", "reality")
            .await
            .unwrap();
        assert_eq!(ctx.recent_messages, vec!["hello from b".to_string()]);
        assert_eq!(sm.session_count(), 2);
    }
}
