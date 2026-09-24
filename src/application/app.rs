use crate::application::retrieval::RetrievalService;
use crate::application::session_manager::{ContextWindow, SessionManager};
use crate::application::sleep::SleepWorker;
use crate::domain::models::{Episode, MemoryNode, MemoryResult, SessionId, TenantId, TimeFilter};
use crate::domain::ports::{LlmClient, MemoryRepository};
use anyhow::Result;
use std::sync::Arc;

pub struct NeurolitheApp {
    memory_repo: Arc<dyn MemoryRepository>,
    llm_client: Arc<dyn LlmClient>,
    retrieval_service: RetrievalService,
    sleep_worker: SleepWorker,
    session_manager: SessionManager,
}

// SAFETY: All fields are either Arc (Send+Sync) or use std::sync::Mutex internally.
unsafe impl Send for NeurolitheApp {}
unsafe impl Sync for NeurolitheApp {}

impl NeurolitheApp {
    pub fn new(
        memory_repo: Arc<dyn MemoryRepository>,
        llm_client: Arc<dyn LlmClient>,
        default_half_life_days: f64,
        working_half_life_days: f64,
    ) -> Self {
        Self {
            memory_repo: memory_repo.clone(),
            llm_client: llm_client.clone(),
            retrieval_service: RetrievalService::new(llm_client.clone(), memory_repo.clone()),
            sleep_worker: SleepWorker::new(
                memory_repo.clone(),
                llm_client.clone(),
                default_half_life_days,
                working_half_life_days,
            ),
            session_manager: SessionManager::new(
                memory_repo.clone(),
                llm_client.clone(),
                4000, // ~4000 token threshold (configurable)
                10,   // keep 10 most recent messages raw
            ),
        }
    }

    /// Run one STM decay sweep (age-prioritization). Scheduled periodically by
    /// the daemon (see `scheduler::run_periodic`); also callable on demand.
    pub async fn run_decay_sweep(&self) -> Result<()> {
        self.sleep_worker.run_decay_sweep().await
    }

    /// Drop `memory.command` idempotency rows older than `older_than_days`.
    /// Piggybacks on the decay sweep cadence; returns how many were removed.
    pub fn sweep_processed_commands(&self, older_than_days: i64) -> Result<usize> {
        self.memory_repo.sweep_processed_commands(older_than_days)
    }

    /// Soft reset — wipe the STM store only. The permanent LTM store lives in a
    /// separate connection/file and is untouched. Destructive.
    pub fn soft_reset(&self) -> Result<()> {
        self.memory_repo.reset_store()
    }

    /// Push dialogue to Short-Term Memory (Flow 1 from blueprint).
    /// Compresses old messages, returns optimized context window,
    /// and queues the new dialogue for background fact extraction.
    pub async fn push_dialogue(
        &self,
        tenant_id: &str,
        session_id: &str,
        new_message: &str,
        ccl: &str,
    ) -> Result<ContextWindow> {
        let (mut ctx, episode_id) = self
            .session_manager
            .push_dialogue(
                &TenantId(tenant_id.to_string()),
                &SessionId(session_id.to_string()),
                new_message,
                ccl,
            )
            .await?;

        // Learn from the new message, attributing facts to the episode it was
        // archived as (the old placeholder id 0 FK-failed every fact — ARC-1).
        // The message is already archived and buffered, so a learning failure
        // must not fail the call (a retry would store it twice — REV-1): it is
        // logged and reported in `learning_error` instead, never swallowed.
        let episode = Episode {
            id: Some(episode_id),
            tenant_id: TenantId(tenant_id.to_string()),
            session_id: SessionId(session_id.to_string()),
            raw_dialogue: new_message.to_string(),
            ccl: ccl.to_string(),
            created_at: None,
        };
        if let Err(e) = self.sleep_worker.process_episode(&episode).await {
            eprintln!(
                "[neurolithe] dialogue archived as episode {episode_id}, but fact extraction failed: {e:#}"
            );
            ctx.learning_error = Some(format!("fact extraction failed: {e:#}"));
        }

        Ok(ctx)
    }

    /// Stores raw memory dialogue (Episode) and extracts facts via the Sleep pipeline.
    /// Used by push_dialogue's background extraction pathway.
    pub async fn store_memory(
        &self,
        tenant_id: &str,
        session_id: &str,
        dialogue: &str,
        ccl: &str,
    ) -> Result<()> {
        let ep = Episode {
            id: None,
            tenant_id: TenantId(tenant_id.to_string()),
            session_id: SessionId(session_id.to_string()),
            raw_dialogue: dialogue.to_string(),
            ccl: ccl.to_string(),
            created_at: None,
        };

        let ep_id = self.memory_repo.store_episode(&ep)?;
        let mut ep_with_id = ep.clone();
        ep_with_id.id = Some(ep_id);

        self.sleep_worker.process_episode(&ep_with_id).await?;
        Ok(())
    }

    /// Store an explicit fact directly (bypasses LLM extraction) — Blueprint store_memory tool
    pub async fn store_explicit_fact(
        &self,
        tenant_id: &str,
        fact_text: &str,
        tags: &[String],
        ccl: &str,
    ) -> Result<()> {
        let embedding = self.llm_client.embed_text(fact_text).await?;

        let node = MemoryNode {
            id: None,
            tenant_id: TenantId(tenant_id.to_string()),
            source_episode_id: None,
            payload: serde_json::json!({
                "fact": fact_text,
                "tags": tags
            }),
            status: "active".into(),
            ccl: ccl.to_string(),
            is_explicit: true,
            support_count: 1,
            relevance_score: 1.0,
            context_key: None,
        };

        self.memory_repo.store_node(&node, &embedding)?;
        Ok(())
    }

    /// Query hybrid memory with temporal filtering and graph traversal
    pub async fn query_memory(
        &self,
        tenant_id: &str,
        query: &str,
        time_filter: &TimeFilter,
        ccl_filter: &[String],
    ) -> Result<Vec<MemoryResult>> {
        self.retrieval_service
            .query(
                &TenantId(tenant_id.to_string()),
                query,
                time_filter,
                ccl_filter,
            )
            .await
    }

    pub async fn register_ccl(&self, tenant_id: &str, name: &str, description: &str) -> Result<()> {
        let def = crate::domain::models::CclDefinition {
            id: None,
            tenant_id: TenantId(tenant_id.to_string()),
            name: name.to_string(),
            description: description.to_string(),
        };
        self.memory_repo.store_ccl_definition(&def)?;
        Ok(())
    }

    pub async fn get_ccl_layers(
        &self,
        tenant_id: &str,
    ) -> Result<Vec<crate::domain::models::CclDefinition>> {
        self.memory_repo
            .get_ccl_definitions(&TenantId(tenant_id.to_string()))
    }

    /// Delete all tenant information
    pub async fn delete_tenant(&self, tenant_id: &str) -> Result<()> {
        self.memory_repo
            .delete_tenant(&TenantId(tenant_id.to_string()))
    }

    /// Export tenant data to a JSON string
    pub async fn export_tenant(&self, tenant_id: &str) -> Result<String> {
        self.memory_repo
            .export_tenant(&TenantId(tenant_id.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::models::CclDefinition;
    use crate::domain::ports::ExtractedFact;
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::repository::SqliteMemoryRepository;
    use crate::infrastructure::schema::init_schema;
    use async_trait::async_trait;

    /// Extracts one fact per dialogue (or fails, when `fail` is set).
    struct StubLlm {
        fail: bool,
    }

    #[async_trait]
    impl LlmClient for StubLlm {
        async fn extract_facts(
            &self,
            dialogue: &str,
            _c: &[CclDefinition],
        ) -> Result<Vec<ExtractedFact>> {
            if self.fail {
                anyhow::bail!("extractor down");
            }
            Ok(vec![ExtractedFact {
                fact: format!("learned: {dialogue}"),
                ccl: "reality".into(),
                tags: vec!["t".into()],
                relationships: vec![],
            }])
        }
        async fn generate_ccl_description(&self, _n: &str, _c: &str) -> Result<String> {
            Ok("d".into())
        }
        async fn embed_text(&self, _t: &str) -> Result<Vec<f32>> {
            Ok(vec![0.5, 0.1, 0.0, 0.0])
        }
        async fn compress_context(&self, _m: &str) -> Result<String> {
            Ok("summary".into())
        }
    }

    fn app(fail: bool) -> NeurolitheApp {
        let conn = init_db(None as Option<&String>).unwrap();
        init_schema(&conn, 4).unwrap();
        NeurolitheApp::new(
            Arc::new(SqliteMemoryRepository::new(conn)),
            Arc::new(StubLlm { fail }),
            7.0,
            30.0 / 1440.0,
        )
    }

    /// ARC-1/DEV-3/QA-1: facts extracted from pushed dialogue are persisted
    /// (the placeholder episode id 0 used to FK-fail every insert, silently).
    #[tokio::test]
    async fn push_dialogue_persists_extracted_facts() {
        let app = app(false);
        app.push_dialogue("t1", "s1", "I moved to Lyon", "reality")
            .await
            .expect("push_dialogue succeeds");

        let export: serde_json::Value =
            serde_json::from_str(&app.export_tenant("t1").await.unwrap()).unwrap();
        let facts = export["extracted_facts"].as_array().unwrap();
        assert_eq!(
            facts.len(),
            1,
            "the extracted fact must be stored: {export}"
        );
        assert_eq!(facts[0]["fact"], "learned: I moved to Lyon");
    }

    /// REV-1: an extraction failure after the message is archived still returns
    /// the context window (success) and reports it in `learning_error` — so a
    /// client never retries and archives the message twice.
    #[tokio::test]
    async fn push_dialogue_reports_learning_error_without_failing() {
        let app = app(true);
        let ctx = app
            .push_dialogue("t1", "s1", "hello", "reality")
            .await
            .expect("archiving succeeded, so the call succeeds");
        assert_eq!(ctx.recent_messages, vec!["hello".to_string()]);
        let err = ctx
            .learning_error
            .clone()
            .expect("learning failure is reported");
        assert!(err.contains("extractor down"), "{err}");

        let json = serde_json::to_value(&ctx).unwrap();
        assert!(
            json["learning_error"]
                .as_str()
                .unwrap()
                .contains("extractor down")
        );
    }

    /// Success responses carry no `learning_error` key at all.
    #[tokio::test]
    async fn push_dialogue_success_has_no_learning_error() {
        let app = app(false);
        let ctx = app
            .push_dialogue("t1", "s1", "hi", "reality")
            .await
            .unwrap();
        assert!(ctx.learning_error.is_none());
        let json = serde_json::to_value(&ctx).unwrap();
        assert!(json.get("learning_error").is_none());
    }

    /// REV-1: with no LLM at all (chat and embeddings fail), the message is
    /// still archived and the context window is still returned.
    #[tokio::test]
    async fn push_dialogue_without_llm_still_returns_context() {
        let conn = init_db(None as Option<&String>).unwrap();
        init_schema(&conn, 4).unwrap();
        let app = NeurolitheApp::new(
            Arc::new(SqliteMemoryRepository::new(conn)),
            Arc::new(NoLlm),
            7.0,
            30.0 / 1440.0,
        );
        let ctx = app
            .push_dialogue("t1", "s1", "remember this", "reality")
            .await
            .expect("no LLM must not fail an archived push");
        assert_eq!(ctx.recent_messages, vec!["remember this".to_string()]);
        assert!(ctx.relevant_facts.is_empty());
        assert!(ctx.learning_error.is_some());
        assert!(!ctx.warnings.is_empty(), "the failed recall is reported");
    }

    /// Every call fails, like an unconfigured provider.
    struct NoLlm;

    #[async_trait]
    impl LlmClient for NoLlm {
        async fn extract_facts(
            &self,
            _d: &str,
            _c: &[CclDefinition],
        ) -> Result<Vec<ExtractedFact>> {
            anyhow::bail!("LLM not configured")
        }
        async fn generate_ccl_description(&self, _n: &str, _c: &str) -> Result<String> {
            anyhow::bail!("LLM not configured")
        }
        async fn embed_text(&self, _t: &str) -> Result<Vec<f32>> {
            anyhow::bail!("LLM not configured")
        }
        async fn compress_context(&self, _m: &str) -> Result<String> {
            anyhow::bail!("LLM not configured")
        }
    }
}
