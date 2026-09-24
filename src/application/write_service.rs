//! Write service — applies `remember` / `forget` commands from `memory.command`.
//!
//! The bus door for an agent to *write* memory. `remember stm`
//! stores a distilled fact into working memory; `remember ltm` mints a synthetic
//! `note_<uuidv7>` `dataId` and places a durable note as a leaf on the knowledge
//! tree (provenance `agent`); `forget` tombstones a `dataId` across both stores.
//!
//! Every write is idempotent on its `commandId` (Kafka is at-least-once): a
//! duplicate delivery is recorded once and skipped thereafter. The id is marked
//! only AFTER the write succeeds, so a mid-write crash re-applies rather than
//! silently drops.

use crate::application::ingestion::IngestionService;
use crate::application::ltm_placement::{DocumentToPlace, LtmPlacement};
use crate::application::reset_service::{
    ForgetCommand, MemoryCommand, RememberCommand, WriteScope,
};
use crate::domain::ltm::{LtmRepository, Provenance};
use crate::domain::models::{Edge, MemoryNode, TenantId};
use crate::domain::ports::{LlmClient, MemoryRepository};
use anyhow::{Result, bail};
use std::sync::Arc;
use uuid::Uuid;

/// Tenant for agent writes: the process's single workspace (a legacy `tenant`
/// field on the command is ignored).
const AGENT_TENANT: &str = crate::domain::models::WORKSPACE_TENANT;
/// Default cognitive-context layer for agent-remembered facts (design §7).
const DEFAULT_CCL: &str = "reality";
/// Provenance source stamped on agent-authored LTM notes.
const NOTE_SOURCE: &str = "agent";
/// Max characters of note text used as the LTM leaf's display name.
const NAME_MAX: usize = 60;

/// What a write did (for logging / traces).
#[derive(Debug, PartialEq)]
pub enum WriteOutcome {
    RememberedStm,
    RememberedLtm {
        data_id: String,
        matched: bool,
    },
    Forgotten {
        data_id: String,
    },
    /// The `commandId` was already applied — nothing done.
    Skipped,
}

/// Applies agent writes to STM + LTM, idempotently.
pub struct WriteService {
    stm: Arc<dyn MemoryRepository>,
    ltm: Arc<dyn LtmRepository>,
    ingestion: Arc<IngestionService>,
    llm: Arc<dyn LlmClient>,
    /// STM embedding dimension, for zero-vector graph-anchor (subject) nodes.
    stm_vector_dim: usize,
    /// Resolved placement threshold for LTM notes (see `domain::thresholds`).
    placement_max_distance: f64,
}

impl WriteService {
    pub fn new(
        stm: Arc<dyn MemoryRepository>,
        ltm: Arc<dyn LtmRepository>,
        ingestion: Arc<IngestionService>,
        llm: Arc<dyn LlmClient>,
        stm_vector_dim: usize,
        thresholds: &crate::domain::thresholds::Thresholds,
    ) -> Self {
        Self {
            stm,
            ltm,
            ingestion,
            llm,
            stm_vector_dim,
            placement_max_distance: thresholds.placement_max_distance.value,
        }
    }

    /// Apply a write command, deduplicated on its `commandId`. Errors bubble so
    /// the consumer can dead-letter without marking the id processed.
    pub async fn handle(&self, cmd: &MemoryCommand) -> Result<WriteOutcome> {
        let command_id = match cmd {
            MemoryCommand::Remember(r) => r.command_id.as_str(),
            MemoryCommand::Forget(f) => f.command_id.as_str(),
            _ => bail!("write service received a non-write command"),
        };

        if self.stm.is_command_processed(command_id)? {
            return Ok(WriteOutcome::Skipped);
        }

        let outcome = match cmd {
            MemoryCommand::Remember(r) => self.remember(r).await?,
            MemoryCommand::Forget(f) => self.forget(f).await?,
            _ => unreachable!("guarded above"),
        };

        // Mark only after success — a failure above retries on redelivery.
        self.stm.mark_command_processed(command_id)?;
        Ok(outcome)
    }

    async fn remember(&self, cmd: &RememberCommand) -> Result<WriteOutcome> {
        match cmd.scope {
            WriteScope::Stm => self.remember_stm(cmd).await,
            WriteScope::Ltm => self.remember_ltm(cmd).await,
        }
    }

    async fn remember_stm(&self, cmd: &RememberCommand) -> Result<WriteOutcome> {
        let Some(fact) = cmd.fact.as_deref() else {
            bail!("remember stm requires a 'fact'");
        };
        let tenant = AGENT_TENANT;
        let ccl = cmd.ccl.as_deref().unwrap_or(DEFAULT_CCL);
        let tenant_id = TenantId(tenant.to_string());

        // The turn node — the action/task. `kind:"turn"` marks it in the graph.
        let embedding = self.llm.embed_text(fact).await?;
        let turn = MemoryNode {
            id: None,
            tenant_id: tenant_id.clone(),
            source_episode_id: None,
            payload: serde_json::json!({ "fact": fact, "kind": "turn", "tags": cmd.tags }),
            status: "active".into(),
            ccl: ccl.to_string(),
            is_explicit: true,
            support_count: 1,
            relevance_score: 1.0,
            context_key: cmd.context_key.clone(),
        };
        let turn_id = self.stm.store_node(&turn, &embedding)?;

        // STM-GRAPH: link this turn to the documents/entities it is about. Each
        // subject is a reused node per (thread, dataId) so turns about the same
        // thing connect through it; an `about` edge is the relation. Only
        // meaningful when the turn is thread-scoped (a working note).
        if let Some(context_key) = cmd.context_key.as_deref() {
            for subject in &cmd.subjects {
                if subject.data_id.is_empty() {
                    continue;
                }
                let subject_id = self
                    .find_or_create_subject(&tenant_id, context_key, ccl, subject)
                    .await?;
                self.stm.store_edge(&Edge {
                    source_id: turn_id,
                    target_id: subject_id,
                    relation: "about".into(),
                    ccl: ccl.to_string(),
                    valid_from: None,
                    valid_until: None,
                    weight: 1.0,
                })?;
            }
        }

        Ok(WriteOutcome::RememberedStm)
    }

    /// Reuse the thread's subject node for a `dataId` (refreshing it so it stays
    /// alive while referenced), or create it. Subject nodes are graph anchors, so
    /// they carry a zero embedding — they're reached via `about` edges, never by
    /// vector search.
    async fn find_or_create_subject(
        &self,
        tenant_id: &TenantId,
        context_key: &str,
        ccl: &str,
        subject: &crate::application::reset_service::SubjectRef,
    ) -> Result<i64> {
        if let Some(existing) =
            self.stm
                .find_working_subject(tenant_id, context_key, &subject.data_id)?
        {
            // A read/reference reinforces — keep the anchor hot for the session.
            self.stm.boost_relevance(&[existing])?;
            return Ok(existing);
        }
        let node = MemoryNode {
            id: None,
            tenant_id: tenant_id.clone(),
            source_episode_id: None,
            payload: serde_json::json!({
                "fact": subject.label,
                "kind": "subject",
                "dataId": subject.data_id,
            }),
            status: "active".into(),
            ccl: ccl.to_string(),
            is_explicit: true,
            support_count: 1,
            relevance_score: 1.0,
            context_key: Some(context_key.to_string()),
        };
        self.stm.store_node(&node, &self.zero_embedding())
    }

    /// A zero vector at the STM embedding dimension, for graph-anchor nodes that
    /// don't need semantic search.
    fn zero_embedding(&self) -> Vec<f32> {
        vec![0.0; self.stm_vector_dim]
    }

    async fn remember_ltm(&self, cmd: &RememberCommand) -> Result<WriteOutcome> {
        let Some(text) = cmd.text.as_deref() else {
            bail!("remember ltm requires a 'text'");
        };
        let data_id = format!("note_{}", Uuid::now_v7());
        let embedding = self.llm.embed_text(text).await?;

        let placement = LtmPlacement::new(self.ltm.clone(), self.placement_max_distance);
        let doc = DocumentToPlace {
            name: leaf_name(text),
            summary: text.to_string(),
            embedding,
            data_id: data_id.clone(),
            provenance: Provenance {
                source: NOTE_SOURCE.to_string(),
                ingested_at: None,
                confidence: 1.0,
            },
        };
        let placed = placement.place(&doc)?;
        Ok(WriteOutcome::RememberedLtm {
            data_id,
            matched: placed.matched,
        })
    }

    async fn forget(&self, cmd: &ForgetCommand) -> Result<WriteOutcome> {
        self.ingestion.forget(&cmd.data_id).await?;
        Ok(WriteOutcome::Forgotten {
            data_id: cmd.data_id.clone(),
        })
    }
}

/// A short human-facing label for an LTM note leaf: the first line, clipped to
/// [`NAME_MAX`] characters (on a char boundary).
fn leaf_name(text: &str) -> String {
    let first_line = text.lines().next().unwrap_or(text).trim();
    match first_line.char_indices().nth(NAME_MAX) {
        Some((byte_idx, _)) => format!("{}…", &first_line[..byte_idx]),
        None => first_line.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::ingestion::tests::text_event;
    use crate::domain::models::CclDefinition;
    use crate::domain::ports::ExtractedFact;
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::ltm_repository::SqliteLtmRepository;
    use crate::infrastructure::repository::SqliteMemoryRepository;
    use crate::infrastructure::schema::{init_ltm_schema, init_schema};
    use async_trait::async_trait;

    const DIM: usize = 8;

    struct StubLlm;
    #[async_trait]
    impl LlmClient for StubLlm {
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
            Ok(vec![0.2; DIM])
        }
        async fn compress_context(&self, _m: &str) -> Result<String> {
            Ok(String::new())
        }
    }

    struct Harness {
        write: WriteService,
        ingest: Arc<IngestionService>,
        stm: Arc<SqliteMemoryRepository>,
        ltm: Arc<SqliteLtmRepository>,
        tenant: TenantId,
    }

    fn harness() -> Harness {
        let stm_conn = init_db(None as Option<&String>).unwrap();
        init_schema(&stm_conn, DIM).unwrap();
        let stm = Arc::new(SqliteMemoryRepository::new(stm_conn));

        let ltm_conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&ltm_conn, DIM).unwrap();
        let ltm = Arc::new(SqliteLtmRepository::new(ltm_conn));
        ltm.seed_spine().unwrap();

        let ingest = Arc::new(IngestionService::new(
            stm.clone() as Arc<dyn MemoryRepository>,
            ltm.clone() as Arc<dyn LtmRepository>,
            Arc::new(StubLlm),
            DIM,
            AGENT_TENANT,
            &crate::domain::thresholds::Thresholds::text_embedding_004(),
        ));
        let write = WriteService::new(
            stm.clone() as Arc<dyn MemoryRepository>,
            ltm.clone() as Arc<dyn LtmRepository>,
            ingest.clone(),
            Arc::new(StubLlm),
            DIM,
            &crate::domain::thresholds::Thresholds::text_embedding_004(),
        );
        Harness {
            write,
            ingest,
            stm,
            ltm,
            tenant: TenantId(AGENT_TENANT.into()),
        }
    }

    fn remember_stm(command_id: &str, fact: &str) -> MemoryCommand {
        MemoryCommand::Remember(RememberCommand {
            command_id: command_id.into(),
            scope: WriteScope::Stm,
            fact: Some(fact.into()),
            text: None,
            ccl: None,
            context_key: None,
            tags: vec![],
            subjects: vec![],
        })
    }

    fn remember_with_subjects(
        command_id: &str,
        fact: &str,
        context_key: &str,
        subjects: &[(&str, &str)],
    ) -> MemoryCommand {
        MemoryCommand::Remember(RememberCommand {
            command_id: command_id.into(),
            scope: WriteScope::Stm,
            fact: Some(fact.into()),
            text: None,
            ccl: Some("working".into()),
            context_key: Some(context_key.into()),
            tags: vec![],
            subjects: subjects
                .iter()
                .map(
                    |(id, label)| crate::application::reset_service::SubjectRef {
                        data_id: (*id).into(),
                        label: (*label).into(),
                    },
                )
                .collect(),
        })
    }

    /// A working turn about a document builds a reused subject node (the graph
    /// anchor). A later turn about the same document reuses it — not a duplicate.
    #[tokio::test]
    async fn remember_stm_subjects_build_and_reuse_the_graph_anchor() {
        let h = harness();
        let ctx = "chat.1";

        h.write
            .handle(&remember_with_subjects(
                "cmd_g1",
                "find my last lululemon purchase",
                ctx,
                &[("doc_LL", "Lululemon order")],
            ))
            .await
            .unwrap();
        let subj1 = h
            .stm
            .find_working_subject(&h.tenant, ctx, "doc_LL")
            .unwrap();
        assert!(subj1.is_some(), "subject anchor created for doc_LL");
        assert!(stm_has(&h, "Lululemon order"), "subject label stored");

        // A second turn about the SAME document reuses the same anchor node
        // (find returns newest-by-id; an equal id proves no duplicate was made).
        h.write
            .handle(&remember_with_subjects(
                "cmd_g2",
                "what is its document id",
                ctx,
                &[("doc_LL", "Lululemon order")],
            ))
            .await
            .unwrap();
        let subj2 = h
            .stm
            .find_working_subject(&h.tenant, ctx, "doc_LL")
            .unwrap();
        assert_eq!(subj1, subj2, "subject reused across turns, not duplicated");

        // A different thread gets its OWN anchor (context isolation).
        h.write
            .handle(&remember_with_subjects(
                "cmd_g3",
                "find lululemon",
                "chat.2",
                &[("doc_LL", "Lululemon order")],
            ))
            .await
            .unwrap();
        let other = h
            .stm
            .find_working_subject(&h.tenant, "chat.2", "doc_LL")
            .unwrap();
        assert!(other.is_some());
        assert_ne!(other, subj1, "each thread anchors its own subject");
    }

    fn stm_has(h: &Harness, needle: &str) -> bool {
        h.stm.export_tenant(&h.tenant).unwrap().contains(needle)
    }

    fn stm_count(h: &Harness, needle: &str) -> usize {
        h.stm
            .export_tenant(&h.tenant)
            .unwrap()
            .matches(needle)
            .count()
    }

    /// Back-compat: today's reset envelopes still parse after adding the write
    /// variants, and the new envelopes parse to their variants.
    #[test]
    fn parses_reset_and_write_envelopes() {
        assert_eq!(
            MemoryCommand::parse(br#"{"command":"reset_soft"}"#).unwrap(),
            MemoryCommand::ResetSoft
        );
        assert_eq!(
            MemoryCommand::parse(br#"{"command":"reset_hard","confirm":"x"}"#).unwrap(),
            MemoryCommand::ResetHard {
                confirm: "x".into()
            }
        );
        let remember = MemoryCommand::parse(
            br#"{"command":"remember","scope":"ltm","commandId":"cmd_1","text":"hi","tags":["a"]}"#,
        )
        .unwrap();
        match remember {
            MemoryCommand::Remember(r) => {
                assert_eq!(r.scope, WriteScope::Ltm);
                assert_eq!(r.command_id, "cmd_1");
                assert_eq!(r.text.as_deref(), Some("hi"));
            }
            other => panic!("expected Remember, got {other:?}"),
        }
        let forget =
            MemoryCommand::parse(br#"{"command":"forget","commandId":"cmd_2","dataId":"doc_9"}"#)
                .unwrap();
        assert_eq!(
            forget,
            MemoryCommand::Forget(ForgetCommand {
                command_id: "cmd_2".into(),
                data_id: "doc_9".into(),
            })
        );
    }

    #[tokio::test]
    async fn remember_stm_stores_a_fact() {
        let h = harness();
        let outcome = h
            .write
            .handle(&remember_stm("cmd_1", "User prefers morning meetings"))
            .await
            .unwrap();
        assert_eq!(outcome, WriteOutcome::RememberedStm);
        assert!(stm_has(&h, "morning meetings"));
    }

    #[tokio::test]
    async fn remember_ltm_mints_a_note_and_places_a_leaf() {
        let h = harness();
        let cmd = MemoryCommand::Remember(RememberCommand {
            command_id: "cmd_1".into(),
            scope: WriteScope::Ltm,
            fact: None,
            text: Some("Project Atlas uses claim-check on Kafka".into()),
            ccl: None,
            context_key: None,
            tags: vec!["atlas".into()],
            subjects: vec![],
        });
        let outcome = h.write.handle(&cmd).await.unwrap();
        let WriteOutcome::RememberedLtm { data_id, .. } = outcome else {
            panic!("expected RememberedLtm, got {outcome:?}");
        };
        assert!(data_id.starts_with("note_"), "synthetic dataId: {data_id}");
        // The leaf is discoverable by its dataId with agent provenance.
        let leaf = h.ltm.get_node_by_data_id(&data_id).unwrap();
        assert!(leaf.is_some(), "note placed as an LTM leaf");
    }

    #[tokio::test]
    async fn remember_stm_with_context_key_lands_a_working_note() {
        let h = harness();
        let cmd = MemoryCommand::Remember(RememberCommand {
            command_id: "cmd_ctx".into(),
            scope: WriteScope::Stm,
            fact: Some("found home-inspection report = doc_42".into()),
            text: None,
            ccl: Some("working".into()),
            context_key: Some("chat.jid:1".into()),
            tags: vec![],
            subjects: vec![],
        });
        let outcome = h.write.handle(&cmd).await.unwrap();
        assert_eq!(outcome, WriteOutcome::RememberedStm);

        // The note is retrievable via the recency backbone under its context.
        let recent = h
            .stm
            .recent_in_context(&h.tenant, &["working".to_string()], "chat.jid:1", 10)
            .unwrap();
        assert_eq!(recent.len(), 1);
        assert!(recent[0].fact.contains("doc_42"));
        assert_eq!(recent[0].ccl, "working");
        assert_eq!(recent[0].context_key.as_deref(), Some("chat.jid:1"));
    }

    #[tokio::test]
    async fn remember_stm_without_fact_errors() {
        let h = harness();
        let bad = MemoryCommand::Remember(RememberCommand {
            command_id: "cmd_1".into(),
            scope: WriteScope::Stm,
            fact: None,
            text: None,
            ccl: None,
            context_key: None,
            tags: vec![],
            subjects: vec![],
        });
        assert!(h.write.handle(&bad).await.is_err());
    }

    #[tokio::test]
    async fn forget_tombstones_across_both_stores() {
        let h = harness();
        // Ingest a document, then forget it by dataId.
        h.ingest.ingest(&text_event("grp_1")).await.unwrap();
        assert!(h.ltm.get_node_by_data_id("grp_1").unwrap().is_some());

        let outcome = h
            .write
            .handle(&MemoryCommand::Forget(ForgetCommand {
                command_id: "cmd_1".into(),
                data_id: "grp_1".into(),
            }))
            .await
            .unwrap();
        assert_eq!(
            outcome,
            WriteOutcome::Forgotten {
                data_id: "grp_1".into()
            }
        );
        assert!(h.ltm.get_node_by_data_id("grp_1").unwrap().is_none());
    }

    #[tokio::test]
    async fn duplicate_command_id_is_skipped() {
        let h = harness();
        let cmd = remember_stm("cmd_dup", "User likes cycling");

        let first = h.write.handle(&cmd).await.unwrap();
        assert_eq!(first, WriteOutcome::RememberedStm);
        assert_eq!(stm_count(&h, "cycling"), 1);

        // Same commandId again → skipped, no second write.
        let second = h.write.handle(&cmd).await.unwrap();
        assert_eq!(second, WriteOutcome::Skipped);
        assert_eq!(stm_count(&h, "cycling"), 1, "no duplicate STM node");
    }

    #[tokio::test]
    async fn sweep_removes_only_old_idempotency_rows() {
        let h = harness();
        h.write
            .handle(&remember_stm("cmd_fresh", "fresh fact"))
            .await
            .unwrap();
        // A 14-day sweep leaves a just-written id intact.
        let removed = h.stm.sweep_processed_commands(14).unwrap();
        assert_eq!(removed, 0);
        assert!(h.stm.is_command_processed("cmd_fresh").unwrap());
    }
}
