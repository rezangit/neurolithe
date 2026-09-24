//! Reset — the disposable-brain operations (Kafka mode).
//!
//! Soft reset wipes STM only (junky working memory → clean slate). Hard reset
//! wipes BOTH stores and re-seeds the spine; the caller then rewinds the
//! feeder so the documents topic replays and LTM (+ STM) is re-derived from the
//! events. Both are destructive.
//!
//! Hard reset is **disabled** unless `NEUROLITHE_RESET_TOKEN` holds a token of
//! at least [`MIN_RESET_TOKEN_LEN`] characters; the command's `confirm` value
//! is compared against it in constant time (SEC-01).

use crate::domain::ltm::{LtmRepository, SpineSeed, default_spine};
use crate::domain::ports::MemoryRepository;
use anyhow::{Result, bail};
use serde::Deserialize;
use std::sync::Arc;

/// A command on the `memory.command` topic. Internally tagged on `command`:
/// `{"command":"reset_soft"}`, `{"command":"reset_hard","confirm":"<token>"}`,
/// `{"command":"remember","scope":"stm",…}`, `{"command":"forget","dataId":…}`.
///
/// The two reset variants are the original wire shape and MUST keep parsing
/// unchanged as `remember`/`forget` are added.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum MemoryCommand {
    ResetSoft,
    ResetHard { confirm: String },
    Remember(RememberCommand),
    Forget(ForgetCommand),
}

/// Which store a `remember` targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteScope {
    Stm,
    Ltm,
}

/// A `remember` write. `scope=stm` uses `fact` (+ optional `ccl`);
/// `scope=ltm` uses `text` (+ optional `tags`). A legacy `tenant` field is
/// accepted and ignored: one workspace per process. Fields are optional at the parse
/// layer and validated per-scope when applied, so one envelope covers both.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RememberCommand {
    pub command_id: String,
    pub scope: WriteScope,
    /// STM fact text (`scope=stm`).
    #[serde(default)]
    pub fact: Option<String>,
    /// LTM note text (`scope=ltm`).
    #[serde(default)]
    pub text: Option<String>,
    /// Cognitive-context layer for an STM fact (defaults to `reality`; agents
    /// pass `working` for situational notes).
    #[serde(default)]
    pub ccl: Option<String>,
    /// Working-memory thread key stamped on an STM note (STM-WORKING-MEMORY
    /// §5a). `None` for ordinary knowledge writes.
    #[serde(default)]
    pub context_key: Option<String>,
    /// Tags for an LTM note (advisory; not vocabulary-enforced in v1).
    #[serde(default)]
    pub tags: Vec<String>,
    /// Documents/entities this working turn is *about* (STM-GRAPH). Each becomes
    /// a reused subject node with an `about` edge from the turn — the relation
    /// that connects a session's turns and marks the focus. Empty for ordinary
    /// facts.
    #[serde(default)]
    pub subjects: Vec<SubjectRef>,
}

/// A subject a working turn refers to: a `dataId` plus a short human label.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubjectRef {
    pub data_id: String,
    #[serde(default)]
    pub label: String,
}

/// A `forget` write — tombstones a `dataId` across both stores.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ForgetCommand {
    pub command_id: String,
    pub data_id: String,
}

impl MemoryCommand {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

/// Environment variable holding the hard-reset confirmation token.
pub const RESET_TOKEN_ENV: &str = "NEUROLITHE_RESET_TOKEN";

/// Shortest accepted hard-reset token. Anything shorter (or unset/blank)
/// leaves hard reset disabled — there is no built-in default token.
pub const MIN_RESET_TOKEN_LEN: usize = 16;

/// Compare two byte strings in time that depends only on their lengths, never
/// on where they first differ (no early exit), so a remote caller can't recover
/// the token byte by byte from response timing.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = u8::from(a.len() != b.len());
    for i in 0..a.len().max(b.len()) {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= x ^ y;
    }
    std::hint::black_box(diff) == 0
}

/// The usable hard-reset token from a raw value: trimmed, and only if it is at
/// least [`MIN_RESET_TOKEN_LEN`] characters. `None` = hard reset disabled.
pub fn usable_reset_token(raw: Option<&str>) -> Option<String> {
    raw.map(str::trim)
        .filter(|t| t.chars().count() >= MIN_RESET_TOKEN_LEN)
        .map(str::to_string)
}

/// Performs soft/hard resets across the two stores. Store wipes only — the
/// feeder offset rewind for hard reset is the consumer's job.
pub struct ResetService {
    stm: Arc<dyn MemoryRepository>,
    ltm: Arc<dyn LtmRepository>,
    /// `None` = hard reset disabled (no/short token configured).
    confirm_token: Option<String>,
    /// The spine re-seeded after a hard reset (`[ltm.spine]` or the default).
    spine: Vec<SpineSeed>,
}

impl ResetService {
    /// `confirm_token` is validated with [`usable_reset_token`]: unset, blank or
    /// shorter than [`MIN_RESET_TOKEN_LEN`] disables hard reset.
    pub fn new(
        stm: Arc<dyn MemoryRepository>,
        ltm: Arc<dyn LtmRepository>,
        confirm_token: Option<&str>,
    ) -> Self {
        Self {
            stm,
            ltm,
            confirm_token: usable_reset_token(confirm_token),
            spine: default_spine(),
        }
    }

    /// Re-seed this spine after a hard reset instead of the default
    /// (`config.ltm.spine_or_default()`).
    pub fn with_spine(mut self, spine: Vec<SpineSeed>) -> Self {
        self.spine = spine;
        self
    }

    /// Build from the process environment ([`RESET_TOKEN_ENV`]).
    pub fn from_env(stm: Arc<dyn MemoryRepository>, ltm: Arc<dyn LtmRepository>) -> Self {
        let raw = std::env::var(RESET_TOKEN_ENV).ok();
        Self::new(stm, ltm, raw.as_deref())
    }

    /// Whether a hard reset can ever succeed (a usable token is configured).
    pub fn hard_reset_enabled(&self) -> bool {
        self.confirm_token.is_some()
    }

    /// Wipe STM only; LTM untouched.
    pub fn soft_reset(&self) -> Result<()> {
        self.stm.reset_store()
    }

    /// Wipe BOTH stores and re-seed the spine. Requires hard reset to be enabled
    /// and `confirm` to equal the configured token (constant-time compare). The
    /// caller rewinds the feeder afterward to relearn from the bus.
    pub fn hard_reset(&self, confirm: &str) -> Result<()> {
        let Some(token) = &self.confirm_token else {
            bail!(
                "hard reset refused: disabled (set {RESET_TOKEN_ENV} to a token of at least \
                 {MIN_RESET_TOKEN_LEN} characters to enable it)"
            );
        };
        if !constant_time_eq(confirm.as_bytes(), token.as_bytes()) {
            bail!("hard reset refused: confirmation token mismatch");
        }
        self.stm.reset_store()?;
        self.ltm.reset_store()?;
        // Restore the backbone; the feeder regrows the leaves.
        self.ltm.seed_spine_from(&self.spine)?;
        Ok(())
    }

    /// Dispatch a parsed command. Returns whether a hard reset occurred (so the
    /// caller knows to rewind the feeder).
    pub fn apply(&self, command: &MemoryCommand) -> Result<ResetKind> {
        match command {
            MemoryCommand::ResetSoft => {
                self.soft_reset()?;
                Ok(ResetKind::Soft)
            }
            MemoryCommand::ResetHard { confirm } => {
                self.hard_reset(confirm)?;
                Ok(ResetKind::Hard)
            }
            // Writes are handled by WriteService, not here; the consumer routes
            // by variant, so this arm is a defensive guard only.
            MemoryCommand::Remember(_) | MemoryCommand::Forget(_) => {
                bail!("reset service received a non-reset command")
            }
        }
    }
}

/// Which reset ran — Hard signals the caller to rewind the feeder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetKind {
    Soft,
    Hard,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::ingestion::tests::text_event;
    use crate::application::ingestion::{DocumentEvent, IngestionService};
    use crate::domain::models::{CclDefinition, TenantId};
    use crate::domain::ports::{ExtractedFact, LlmClient};
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::ltm_repository::SqliteLtmRepository;
    use crate::infrastructure::repository::SqliteMemoryRepository;
    use crate::infrastructure::schema::{init_ltm_schema, init_schema};
    use async_trait::async_trait;

    const DIM: usize = 8;
    const TOKEN: &str = "CONFIRM-WIPE-0123456789";
    const TENANT: &str = "default";

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
            Ok(vec![0.25; DIM])
        }
        async fn compress_context(&self, m: &str) -> Result<String> {
            Ok(format!("summary[{}]", m.len()))
        }
    }

    fn event(id: &str) -> DocumentEvent {
        text_event(id)
    }

    struct Harness {
        reset: ResetService,
        ingest: IngestionService,
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

        let ingest = IngestionService::new(
            stm.clone() as Arc<dyn MemoryRepository>,
            ltm.clone() as Arc<dyn LtmRepository>,
            Arc::new(StubLlm),
            DIM,
            TENANT,
        );
        let reset = ResetService::new(
            stm.clone() as Arc<dyn MemoryRepository>,
            ltm.clone() as Arc<dyn LtmRepository>,
            Some(TOKEN),
        );
        Harness {
            reset,
            ingest,
            stm,
            ltm,
            tenant: TenantId(TENANT.into()),
        }
    }

    fn stm_has(h: &Harness, id: &str) -> bool {
        h.stm.export_tenant(&h.tenant).unwrap().contains(id)
    }

    #[test]
    fn test_command_parsing() {
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
        assert!(MemoryCommand::parse(b"not json").is_err());

        // A legacy `tenant` field still parses (and is ignored).
        let legacy = MemoryCommand::parse(
            br#"{"command":"remember","commandId":"c1","scope":"stm","fact":"f","tenant":"other"}"#,
        )
        .unwrap();
        assert!(matches!(legacy, MemoryCommand::Remember(r) if r.fact.as_deref() == Some("f")));
    }

    /// Soft reset empties STM only; the LTM leaf survives.
    #[tokio::test]
    async fn test_soft_reset_empties_stm_only() {
        let h = harness();
        h.ingest.ingest(&event("grp_1")).await.unwrap();
        assert!(stm_has(&h, "grp_1"));
        assert!(h.ltm.get_node_by_data_id("grp_1").unwrap().is_some());

        h.reset.apply(&MemoryCommand::ResetSoft).unwrap();

        assert!(!stm_has(&h, "grp_1"), "STM wiped");
        assert!(
            h.ltm.get_node_by_data_id("grp_1").unwrap().is_some(),
            "LTM survives soft reset"
        );
    }

    /// Hard reset empties both stores (spine restored), then a replayed
    /// document re-populates LTM + STM.
    #[tokio::test]
    async fn test_hard_reset_empties_both_then_replay_repopulates() {
        let h = harness();
        h.ingest.ingest(&event("grp_1")).await.unwrap();

        let kind = h
            .reset
            .apply(&MemoryCommand::ResetHard {
                confirm: TOKEN.into(),
            })
            .unwrap();
        assert_eq!(kind, ResetKind::Hard);

        // Both emptied of the document; the curated spine is back.
        assert!(!stm_has(&h, "grp_1"));
        assert!(h.ltm.get_node_by_data_id("grp_1").unwrap().is_none());
        assert!(!h.ltm.get_roots().unwrap().is_empty(), "spine re-seeded");

        // Replay (feeder backfill) re-derives both.
        h.ingest.ingest(&event("grp_1")).await.unwrap();
        assert!(stm_has(&h, "grp_1"));
        assert!(h.ltm.get_node_by_data_id("grp_1").unwrap().is_some());
    }

    /// After a hard reset the re-seeded spine is un-embedded (so a document would
    /// fall to the inbox); once the daemon's post-reset step embeds it, a
    /// replayed document files under a concept. Regression guard: a reset must
    /// not silently re-break placement (field-report §3).
    #[tokio::test]
    async fn test_hard_reset_then_embed_spine_files_under_concept() {
        use crate::application::ingestion::IngestOutcome;
        use crate::application::ltm_placement::embed_spine_concepts;

        let h = harness();
        h.reset.hard_reset(TOKEN).unwrap();

        // Un-embedded spine → the first replayed doc has no concept to match.
        let before = h.ingest.ingest(&event("grp_before")).await.unwrap();
        assert!(
            matches!(before, IngestOutcome::Ingested { matched: false, .. }),
            "without spine vectors a document falls to the inbox"
        );

        // The daemon's post-hard-reset step: embed the curated spine.
        let ltm_dyn: Arc<dyn LtmRepository> = h.ltm.clone();
        let llm_dyn: Arc<dyn LlmClient> = Arc::new(StubLlm);
        let n = embed_spine_concepts(&ltm_dyn, &llm_dyn, DIM).await.unwrap();
        assert!(n > 0, "spine concepts were embedded");

        // Now a document matches a concept instead of the inbox.
        let after = h.ingest.ingest(&event("grp_after")).await.unwrap();
        assert!(
            matches!(after, IngestOutcome::Ingested { matched: true, .. }),
            "after embedding, a document files under a concept"
        );
    }

    /// Hard reset with the wrong token is refused and changes nothing.
    #[tokio::test]
    async fn test_hard_reset_wrong_token_refused() {
        let h = harness();
        h.ingest.ingest(&event("grp_1")).await.unwrap();

        let result = h.reset.hard_reset("WRONG");
        assert!(result.is_err(), "wrong token refused");

        // Nothing was wiped.
        assert!(stm_has(&h, "grp_1"));
        assert!(h.ltm.get_node_by_data_id("grp_1").unwrap().is_some());
    }

    /// SEC-01: with no token, a blank one, or one shorter than the minimum,
    /// hard reset is disabled — even a `confirm` equal to that value (or to
    /// the old built-in `RESET-DISABLED` default) is refused and nothing is
    /// wiped.
    #[tokio::test]
    async fn test_hard_reset_disabled_without_a_strong_token() {
        let h = harness();
        h.ingest.ingest(&event("grp_1")).await.unwrap();

        for configured in [None, Some(""), Some("   "), Some("fifteen-chars!!")] {
            let reset = ResetService::new(
                h.stm.clone() as Arc<dyn MemoryRepository>,
                h.ltm.clone() as Arc<dyn LtmRepository>,
                configured,
            );
            assert!(!reset.hard_reset_enabled(), "{configured:?} must disable");
            for confirm in ["", "RESET-DISABLED", configured.unwrap_or("")] {
                let err = reset
                    .hard_reset(confirm)
                    .expect_err("hard reset must be refused while disabled");
                assert!(err.to_string().contains("disabled"), "{err}");
            }
        }
        assert!(stm_has(&h, "grp_1"), "nothing was wiped");
        assert!(h.ltm.get_node_by_data_id("grp_1").unwrap().is_some());
    }

    /// A token of exactly the minimum length enables hard reset; surrounding
    /// whitespace in the configured value is ignored.
    #[test]
    fn test_usable_reset_token_rules() {
        assert_eq!(usable_reset_token(None), None);
        assert_eq!(usable_reset_token(Some("x".repeat(15).as_str())), None);
        let sixteen = "y".repeat(MIN_RESET_TOKEN_LEN);
        assert_eq!(usable_reset_token(Some(&sixteen)), Some(sixteen.clone()));
        assert_eq!(
            usable_reset_token(Some(&format!("  {sixteen}\n"))),
            Some(sixteen)
        );
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"same-token-value", b"same-token-value"));
        assert!(!constant_time_eq(b"same-token-value", b"same-token-valuf"));
        assert!(!constant_time_eq(b"prefix", b"prefix-longer"));
        assert!(!constant_time_eq(b"prefix-longer", b"prefix"));
        assert!(!constant_time_eq(b"a", b""));
    }

    /// A hard reset re-seeds the configured spine (`[ltm.spine]`), not the
    /// built-in default.
    #[test]
    fn test_hard_reset_reseeds_configured_spine() {
        let h = harness();
        let reset = ResetService::new(
            h.stm.clone() as Arc<dyn MemoryRepository>,
            h.ltm.clone() as Arc<dyn LtmRepository>,
            Some(TOKEN),
        )
        .with_spine(vec![SpineSeed::new("research", "Papers and notes")]);
        reset.hard_reset(TOKEN).unwrap();

        let root = h.ltm.get_roots().unwrap()[0].id.unwrap();
        let names: Vec<String> = h
            .ltm
            .get_children(root)
            .unwrap()
            .into_iter()
            .map(|n| n.name)
            .collect();
        assert!(names.contains(&"research".to_string()), "{names:?}");
        assert!(names.contains(&"inbox".to_string()), "{names:?}");
        assert!(
            !names.contains(&"notes".to_string()),
            "default not used: {names:?}"
        );
    }
}
