use crate::domain::cognition::conflict_resolver::{AdaptationResult, ConflictResolver};
use crate::domain::decay::DecayEngine;
use crate::domain::ports::{ExtractedFact, LlmClient, MemoryRepository};
use anyhow::Result;
use std::sync::Arc;

/// At most this many facts are accepted from one extraction.
pub const MAX_FACTS_PER_EXTRACTION: usize = 32;
/// At most this many relationships are accepted from one extraction, in total
/// across its facts.
pub const MAX_RELATIONSHIPS_PER_EXTRACTION: usize = 32;

/// Truncate an extraction to [`MAX_FACTS_PER_EXTRACTION`] facts and
/// [`MAX_RELATIONSHIPS_PER_EXTRACTION`] relationships (kept in order), logging
/// when anything is dropped.
fn cap_extraction(facts: &mut Vec<ExtractedFact>, episode_id: Option<i64>) {
    let total_facts = facts.len();
    facts.truncate(MAX_FACTS_PER_EXTRACTION);

    let mut budget = MAX_RELATIONSHIPS_PER_EXTRACTION;
    let mut dropped_rels = 0;
    for fact in facts.iter_mut() {
        let keep = fact.relationships.len().min(budget);
        dropped_rels += fact.relationships.len() - keep;
        fact.relationships.truncate(keep);
        budget -= keep;
    }

    let dropped_facts = total_facts - facts.len();
    if dropped_facts > 0 || dropped_rels > 0 {
        tracing::warn!(
            "extraction capped for episode {episode_id:?}: dropped {dropped_facts} fact(s), {dropped_rels} relationship(s)"
        );
    }
}

pub struct SleepWorker {
    memory_repo: Arc<dyn MemoryRepository>,
    llm_client: Arc<dyn LlmClient>,
    decay_engine: DecayEngine,
    conflict_resolver: ConflictResolver,
}

impl SleepWorker {
    pub fn new(
        memory_repo: Arc<dyn MemoryRepository>,
        llm_client: Arc<dyn LlmClient>,
        default_half_life_days: f64,
        working_half_life_days: f64,
    ) -> Self {
        Self {
            memory_repo,
            llm_client,
            decay_engine: DecayEngine::with_half_lives(
                default_half_life_days,
                working_half_life_days,
            ),
            conflict_resolver: ConflictResolver::new(),
        }
    }

    /// Triggers the background decay process across the database
    pub async fn run_decay_sweep(&self) -> Result<()> {
        self.memory_repo.sweep_decay(&self.decay_engine)?;
        Ok(())
    }

    /// Processes un-extracted episodes using the full Sleep pipeline:
    /// 1. Extract facts (with relationships + temporal bounds)
    /// 2. For each fact, run Tri-Modal Conflict Resolution
    /// 3. Create edges for any extracted relationships
    pub async fn process_episode(&self, episode: &crate::domain::models::Episode) -> Result<()> {
        let valid_ccls = self.memory_repo.get_ccl_definitions(&episode.tenant_id)?;
        let mut extracted_facts = self
            .llm_client
            .extract_facts(&episode.raw_dialogue, &valid_ccls)
            .await?;

        // Bound the work one message can trigger: every fact and relationship
        // costs an embedding call plus writes (REV-2 / SEC-12).
        cap_extraction(&mut extracted_facts, episode.id);

        let mut known_ccl_names: std::collections::HashSet<String> =
            valid_ccls.into_iter().map(|c| c.name).collect();

        for fact in extracted_facts {
            if !known_ccl_names.contains(&fact.ccl) {
                let context = format!("Fact: {}", fact.fact);
                let desc = self
                    .llm_client
                    .generate_ccl_description(&fact.ccl, &context)
                    .await
                    .unwrap_or_else(|_| "Auto-generated cognitive layer".to_string());
                let new_def = crate::domain::models::CclDefinition {
                    id: None,
                    tenant_id: episode.tenant_id.clone(),
                    name: fact.ccl.clone(),
                    description: desc,
                };
                self.memory_repo.store_ccl_definition(&new_def)?;
                known_ccl_names.insert(fact.ccl.clone());
            }
            let embedding = self.llm_client.embed_text(&fact.fact).await?;
            let payload = serde_json::json!({
                "fact": fact.fact,
                "tags": fact.tags
            });

            // Tri-Modal Conflict Resolution
            let source_node_id = match self.conflict_resolver.resolve(
                &self.memory_repo,
                &embedding,
                &episode.tenant_id,
                &payload,
            )? {
                AdaptationResult::Assimilated(existing_id) => {
                    // Fact already exists, support was boosted
                    existing_id
                }
                AdaptationResult::AccommodatedModify(existing_id) => {
                    // Similar fact was updated with merged payload
                    existing_id
                }
                AdaptationResult::AccommodateCreate => {
                    // No match — create a new node
                    let node = crate::domain::models::MemoryNode {
                        id: None,
                        tenant_id: episode.tenant_id.clone(),
                        source_episode_id: episode.id,
                        payload: payload.clone(),
                        status: "active".into(),
                        ccl: fact.ccl.clone(),
                        is_explicit: false,
                        support_count: 1,
                        relevance_score: 1.0,
                        context_key: None,
                    };
                    self.memory_repo.store_node(&node, &embedding)?
                }
            };

            // Create edges for any extracted relationships
            for rel in &fact.relationships {
                let target_embedding = self.llm_client.embed_text(&rel.target_entity).await?;
                let target_payload = serde_json::json!({
                    "fact": rel.target_entity,
                    "tags": ["entity"]
                });

                // Also resolve target entities through conflict resolution
                let target_node_id = match self.conflict_resolver.resolve(
                    &self.memory_repo,
                    &target_embedding,
                    &episode.tenant_id,
                    &target_payload,
                )? {
                    AdaptationResult::Assimilated(id)
                    | AdaptationResult::AccommodatedModify(id) => id,
                    AdaptationResult::AccommodateCreate => {
                        let target_node = crate::domain::models::MemoryNode {
                            id: None,
                            tenant_id: episode.tenant_id.clone(),
                            source_episode_id: episode.id,
                            payload: target_payload,
                            status: "active".into(),
                            ccl: fact.ccl.clone(),
                            is_explicit: false,
                            support_count: 1,
                            relevance_score: 1.0,
                            context_key: None,
                        };
                        self.memory_repo
                            .store_node(&target_node, &target_embedding)?
                    }
                };

                let edge = crate::domain::models::Edge {
                    source_id: source_node_id,
                    target_id: target_node_id,
                    relation: rel.relation.clone(),
                    ccl: fact.ccl.clone(),
                    valid_from: rel.valid_from.clone(),
                    valid_until: rel.valid_until.clone(),
                    weight: 1.0,
                };
                self.memory_repo.store_edge(&edge)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::models::{CclDefinition, Episode, SessionId, TenantId};
    use crate::domain::ports::ExtractedRelationship;
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::repository::SqliteMemoryRepository;
    use crate::infrastructure::schema::init_schema;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Extracts 100 facts with 5 relationships each; counts embedding calls.
    struct FloodLlm {
        embeds: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl LlmClient for FloodLlm {
        async fn extract_facts(
            &self,
            _d: &str,
            _c: &[CclDefinition],
        ) -> Result<Vec<ExtractedFact>> {
            Ok((0..100)
                .map(|i| ExtractedFact {
                    fact: format!("fact {i}"),
                    ccl: "reality".into(),
                    tags: vec![],
                    relationships: (0..5)
                        .map(|j| ExtractedRelationship {
                            target_entity: format!("entity {i}-{j}"),
                            relation: "rel".into(),
                            ccl: "reality".into(),
                            valid_from: None,
                            valid_until: None,
                        })
                        .collect(),
                })
                .collect())
        }
        async fn generate_ccl_description(&self, _n: &str, _c: &str) -> Result<String> {
            Ok("d".into())
        }
        async fn embed_text(&self, _t: &str) -> Result<Vec<f32>> {
            let n = self.embeds.fetch_add(1, Ordering::SeqCst);
            // Distinct, well-separated vectors so nothing merges.
            Ok(vec![n as f32, 1.0, 0.0, 0.0])
        }
        async fn compress_context(&self, _m: &str) -> Result<String> {
            Ok(String::new())
        }
    }

    /// REV-2: one message can trigger at most MAX_FACTS + MAX_RELATIONSHIPS
    /// embedding calls, however much the extractor returns.
    #[tokio::test]
    async fn extraction_is_capped() {
        let conn = init_db(None as Option<&String>).unwrap();
        init_schema(&conn, 4).unwrap();
        let repo: Arc<dyn MemoryRepository> = Arc::new(SqliteMemoryRepository::new(conn));
        let llm = Arc::new(FloodLlm {
            embeds: AtomicUsize::new(0),
        });
        let worker = SleepWorker::new(repo.clone(), llm.clone(), 7.0, 1.0);
        let tenant = TenantId("t".into());
        let ep = repo
            .store_episode(&Episode {
                id: None,
                tenant_id: tenant.clone(),
                session_id: SessionId("s".into()),
                raw_dialogue: "a flood".into(),
                ccl: "reality".into(),
                created_at: None,
            })
            .unwrap();
        let episode = Episode {
            id: Some(ep),
            tenant_id: tenant,
            session_id: SessionId("s".into()),
            raw_dialogue: "a flood".into(),
            ccl: "reality".into(),
            created_at: None,
        };

        worker.process_episode(&episode).await.unwrap();

        assert_eq!(
            llm.embeds.load(Ordering::SeqCst),
            MAX_FACTS_PER_EXTRACTION + MAX_RELATIONSHIPS_PER_EXTRACTION
        );
    }
}
