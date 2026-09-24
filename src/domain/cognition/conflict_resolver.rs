use crate::domain::models::TenantId;
use crate::domain::ports::MemoryRepository;
use anyhow::{Result, anyhow};
use std::sync::Arc;

/// The three adaptation modes from cognitive psychology:
/// - Assimilate: Fact already exists, boost its support
/// - AccommodateModify: Similar fact exists but with new info, update it
/// - AccommodateCreate: No match found, create new node
pub enum AdaptationResult {
    /// Existing node was reinforced (its support_count was incremented)
    Assimilated(i64),
    /// Existing node was modified with updated payload
    AccommodatedModify(i64),
    /// A brand new node should be created
    AccommodateCreate,
}

pub struct ConflictResolver {
    /// Vector distance (sqlite-vec default metric, L2) at or below which a new
    /// fact is the *same* fact as its neighbour: the neighbour is reinforced and
    /// its text is kept.
    pub assimilation_threshold: f64,
    /// Vector distance (L2) at or below which a new fact *refines* its
    /// neighbour: the neighbour takes the new text (and the new embedding).
    /// Beyond this, a new node is created.
    pub accommodation_threshold: f64,
}

/// How many neighbours are inspected when looking for a merge partner. More
/// than one, so an ineligible nearest neighbour (a different document or node
/// kind) does not hide an eligible one just behind it.
const MERGE_CANDIDATES: usize = 5;

impl Default for ConflictResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// The payload fields that identify *what* a node is. Two nodes may only be
/// merged when these agree: a document (`dataId`) is never folded into another
/// document or into a dialogue fact, and a graph `kind` (turn/subject) is never
/// folded into a plain fact (ARC-2).
fn identity(payload: &serde_json::Value) -> (Option<&str>, Option<&str>) {
    (
        payload.get("dataId").and_then(|v| v.as_str()),
        payload.get("kind").and_then(|v| v.as_str()),
    )
}

fn fact_text(payload: &serde_json::Value) -> &str {
    payload.get("fact").and_then(|f| f.as_str()).unwrap_or("")
}

/// Union of the `tags` arrays of `a` and `b` (order-preserving, `a` first).
fn union_tags(a: &serde_json::Value, b: &serde_json::Value) -> Vec<serde_json::Value> {
    let mut all: Vec<serde_json::Value> = a
        .get("tags")
        .and_then(|t| t.as_array())
        .cloned()
        .unwrap_or_default();
    for t in b
        .get("tags")
        .and_then(|t| t.as_array())
        .into_iter()
        .flatten()
    {
        if !all.contains(t) {
            all.push(t.clone());
        }
    }
    all
}

/// `existing` with every key of `new` overlaid (new values win) and `tags`
/// unioned — so keys only the existing node carries (e.g. `dataId`,
/// `kind`) survive a merge instead of being dropped.
fn merge_payload(existing: &serde_json::Value, new: &serde_json::Value) -> serde_json::Value {
    let mut merged = match existing {
        serde_json::Value::Object(map) => map.clone(),
        _ => serde_json::Map::new(),
    };
    if let serde_json::Value::Object(new_map) = new {
        for (k, v) in new_map {
            merged.insert(k.clone(), v.clone());
        }
    }
    let tags = union_tags(existing, new);
    if !tags.is_empty() {
        merged.insert("tags".into(), serde_json::Value::Array(tags));
    }
    serde_json::Value::Object(merged)
}

impl ConflictResolver {
    /// The legacy `text-embedding-004` thresholds. Production code resolves
    /// thresholds per embedding model and uses [`Self::from_thresholds`].
    pub fn new() -> Self {
        Self::from_thresholds(&crate::domain::thresholds::Thresholds::text_embedding_004())
    }

    /// A resolver using the resolved (config / per-model) thresholds.
    pub fn from_thresholds(thresholds: &crate::domain::thresholds::Thresholds) -> Self {
        Self {
            assimilation_threshold: thresholds.assimilation.value,
            accommodation_threshold: thresholds.accommodation.value,
        }
    }

    /// Determine how to adapt a new fact given existing knowledge, applying the
    /// change to the repository for the assimilate/modify cases.
    ///
    /// - Only neighbours with the same identity (`dataId` + `kind`) are merge
    ///   candidates; anything else leads to a new node.
    /// - **Assimilate** (identical text, or distance ≤ `assimilation_threshold`):
    ///   reinforce the neighbour, keep its text and embedding, union the tags.
    /// - **Accommodate-modify** (distance ≤ `accommodation_threshold`): the
    ///   neighbour takes the new text, the other payload keys are merged, and
    ///   its embedding is replaced by `embedding` in the same transaction.
    pub fn resolve(
        &self,
        memory_repo: &Arc<dyn MemoryRepository>,
        embedding: &[f32],
        tenant_id: &TenantId,
        new_payload: &serde_json::Value,
    ) -> Result<AdaptationResult> {
        let candidates = memory_repo.find_similar_nodes(
            embedding,
            tenant_id,
            self.accommodation_threshold,
            MERGE_CANDIDATES,
        )?;

        let new_identity = identity(new_payload);
        let Some((closest, distance)) = candidates
            .into_iter()
            .find(|(node, _)| identity(&node.payload) == new_identity)
        else {
            return Ok(AdaptationResult::AccommodateCreate);
        };
        let closest_id = closest
            .id
            .ok_or_else(|| anyhow!("similar node returned without an id"))?;

        if fact_text(&closest.payload) == fact_text(new_payload)
            || distance <= self.assimilation_threshold
        {
            // Same fact: reinforce, keep the existing text (so the stored
            // embedding stays valid), and pick up any new tags.
            let tags = union_tags(&closest.payload, new_payload);
            let existing_tags = closest
                .payload
                .get("tags")
                .and_then(|t| t.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            if tags.len() > existing_tags
                && let serde_json::Value::Object(mut map) = closest.payload.clone()
            {
                map.insert("tags".into(), serde_json::Value::Array(tags));
                memory_repo
                    .update_node_support(closest_id, Some(&serde_json::Value::Object(map)))?;
            } else {
                memory_repo.update_node_support(closest_id, None)?;
            }
            return Ok(AdaptationResult::Assimilated(closest_id));
        }

        // Refinement: new text wins, other keys are preserved, and the vector is
        // replaced so it matches the new text.
        let merged = merge_payload(&closest.payload, new_payload);
        memory_repo.update_node_content(closest_id, &merged, embedding)?;
        Ok(AdaptationResult::AccommodatedModify(closest_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::domain::models::MemoryNode;
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::repository::SqliteMemoryRepository;
    use crate::infrastructure::schema::init_schema;
    use serde_json::json;

    #[test]
    fn test_default_thresholds() {
        let resolver = ConflictResolver::new();
        assert!(resolver.assimilation_threshold < resolver.accommodation_threshold);
    }

    const E: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    /// L2 distance 0.25 from `E`: inside accommodation (0.35), outside assimilation (0.15).
    const E_NEAR: [f32; 4] = [1.0, 0.25, 0.0, 0.0];
    /// L2 distance 0.10 from `E`: inside assimilation.
    const E_SAME: [f32; 4] = [1.0, 0.10, 0.0, 0.0];

    fn tenant() -> TenantId {
        TenantId("t".into())
    }

    fn setup(payload: serde_json::Value) -> (Arc<dyn MemoryRepository>, i64) {
        let conn = init_db(None as Option<&String>).unwrap();
        init_schema(&conn, 4).unwrap();
        let repo: Arc<dyn MemoryRepository> = Arc::new(SqliteMemoryRepository::new(conn));
        let node = MemoryNode {
            id: None,
            tenant_id: tenant(),
            source_episode_id: None,
            payload,
            status: "active".into(),
            ccl: "reality".into(),
            is_explicit: false,
            support_count: 1,
            relevance_score: 1.0,
            context_key: None,
        };
        let id = repo.store_node(&node, &E).unwrap();
        (repo, id)
    }

    fn nearest(repo: &Arc<dyn MemoryRepository>, e: &[f32]) -> Option<(MemoryNode, f64)> {
        repo.find_similar_nodes(e, &tenant(), 0.01, 1)
            .unwrap()
            .into_iter()
            .next()
    }

    /// ARC-2/DEV-10: an accommodate-modify replaces the text AND the vector, so
    /// the node is found by what it now says — not by its old meaning.
    #[test]
    fn modify_reembeds_the_node() {
        let (repo, id) = setup(json!({ "fact": "Alice lives in Paris" }));
        let result = ConflictResolver::new()
            .resolve(
                &repo,
                &E_NEAR,
                &tenant(),
                &json!({ "fact": "Alice lives in Lyon" }),
            )
            .unwrap();
        assert!(matches!(result, AdaptationResult::AccommodatedModify(m) if m == id));

        let (node, _) = nearest(&repo, &E_NEAR).expect("found by its new embedding");
        assert_eq!(node.payload["fact"], "Alice lives in Lyon");
        assert!(
            nearest(&repo, &E).is_none(),
            "old embedding must be replaced"
        );
    }

    /// ARC-2: a merge keeps keys only the existing node carries and unions tags.
    #[test]
    fn modify_preserves_existing_payload_keys() {
        let (repo, id) = setup(json!({
            "fact": "Invoice from ACME", "dataId": "doc_1", "source": "scan", "tags": ["invoice"]
        }));
        ConflictResolver::new()
            .resolve(
                &repo,
                &E_NEAR,
                &tenant(),
                &json!({ "fact": "Invoice from ACME, paid", "dataId": "doc_1", "tags": ["paid"] }),
            )
            .unwrap();

        let (node, _) = nearest(&repo, &E_NEAR).unwrap();
        assert_eq!(node.id, Some(id));
        assert_eq!(node.payload["fact"], "Invoice from ACME, paid");
        assert_eq!(node.payload["dataId"], "doc_1");
        assert_eq!(node.payload["source"], "scan");
        assert_eq!(node.payload["tags"], json!(["invoice", "paid"]));
    }

    /// ARC-2: two different documents are never merged, however close.
    #[test]
    fn different_data_ids_never_merge() {
        let (repo, id) = setup(json!({ "fact": "Receipt", "dataId": "doc_1" }));
        let result = ConflictResolver::new()
            .resolve(
                &repo,
                &E,
                &tenant(),
                &json!({ "fact": "Receipt", "dataId": "doc_2" }),
            )
            .unwrap();
        assert!(matches!(result, AdaptationResult::AccommodateCreate));

        let (node, _) = nearest(&repo, &E).unwrap();
        assert_eq!(node.id, Some(id));
        assert_eq!(
            node.payload["dataId"], "doc_1",
            "existing document untouched"
        );
    }

    /// ARC-2: a dialogue fact (no dataId) never overwrites a document node.
    #[test]
    fn plain_fact_never_merges_into_a_document() {
        let (repo, _) = setup(json!({ "fact": "Receipt", "dataId": "doc_1" }));
        let result = ConflictResolver::new()
            .resolve(
                &repo,
                &E_NEAR,
                &tenant(),
                &json!({ "fact": "Receipt total" }),
            )
            .unwrap();
        assert!(matches!(result, AdaptationResult::AccommodateCreate));
    }

    /// `assimilation_threshold` is live: within it, a differently-worded fact is
    /// the same fact — reinforced, text and vector kept.
    #[test]
    fn within_assimilation_threshold_keeps_existing_text() {
        let (repo, id) = setup(json!({ "fact": "Bob likes tea", "tags": ["pref"] }));
        let result = ConflictResolver::new()
            .resolve(
                &repo,
                &E_SAME,
                &tenant(),
                &json!({ "fact": "Bob enjoys tea", "tags": ["drink"] }),
            )
            .unwrap();
        assert!(matches!(result, AdaptationResult::Assimilated(a) if a == id));

        let (node, _) = nearest(&repo, &E).expect("vector unchanged");
        assert_eq!(node.payload["fact"], "Bob likes tea");
        assert_eq!(node.payload["tags"], json!(["pref", "drink"]));
        assert_eq!(node.support_count, 2);
    }

    fn thresholds(assimilation: f64, accommodation: f64) -> crate::domain::thresholds::Thresholds {
        crate::domain::thresholds::Thresholds::resolve(
            "unknown:model",
            crate::domain::thresholds::ThresholdOverrides {
                placement_max_distance: None,
                assimilation: Some(assimilation),
                accommodation: Some(accommodation),
            },
        )
        .unwrap()
        .0
    }

    /// The resolver uses the resolved thresholds, not built-in constants: a
    /// neighbour at 0.25 is modified under 0.15/0.35 but is a separate fact
    /// under 0.10/0.20, and a neighbour at 0.10 is only assimilated when the
    /// assimilation threshold covers it.
    #[test]
    fn resolved_thresholds_drive_the_decision() {
        let (repo, _) = setup(json!({ "fact": "Alice lives in Paris" }));
        let strict = ConflictResolver::from_thresholds(&thresholds(0.10, 0.20));
        let result = strict
            .resolve(
                &repo,
                &E_NEAR,
                &tenant(),
                &json!({ "fact": "Alice lives in Lyon" }),
            )
            .unwrap();
        assert!(matches!(result, AdaptationResult::AccommodateCreate));

        let (repo, id) = setup(json!({ "fact": "Bob likes tea" }));
        let tight = ConflictResolver::from_thresholds(&thresholds(0.05, 0.20));
        let result = tight
            .resolve(
                &repo,
                &E_SAME,
                &tenant(),
                &json!({ "fact": "Bob enjoys tea" }),
            )
            .unwrap();
        assert!(
            matches!(result, AdaptationResult::AccommodatedModify(m) if m == id),
            "0.10 is beyond assimilation 0.05 but within accommodation 0.20"
        );
    }
}
