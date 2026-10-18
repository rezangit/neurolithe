//! LTM placement — files a new document into the knowledge tree and keeps
//! rolling summaries in sync.
//!
//! V2 is deliberately **simple**: best-match-or-inbox placement and a
//! concatenate-and-truncate roll-up. A document is filed under its nearest
//! concept when that concept is within `placement_max_distance` and, if the
//! margin rule is on (`placement_min_margin` > 0), at least that much closer
//! than the runner-up concept; otherwise it goes to the inbox. A root (`root`,
//! a node without a parent) is never a placement target. The hard "grow the right way" logic
//! (split fat nodes, merge similar branches, spawn new branches) is deferred —
//! see `V2-DESIGN.md` §3.2.

use crate::domain::ltm::{
    DocumentBody, LtmRepository, Provenance, TreeEdge, TreeNode, TreeNodeKind,
};
use crate::domain::ports::Embedder;
use crate::infrastructure::embedding_codec::check_dim;
use anyhow::{Result, anyhow};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Embed every concept node that lacks a placement vector, deriving the vector
/// from the concept's curated identity (`name: summary`) — **not** its rolling
/// summary, so filing targets stay stable as documents accumulate. Idempotent:
/// embeds only the missing ones, so steady-state startups make zero LLM calls.
/// Returns how many concepts were embedded. The daemon runs this after
/// `seed_spine` so placement has live match targets.
pub async fn embed_spine_concepts(
    ltm: &Arc<dyn LtmRepository>,
    embedder: &Arc<dyn Embedder>,
    expected_dim: usize,
) -> Result<usize> {
    let mut embedded = 0;
    for node in ltm.concepts_missing_embedding()? {
        let id = node.id.expect("stored node has id");
        let text = if node.summary.trim().is_empty() {
            node.name.clone()
        } else {
            format!("{}: {}", node.name, node.summary)
        };
        let embedding = embedder.embed_text(&text).await?;
        check_dim(
            &embedding,
            expected_dim,
            &format!("embedder (concept '{}')", node.name),
        )?;
        ltm.set_concept_embedding(id, &embedding)?;
        embedded += 1;
    }
    Ok(embedded)
}

/// Cap on a rolled summary's length (characters) before truncation.
const DEFAULT_MAX_SUMMARY_LEN: usize = 600;

/// A document ready to be filed: its distilled meaning + its archive pointer.
pub struct DocumentToPlace {
    /// Short human-facing label for the leaf node.
    pub name: String,
    /// Distilled summary (becomes the leaf's summary + roll-up input).
    pub summary: String,
    /// Embedding of the summary (LTM dimension). Used both to locate the best
    /// concept (placement) and — stored on the leaf — to make the document
    /// recallable by meaning.
    pub embedding: Vec<f32>,
    /// Source-document reference (`dataId`).
    pub data_id: String,
    pub provenance: Provenance,
}

/// Where a document landed.
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    pub leaf_node_id: i64,
    pub parent_id: i64,
    /// True if matched a concept; false if it fell to the inbox.
    pub matched: bool,
}

/// A concept a document could be filed under, with its vector distance.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConceptMatch {
    pub id: i64,
    pub name: String,
    pub distance: f64,
}

/// The two nearest placement targets for an embedding. Roots (nodes without a
/// parent, e.g. `root`) are never targets.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Candidates {
    pub best: Option<ConceptMatch>,
    pub runner_up: Option<ConceptMatch>,
}

impl Candidates {
    /// How much closer the best concept is than the runner-up, when both exist.
    pub fn margin(&self) -> Option<f64> {
        Some(self.runner_up.as_ref()?.distance - self.best.as_ref()?.distance)
    }
}

/// What placement decides for a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementDecision {
    /// File under the best concept.
    File,
    /// No concept within `placement_max_distance`: the inbox.
    InboxTooFar,
    /// The best concept is within the cutoff, but doesn't beat the runner-up
    /// by `placement_min_margin`: the inbox.
    InboxAmbiguous,
}

/// Files documents into the LTM tree and rolls summaries up the affected
/// ancestors.
pub struct LtmPlacement {
    repo: Arc<dyn LtmRepository>,
    max_distance: f64,
    /// The margin rule (`0` = off).
    min_margin: f64,
    max_summary_len: usize,
}

impl LtmPlacement {
    /// `max_distance`: the placement threshold — a document attaches to its
    /// nearest concept only within this vector distance (resolved per embedding
    /// model or from `[ltm] placement_max_distance`; see `domain::thresholds`).
    pub fn new(repo: Arc<dyn LtmRepository>, max_distance: f64) -> Self {
        Self {
            repo,
            max_distance,
            min_margin: 0.0,
            max_summary_len: DEFAULT_MAX_SUMMARY_LEN,
        }
    }

    /// Placement with the resolved cutoff and margin rule (see
    /// `domain::thresholds`).
    pub fn from_thresholds(
        repo: Arc<dyn LtmRepository>,
        thresholds: &crate::domain::thresholds::Thresholds,
    ) -> Self {
        Self::new(repo, thresholds.placement_max_distance.value)
            .with_min_margin(thresholds.placement_min_margin.value)
    }

    /// Set the margin rule: file only if the best concept is at least
    /// `min_margin` closer than the runner-up. `0` turns it off.
    pub fn with_min_margin(mut self, min_margin: f64) -> Self {
        self.min_margin = min_margin;
        self
    }

    /// The two nearest concepts to `embedding`, roots excluded, at any
    /// distance (the cutoff is applied by [`Self::decide`]).
    pub fn candidates(&self, embedding: &[f32]) -> Result<Candidates> {
        let roots: std::collections::HashSet<i64> = self
            .repo
            .get_roots()?
            .into_iter()
            .filter_map(|n| n.id)
            .collect();
        // Over-fetch by the number of roots: the KNN cut runs before we drop
        // them, and an embedded root must not push the runner-up out.
        let mut matches = self
            .repo
            .find_similar_concepts(embedding, f64::MAX, 2 + roots.len())?
            .into_iter()
            .filter_map(|(node, distance)| {
                let id = node.id?;
                (!roots.contains(&id)).then_some(ConceptMatch {
                    id,
                    name: node.name,
                    distance,
                })
            });
        Ok(Candidates {
            best: matches.next(),
            runner_up: matches.next(),
        })
    }

    /// Apply the cutoff and the margin rule to `candidates`.
    pub fn decide(&self, candidates: &Candidates) -> PlacementDecision {
        let Some(best) = &candidates.best else {
            return PlacementDecision::InboxTooFar;
        };
        if best.distance > self.max_distance {
            return PlacementDecision::InboxTooFar;
        }
        match candidates.margin() {
            Some(margin) if self.min_margin > 0.0 && margin < self.min_margin => {
                PlacementDecision::InboxAmbiguous
            }
            _ => PlacementDecision::File,
        }
    }

    /// The concept to file under, or `None` for the inbox.
    fn target(&self, embedding: &[f32]) -> Result<Option<i64>> {
        let candidates = self.candidates(embedding)?;
        Ok(match (self.decide(&candidates), candidates.best) {
            (PlacementDecision::File, Some(best)) => Some(best.id),
            _ => None,
        })
    }

    /// File a document: attach it under the best-matching concept (vector within
    /// threshold) or, failing that, under the inbox; then roll its summary up
    /// into the ancestors. Returns where it landed.
    pub fn place(&self, doc: &DocumentToPlace) -> Result<Placement> {
        self.place_with_body(doc, None)
    }

    /// [`place`](Self::place), storing the document's full `body` (for
    /// keyword recall) in the same transaction as the leaf.
    pub fn place_with_body(
        &self,
        doc: &DocumentToPlace,
        body: Option<&DocumentBody>,
    ) -> Result<Placement> {
        // 1. Best concept match (cutoff + margin rule, never a root), or fall
        //    back to the inbox.
        let (parent_id, matched) = match self.target(&doc.embedding)? {
            Some(concept_id) => (concept_id, true),
            None => {
                let inbox = self
                    .repo
                    .get_inbox()?
                    .ok_or_else(|| anyhow!("no inbox node — spine not seeded"))?;
                (inbox.id.expect("stored node has id"), false)
            }
        };

        // 2. File the leaf: node, dataId link, provenance and parent edge in one
        //    transaction, so a crash can't leave an unlinked leaf that no later
        //    replace/forget of this dataId would find. The leaf IS
        //    vector-indexed (its summary embedding), so `recall` can land on the
        //    document directly by meaning — essential while docs pile in the
        //    inbox before the tree grows real concepts around them. This does
        //    NOT affect placement: step 1 uses the concept-only
        //    `find_similar_concepts`, so an embedded leaf can never crowd out a
        //    concept match and misroute a new document to the inbox. An unset
        //    `provenance.ingested_at` is stamped from the node's DB-assigned
        //    `created_at` (no wall-clock dependency).
        let leaf_node_id = self.repo.create_leaf_node(
            &TreeNode::new(doc.name.clone(), doc.summary.clone(), TreeNodeKind::Leaf),
            &doc.embedding,
            &doc.data_id,
            &doc.provenance,
            parent_id,
            body,
        )?;

        // 3. Roll the new summary up into the parent and its ancestors.
        self.roll_up_from(parent_id)?;

        Ok(Placement {
            leaf_node_id,
            parent_id,
            matched,
        })
    }

    /// Inbox gardener: re-file inbox documents that now match a concept, using
    /// their **stored** leaf embeddings (no LLM, no re-distill). Lets a threshold
    /// change or a newly-added spine branch re-home previously-inboxed docs
    /// without a full replay. Idempotent — only leaves whose nearest concept is
    /// now within threshold move; the ambiguous tail stays in the inbox. Returns
    /// how many documents were re-homed.
    pub fn garden_inbox(&self) -> Result<usize> {
        let Some(inbox) = self.repo.get_inbox()? else {
            return Ok(0);
        };
        let inbox_id = inbox.id.expect("stored node has id");

        let mut touched: std::collections::HashSet<i64> = std::collections::HashSet::new();
        let mut moved = 0;
        for (leaf_id, embedding) in self.repo.inbox_leaf_embeddings()? {
            let Some(concept_id) = self.target(&embedding)? else {
                continue; // still no clear concept within threshold — leave in inbox
            };
            self.repo.remove_edge(inbox_id, leaf_id)?;
            self.repo.add_edge(&TreeEdge::new(concept_id, leaf_id))?;
            touched.insert(concept_id);
            moved += 1;
        }

        // Re-roll the inbox (it shrank) and every concept that gained a leaf.
        if moved > 0 {
            self.roll_up_from(inbox_id)?;
            for concept_id in touched {
                self.roll_up_from(concept_id)?;
            }
        }
        Ok(moved)
    }

    /// Tombstone: forget a `dataId` — remove **every** leaf carrying it and
    /// re-roll the affected ancestor summaries. Returns false if the document
    /// was not in the tree.
    pub fn forget(&self, data_id: &str) -> Result<bool> {
        let removed = self.remove_leaves(data_id, None)?;
        // The last leaf's delete drops the body (a trigger), but a body no
        // leaf carried (an ingest that failed before filing) must go too: a
        // forgotten document's text may not stay on disk (W3R-1).
        self.repo.delete_document_body(data_id)?;
        Ok(removed > 0)
    }

    /// Upsert: file `doc`, then remove every *other* leaf with the same
    /// `data_id` and re-roll the summaries of the parents they left (the new
    /// parent included, so its title list no longer shows the old version).
    ///
    /// Ordering keeps it safe without a cross-call transaction: the new version
    /// is committed before any old one is deleted, so a crash in between leaves
    /// a duplicate (never a loss), and the next replace/forget of that `data_id`
    /// removes all stale copies. Returns the placement and how many old leaves
    /// were replaced.
    pub fn replace(&self, doc: &DocumentToPlace) -> Result<(Placement, usize)> {
        self.replace_with_body(doc, None)
    }

    /// [`replace`](Self::replace), storing the new version's full `body` in
    /// the same transaction as its leaf, so a failed filing never leaves the
    /// new text beside the old leaf (W3R-2).
    pub fn replace_with_body(
        &self,
        doc: &DocumentToPlace,
        body: Option<&DocumentBody>,
    ) -> Result<(Placement, usize)> {
        let placed = self.place_with_body(doc, body)?;
        let replaced = self.remove_leaves(&doc.data_id, Some(placed.leaf_node_id))?;
        Ok((placed, replaced))
    }

    /// Delete every leaf node carrying `data_id` except `keep`, then re-roll
    /// each affected parent's ancestors (once). Returns how many leaves were
    /// deleted.
    ///
    /// Each leaf is deleted in the same transaction that rewrites its parents'
    /// summaries without it (W2R-2). Once a leaf is gone, a redelivery can no
    /// longer find the parents it had, so a separate roll-up that a crash cut
    /// off would leave them listing the removed document for good. Ancestors
    /// above the parents list their children's names, not summaries, so the
    /// roll-up afterwards only refreshes them for completeness.
    fn remove_leaves(&self, data_id: &str, keep: Option<i64>) -> Result<usize> {
        let mut parents: Vec<i64> = Vec::new();
        let mut removed = 0;
        for leaf in self.repo.get_nodes_by_data_id(data_id)? {
            let leaf_id = leaf.id.expect("stored node has id");
            if Some(leaf_id) == keep {
                continue;
            }
            let mut summaries = Vec::new();
            for parent in self.repo.get_parents(leaf_id)? {
                let pid = parent.id.expect("stored node has id");
                let siblings: Vec<TreeNode> = self
                    .repo
                    .get_children(pid)?
                    .into_iter()
                    .filter(|c| c.id != Some(leaf_id))
                    .collect();
                summaries.push((pid, collection_summary(&siblings, self.max_summary_len)));
                if !parents.contains(&pid) {
                    parents.push(pid);
                }
            }
            self.repo.delete_node_with_summaries(leaf_id, &summaries)?;
            removed += 1;
        }
        for pid in parents {
            self.roll_up_from(pid)?;
        }
        Ok(removed)
    }

    /// Recompute `start`'s summary and every ancestor's, children-before-parents
    /// (so each parent sees its children's fresh summaries). Multi-parent safe.
    fn roll_up_from(&self, start: i64) -> Result<()> {
        // BFS upward, tracking each node's min distance from `start`.
        let mut dist: HashMap<i64, usize> = HashMap::new();
        let mut queue: VecDeque<i64> = VecDeque::new();
        dist.insert(start, 0);
        queue.push_back(start);
        while let Some(node_id) = queue.pop_front() {
            let d = dist[&node_id];
            for parent in self.repo.get_parents(node_id)? {
                let pid = parent.id.expect("stored node has id");
                let nd = d + 1;
                if dist.get(&pid).is_none_or(|&old| nd < old) {
                    dist.insert(pid, nd);
                    queue.push_back(pid);
                }
            }
        }

        // Recompute in increasing distance order: children before parents.
        let mut ordered: Vec<(i64, usize)> = dist.into_iter().collect();
        ordered.sort_by_key(|&(_, d)| d);
        for (node_id, _) in ordered {
            self.recompute_summary(node_id)?;
        }
        Ok(())
    }

    /// A container node's rolling summary **describes its collection** — a count
    /// plus the titles of its children — rather than quoting one child's text.
    ///
    /// The old behaviour concatenated children's full summaries and truncated to
    /// 600 chars, so a folder of 79 documents "became" doc #1's summary, and that
    /// text propagated up to the root (field-report §4). Listing titles keeps the
    /// summary faithful to the whole set and stops one document from poisoning
    /// ancestor text. (Placement is unaffected either way: concept match vectors
    /// come from the curated identity, not this rolling summary.) No children →
    /// empty.
    fn recompute_summary(&self, node_id: i64) -> Result<()> {
        let children = self.repo.get_children(node_id)?;
        let new_summary = collection_summary(&children, self.max_summary_len);

        if let Some(node) = self.repo.get_node(node_id)?
            && node.summary != new_summary
        {
            self.repo.update_summary(node_id, &new_summary)?;
        }
        Ok(())
    }
}

/// Truncate `text` to at most `max_len` characters on a char boundary, marking
/// the cut with an ellipsis.
fn condense(text: &str, max_len: usize) -> String {
    if text.chars().count() <= max_len {
        return text.to_string();
    }
    let head: String = text.chars().take(max_len.saturating_sub(1)).collect();
    format!("{head}…")
}

/// A short label for a child in a collection summary: its name (leaf title or
/// concept name), falling back to the first line of its summary.
fn child_label(node: &TreeNode) -> String {
    let name = node.name.trim();
    let raw = if name.is_empty() {
        node.summary.lines().next().unwrap_or("").trim()
    } else {
        name
    };
    condense(raw, 60)
}

/// Build a collection descriptor: `"{n} items: title; title; …"`, bounded to
/// `max_len` chars. Empty when there are no children.
fn collection_summary(children: &[TreeNode], max_len: usize) -> String {
    if children.is_empty() {
        return String::new();
    }
    let noun = if children.len() == 1 { "item" } else { "items" };
    let labels: Vec<String> = children
        .iter()
        .map(child_label)
        .filter(|l| !l.is_empty())
        .collect();
    let body = format!("{} {}: {}", children.len(), noun, labels.join("; "));
    condense(&body, max_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ltm::Leaf;
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::ltm_repository::SqliteLtmRepository;
    use crate::infrastructure::schema::init_ltm_schema;

    const DIM: usize = 4;

    fn provenance() -> Provenance {
        Provenance {
            source: "test".into(),
            ingested_at: None,
            confidence: 1.0,
        }
    }

    /// A tree with root -> "job" (a concept embedded at [1,0,0,0]) and an inbox.
    /// Returns the service plus the concept/root/inbox ids.
    struct Fixture {
        svc: LtmPlacement,
        repo: Arc<SqliteLtmRepository>,
        root: i64,
        concept: i64,
        inbox: i64,
    }

    fn fixture() -> Fixture {
        let conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&conn, DIM).unwrap();
        let repo = Arc::new(SqliteLtmRepository::new(conn));

        let root = repo
            .create_node(
                &TreeNode::new("root", "root seed", TreeNodeKind::Spine),
                None,
            )
            .unwrap();
        let concept = repo
            .create_node(
                &TreeNode::new("job", "job seed", TreeNodeKind::Spine),
                Some(&[1.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        repo.add_edge(&TreeEdge::new(root, concept)).unwrap();
        let inbox = repo
            .create_node(&TreeNode::new("inbox", "", TreeNodeKind::Inbox), None)
            .unwrap();
        repo.add_edge(&TreeEdge::new(root, inbox)).unwrap();

        let svc = LtmPlacement::new(
            repo.clone() as Arc<dyn LtmRepository>,
            crate::domain::thresholds::TEXT_EMBEDDING_004.placement_max_distance,
        );
        Fixture {
            svc,
            repo,
            root,
            concept,
            inbox,
        }
    }

    /// A container summary describes the collection (count + child titles), and
    /// stays bounded — never a quote of one child's full text.
    #[test]
    fn test_collection_summary_lists_children() {
        let children = vec![
            TreeNode::new("Annual Budget Report", "long body a", TreeNodeKind::Leaf),
            TreeNode::new("Welcome Newsletter", "long body b", TreeNodeKind::Leaf),
        ];
        let s = collection_summary(&children, 600);
        assert!(s.starts_with("2 items: "), "has a count header: {s}");
        assert!(s.contains("Annual Budget Report"));
        assert!(s.contains("Welcome Newsletter"));
        assert!(!s.contains("long body"), "does not quote child bodies");

        // Singular header + empty case.
        assert!(collection_summary(&children[..1], 600).starts_with("1 item: "));
        assert_eq!(collection_summary(&[], 600), "");
    }

    /// A document whose embedding matches a concept attaches under that concept.
    #[test]
    fn test_high_similarity_attaches_under_match() {
        let f = fixture();
        let doc = DocumentToPlace {
            name: "metro project".into(),
            summary: "BestBuy project Metro notes".into(),
            embedding: vec![1.0, 0.0, 0.0, 0.0], // distance 0 to the concept
            data_id: "doc_metro".into(),
            provenance: provenance(),
        };

        let placement = f.svc.place(&doc).unwrap();
        assert!(placement.matched, "should match the concept");
        assert_eq!(placement.parent_id, f.concept);

        let parents = f.repo.get_parents(placement.leaf_node_id).unwrap();
        assert_eq!(parents.len(), 1);
        assert_eq!(parents[0].id, Some(f.concept));
    }

    /// The inbox gardener re-homes an inbox document whose stored embedding now
    /// matches a concept, leaves an unmatched one alone, and is idempotent.
    #[test]
    fn test_garden_inbox_rehomes_matching_leaf_only() {
        let f = fixture();

        // A doc in the inbox, embedded like the concept → should be re-homed.
        let match_leaf = f
            .repo
            .create_node(
                &TreeNode::new("job doc", "about the job", TreeNodeKind::Leaf),
                Some(&[1.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        f.repo
            .create_leaf(&Leaf {
                tree_node_id: match_leaf,
                data_id: "doc_match".into(),
                provenance: provenance(),
            })
            .unwrap();
        f.repo
            .add_edge(&TreeEdge::new(f.inbox, match_leaf))
            .unwrap();

        // A doc in the inbox, embedded far from any concept → should stay.
        let stray_leaf = f
            .repo
            .create_node(
                &TreeNode::new("stray", "unrelated", TreeNodeKind::Leaf),
                Some(&[0.0, 0.0, 0.0, 1.0]),
            )
            .unwrap();
        f.repo
            .create_leaf(&Leaf {
                tree_node_id: stray_leaf,
                data_id: "doc_stray".into(),
                provenance: provenance(),
            })
            .unwrap();
        f.repo
            .add_edge(&TreeEdge::new(f.inbox, stray_leaf))
            .unwrap();

        let moved = f.svc.garden_inbox().unwrap();
        assert_eq!(moved, 1, "only the matching leaf is re-homed");

        let mp = f.repo.get_parents(match_leaf).unwrap();
        assert_eq!(mp.len(), 1);
        assert_eq!(
            mp[0].id,
            Some(f.concept),
            "matching leaf now under the concept"
        );

        let sp = f.repo.get_parents(stray_leaf).unwrap();
        assert_eq!(sp.len(), 1);
        assert_eq!(sp[0].id, Some(f.inbox), "stray leaf stays in the inbox");

        // Idempotent: a second pass moves nothing.
        assert_eq!(f.svc.garden_inbox().unwrap(), 0);
    }

    /// A document with no nearby concept falls to the inbox.
    #[test]
    fn test_low_similarity_falls_to_inbox() {
        let f = fixture();
        let doc = DocumentToPlace {
            name: "stray".into(),
            summary: "unrelated content".into(),
            embedding: vec![0.0, 0.0, 0.0, 1.0], // distance sqrt(2) > threshold
            data_id: "doc_stray".into(),
            provenance: provenance(),
        };

        let placement = f.svc.place(&doc).unwrap();
        assert!(!placement.matched, "should not match");
        assert_eq!(placement.parent_id, f.inbox);
    }

    /// After attaching, the parent concept's rolling summary describes its
    /// document collection (count + the leaf's title), and the ancestor (root)
    /// summary lists its child branches — a table-of-contents roll-up, not a
    /// quote of one document.
    #[test]
    fn test_ancestor_summaries_roll_up_on_attach() {
        let f = fixture();
        let doc = DocumentToPlace {
            name: "Metro rollout plan".into(),
            summary: "Metro rollout details".into(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            data_id: "doc_metro".into(),
            provenance: provenance(),
        };
        f.svc.place(&doc).unwrap();

        // The concept now reads as a 1-document collection, titled by the leaf.
        let concept = f.repo.get_node(f.concept).unwrap().unwrap();
        assert!(
            concept.summary.contains("1 item") && concept.summary.contains("Metro rollout plan"),
            "concept describes its collection: {:?}",
            concept.summary
        );
        // The root lists its child branch (by name), not the document's text.
        let root = f.repo.get_node(f.root).unwrap().unwrap();
        assert!(
            root.summary.contains("job"),
            "root lists its branches: {:?}",
            root.summary
        );
        assert!(
            !root.summary.contains("Metro rollout details"),
            "one document must not poison the root summary: {:?}",
            root.summary
        );
    }

    /// Forgetting a dataId removes its leaf and re-rolls the ancestors so the
    /// document no longer appears in their summaries.
    #[test]
    fn test_forget_removes_leaf_and_rerolls() {
        let f = fixture();
        let doc = DocumentToPlace {
            name: "metro".into(),
            summary: "Metro rollout".into(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            data_id: "doc_metro".into(),
            provenance: provenance(),
        };
        let placement = f.svc.place(&doc).unwrap();

        let removed = f.svc.forget("doc_metro").unwrap();
        assert!(removed, "document was present");

        // Leaf is gone from the tree and from the dataId index.
        assert!(f.repo.get_node(placement.leaf_node_id).unwrap().is_none());
        assert!(f.repo.get_node_by_data_id("doc_metro").unwrap().is_none());

        // Ancestors re-rolled: the document's summary no longer appears.
        let concept = f.repo.get_node(f.concept).unwrap().unwrap();
        assert!(
            !concept.summary.contains("Metro rollout"),
            "concept re-rolled: {:?}",
            concept.summary
        );
        let root = f.repo.get_node(f.root).unwrap().unwrap();
        assert!(!root.summary.contains("Metro rollout"));

        // Forgetting an unknown dataId is a no-op false.
        assert!(!f.svc.forget("doc_unknown").unwrap());
    }

    /// A second document identical to the first still routes to the concept,
    /// not the inbox — because leaves are not vector-indexed, the first doc's
    /// leaf can't crowd out the concept match.
    #[test]
    fn test_second_similar_doc_still_matches_concept() {
        let f = fixture();
        let make = |data_id: &str| DocumentToPlace {
            name: "metro".into(),
            summary: format!("Metro rollout {data_id}"),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            data_id: data_id.into(),
            provenance: provenance(),
        };

        let first = f.svc.place(&make("doc_1")).unwrap();
        assert!(first.matched);
        assert_eq!(first.parent_id, f.concept);

        let second = f.svc.place(&make("doc_2")).unwrap();
        assert!(second.matched, "second similar doc must still match");
        assert_eq!(
            second.parent_id, f.concept,
            "second doc must not fall to the inbox"
        );

        // Both leaves now hang under the concept.
        let children = f.repo.get_children(f.concept).unwrap();
        assert_eq!(children.len(), 2);
    }

    // --- Crash safety of leaf removal (W2R-2) ---------------------------------

    /// An LTM repository that "crashes" right after the first leaf delete
    /// commits: that delete is kept, and every later call fails, as if the
    /// process died there. Reads before the crash pass through.
    struct CrashAfterDelete {
        inner: Arc<SqliteLtmRepository>,
        crashed: std::cell::Cell<bool>,
    }

    impl CrashAfterDelete {
        fn alive(&self) -> Result<()> {
            if self.crashed.get() {
                anyhow::bail!("simulated crash");
            }
            Ok(())
        }
    }

    impl LtmRepository for CrashAfterDelete {
        fn create_node(&self, node: &TreeNode, embedding: Option<&[f32]>) -> Result<i64> {
            self.alive()?;
            self.inner.create_node(node, embedding)
        }
        fn add_edge(&self, edge: &TreeEdge) -> Result<()> {
            self.alive()?;
            self.inner.add_edge(edge)
        }
        fn create_leaf(&self, leaf: &Leaf) -> Result<()> {
            self.alive()?;
            self.inner.create_leaf(leaf)
        }
        fn create_leaf_node(
            &self,
            node: &TreeNode,
            embedding: &[f32],
            data_id: &str,
            provenance: &Provenance,
            parent_id: i64,
            body: Option<&DocumentBody>,
        ) -> Result<i64> {
            self.alive()?;
            self.inner
                .create_leaf_node(node, embedding, data_id, provenance, parent_id, body)
        }
        fn delete_document_body(&self, data_id: &str) -> Result<()> {
            self.alive()?;
            self.inner.delete_document_body(data_id)
        }
        fn get_node(&self, id: i64) -> Result<Option<TreeNode>> {
            self.alive()?;
            self.inner.get_node(id)
        }
        fn get_children(&self, parent_id: i64) -> Result<Vec<TreeNode>> {
            self.alive()?;
            self.inner.get_children(parent_id)
        }
        fn get_parents(&self, child_id: i64) -> Result<Vec<TreeNode>> {
            self.alive()?;
            self.inner.get_parents(child_id)
        }
        fn get_node_by_data_id(&self, data_id: &str) -> Result<Option<TreeNode>> {
            self.alive()?;
            self.inner.get_node_by_data_id(data_id)
        }
        fn get_nodes_by_data_id(&self, data_id: &str) -> Result<Vec<TreeNode>> {
            self.alive()?;
            self.inner.get_nodes_by_data_id(data_id)
        }
        fn find_similar_concepts(
            &self,
            embedding: &[f32],
            max_distance: f64,
            limit: usize,
        ) -> Result<Vec<(TreeNode, f64)>> {
            self.alive()?;
            self.inner
                .find_similar_concepts(embedding, max_distance, limit)
        }
        fn find_similar_any(
            &self,
            embedding: &[f32],
            max_distance: f64,
            limit: usize,
        ) -> Result<Vec<(TreeNode, f64)>> {
            self.alive()?;
            self.inner.find_similar_any(embedding, max_distance, limit)
        }
        fn get_leaf(&self, tree_node_id: i64) -> Result<Option<Leaf>> {
            self.alive()?;
            self.inner.get_leaf(tree_node_id)
        }
        fn get_inbox(&self) -> Result<Option<TreeNode>> {
            self.alive()?;
            self.inner.get_inbox()
        }
        fn update_summary(&self, node_id: i64, summary: &str) -> Result<()> {
            self.alive()?;
            self.inner.update_summary(node_id, summary)
        }
        fn set_concept_embedding(&self, node_id: i64, embedding: &[f32]) -> Result<()> {
            self.alive()?;
            self.inner.set_concept_embedding(node_id, embedding)
        }
        fn concepts_missing_embedding(&self) -> Result<Vec<TreeNode>> {
            self.alive()?;
            self.inner.concepts_missing_embedding()
        }
        fn sample_leaf_embeddings(&self, sample: usize) -> Result<Vec<(i64, String, Vec<f32>)>> {
            self.alive()?;
            self.inner.sample_leaf_embeddings(sample)
        }
        fn delete_node(&self, node_id: i64) -> Result<()> {
            self.alive()?;
            self.inner.delete_node(node_id)?;
            self.crashed.set(true);
            Ok(())
        }
        fn delete_node_with_summaries(
            &self,
            node_id: i64,
            summaries: &[(i64, String)],
        ) -> Result<()> {
            self.alive()?;
            self.inner.delete_node_with_summaries(node_id, summaries)?;
            self.crashed.set(true);
            Ok(())
        }
        fn remove_edge(&self, parent_id: i64, child_id: i64) -> Result<()> {
            self.alive()?;
            self.inner.remove_edge(parent_id, child_id)
        }
        fn inbox_leaf_embeddings(&self) -> Result<Vec<(i64, Vec<f32>)>> {
            self.alive()?;
            self.inner.inbox_leaf_embeddings()
        }
        fn seed_spine_from(&self, seeds: &[crate::domain::ltm::SpineSeed]) -> Result<()> {
            self.alive()?;
            self.inner.seed_spine_from(seeds)
        }
        fn reset_store(&self) -> Result<()> {
            self.alive()?;
            self.inner.reset_store()
        }
        fn get_roots(&self) -> Result<Vec<TreeNode>> {
            self.alive()?;
            self.inner.get_roots()
        }
        fn get_child_leaves(&self, parent_id: i64) -> Result<Vec<Leaf>> {
            self.alive()?;
            self.inner.get_child_leaves(parent_id)
        }
        fn ltm_stats(&self) -> Result<crate::domain::ltm::LtmStats> {
            self.alive()?;
            self.inner.ltm_stats()
        }
        fn put_document_body(&self, data_id: &str, title: &str, body: &str) -> Result<()> {
            self.alive()?;
            self.inner.put_document_body(data_id, title, body)
        }
        fn search_document_bodies(&self, match_query: &str, limit: usize) -> Result<Vec<i64>> {
            self.alive()?;
            self.inner.search_document_bodies(match_query, limit)
        }
        fn leaf_distances(&self, leaf_ids: &[i64], embedding: &[f32]) -> Result<HashMap<i64, f64>> {
            self.alive()?;
            self.inner.leaf_distances(leaf_ids, embedding)
        }
    }

    fn lease(name: &str, embedding: [f32; 4]) -> DocumentToPlace {
        DocumentToPlace {
            name: name.into(),
            summary: format!("{name} body"),
            embedding: embedding.to_vec(),
            data_id: "doc_lease".into(),
            provenance: provenance(),
        }
    }

    fn summary_of(repo: &SqliteLtmRepository, id: i64) -> String {
        repo.get_node(id).unwrap().unwrap().summary
    }

    /// W2R-2: a crash right after the old version's leaf is deleted (before any
    /// roll-up runs) must not leave its old parent listing the removed
    /// document, even after the redelivered update completes. The old leaf is
    /// gone by then, so the redelivery alone can't find that parent again.
    #[test]
    fn test_crash_after_leaf_delete_leaves_no_stale_parent_summary() {
        let f = fixture();
        // v1 lands in the inbox (far from the concept).
        f.svc
            .place(&lease("Old lease", [0.0, 0.0, 0.0, 1.0]))
            .unwrap();
        assert!(summary_of(&f.repo, f.inbox).contains("Old lease"));

        // v2 matches the concept; the process dies right after v1's leaf is
        // deleted.
        let crashy = Arc::new(CrashAfterDelete {
            inner: f.repo.clone(),
            crashed: std::cell::Cell::new(false),
        });
        let svc = LtmPlacement::new(
            crashy as Arc<dyn LtmRepository>,
            crate::domain::thresholds::TEXT_EMBEDDING_004.placement_max_distance,
        );
        let v2 = lease("New lease", [1.0, 0.0, 0.0, 0.0]);
        assert!(svc.replace(&v2).is_err(), "the simulated crash surfaces");

        // After the restart, the redelivered v2 completes.
        f.svc.replace(&v2).unwrap();
        let leaves = f.repo.get_nodes_by_data_id("doc_lease").unwrap();
        assert_eq!(leaves.len(), 1, "{leaves:?}");
        assert!(summary_of(&f.repo, f.concept).contains("New lease"));
        let inbox = summary_of(&f.repo, f.inbox);
        assert!(
            !inbox.contains("Old lease"),
            "old parent still lists the removed document: {inbox:?}"
        );
    }

    /// The same for a tombstone: a crash right after the delete doesn't leave
    /// the parent listing the forgotten document.
    #[test]
    fn test_crash_after_forget_delete_leaves_no_stale_parent_summary() {
        let f = fixture();
        f.svc
            .place(&lease("Old lease", [0.0, 0.0, 0.0, 1.0]))
            .unwrap();
        let crashy = Arc::new(CrashAfterDelete {
            inner: f.repo.clone(),
            crashed: std::cell::Cell::new(false),
        });
        let svc = LtmPlacement::new(
            crashy as Arc<dyn LtmRepository>,
            crate::domain::thresholds::TEXT_EMBEDDING_004.placement_max_distance,
        );
        assert!(
            svc.forget("doc_lease").is_err(),
            "the simulated crash surfaces"
        );

        // The redelivered tombstone finds nothing left to delete.
        assert!(!f.svc.forget("doc_lease").unwrap());
        let inbox = summary_of(&f.repo, f.inbox);
        assert!(!inbox.contains("Old lease"), "{inbox:?}");
    }

    // --- E3: placement targets and the margin rule ----------------------------

    fn doc_at(name: &str, data_id: &str, embedding: [f32; 4]) -> DocumentToPlace {
        DocumentToPlace {
            name: name.into(),
            summary: format!("{name} body"),
            embedding: embedding.to_vec(),
            data_id: data_id.into(),
            provenance: provenance(),
        }
    }

    /// E3a: the root is never a placement target, even when it's the nearest
    /// embedded node. The document goes to the nearest real concept within
    /// the cutoff, or to the inbox.
    #[test]
    fn test_root_is_never_a_placement_target() {
        let f = fixture();
        // The root sits exactly where the documents are.
        f.repo
            .set_concept_embedding(f.root, &[0.0, 1.0, 0.0, 0.0])
            .unwrap();
        f.repo
            .set_concept_embedding(f.concept, &[0.0, 0.8, 0.0, 0.0])
            .unwrap();

        // Concept 0.2 away, root 0: files under the concept.
        let near = f
            .svc
            .place(&doc_at("near", "doc_near", [0.0, 1.0, 0.0, 0.0]))
            .unwrap();
        assert_eq!(near.parent_id, f.concept);
        assert!(near.matched);

        // Only the root within the cutoff: the inbox, not the root.
        f.repo
            .set_concept_embedding(f.concept, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        let far = f
            .svc
            .place(&doc_at("far", "doc_far", [0.0, 1.0, 0.0, 0.0]))
            .unwrap();
        assert_eq!(far.parent_id, f.inbox);
        assert!(!far.matched);
        assert!(f.repo.get_child_leaves(f.root).unwrap().is_empty());
    }

    /// E3a: the inbox gardener never moves a document under the root either.
    #[test]
    fn test_garden_inbox_never_moves_to_root() {
        let f = fixture();
        let leaf = f
            .svc
            .place(&doc_at("stray", "doc_stray", [0.0, 1.0, 0.0, 0.0]))
            .unwrap();
        assert_eq!(leaf.parent_id, f.inbox);
        // Later the root gets a vector right on top of the document.
        f.repo
            .set_concept_embedding(f.root, &[0.0, 1.0, 0.0, 0.0])
            .unwrap();

        assert_eq!(f.svc.garden_inbox().unwrap(), 0);
        let parents = f.repo.get_parents(leaf.leaf_node_id).unwrap();
        assert_eq!(parents.len(), 1);
        assert_eq!(parents[0].id, Some(f.inbox));
    }

    fn concept_at(name: &str, distance: f64) -> ConceptMatch {
        ConceptMatch {
            id: name.len() as i64,
            name: name.into(),
            distance,
        }
    }

    /// E3b: the decision table. Cutoff first; then the margin rule, which only
    /// applies when it's on and a runner-up exists.
    #[test]
    fn test_decide_applies_cutoff_then_margin() {
        let repo = fixture().repo as Arc<dyn LtmRepository>;
        let off = LtmPlacement::new(repo.clone(), 1.0);
        let on = LtmPlacement::new(repo, 1.0).with_min_margin(0.05);
        let both = |best: f64, runner: f64| Candidates {
            best: Some(concept_at("best", best)),
            runner_up: Some(concept_at("runner_up", runner)),
        };

        for p in [&off, &on] {
            assert_eq!(
                p.decide(&Candidates::default()),
                PlacementDecision::InboxTooFar
            );
            assert_eq!(p.decide(&both(1.01, 1.5)), PlacementDecision::InboxTooFar);
            // A lone concept has no runner-up to be ambiguous with.
            let lone = Candidates {
                best: Some(concept_at("best", 0.9)),
                runner_up: None,
            };
            assert_eq!(p.decide(&lone), PlacementDecision::File);
        }
        assert_eq!(off.decide(&both(0.90, 0.91)), PlacementDecision::File);
        assert_eq!(
            on.decide(&both(0.90, 0.94)),
            PlacementDecision::InboxAmbiguous
        );
        assert_eq!(on.decide(&both(0.90, 0.96)), PlacementDecision::File);
        // The runner-up may sit beyond the cutoff; it still counts.
        assert_eq!(
            on.decide(&both(0.98, 1.02)),
            PlacementDecision::InboxAmbiguous
        );
        assert_eq!(
            both(0.90, 0.94).margin().map(|m| (m * 100.0).round()),
            Some(4.0)
        );
    }

    /// E3b: with the margin rule on, a document that is nearly as close to a
    /// second concept goes to the inbox; a clear winner is still filed. The
    /// runner-up is found even when it's beyond the cutoff.
    #[test]
    fn test_margin_rule_sends_ambiguous_documents_to_the_inbox() {
        let f = fixture();
        let home = f
            .repo
            .create_node(
                &TreeNode::new("home", "home seed", TreeNodeKind::Spine),
                Some(&[1.0, 0.04, 0.0, 0.0]),
            )
            .unwrap();
        f.repo.add_edge(&TreeEdge::new(f.root, home)).unwrap();
        let svc = |margin: f64| {
            LtmPlacement::new(f.repo.clone() as Arc<dyn LtmRepository>, 1.10)
                .with_min_margin(margin)
        };
        let doc = |id: &str| doc_at("contract", id, [1.0, 0.0, 0.0, 0.0]);

        // "job" at 0, "home" at 0.04: a 0.04 margin.
        let c = svc(0.05).candidates(&doc("x").embedding).unwrap();
        assert_eq!(c.best.as_ref().map(|b| b.id), Some(f.concept));
        assert_eq!(c.runner_up.as_ref().map(|r| r.id), Some(home));

        let ambiguous = svc(0.05).place(&doc("doc_a")).unwrap();
        assert_eq!(ambiguous.parent_id, f.inbox);
        assert!(!ambiguous.matched);
        let clear = svc(0.03).place(&doc("doc_b")).unwrap();
        assert_eq!(clear.parent_id, f.concept);
        let off = svc(0.0).place(&doc("doc_c")).unwrap();
        assert_eq!(off.parent_id, f.concept);

        // The gardener applies the same rule: with the rule on, the ambiguous
        // document stays put.
        assert_eq!(svc(0.05).garden_inbox().unwrap(), 0);
        assert_eq!(svc(0.0).garden_inbox().unwrap(), 1);
    }

    /// `from_thresholds` takes both the cutoff and the margin.
    #[test]
    fn test_from_thresholds_carries_the_margin() {
        use crate::domain::thresholds::{ThresholdOverrides, Thresholds};
        let (t, _) = Thresholds::resolve(
            "local:bge-small-en-v1.5",
            ThresholdOverrides {
                placement_max_distance: Some(0.5),
                placement_min_margin: Some(0.07),
                ..Default::default()
            },
        )
        .unwrap();
        let p = LtmPlacement::from_thresholds(fixture().repo as Arc<dyn LtmRepository>, &t);
        assert_eq!(p.max_distance, 0.5);
        assert_eq!(p.min_margin, 0.07);
    }
}
