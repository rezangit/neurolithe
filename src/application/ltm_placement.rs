//! LTM placement — files a new document into the knowledge tree and keeps
//! rolling summaries in sync.
//!
//! V2 is deliberately **simple**: best-match-or-inbox placement and a
//! concatenate-and-truncate roll-up. The hard "grow the right way" logic
//! (split fat nodes, merge similar branches, spawn new branches) is deferred —
//! see `V2-DESIGN.md` §3.2.

use crate::domain::ltm::{Leaf, LtmRepository, Provenance, TreeEdge, TreeNode, TreeNodeKind};
use crate::domain::ports::LlmClient;
use anyhow::{Result, anyhow, bail};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Max L2 distance for a document to count as matching a concept node; beyond
/// it, the document falls to the inbox.
///
/// Tuned from **measured** distances on real data (`placement_debug` over the
/// live corpus): document→concept L2 distances cluster in ~[0.91, 1.16] (median
/// 1.05) for unit-normalized `text-embedding-004`, i.e. cosine ~0.33–0.59 — much
/// larger than a category label suggests. 1.10 (cosine ≈ 0.40) files the
/// confident majority under their nearest branch while leaving the ambiguous
/// tail in the inbox. Earlier guesses (0.5, then 0.85) were below the whole
/// distribution and filed 100% to the inbox (field-report §3).
const DEFAULT_MAX_DISTANCE: f64 = 1.10;

/// Embed every concept node that lacks a placement vector, deriving the vector
/// from the concept's curated identity (`name: summary`) — **not** its rolling
/// summary, so filing targets stay stable as documents accumulate. Idempotent:
/// embeds only the missing ones, so steady-state startups make zero LLM calls.
/// Returns how many concepts were embedded. The daemon runs this after
/// `seed_spine` so placement has live match targets.
pub async fn embed_spine_concepts(
    ltm: &Arc<dyn LtmRepository>,
    llm: &Arc<dyn LlmClient>,
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
        let embedding = llm.embed_text(&text).await?;
        if embedding.len() != expected_dim {
            bail!(
                "concept '{}' embedding dim {} != LTM dim {}",
                node.name,
                embedding.len(),
                expected_dim
            );
        }
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

/// Files documents into the LTM tree and rolls summaries up the affected
/// ancestors.
pub struct LtmPlacement {
    repo: Arc<dyn LtmRepository>,
    max_distance: f64,
    max_summary_len: usize,
}

impl LtmPlacement {
    pub fn new(repo: Arc<dyn LtmRepository>) -> Self {
        Self {
            repo,
            max_distance: DEFAULT_MAX_DISTANCE,
            max_summary_len: DEFAULT_MAX_SUMMARY_LEN,
        }
    }

    /// File a document: attach it under the best-matching concept (vector within
    /// threshold) or, failing that, under the inbox; then roll its summary up
    /// into the ancestors. Returns where it landed.
    pub fn place(&self, doc: &DocumentToPlace) -> Result<Placement> {
        // 1. Best concept match, or fall back to the inbox.
        let best = self
            .repo
            .find_similar_concepts(&doc.embedding, self.max_distance, 1)?
            .into_iter()
            .next();
        let (parent_id, matched) = match best {
            Some((node, _dist)) => (node.id.expect("stored node has id"), true),
            None => {
                let inbox = self
                    .repo
                    .get_inbox()?
                    .ok_or_else(|| anyhow!("no inbox node — spine not seeded"))?;
                (inbox.id.expect("stored node has id"), false)
            }
        };

        // 2. Create the leaf node and its dataId link. The leaf IS vector-indexed
        //    (its summary embedding), so `recall` can land on the document
        //    directly by meaning — essential while docs pile in the inbox before
        //    the tree grows real concepts around them. This does NOT affect
        //    placement: step 1 uses the concept-only `find_similar_concepts`, so
        //    an embedded leaf can never crowd out a concept match and misroute a
        //    new document to the inbox.
        let leaf_node_id = self.repo.create_node(
            &TreeNode::new(doc.name.clone(), doc.summary.clone(), TreeNodeKind::Leaf),
            Some(&doc.embedding),
        )?;
        // Stamp the ingest time from the DB-assigned `created_at` when the caller
        // didn't supply one, so `provenance.ingested_at` is populated everywhere
        // it surfaces (listings, recall, trace) — it was null on every leaf
        // before (field-report §5). No wall-clock dependency: the tree node's
        // created_at is the authoritative ingest instant.
        let mut provenance = doc.provenance.clone();
        if provenance.ingested_at.is_none() {
            provenance.ingested_at = self.repo.get_node(leaf_node_id)?.and_then(|n| n.created_at);
        }
        self.repo.create_leaf(&Leaf {
            tree_node_id: leaf_node_id,
            data_id: doc.data_id.clone(),
            provenance,
        })?;
        self.repo
            .add_edge(&TreeEdge::new(parent_id, leaf_node_id))?;

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
            let Some((concept, _dist)) = self
                .repo
                .find_similar_concepts(&embedding, self.max_distance, 1)?
                .into_iter()
                .next()
            else {
                continue; // still no concept within threshold — leave in inbox
            };
            let concept_id = concept.id.expect("stored node has id");
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
        let placed = self.place(doc)?;
        let replaced = self.remove_leaves(&doc.data_id, Some(placed.leaf_node_id))?;
        Ok((placed, replaced))
    }

    /// Delete every leaf node carrying `data_id` except `keep`, then re-roll
    /// each affected parent (once). Returns how many leaves were deleted.
    fn remove_leaves(&self, data_id: &str, keep: Option<i64>) -> Result<usize> {
        let mut parents: Vec<i64> = Vec::new();
        let mut removed = 0;
        for leaf in self.repo.get_nodes_by_data_id(data_id)? {
            let leaf_id = leaf.id.expect("stored node has id");
            if Some(leaf_id) == keep {
                continue;
            }
            // Capture the parents before deletion so we know what to re-roll.
            for parent in self.repo.get_parents(leaf_id)? {
                let pid = parent.id.expect("stored node has id");
                if !parents.contains(&pid) {
                    parents.push(pid);
                }
            }
            self.repo.delete_node(leaf_id)?;
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

        let svc = LtmPlacement::new(repo.clone() as Arc<dyn LtmRepository>);
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
}
