//! LTM retrieval — navigate the knowledge tree: map -> expand -> drill, plus
//! vector recall.
//!
//! Results are **reference-returning**: leaves surface their `dataId` +
//! provenance (the opposite of STM's id-hiding `MemoryResult`), so the agent
//! can fetch originals from Ledger -> Pithos. See `V2-DESIGN.md` §3.3.

use crate::domain::ltm::{LtmRepository, Provenance, TreeNode, TreeNodeKind};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Recall applies **no** distance cutoff: the nearest `k` nodes are always
/// returned, ordered by distance (which each hit reports). A fixed L2 cutoff
/// assumes unit-norm embeddings, which nothing enforces — with an un-normalised
/// embedder it silently empties every recall (REV-4). Output stays bounded by
/// [`RECALL_MAX_K`] and the per-hit child/leaf caps.
const RECALL_NO_DISTANCE_CAP: f64 = f64::MAX;

/// Upper bound on hits per recall, whatever `k` the caller asks for.
pub const RECALL_MAX_K: usize = 50;

/// Per hit, at most this many child concepts and this many document leaves are
/// returned, so a broad concept with thousands of documents cannot flood the
/// caller's context (ARC-15). Use `drill`/`inspect_node` to page further.
pub const RECALL_MAX_CHILDREN: usize = 20;
pub const RECALL_MAX_LEAVES: usize = 20;

/// A concept/leaf node, flattened for output (no internal timestamps).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeView {
    pub id: i64,
    pub name: String,
    pub summary: String,
    pub kind: TreeNodeKind,
}

impl From<&TreeNode> for NodeView {
    fn from(node: &TreeNode) -> Self {
        Self {
            id: node.id.expect("stored node has id"),
            name: node.name.clone(),
            summary: node.summary.clone(),
            kind: node.kind,
        }
    }
}

/// A node plus its (concept) descendants down to a depth — the "table of
/// contents". Leaves (documents) are omitted; reach them via [`LtmRetrieval::drill`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapNode {
    pub id: i64,
    pub name: String,
    pub summary: String,
    pub kind: TreeNodeKind,
    pub children: Vec<MapNode>,
}

/// A reference to a document leaf: its archive pointer + provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeafRef {
    pub data_id: String,
    pub provenance: Provenance,
}

/// A node and its direct children (concepts and leaves).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpandResult {
    pub node: NodeView,
    pub children: Vec<NodeView>,
}

/// A node and the documents attached directly beneath it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrillResult {
    pub node: NodeView,
    pub leaves: Vec<LeafRef>,
}

/// Vector recall: the located entry concept, framed by its ancestors (up) and
/// detailed by its children + document leaves (down).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallResult {
    pub entry: NodeView,
    pub ancestors: Vec<NodeView>,
    pub children: Vec<NodeView>,
    pub leaves: Vec<LeafRef>,
    pub distance: f64,
}

/// Read-only navigation over the LTM tree.
pub struct LtmRetrieval {
    repo: Arc<dyn LtmRepository>,
}

impl LtmRetrieval {
    pub fn new(repo: Arc<dyn LtmRepository>) -> Self {
        Self { repo }
    }

    /// The top `depth` concept layers from the roots — a compact map of the
    /// person within a bounded token budget. `depth` = number of layers
    /// (1 = roots only). Leaf/document nodes are excluded.
    pub fn map(&self, depth: usize) -> Result<Vec<MapNode>> {
        if depth == 0 {
            return Ok(Vec::new());
        }
        let roots = self.repo.get_roots()?;
        roots.iter().map(|n| self.build_map(n, depth)).collect()
    }

    fn build_map(&self, node: &TreeNode, remaining: usize) -> Result<MapNode> {
        let node_id = node.id.expect("stored node has id");
        let children = if remaining <= 1 {
            Vec::new()
        } else {
            self.repo
                .get_children(node_id)?
                .iter()
                .filter(|c| c.kind != TreeNodeKind::Leaf)
                .map(|c| self.build_map(c, remaining - 1))
                .collect::<Result<Vec<_>>>()?
        };
        Ok(MapNode {
            id: node_id,
            name: node.name.clone(),
            summary: node.summary.clone(),
            kind: node.kind,
            children,
        })
    }

    /// A branch of the tree: the node and its concept descendants down to
    /// `depth` (leaves excluded, like [`map`](Self::map)). `None` if the node
    /// does not exist.
    pub fn subtree(&self, node_id: i64, depth: usize) -> Result<Option<MapNode>> {
        if depth == 0 {
            return Ok(None);
        }
        match self.repo.get_node(node_id)? {
            Some(node) => Ok(Some(self.build_map(&node, depth)?)),
            None => Ok(None),
        }
    }

    /// All distinct ancestors of a node (nearest first) as views — for framing.
    pub fn ancestors(&self, node_id: i64) -> Result<Vec<NodeView>> {
        Ok(self
            .collect_ancestors(node_id)?
            .iter()
            .map(NodeView::from)
            .collect())
    }

    /// A node's direct children (concepts and leaves) with their summaries.
    pub fn expand(&self, node_id: i64) -> Result<Option<ExpandResult>> {
        let Some(node) = self.repo.get_node(node_id)? else {
            return Ok(None);
        };
        let children = self
            .repo
            .get_children(node_id)?
            .iter()
            .map(NodeView::from)
            .collect();
        Ok(Some(ExpandResult {
            node: NodeView::from(&node),
            children,
        }))
    }

    /// A node's detail plus the documents (`dataId` + provenance) attached
    /// directly beneath it.
    pub fn drill(&self, node_id: i64) -> Result<Option<DrillResult>> {
        let Some(node) = self.repo.get_node(node_id)? else {
            return Ok(None);
        };
        let leaves = self
            .repo
            .get_child_leaves(node_id)?
            .into_iter()
            .map(|l| LeafRef {
                data_id: l.data_id,
                provenance: l.provenance,
            })
            .collect();
        Ok(Some(DrillResult {
            node: NodeView::from(&node),
            leaves,
        }))
    }

    /// Vector-locate the `k` nearest nodes to `embedding` — concepts **and**
    /// document leaves, with no distance cutoff — nearest
    /// first, and assemble context for each. A concept hit returns its
    /// ancestors + (bounded) children + (bounded) child document leaves. A leaf
    /// hit returns the document itself (its `dataId`/provenance) framed by its
    /// ancestors, so a document is recallable by meaning even while it sits
    /// under the inbox. Empty only when nothing is vector-indexed.
    pub fn recall(&self, embedding: &[f32], k: usize) -> Result<Vec<RecallResult>> {
        let k = k.clamp(1, RECALL_MAX_K);
        self.repo
            .find_similar_any(embedding, RECALL_NO_DISTANCE_CAP, k)?
            .into_iter()
            .map(|(entry, distance)| self.assemble_hit(entry, distance))
            .collect()
    }

    fn assemble_hit(&self, entry: TreeNode, distance: f64) -> Result<RecallResult> {
        let entry_id = entry.id.expect("stored node has id");

        let ancestors = self
            .collect_ancestors(entry_id)?
            .iter()
            .map(NodeView::from)
            .collect();

        let to_ref = |l: crate::domain::ltm::Leaf| LeafRef {
            data_id: l.data_id,
            provenance: l.provenance,
        };

        // A leaf hit surfaces the document directly; a concept hit surfaces its
        // children + attached document leaves, each bounded.
        let (children, leaves) = if entry.kind == TreeNodeKind::Leaf {
            let leaf = self
                .repo
                .get_leaf(entry_id)?
                .map(to_ref)
                .into_iter()
                .collect();
            (Vec::new(), leaf)
        } else {
            let children = self
                .repo
                .get_children(entry_id)?
                .iter()
                .take(RECALL_MAX_CHILDREN)
                .map(NodeView::from)
                .collect();
            let leaves = self
                .repo
                .get_child_leaves(entry_id)?
                .into_iter()
                .take(RECALL_MAX_LEAVES)
                .map(to_ref)
                .collect();
            (children, leaves)
        };

        Ok(RecallResult {
            entry: NodeView::from(&entry),
            ancestors,
            children,
            leaves,
            distance,
        })
    }

    /// All distinct ancestors of a node (BFS upward), nearest first.
    fn collect_ancestors(&self, node_id: i64) -> Result<Vec<TreeNode>> {
        use std::collections::{HashSet, VecDeque};
        let mut seen: HashSet<i64> = HashSet::new();
        let mut out: Vec<TreeNode> = Vec::new();
        let mut queue: VecDeque<i64> = VecDeque::new();
        queue.push_back(node_id);
        while let Some(current) = queue.pop_front() {
            for parent in self.repo.get_parents(current)? {
                let pid = parent.id.expect("stored node has id");
                if seen.insert(pid) {
                    queue.push_back(pid);
                    out.push(parent);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ltm::{Leaf, TreeEdge};
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::ltm_repository::SqliteLtmRepository;
    use crate::infrastructure::schema::init_ltm_schema;

    const DIM: usize = 4;

    /// root -> {jobs (embedded [1,0,0,0]), learning}; jobs has one document leaf.
    struct Fixture {
        ret: LtmRetrieval,
        root: i64,
        jobs: i64,
        learning: i64,
        leaf: i64,
    }

    fn fixture() -> Fixture {
        let conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&conn, DIM).unwrap();
        let repo = Arc::new(SqliteLtmRepository::new(conn));

        let root = repo
            .create_node(&TreeNode::new("root", "root", TreeNodeKind::Spine), None)
            .unwrap();
        let jobs = repo
            .create_node(
                &TreeNode::new("job", "work", TreeNodeKind::Spine),
                Some(&[1.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        let learning = repo
            .create_node(
                &TreeNode::new("learning", "study", TreeNodeKind::Spine),
                None,
            )
            .unwrap();
        repo.add_edge(&TreeEdge::new(root, jobs)).unwrap();
        repo.add_edge(&TreeEdge::new(root, learning)).unwrap();

        // A document leaf under jobs.
        let leaf = repo
            .create_node(
                &TreeNode::new("metro", "Metro project", TreeNodeKind::Leaf),
                None,
            )
            .unwrap();
        repo.create_leaf(&Leaf {
            tree_node_id: leaf,
            data_id: "doc_metro".into(),
            provenance: Provenance {
                source: "scanner".into(),
                ingested_at: None,
                confidence: 0.8,
            },
        })
        .unwrap();
        repo.add_edge(&TreeEdge::new(jobs, leaf)).unwrap();

        let ret = LtmRetrieval::new(repo as Arc<dyn LtmRepository>);
        Fixture {
            ret,
            root,
            jobs,
            learning,
            leaf,
        }
    }

    /// map(depth) returns only the top `depth` layers, excluding document leaves.
    #[test]
    fn test_map_respects_depth() {
        let f = fixture();

        // depth 1: just the root, no children.
        let m1 = f.ret.map(1).unwrap();
        assert_eq!(m1.len(), 1);
        assert_eq!(m1[0].id, f.root);
        assert!(m1[0].children.is_empty(), "depth 1 has no children");

        // depth 2: root + its two concept branches (the leaf is excluded).
        let m2 = f.ret.map(2).unwrap();
        assert_eq!(m2[0].children.len(), 2);
        let names: Vec<_> = m2[0].children.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"job"));
        assert!(names.contains(&"learning"));

        // depth 3: jobs still has no concept children (its only child is a leaf).
        let m3 = f.ret.map(3).unwrap();
        let jobs = m3[0]
            .children
            .iter()
            .find(|c| c.name == "job")
            .expect("job present");
        assert!(jobs.children.is_empty(), "leaves never appear in the map");
    }

    /// expand returns a node's direct children (including leaves).
    #[test]
    fn test_expand_returns_direct_children() {
        let f = fixture();

        let root = f.ret.expand(f.root).unwrap().expect("root exists");
        assert_eq!(root.children.len(), 2);

        let jobs = f.ret.expand(f.jobs).unwrap().expect("jobs exists");
        assert_eq!(jobs.children.len(), 1);
        assert_eq!(jobs.children[0].kind, TreeNodeKind::Leaf);
        assert_eq!(jobs.children[0].id, f.leaf);

        // learning has no children; an unknown node yields None.
        assert!(
            f.ret
                .expand(f.learning)
                .unwrap()
                .unwrap()
                .children
                .is_empty()
        );
        assert!(f.ret.expand(9999).unwrap().is_none());
    }

    /// drill returns the documents (dataId + provenance) beneath a node.
    #[test]
    fn test_drill_returns_leaves_with_data_id() {
        let f = fixture();

        let drilled = f.ret.drill(f.jobs).unwrap().expect("jobs exists");
        assert_eq!(drilled.node.id, f.jobs);
        assert_eq!(drilled.leaves.len(), 1);
        assert_eq!(drilled.leaves[0].data_id, "doc_metro");
        assert_eq!(drilled.leaves[0].provenance.source, "scanner");

        // A concept with no documents drills to an empty leaf list.
        assert!(f.ret.drill(f.learning).unwrap().unwrap().leaves.is_empty());
    }

    /// A vector query lands on the matching seeded concept, framed by its
    /// ancestor and detailed by its document leaf.
    #[test]
    fn test_recall_lands_on_seeded_concept() {
        let f = fixture();

        let recalled = f
            .ret
            .recall(&[1.0, 0.0, 0.0, 0.0], 1)
            .unwrap()
            .into_iter()
            .next()
            .expect("a concept exists");
        assert_eq!(recalled.entry.id, f.jobs, "vector entry is the job concept");
        assert!(recalled.distance < 1e-6, "exact match has ~0 distance");

        // Up: framed by the root ancestor.
        let ancestor_ids: Vec<_> = recalled.ancestors.iter().map(|n| n.id).collect();
        assert!(ancestor_ids.contains(&f.root));

        // Down: the document leaf is surfaced as a reference.
        assert_eq!(recalled.leaves.len(), 1);
        assert_eq!(recalled.leaves[0].data_id, "doc_metro");
    }

    /// A document leaf embedded nearer than any concept is recalled **directly**:
    /// the entry is the leaf and it carries its own `dataId`. This is what makes
    /// inbox documents reachable by meaning before the tree grows concepts.
    #[test]
    fn test_recall_returns_document_when_leaf_is_nearest() {
        let conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&conn, DIM).unwrap();
        let repo = Arc::new(SqliteLtmRepository::new(conn));

        // A concept off in one direction; a document leaf embedded in another.
        let root = repo
            .create_node(&TreeNode::new("root", "root", TreeNodeKind::Spine), None)
            .unwrap();
        let concept = repo
            .create_node(
                &TreeNode::new("job", "work", TreeNodeKind::Spine),
                Some(&[1.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        repo.add_edge(&TreeEdge::new(root, concept)).unwrap();
        let leaf = repo
            .create_node(
                &TreeNode::new("note", "a durable metis note", TreeNodeKind::Leaf),
                Some(&[0.0, 1.0, 0.0, 0.0]),
            )
            .unwrap();
        repo.create_leaf(&Leaf {
            tree_node_id: leaf,
            data_id: "note_1".into(),
            provenance: Provenance {
                source: "metis".into(),
                ingested_at: None,
                confidence: 1.0,
            },
        })
        .unwrap();
        repo.add_edge(&TreeEdge::new(root, leaf)).unwrap();

        let ret = LtmRetrieval::new(repo as Arc<dyn LtmRepository>);
        let recalled = ret
            .recall(&[0.0, 1.0, 0.0, 0.0], 1)
            .unwrap()
            .into_iter()
            .next()
            .expect("the leaf is vector-indexed");

        // Nearest node is the document leaf → entry is the leaf, carrying its dataId.
        assert_eq!(recalled.entry.id, leaf);
        assert_eq!(recalled.entry.kind, TreeNodeKind::Leaf);
        assert_eq!(recalled.leaves.len(), 1);
        assert_eq!(recalled.leaves[0].data_id, "note_1");
        assert_eq!(recalled.leaves[0].provenance.source, "metis");
        // Framed by its ancestor for context.
        assert!(recalled.ancestors.iter().any(|n| n.id == root));
    }

    fn leaf_under(
        repo: &Arc<SqliteLtmRepository>,
        parent: i64,
        data_id: &str,
        emb: Option<&[f32]>,
    ) {
        let leaf = repo
            .create_node(&TreeNode::new(data_id, "doc", TreeNodeKind::Leaf), emb)
            .unwrap();
        repo.create_leaf(&Leaf {
            tree_node_id: leaf,
            data_id: data_id.into(),
            provenance: Provenance {
                source: "test".into(),
                ingested_at: None,
                confidence: 1.0,
            },
        })
        .unwrap();
        repo.add_edge(&TreeEdge::new(parent, leaf)).unwrap();
    }

    /// ARC-15: recall returns the top-k hits, nearest first (not just one).
    #[test]
    fn test_recall_returns_top_k_nearest_first() {
        let conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&conn, DIM).unwrap();
        let repo = Arc::new(SqliteLtmRepository::new(conn));
        let root = repo
            .create_node(&TreeNode::new("root", "root", TreeNodeKind::Spine), None)
            .unwrap();
        let mut ids = Vec::new();
        for (name, emb) in [
            ("near", [1.0, 0.0, 0.0, 0.0]),
            ("mid", [1.0, 0.5, 0.0, 0.0]),
            ("far", [1.0, 0.9, 0.0, 0.0]),
        ] {
            let id = repo
                .create_node(&TreeNode::new(name, name, TreeNodeKind::Spine), Some(&emb))
                .unwrap();
            repo.add_edge(&TreeEdge::new(root, id)).unwrap();
            ids.push(id);
        }
        let ret = LtmRetrieval::new(repo as Arc<dyn LtmRepository>);

        let two: Vec<i64> = ret
            .recall(&[1.0, 0.0, 0.0, 0.0], 2)
            .unwrap()
            .iter()
            .map(|r| r.entry.id)
            .collect();
        assert_eq!(two, vec![ids[0], ids[1]], "top-2, nearest first");

        let all = ret.recall(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(all.len(), 3);
        assert!(all.windows(2).all(|w| w[0].distance <= w[1].distance));
    }

    /// REV-4: with un-normalised embeddings, distances are large; recall must
    /// still return the nearest nodes rather than an empty result.
    #[test]
    fn test_recall_has_no_distance_cutoff() {
        let conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&conn, DIM).unwrap();
        let repo = Arc::new(SqliteLtmRepository::new(conn));
        let far = repo
            .create_node(
                &TreeNode::new("far", "far", TreeNodeKind::Spine),
                Some(&[40.0, 30.0, 0.0, 0.0]),
            )
            .unwrap();
        let ret = LtmRetrieval::new(repo as Arc<dyn LtmRepository>);

        let hits = ret.recall(&[0.0, 0.0, 3.0, 4.0], 5).unwrap();
        assert_eq!(hits.len(), 1, "nearest node returned despite distance ~50");
        assert_eq!(hits[0].entry.id, far);
        assert!(hits[0].distance > 10.0);
    }

    /// ARC-15: a concept hit carries a bounded number of document leaves.
    #[test]
    fn test_recall_bounds_child_leaves() {
        let conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&conn, DIM).unwrap();
        let repo = Arc::new(SqliteLtmRepository::new(conn));
        let concept = repo
            .create_node(
                &TreeNode::new("bills", "bills", TreeNodeKind::Spine),
                Some(&[1.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        for i in 0..(RECALL_MAX_LEAVES + 15) {
            leaf_under(&repo, concept, &format!("doc_{i}"), None);
        }
        let ret = LtmRetrieval::new(repo as Arc<dyn LtmRepository>);

        let hit = ret.recall(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].leaves.len(), RECALL_MAX_LEAVES);
        assert!(hit[0].children.len() <= RECALL_MAX_CHILDREN);
    }
}
