//! Long-Term Memory (LTM) domain — the permanent knowledge tree.
//!
//! LTM is a poly-hierarchy (DAG) of concept nodes with documents attached as
//! leaves by `dataId` reference. It never decays. The actual content stays in
//! the source system; LTM holds *meaning* (rolling summaries + embeddings) and the
//! `dataId` pointer only.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// What a tree node represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TreeNodeKind {
    /// Curated backbone — the user's main branches, seeded once.
    Spine,
    /// AI-created concept node grown under the spine.
    Grown,
    /// Holding area for documents that found no good concept match.
    Inbox,
    /// A document, attached by `dataId` (see [`Leaf`]).
    Leaf,
}

impl TreeNodeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            TreeNodeKind::Spine => "spine",
            TreeNodeKind::Grown => "grown",
            TreeNodeKind::Inbox => "inbox",
            TreeNodeKind::Leaf => "leaf",
        }
    }

    pub fn from_db_str(s: &str) -> Result<Self> {
        Ok(match s {
            "spine" => TreeNodeKind::Spine,
            "grown" => TreeNodeKind::Grown,
            "inbox" => TreeNodeKind::Inbox,
            "leaf" => TreeNodeKind::Leaf,
            other => anyhow::bail!("unknown tree_node kind: {other}"),
        })
    }
}

/// A concept (or leaf) node in the knowledge tree.
///
/// Concept nodes carry a rolling `summary` (indexed in `fts_ltm`) and an
/// embedding of that summary (in `vec_ltm`). `permanent` is always true — the
/// decay path must never touch LTM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeNode {
    pub id: Option<i64>,
    pub name: String,
    pub summary: String,
    pub kind: TreeNodeKind,
    pub permanent: bool,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
}

impl TreeNode {
    /// A new permanent node with no DB-assigned id/timestamps yet.
    pub fn new(name: impl Into<String>, summary: impl Into<String>, kind: TreeNodeKind) -> Self {
        Self {
            id: None,
            name: name.into(),
            summary: summary.into(),
            kind,
            permanent: true,
            created_at: None,
            updated_at: None,
        }
    }
}

/// A directed parent → child edge. A child may have several parents (DAG), so
/// `(parent_id, child_id)` together identify the edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeEdge {
    pub parent_id: i64,
    pub child_id: i64,
    pub weight: f64,
}

impl TreeEdge {
    pub fn new(parent_id: i64, child_id: i64) -> Self {
        Self {
            parent_id,
            child_id,
            weight: 1.0,
        }
    }
}

/// Where a leaf came from: source, ingest time, and the placement confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub source: String,
    pub ingested_at: Option<String>,
    pub confidence: f64,
}

/// A document attached to a leaf tree node by `dataId` reference. The bytes
/// live in the source system; LTM stores only the pointer + provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Leaf {
    pub tree_node_id: i64,
    pub data_id: String,
    pub provenance: Provenance,
}

/// One curated spine branch to seed, from config `[ltm.spine]`. `path` is
/// `/`-separated and relative to the root (e.g. `"work/projects"`); missing
/// intermediate branches are created with an empty description. The inbox is
/// always created and need not be listed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpineSeed {
    pub path: String,
    #[serde(default)]
    pub description: String,
}

impl SpineSeed {
    pub fn new(path: &str, description: &str) -> Self {
        Self {
            path: path.to_string(),
            description: description.to_string(),
        }
    }

    /// The path's branch names, ignoring empty segments and surrounding space.
    pub fn segments(&self) -> Vec<&str> {
        self.path
            .split('/')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect()
    }
}

/// The generic default spine: `root` → `inbox` (implicit), `notes`,
/// `documents`. Deliberately minimal — the tree grows from use, and users
/// shape it via `[ltm.spine]`.
pub fn default_spine() -> Vec<SpineSeed> {
    vec![
        SpineSeed::new("notes", "Notes, ideas, decisions, and facts worth keeping."),
        SpineSeed::new(
            "documents",
            "Documents, references, and longer texts filed for later recall.",
        ),
    ]
}

/// Persistence port for the LTM knowledge tree. Synchronous, like
/// [`crate::domain::ports::MemoryRepository`]; the SQLite impl serializes
/// access internally.
pub trait LtmRepository {
    /// Insert a tree node; if `embedding` is given it is stored in `vec_ltm`
    /// (must match the LTM vector dimension). Returns the new node id.
    fn create_node(&self, node: &TreeNode, embedding: Option<&[f32]>) -> Result<i64>;

    /// Add a parent → child edge (idempotent on the pair).
    fn add_edge(&self, edge: &TreeEdge) -> Result<()>;

    /// Attach a document leaf (`tree_node_id` → `data_id` + provenance).
    fn create_leaf(&self, leaf: &Leaf) -> Result<()>;

    /// Fetch a node by id.
    fn get_node(&self, id: i64) -> Result<Option<TreeNode>>;

    /// Direct children of a node.
    fn get_children(&self, parent_id: i64) -> Result<Vec<TreeNode>>;

    /// Direct parents of a node (may be several — DAG).
    fn get_parents(&self, child_id: i64) -> Result<Vec<TreeNode>>;

    /// The leaf node carrying a given `data_id`, if any.
    fn get_node_by_data_id(&self, data_id: &str) -> Result<Option<TreeNode>>;

    /// **Every** leaf node carrying `data_id`, oldest first. Normally at most
    /// one; more only after an interrupted replace, which callers then heal.
    fn get_nodes_by_data_id(&self, data_id: &str) -> Result<Vec<TreeNode>>;

    /// Concept nodes (`spine`/`grown`) whose summary embedding is within
    /// `max_distance` of `embedding`, nearest first. Used by placement to find
    /// the best home for a new document. Leaves and the inbox are never match
    /// targets. Returns each match with its vector distance.
    fn find_similar_concepts(
        &self,
        embedding: &[f32],
        max_distance: f64,
        limit: usize,
    ) -> Result<Vec<(TreeNode, f64)>>;

    /// Nearest embedded nodes across concepts **and document leaves** (unlike
    /// [`find_similar_concepts`](Self::find_similar_concepts), which is
    /// concept-only for placement). Used by recall so a document is reachable by
    /// meaning even while it sits under the inbox (before the tree grows real
    /// concepts around it).
    fn find_similar_any(
        &self,
        embedding: &[f32],
        max_distance: f64,
        limit: usize,
    ) -> Result<Vec<(TreeNode, f64)>>;

    /// The document leaf attached to a given tree node, if that node is a leaf.
    fn get_leaf(&self, tree_node_id: i64) -> Result<Option<Leaf>>;

    /// The inbox holding node (the placement fallback), if seeded.
    fn get_inbox(&self) -> Result<Option<TreeNode>>;

    /// Replace a node's rolling summary (and bump `updated_at`).
    fn update_summary(&self, node_id: i64, summary: &str) -> Result<()>;

    /// Set (or replace) a concept node's placement embedding in `vec_ltm`.
    /// Concept vectors are the stable anchors placement matches against; they are
    /// derived from the concept's curated identity, **not** its rolling summary,
    /// so filing does not drift as documents accumulate.
    fn set_concept_embedding(&self, node_id: i64, embedding: &[f32]) -> Result<()>;

    /// Concept nodes (`spine`/`grown`) that have no `vec_ltm` embedding yet — the
    /// ones placement is blind to. The daemon embeds these at startup so the
    /// spine is a live set of match targets (fixes the "everything falls to the
    /// inbox" bug: an un-embedded spine can never match).
    fn concepts_missing_embedding(&self) -> Result<Vec<TreeNode>>;

    /// Placement calibration: for up to `sample` already-embedded document
    /// leaves, the distance to their **nearest concept** (ignoring any placement
    /// threshold). Reveals the real distance scale so the placement threshold can
    /// be tuned to actual data instead of guessed. Read-only; no LLM calls.
    fn placement_calibration(&self, sample: usize) -> Result<Vec<PlacementProbe>>;

    /// Delete a node and everything attached to it: its edges (either
    /// direction), its vector, and its leaf rows. Used by tombstone/forget.
    fn delete_node(&self, node_id: i64) -> Result<()>;

    /// Remove a single parent → child edge (no-op if absent). Used by the inbox
    /// gardener to re-home a leaf from the inbox to a concept.
    fn remove_edge(&self, parent_id: i64, child_id: i64) -> Result<()>;

    /// Document leaves currently under the inbox, each with its **stored**
    /// embedding — so the gardener can re-run placement on them without
    /// re-embedding (no LLM). Used to re-home the inbox after a threshold change
    /// or a new spine branch, avoiding a full replay.
    fn inbox_leaf_embeddings(&self) -> Result<Vec<(i64, Vec<f32>)>>;

    /// Ensure `root`, the inbox, and every branch in `seeds` exist, creating
    /// only the missing ones (additive & idempotent — existing branches keep
    /// their leaves and summaries). Re-run after a hard reset.
    fn seed_spine_from(&self, seeds: &[SpineSeed]) -> Result<()>;

    /// [`seed_spine_from`](Self::seed_spine_from) with [`default_spine`].
    fn seed_spine(&self) -> Result<()> {
        self.seed_spine_from(&default_spine())
    }

    /// Wipe the entire tree (nodes, edges, leaves, vectors), keeping the schema.
    /// Used by hard reset; the caller re-seeds the spine afterward.
    fn reset_store(&self) -> Result<()>;

    /// Top-level nodes — those with no parent edge. The map starts here.
    fn get_roots(&self) -> Result<Vec<TreeNode>>;

    /// The document leaves attached directly under a node (its leaf children),
    /// each with `data_id` + provenance. Used by drill/recall.
    fn get_child_leaves(&self, parent_id: i64) -> Result<Vec<Leaf>>;

    /// CT-scan statistics for the LTM tree (slice 9).
    fn ltm_stats(&self) -> Result<LtmStats>;
}

/// One placement-calibration reading: a document leaf, the nearest concept, and
/// the raw vector distance between them (the number placement's threshold gates).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlacementProbe {
    pub leaf_title: String,
    pub nearest_concept: String,
    pub distance: f64,
}

/// A snapshot of LTM tree health for the metrics CT scan.
#[derive(Debug, Clone, PartialEq)]
pub struct LtmStats {
    pub tree_nodes: i64,
    /// Document leaf nodes (kind = leaf).
    pub leaves: i64,
    pub edges: i64,
    /// Documents sitting in the inbox (unsorted — a placement-quality signal).
    pub inbox_docs: i64,
    /// Leaf nodes with no parent edge (disconnected — a health signal; normally 0).
    pub orphan_leaves: i64,
    /// Longest root-to-node path (tree height).
    pub max_depth: i64,
    pub db_size_bytes: i64,
}
