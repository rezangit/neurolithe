//! SQLite implementation of the LTM knowledge-tree repository.
//!
//! Mirrors `SqliteMemoryRepository`: owns one `rusqlite::Connection` (to the
//! LTM database) and serializes access. Concept-node embeddings go into the
//! `vec_ltm` sqlite-vec index; summaries are kept in `fts_ltm` via triggers.

use crate::domain::ltm::{
    Leaf, LtmRepository, Provenance, SpineSeed, TreeEdge, TreeNode, TreeNodeKind,
};
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};

pub struct SqliteLtmRepository {
    conn: Connection,
    /// Keeps the workspace marked "open" (its shared advisory lock) for
    /// exactly as long as this connection lives — i.e. as long as anything
    /// still holds the repository.
    _lease: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
}

impl SqliteLtmRepository {
    pub fn new(conn: Connection) -> Self {
        Self { conn, _lease: None }
    }

    /// Tie a workspace lease to this repository's connection lifetime.
    pub fn with_lease(mut self, lease: std::sync::Arc<dyn std::any::Any + Send + Sync>) -> Self {
        self._lease = Some(lease);
        self
    }

    /// The id of the first node with this exact `name` (used by additive spine
    /// seeding to avoid duplicating a branch/root that already exists).
    fn find_node_by_name(&self, name: &str) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id FROM tree_nodes WHERE name = ?1 ORDER BY id LIMIT 1",
                params![name],
                |r| r.get::<_, i64>(0),
            )
            .optional()?)
    }

    /// The concept (non-leaf) child of `parent_id` named `name`, if any.
    fn child_by_name(&self, parent_id: i64, name: &str) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row(
                "SELECT n.id FROM tree_edges e JOIN tree_nodes n ON n.id = e.child_id
                 WHERE e.parent_id = ?1 AND n.name = ?2 AND n.kind IN ('spine', 'grown')
                 ORDER BY n.id LIMIT 1",
                params![parent_id, name],
                |r| r.get::<_, i64>(0),
            )
            .optional()?)
    }

    /// Map a `tree_nodes` row (id, name, summary, kind, permanent, created_at,
    /// updated_at) to a [`TreeNode`].
    fn row_to_node(row: &rusqlite::Row) -> rusqlite::Result<TreeNode> {
        let kind_str: String = row.get(3)?;
        let kind = TreeNodeKind::from_db_str(&kind_str).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, e.into())
        })?;
        Ok(TreeNode {
            id: Some(row.get(0)?),
            name: row.get(1)?,
            summary: row.get(2)?,
            kind,
            permanent: row.get(4)?,
            created_at: row.get(5)?,
            updated_at: row.get(6)?,
        })
    }
}

impl LtmRepository for SqliteLtmRepository {
    /// Seeds `root`, the inbox, and each configured branch by *path*, creating
    /// only what is missing. Branch summaries are the curated definition of the
    /// concept; startup embeds them (`concepts_missing_embedding`) so placement
    /// has real match targets. Nested paths (`a/b`) create `a` first; a branch
    /// is identified by name **under its parent**, so the same name may appear
    /// in different subtrees.
    fn seed_spine_from(&self, seeds: &[SpineSeed]) -> Result<()> {
        let root = match self.find_node_by_name("root")? {
            Some(id) => id,
            None => self.create_node(
                &TreeNode::new("root", "Root of the knowledge tree.", TreeNodeKind::Spine),
                None,
            )?,
        };

        for seed in seeds {
            let segments = seed.segments();
            let mut parent = root;
            for (i, name) in segments.iter().enumerate() {
                let is_last = i + 1 == segments.len();
                parent = match self.child_by_name(parent, name)? {
                    Some(id) => id,
                    None => {
                        let summary = if is_last {
                            seed.description.as_str()
                        } else {
                            ""
                        };
                        let child = self.create_node(
                            &TreeNode::new(*name, summary, TreeNodeKind::Spine),
                            None,
                        )?;
                        self.add_edge(&TreeEdge::new(parent, child))?;
                        child
                    }
                };
            }
        }

        // The holding area for documents that find no good concept match.
        if self.get_inbox()?.is_none() {
            let inbox = self.create_node(
                &TreeNode::new(
                    "inbox",
                    "Unsorted documents awaiting placement.",
                    TreeNodeKind::Inbox,
                ),
                None,
            )?;
            self.add_edge(&TreeEdge::new(root, inbox))?;
        }

        Ok(())
    }

    fn reset_store(&self) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        // FK-safe order; deleting tree_nodes fires the fts_ltm delete trigger.
        tx.execute("DELETE FROM tree_edges", [])?;
        tx.execute("DELETE FROM vec_ltm", [])?;
        tx.execute("DELETE FROM vec_leaves", [])?;
        tx.execute("DELETE FROM leaves", [])?;
        tx.execute("DELETE FROM tree_nodes", [])?;
        tx.commit()?;
        Ok(())
    }

    fn create_node(&self, node: &TreeNode, embedding: Option<&[f32]>) -> Result<i64> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO tree_nodes (name, summary, kind, permanent) VALUES (?1, ?2, ?3, ?4)",
            params![node.name, node.summary, node.kind.as_str(), node.permanent],
        )?;
        let node_id = tx.last_insert_rowid();

        if let Some(embedding) = embedding {
            let embedding_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    embedding.as_ptr() as *const u8,
                    std::mem::size_of_val(embedding),
                )
            };
            // Concepts and document leaves live in separate vector indexes (see
            // schema): placement searches concepts only, recall searches both.
            let table = if node.kind == TreeNodeKind::Leaf {
                "vec_leaves"
            } else {
                "vec_ltm"
            };
            tx.execute(
                &format!("INSERT INTO {table}(node_id, embedding) VALUES (?1, ?2)"),
                params![node_id, embedding_bytes],
            )?;
        }

        tx.commit()?;
        Ok(node_id)
    }

    fn add_edge(&self, edge: &TreeEdge) -> Result<()> {
        // INSERT OR IGNORE: re-adding the same parent->child pair is a no-op.
        self.conn.execute(
            "INSERT OR IGNORE INTO tree_edges (parent_id, child_id, weight) VALUES (?1, ?2, ?3)",
            params![edge.parent_id, edge.child_id, edge.weight],
        )?;
        Ok(())
    }

    fn create_leaf(&self, leaf: &Leaf) -> Result<()> {
        let provenance_json = serde_json::to_string(&leaf.provenance)?;
        self.conn.execute(
            "INSERT OR REPLACE INTO leaves (tree_node_id, data_id, provenance) VALUES (?1, ?2, ?3)",
            params![leaf.tree_node_id, leaf.data_id, provenance_json],
        )?;
        Ok(())
    }

    fn get_node(&self, id: i64) -> Result<Option<TreeNode>> {
        let node = self
            .conn
            .query_row(
                "SELECT id, name, summary, kind, permanent, created_at, updated_at
                 FROM tree_nodes WHERE id = ?1",
                params![id],
                Self::row_to_node,
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;
        Ok(node)
    }

    fn get_children(&self, parent_id: i64) -> Result<Vec<TreeNode>> {
        let mut stmt = self.conn.prepare(
            "SELECT n.id, n.name, n.summary, n.kind, n.permanent, n.created_at, n.updated_at
             FROM tree_nodes n
             JOIN tree_edges e ON n.id = e.child_id
             WHERE e.parent_id = ?1
             ORDER BY n.id",
        )?;
        let rows = stmt.query_map(params![parent_id], Self::row_to_node)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn get_parents(&self, child_id: i64) -> Result<Vec<TreeNode>> {
        let mut stmt = self.conn.prepare(
            "SELECT n.id, n.name, n.summary, n.kind, n.permanent, n.created_at, n.updated_at
             FROM tree_nodes n
             JOIN tree_edges e ON n.id = e.parent_id
             WHERE e.child_id = ?1
             ORDER BY n.id",
        )?;
        let rows = stmt.query_map(params![child_id], Self::row_to_node)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn get_node_by_data_id(&self, data_id: &str) -> Result<Option<TreeNode>> {
        let node = self
            .conn
            .query_row(
                "SELECT n.id, n.name, n.summary, n.kind, n.permanent, n.created_at, n.updated_at
                 FROM tree_nodes n
                 JOIN leaves l ON n.id = l.tree_node_id
                 WHERE l.data_id = ?1
                 LIMIT 1",
                params![data_id],
                Self::row_to_node,
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;
        Ok(node)
    }

    fn get_nodes_by_data_id(&self, data_id: &str) -> Result<Vec<TreeNode>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT n.id, n.name, n.summary, n.kind, n.permanent, n.created_at, n.updated_at
             FROM tree_nodes n
             JOIN leaves l ON n.id = l.tree_node_id
             WHERE l.data_id = ?1
             ORDER BY n.id",
        )?;
        let nodes = stmt
            .query_map(params![data_id], Self::row_to_node)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(nodes)
    }

    fn find_similar_concepts(
        &self,
        embedding: &[f32],
        max_distance: f64,
        limit: usize,
    ) -> Result<Vec<(TreeNode, f64)>> {
        let embedding_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                embedding.as_ptr() as *const u8,
                std::mem::size_of_val(embedding),
            )
        };

        // Node columns first (mapped by row_to_node), then the vector distance.
        let mut stmt = self.conn.prepare(
            "SELECT n.id, n.name, n.summary, n.kind, n.permanent, n.created_at, n.updated_at,
                    v.distance
             FROM vec_ltm v
             JOIN tree_nodes n ON n.id = v.node_id
             WHERE v.embedding MATCH ?1 AND k = ?2
               AND n.kind IN ('spine', 'grown')
               AND v.distance <= ?3
             ORDER BY v.distance ASC",
        )?;

        let rows = stmt.query_map(
            params![embedding_bytes, limit as i64, max_distance],
            |row| {
                let node = Self::row_to_node(row)?;
                let distance: f64 = row.get(7)?;
                Ok((node, distance))
            },
        )?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn find_similar_any(
        &self,
        embedding: &[f32],
        max_distance: f64,
        limit: usize,
    ) -> Result<Vec<(TreeNode, f64)>> {
        let embedding_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                embedding.as_ptr() as *const u8,
                std::mem::size_of_val(embedding),
            )
        };

        // Search the concept index and the leaf index independently (they are
        // separate vec0 tables), then merge by distance. Each is queried for the
        // full `limit` so a nearer leaf and a nearer concept both get a fair shot.
        let query_index = |table: &str| -> rusqlite::Result<Vec<(TreeNode, f64)>> {
            let sql = format!(
                "SELECT n.id, n.name, n.summary, n.kind, n.permanent, n.created_at, n.updated_at,
                        v.distance
                 FROM {table} v
                 JOIN tree_nodes n ON n.id = v.node_id
                 WHERE v.embedding MATCH ?1 AND k = ?2
                   AND v.distance <= ?3
                 ORDER BY v.distance ASC"
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map(
                params![embedding_bytes, limit as i64, max_distance],
                |row| {
                    let node = Self::row_to_node(row)?;
                    let distance: f64 = row.get(7)?;
                    Ok((node, distance))
                },
            )?;
            rows.collect()
        };

        let mut merged = query_index("vec_ltm")?;
        merged.extend(query_index("vec_leaves")?);
        merged.sort_by(|a, b| a.1.total_cmp(&b.1));
        merged.truncate(limit);
        Ok(merged)
    }

    fn get_leaf(&self, tree_node_id: i64) -> Result<Option<Leaf>> {
        let mut stmt = self.conn.prepare(
            "SELECT tree_node_id, data_id, provenance FROM leaves WHERE tree_node_id = ?1",
        )?;
        let mut rows = stmt.query(params![tree_node_id])?;
        if let Some(row) = rows.next()? {
            let tree_node_id: i64 = row.get(0)?;
            let data_id: String = row.get(1)?;
            let provenance_json: String = row.get(2)?;
            let provenance: Provenance = serde_json::from_str(&provenance_json)?;
            Ok(Some(Leaf {
                tree_node_id,
                data_id,
                provenance,
            }))
        } else {
            Ok(None)
        }
    }

    fn get_inbox(&self) -> Result<Option<TreeNode>> {
        let node = self
            .conn
            .query_row(
                "SELECT id, name, summary, kind, permanent, created_at, updated_at
                 FROM tree_nodes WHERE kind = 'inbox' ORDER BY id LIMIT 1",
                [],
                Self::row_to_node,
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;
        Ok(node)
    }

    fn update_summary(&self, node_id: i64, summary: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE tree_nodes SET summary = ?1, updated_at = CURRENT_TIMESTAMP WHERE id = ?2",
            params![summary, node_id],
        )?;
        Ok(())
    }

    fn set_concept_embedding(&self, node_id: i64, embedding: &[f32]) -> Result<()> {
        let embedding_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                embedding.as_ptr() as *const u8,
                std::mem::size_of_val(embedding),
            )
        };
        // vec0 rows are replaced by delete-then-insert on the primary key.
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM vec_ltm WHERE node_id = ?1", params![node_id])?;
        tx.execute(
            "INSERT INTO vec_ltm(node_id, embedding) VALUES (?1, ?2)",
            params![node_id, embedding_bytes],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn remove_edge(&self, parent_id: i64, child_id: i64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM tree_edges WHERE parent_id = ?1 AND child_id = ?2",
            params![parent_id, child_id],
        )?;
        Ok(())
    }

    fn inbox_leaf_embeddings(&self) -> Result<Vec<(i64, Vec<f32>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT vl.node_id, vl.embedding
             FROM vec_leaves vl
             JOIN tree_edges e ON e.child_id = vl.node_id
             JOIN tree_nodes p ON p.id = e.parent_id
             WHERE p.kind = 'inbox'",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, blob) = row?;
            // vec0 stores float32 little-endian; decode back to Vec<f32>.
            let emb: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            out.push((id, emb));
        }
        Ok(out)
    }

    fn placement_calibration(
        &self,
        sample: usize,
    ) -> Result<Vec<crate::domain::ltm::PlacementProbe>> {
        // Sample embedded leaves (node_id + raw embedding blob).
        let leaves: Vec<(i64, Vec<u8>)> = {
            let mut stmt = self
                .conn
                .prepare("SELECT node_id, embedding FROM vec_leaves LIMIT ?1")?;
            stmt.query_map(params![sample as i64], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
        };

        let mut probes = Vec::new();
        for (leaf_id, embedding) in leaves {
            let leaf_title: String = self
                .conn
                .query_row(
                    "SELECT name FROM tree_nodes WHERE id = ?1",
                    params![leaf_id],
                    |r| r.get(0),
                )
                .optional()?
                .unwrap_or_default();
            // Nearest concept to this leaf's embedding, threshold-free.
            let nearest = self
                .conn
                .query_row(
                    "SELECT n.name, v.distance
                     FROM vec_ltm v JOIN tree_nodes n ON n.id = v.node_id
                     WHERE v.embedding MATCH ?1 AND k = 1",
                    params![embedding],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)),
                )
                .optional()?;
            if let Some((nearest_concept, distance)) = nearest {
                probes.push(crate::domain::ltm::PlacementProbe {
                    leaf_title,
                    nearest_concept,
                    distance,
                });
            }
        }
        Ok(probes)
    }

    fn concepts_missing_embedding(&self) -> Result<Vec<TreeNode>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, summary, kind, permanent, created_at, updated_at
             FROM tree_nodes
             WHERE kind IN ('spine', 'grown')
               AND id NOT IN (SELECT node_id FROM vec_ltm)
             ORDER BY id",
        )?;
        let rows = stmt.query_map([], Self::row_to_node)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn delete_node(&self, node_id: i64) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        // Edges first (FK to tree_nodes), then vector + leaves, then the node
        // itself (fires the fts_ltm delete trigger).
        tx.execute(
            "DELETE FROM tree_edges WHERE parent_id = ?1 OR child_id = ?1",
            params![node_id],
        )?;
        tx.execute("DELETE FROM vec_ltm WHERE node_id = ?1", params![node_id])?;
        tx.execute(
            "DELETE FROM vec_leaves WHERE node_id = ?1",
            params![node_id],
        )?;
        tx.execute(
            "DELETE FROM leaves WHERE tree_node_id = ?1",
            params![node_id],
        )?;
        tx.execute("DELETE FROM tree_nodes WHERE id = ?1", params![node_id])?;
        tx.commit()?;
        Ok(())
    }

    fn get_roots(&self) -> Result<Vec<TreeNode>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, summary, kind, permanent, created_at, updated_at
             FROM tree_nodes n
             WHERE NOT EXISTS (SELECT 1 FROM tree_edges e WHERE e.child_id = n.id)
             ORDER BY n.id",
        )?;
        let rows = stmt.query_map([], Self::row_to_node)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn get_child_leaves(&self, parent_id: i64) -> Result<Vec<Leaf>> {
        let mut stmt = self.conn.prepare(
            "SELECT l.tree_node_id, l.data_id, l.provenance
             FROM leaves l
             JOIN tree_edges e ON l.tree_node_id = e.child_id
             WHERE e.parent_id = ?1
             ORDER BY l.tree_node_id",
        )?;
        let rows = stmt.query_map(params![parent_id], |row| {
            let tree_node_id: i64 = row.get(0)?;
            let data_id: String = row.get(1)?;
            let provenance_json: String = row.get(2)?;
            Ok((tree_node_id, data_id, provenance_json))
        })?;

        let mut leaves = Vec::new();
        for row in rows {
            let (tree_node_id, data_id, provenance_json) = row?;
            let provenance: Provenance = serde_json::from_str(&provenance_json)?;
            leaves.push(Leaf {
                tree_node_id,
                data_id,
                provenance,
            });
        }
        Ok(leaves)
    }

    fn ltm_stats(&self) -> Result<crate::domain::ltm::LtmStats> {
        let count =
            |sql: &str| -> rusqlite::Result<i64> { self.conn.query_row(sql, [], |r| r.get(0)) };

        let tree_nodes = count("SELECT COUNT(*) FROM tree_nodes")?;
        let leaves = count("SELECT COUNT(*) FROM tree_nodes WHERE kind = 'leaf'")?;
        let edges = count("SELECT COUNT(*) FROM tree_edges")?;
        let inbox_docs = count(
            "SELECT COUNT(*) FROM tree_edges e
             JOIN tree_nodes p ON e.parent_id = p.id
             JOIN tree_nodes c ON e.child_id = c.id
             WHERE p.kind = 'inbox' AND c.kind = 'leaf'",
        )?;
        let orphan_leaves = count(
            "SELECT COUNT(*) FROM tree_nodes n
             WHERE n.kind = 'leaf'
               AND NOT EXISTS (SELECT 1 FROM tree_edges e WHERE e.child_id = n.id)",
        )?;

        // Longest root-to-node path. Roots are parentless; descend the DAG.
        let max_depth: i64 = self.conn.query_row(
            "WITH RECURSIVE d(id, depth) AS (
                 SELECT id, 0 FROM tree_nodes n
                 WHERE NOT EXISTS (SELECT 1 FROM tree_edges e WHERE e.child_id = n.id)
                 UNION ALL
                 SELECT e.child_id, d.depth + 1 FROM d JOIN tree_edges e ON e.parent_id = d.id
             )
             SELECT COALESCE(MAX(depth), 0) FROM d",
            [],
            |r| r.get(0),
        )?;

        Ok(crate::domain::ltm::LtmStats {
            tree_nodes,
            leaves,
            edges,
            inbox_docs,
            orphan_leaves,
            max_depth,
            db_size_bytes: crate::infrastructure::database::db_size_bytes(&self.conn)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ltm::Provenance;
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::schema::init_ltm_schema;

    const DIM: usize = 4;

    fn setup() -> SqliteLtmRepository {
        let conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&conn, DIM).unwrap();
        SqliteLtmRepository::new(conn)
    }

    /// A node can sit under several parents (DAG), and children/parents read
    /// back correctly from either direction.
    #[test]
    fn test_node_with_two_parents() {
        let repo = setup();

        let p1 = repo
            .create_node(&TreeNode::new("job", "", TreeNodeKind::Spine), None)
            .unwrap();
        let p2 = repo
            .create_node(
                &TreeNode::new("self-improvement", "", TreeNodeKind::Spine),
                None,
            )
            .unwrap();
        let child = repo
            .create_node(&TreeNode::new("leadership", "", TreeNodeKind::Grown), None)
            .unwrap();

        repo.add_edge(&TreeEdge::new(p1, child)).unwrap();
        repo.add_edge(&TreeEdge::new(p2, child)).unwrap();
        // Re-adding an existing edge is a no-op.
        repo.add_edge(&TreeEdge::new(p1, child)).unwrap();

        let parents = repo.get_parents(child).unwrap();
        assert_eq!(parents.len(), 2, "child has two parents");
        let parent_names: Vec<_> = parents.iter().map(|n| n.name.as_str()).collect();
        assert!(parent_names.contains(&"job"));
        assert!(parent_names.contains(&"self-improvement"));

        let p1_children = repo.get_children(p1).unwrap();
        assert_eq!(p1_children.len(), 1);
        assert_eq!(p1_children[0].name, "leadership");
        assert_eq!(p1_children[0].kind, TreeNodeKind::Grown);
    }

    /// A leaf links a tree node to a `dataId`, found via get-by-dataId with its
    /// provenance round-tripped.
    #[test]
    fn test_leaf_links_to_data_id() {
        let repo = setup();

        let leaf_node = repo
            .create_node(
                &TreeNode::new(
                    "2019 insurance letter",
                    "health insurance",
                    TreeNodeKind::Leaf,
                ),
                Some(&[0.1_f32; DIM]),
            )
            .unwrap();

        repo.create_leaf(&Leaf {
            tree_node_id: leaf_node,
            data_id: "doc_01HX".into(),
            provenance: Provenance {
                source: "scanner".into(),
                ingested_at: Some("2026-06-28T00:00:00Z".into()),
                confidence: 0.9,
            },
        })
        .unwrap();

        let found = repo.get_node_by_data_id("doc_01HX").unwrap();
        assert!(found.is_some(), "leaf is found by dataId");
        let found = found.unwrap();
        assert_eq!(found.id, Some(leaf_node));
        assert_eq!(found.kind, TreeNodeKind::Leaf);

        // An unknown dataId yields nothing.
        assert!(repo.get_node_by_data_id("doc_missing").unwrap().is_none());
    }

    /// `init_ltm_schema` is safe to run repeatedly (idempotent migrations).
    #[test]
    fn test_schema_idempotent() {
        let conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&conn, DIM).unwrap();
        // A second run must not error (CREATE ... IF NOT EXISTS / triggers).
        init_ltm_schema(&conn, DIM).unwrap();

        let tables: Vec<String> = {
            let mut stmt = conn
                .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                .unwrap();
            stmt.query_map([], |r| r.get(0))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap()
        };
        for t in [
            "tree_nodes",
            "tree_edges",
            "leaves",
            "vec_ltm",
            "vec_leaves",
            "fts_ltm",
        ] {
            assert!(tables.contains(&t.to_string()), "{t} should exist");
        }
    }

    /// The default spine is generic: root → notes, documents, inbox — no
    /// personal-life branches. Re-seeding creates no duplicates.
    #[test]
    fn test_spine_seeded_once() {
        let repo = setup();

        repo.seed_spine().unwrap();
        let count = |repo: &SqliteLtmRepository| -> i64 {
            repo.conn
                .query_row("SELECT COUNT(*) FROM tree_nodes", [], |r| r.get(0))
                .unwrap()
        };
        // root + notes + documents + inbox.
        assert_eq!(count(&repo), 4);

        repo.seed_spine().unwrap();
        assert_eq!(count(&repo), 4, "re-seed creates no duplicates");

        let root = repo
            .find_node_by_name("root")
            .unwrap()
            .expect("root exists");
        let mut names: Vec<String> = repo
            .get_children(root)
            .unwrap()
            .into_iter()
            .map(|n| n.name)
            .collect();
        names.sort();
        assert_eq!(names, vec!["documents", "inbox", "notes"]);
        for legacy in ["job", "health", "family", "investment"] {
            assert!(
                repo.find_node_by_name(legacy).unwrap().is_none(),
                "{legacy}"
            );
        }
    }

    /// A configured spine seeds nested paths (creating intermediate branches),
    /// keeps descriptions on the leaf segment, and is additive: an older tree
    /// keeps its branches and reuses its root and inbox.
    #[test]
    fn test_seed_spine_from_config_is_nested_and_additive() {
        let repo = setup();
        let root = repo
            .create_node(&TreeNode::new("root", "root", TreeNodeKind::Spine), None)
            .unwrap();
        let job = repo
            .create_node(&TreeNode::new("job", "work", TreeNodeKind::Spine), None)
            .unwrap();
        repo.add_edge(&TreeEdge::new(root, job)).unwrap();
        let inbox = repo
            .create_node(&TreeNode::new("inbox", "", TreeNodeKind::Inbox), None)
            .unwrap();
        repo.add_edge(&TreeEdge::new(root, inbox)).unwrap();

        let seeds = vec![
            SpineSeed::new("work/projects", "Active projects."),
            SpineSeed::new(" work / clients ", "Client accounts."),
            SpineSeed::new("reading", "Books and papers."),
        ];
        repo.seed_spine_from(&seeds).unwrap();
        repo.seed_spine_from(&seeds).unwrap(); // idempotent

        assert_eq!(repo.find_node_by_name("root").unwrap(), Some(root));
        let mut top: Vec<String> = repo
            .get_children(root)
            .unwrap()
            .into_iter()
            .map(|n| n.name)
            .collect();
        top.sort();
        assert_eq!(top, vec!["inbox", "job", "reading", "work"]);

        let work = repo.find_node_by_name("work").unwrap().unwrap();
        let mut sub: Vec<(String, String)> = repo
            .get_children(work)
            .unwrap()
            .into_iter()
            .map(|n| (n.name, n.summary))
            .collect();
        sub.sort();
        assert_eq!(
            sub,
            vec![
                ("clients".to_string(), "Client accounts.".to_string()),
                ("projects".to_string(), "Active projects.".to_string()),
            ]
        );
    }

    /// Concept embeddings can be set and are then no longer reported as missing;
    /// leaves are never treated as placement concepts.
    #[test]
    fn test_concept_embedding_set_and_missing_report() {
        let repo = setup();
        let job = repo
            .create_node(&TreeNode::new("job", "work", TreeNodeKind::Spine), None)
            .unwrap();
        let grown = repo
            .create_node(
                &TreeNode::new("metro", "project", TreeNodeKind::Grown),
                None,
            )
            .unwrap();
        // A leaf with an embedding must never appear in the concept-missing list.
        repo.create_node(
            &TreeNode::new("doc", "a document", TreeNodeKind::Leaf),
            Some(&[0.1_f32; DIM]),
        )
        .unwrap();

        let missing: Vec<_> = repo
            .concepts_missing_embedding()
            .unwrap()
            .into_iter()
            .map(|n| n.id.unwrap())
            .collect();
        assert!(missing.contains(&job) && missing.contains(&grown));
        assert_eq!(missing.len(), 2, "only the two un-embedded concepts");

        repo.set_concept_embedding(job, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        // Replacing is allowed (delete-then-insert on the vec0 key).
        repo.set_concept_embedding(job, &[0.0, 1.0, 0.0, 0.0])
            .unwrap();

        let still_missing = repo.concepts_missing_embedding().unwrap();
        assert_eq!(still_missing.len(), 1, "job now embedded");
        assert_eq!(still_missing[0].id, Some(grown));

        // The embedded concept is now a live placement target.
        let hit = repo
            .find_similar_concepts(&[0.0, 1.0, 0.0, 0.0], 0.1, 1)
            .unwrap();
        assert_eq!(hit[0].0.id, Some(job));
    }
}
