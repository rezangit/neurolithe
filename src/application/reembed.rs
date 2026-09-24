//! `neurolithe reembed`: re-embed a workspace with the current embedder
//! (Phase 2 §4).
//!
//! Needed whenever the embedding model or dimension changes — a store's vectors
//! are only searchable with the model that produced them, so
//! [`crate::application::migrations::prepare_store`] refuses a mismatched store
//! and points here.
//!
//! What gets re-embedded (the same texts the write paths embed):
//! - STM: every active fact's text (`payload.fact`); graph anchors
//!   (`kind = subject`) and archived nodes stay unindexed.
//! - LTM concepts (`spine`/`grown`): `"name: summary"` (as spine embedding does).
//! - LTM document leaves: their stored summary (or name if empty).
//!
//! Both stores are backed up first (`VACUUM INTO`). Then the embeddings for
//! **both** stores are computed before anything is written, so an embedder
//! failure (down, wrong dimension) — on either store — leaves the whole
//! workspace exactly as it was. Only then is each store rewritten, back to back,
//! in its own transaction (vec tables dropped, rebuilt at the new dimension and
//! filled; `meta` updated). The two files cannot share one transaction (WAL
//! mode gives no atomic commit across attached databases), so the only
//! non-atomic window is a failure of the *second* write itself (e.g. disk
//! full); the pre-reembed backups cover that case.

use crate::application::migrations::{StoreKind, backup_store, migrate_store, record_embedding};
use crate::domain::ports::{EmbeddingIdentity, LlmClient};
use crate::infrastructure::database::init_db;
use crate::infrastructure::schema::drop_vec_tables;
use anyhow::{Context, Result, bail};
use rusqlite::{Connection, params};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// What a re-embed did.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct ReembedReport {
    pub stm_nodes: usize,
    pub ltm_concepts: usize,
    pub ltm_leaves: usize,
    /// Pre-reembed (and pre-migration) backups, one or more per store.
    pub backups: Vec<PathBuf>,
}

/// Re-embed both stores of a workspace with `embedder`, recording `identity`
/// in their `meta`. Opens the files itself (migrating them first if needed —
/// with the usual automatic backup) and backs each up before re-embedding.
pub async fn reembed_workspace(
    stm_path: &Path,
    ltm_path: &Path,
    embedder: Arc<dyn LlmClient>,
    identity: &EmbeddingIdentity,
) -> Result<ReembedReport> {
    let mut report = ReembedReport::default();

    let stm = open_for_reembed(stm_path, StoreKind::Stm, identity, &mut report)?;
    let ltm = open_for_reembed(ltm_path, StoreKind::Ltm, identity, &mut report)?;

    // Phase 1 — embed everything (no writes).
    let stm_vectors = embed_stm(&stm, embedder.as_ref(), identity)
        .await
        .with_context(|| format!("re-embedding STM store '{}'", stm_path.display()))?;
    let ltm_vectors = embed_ltm(&ltm, embedder.as_ref(), identity)
        .await
        .with_context(|| format!("re-embedding LTM store '{}'", ltm_path.display()))?;

    // Phase 2 — write both stores back to back.
    report.stm_nodes = write_stm(&stm, identity, &stm_vectors)
        .with_context(|| format!("writing STM store '{}'", stm_path.display()))?;
    let (concepts, leaves) = write_ltm(&ltm, identity, &ltm_vectors).with_context(|| {
        format!(
            "writing LTM store '{}' (STM was already re-embedded; restore it from {:?} if needed)",
            ltm_path.display(),
            report.backups
        )
    })?;
    report.ltm_concepts = concepts;
    report.ltm_leaves = leaves;
    Ok(report)
}

/// CLI entry point: re-embed the workspace in `workspace_dir`
/// (`stm.sqlite` + `ltm.sqlite`) with `embedder`, whose identity (model id +
/// dimension) is recorded in the stores' meta.
pub async fn reembed_workspace_dir(
    workspace_dir: &Path,
    embedder: Arc<dyn LlmClient>,
) -> Result<ReembedReport> {
    let identity = crate::domain::ports::embedding_identity(embedder.as_ref())
        .await
        .context("determining the embedder's model and dimension")?;
    reembed_workspace(
        &workspace_dir.join("stm.sqlite"),
        &workspace_dir.join("ltm.sqlite"),
        embedder,
        &identity,
    )
    .await
}

fn open_for_reembed(
    path: &Path,
    kind: StoreKind,
    identity: &EmbeddingIdentity,
    report: &mut ReembedReport,
) -> Result<Connection> {
    if !path.exists() {
        bail!(
            "{} store '{}' does not exist",
            kind.as_str().to_uppercase(),
            path.display()
        );
    }
    let conn = init_db(Some(&path))
        .with_context(|| format!("opening {} store '{}'", kind.as_str(), path.display()))?;
    let migrated = migrate_store(&conn, kind, Some(path), identity.dim)?;
    report.backups.extend(migrated.backup);
    report
        .backups
        .push(backup_store(&conn, path, "reembed").context("backing up before re-embed")?);
    Ok(conn)
}

/// Embed `texts` in order, checking each vector's dimension.
async fn embed_all(
    embedder: &dyn LlmClient,
    identity: &EmbeddingIdentity,
    items: Vec<(i64, String)>,
) -> Result<Vec<(i64, Vec<f32>)>> {
    let mut out = Vec::with_capacity(items.len());
    for (id, text) in items {
        let embedding = embedder.embed_text(&text).await?;
        if embedding.len() != identity.dim {
            bail!(
                "embedder returned {} dimensions, expected {} ({})",
                embedding.len(),
                identity.dim,
                identity.model
            );
        }
        out.push((id, embedding));
    }
    Ok(out)
}

fn to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn insert_vectors(conn: &Connection, table: &str, rows: &[(i64, Vec<f32>)]) -> Result<()> {
    let mut stmt = conn.prepare(&format!(
        "INSERT INTO {table} (node_id, embedding) VALUES (?1, ?2)"
    ))?;
    for (id, embedding) in rows {
        stmt.execute(params![id, to_bytes(embedding)])?;
    }
    Ok(())
}

fn query_texts(conn: &Connection, sql: &str) -> Result<Vec<(i64, String)>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows
        .into_iter()
        .filter(|(_, t)| !t.trim().is_empty())
        .collect())
}

/// Embeddings for an (already migrated) STM store: every active fact except
/// graph anchors. Reads only.
async fn embed_stm(
    conn: &Connection,
    embedder: &dyn LlmClient,
    identity: &EmbeddingIdentity,
) -> Result<Vec<(i64, Vec<f32>)>> {
    let texts = query_texts(
        conn,
        "SELECT id, COALESCE(json_extract(payload, '$.fact'), '') FROM nodes
         WHERE status = 'active'
           AND COALESCE(json_extract(payload, '$.kind'), '') != 'subject'
         ORDER BY id",
    )?;
    embed_all(embedder, identity, texts).await
}

/// Rebuild the STM vector index from `vectors` in one transaction. Returns how
/// many facts were indexed.
fn write_stm(
    conn: &Connection,
    identity: &EmbeddingIdentity,
    vectors: &[(i64, Vec<f32>)],
) -> Result<usize> {
    let tx = conn.unchecked_transaction()?;
    drop_vec_tables(&tx, StoreKind::Stm.vec_tables())?;
    StoreKind::Stm.create_vec_tables(&tx, identity.dim)?;
    insert_vectors(&tx, "vec_nodes", vectors)?;
    record_embedding(&tx, identity)?;
    tx.commit()?;
    Ok(vectors.len())
}

/// Concept and leaf embeddings for an (already migrated) LTM store. Reads only.
struct LtmVectors {
    concepts: Vec<(i64, Vec<f32>)>,
    leaves: Vec<(i64, Vec<f32>)>,
}

async fn embed_ltm(
    conn: &Connection,
    embedder: &dyn LlmClient,
    identity: &EmbeddingIdentity,
) -> Result<LtmVectors> {
    let concepts = query_texts(
        conn,
        "SELECT id, CASE WHEN trim(summary) = '' THEN name ELSE name || ': ' || summary END
         FROM tree_nodes WHERE kind IN ('spine', 'grown') ORDER BY id",
    )?;
    let leaves = query_texts(
        conn,
        "SELECT id, CASE WHEN trim(summary) = '' THEN name ELSE summary END
         FROM tree_nodes WHERE kind = 'leaf' ORDER BY id",
    )?;
    Ok(LtmVectors {
        concepts: embed_all(embedder, identity, concepts).await?,
        leaves: embed_all(embedder, identity, leaves).await?,
    })
}

/// Rebuild the LTM vector indexes in one transaction. Returns (concepts,
/// leaves) indexed.
fn write_ltm(
    conn: &Connection,
    identity: &EmbeddingIdentity,
    vectors: &LtmVectors,
) -> Result<(usize, usize)> {
    let tx = conn.unchecked_transaction()?;
    drop_vec_tables(&tx, StoreKind::Ltm.vec_tables())?;
    StoreKind::Ltm.create_vec_tables(&tx, identity.dim)?;
    insert_vectors(&tx, "vec_ltm", &vectors.concepts)?;
    insert_vectors(&tx, "vec_leaves", &vectors.leaves)?;
    record_embedding(&tx, identity)?;
    tx.commit()?;
    Ok((vectors.concepts.len(), vectors.leaves.len()))
}

/// Re-embed a single (already migrated) STM store: embed, then rewrite.
pub async fn reembed_stm(
    conn: &Connection,
    embedder: &dyn LlmClient,
    identity: &EmbeddingIdentity,
) -> Result<usize> {
    let vectors = embed_stm(conn, embedder, identity).await?;
    write_stm(conn, identity, &vectors)
}

/// Re-embed a single (already migrated) LTM store: embed, then rewrite.
pub async fn reembed_ltm(
    conn: &Connection,
    embedder: &dyn LlmClient,
    identity: &EmbeddingIdentity,
) -> Result<(usize, usize)> {
    let vectors = embed_ltm(conn, embedder, identity).await?;
    write_ltm(conn, identity, &vectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::migrations::{get_meta, prepare_store};
    use crate::domain::ltm::{LtmRepository, Provenance, TreeNode, TreeNodeKind};
    use crate::domain::models::CclDefinition;
    use crate::domain::ports::ExtractedFact;
    use crate::infrastructure::ltm_repository::SqliteLtmRepository;
    use crate::infrastructure::schema::{init_ltm_schema, init_schema, vec_table_dim};
    use async_trait::async_trait;

    /// Deterministic embedder of a fixed dimension; can be told to fail.
    struct DimEmbedder {
        dim: usize,
        fail: bool,
    }

    #[async_trait]
    impl LlmClient for DimEmbedder {
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
        async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            if self.fail {
                bail!("embedder down");
            }
            let mut v = vec![0.0; self.dim];
            v[text.len() % self.dim] = 1.0;
            Ok(v)
        }
        async fn compress_context(&self, _m: &str) -> Result<String> {
            Ok(String::new())
        }
    }

    fn ident(model: &str, dim: usize) -> EmbeddingIdentity {
        EmbeddingIdentity {
            provider: "test".into(),
            model: model.into(),
            dim,
        }
    }

    fn count(conn: &Connection, table: &str) -> i64 {
        conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    }

    /// Build a legacy workspace at 8 dims: STM with an active fact, an
    /// archived fact and an anchor; LTM with a spine concept and a leaf.
    fn legacy_workspace(dir: &tempfile::TempDir) -> (PathBuf, PathBuf) {
        let stm_path = dir.path().join("stm.sqlite");
        let ltm_path = dir.path().join("ltm.sqlite");
        {
            let stm = init_db(Some(&stm_path)).unwrap();
            init_schema(&stm, 8).unwrap();
            for (fact, status, kind) in [
                ("active fact", "active", "fact"),
                ("old fact", "archived", "fact"),
                ("anchor", "active", "subject"),
            ] {
                stm.execute(
                    "INSERT INTO nodes (tenant_id, payload, status)
                     VALUES ('legacy', json_object('fact', ?1, 'kind', ?2), ?3)",
                    params![fact, kind, status],
                )
                .unwrap();
            }
            let ltm = init_db(Some(&ltm_path)).unwrap();
            init_ltm_schema(&ltm, 8).unwrap();
            let repo = SqliteLtmRepository::new(ltm);
            repo.seed_spine().unwrap();
            let leaf = repo
                .create_node(
                    &TreeNode::new("doc", "a stored summary", TreeNodeKind::Leaf),
                    None,
                )
                .unwrap();
            repo.create_leaf(&crate::domain::ltm::Leaf {
                tree_node_id: leaf,
                data_id: "d1".into(),
                provenance: Provenance {
                    source: "t".into(),
                    ingested_at: None,
                    confidence: 1.0,
                },
            })
            .unwrap();
        }
        (stm_path, ltm_path)
    }

    /// Re-embedding switches a workspace to a new model/dimension: vec tables
    /// are rebuilt at the new size, the right rows are indexed, meta is
    /// updated, backups exist — and afterwards the store opens cleanly.
    #[tokio::test]
    async fn reembed_rebuilds_vectors_at_new_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let (stm_path, ltm_path) = legacy_workspace(&dir);
        let new = ident("new-model", 4);

        // Before: the 8-dim store is refused for a 4-dim embedder... (it has no
        // vectors yet, so seed one to make the mismatch real)
        {
            let stm = init_db(Some(&stm_path)).unwrap();
            stm.execute(
                "INSERT INTO vec_nodes (node_id, embedding) VALUES (1, ?1)",
                [to_bytes(&[1.0; 8])],
            )
            .unwrap();
            let err = prepare_store(&stm, StoreKind::Stm, Some(&stm_path), &new)
                .unwrap_err()
                .to_string();
            assert!(err.contains("reembed"), "{err}");
        }

        let report = reembed_workspace(
            &stm_path,
            &ltm_path,
            Arc::new(DimEmbedder {
                dim: 4,
                fail: false,
            }),
            &new,
        )
        .await
        .unwrap();

        assert_eq!(report.stm_nodes, 1, "only the active non-anchor fact");
        assert_eq!(report.ltm_concepts, 3, "root + notes + documents");
        assert_eq!(report.ltm_leaves, 1);
        assert!(report.backups.len() >= 2);
        assert!(report.backups.iter().all(|b| b.exists()));

        let stm = init_db(Some(&stm_path)).unwrap();
        let ltm = init_db(Some(&ltm_path)).unwrap();
        assert_eq!(vec_table_dim(&stm, "vec_nodes").unwrap(), Some(4));
        assert_eq!(vec_table_dim(&ltm, "vec_ltm").unwrap(), Some(4));
        assert_eq!(vec_table_dim(&ltm, "vec_leaves").unwrap(), Some(4));
        assert_eq!(count(&stm, "vec_nodes"), 1);
        assert_eq!(count(&ltm, "vec_leaves"), 1);
        assert_eq!(
            get_meta(&stm, "embedding_model").unwrap().as_deref(),
            Some("new-model")
        );
        assert_eq!(
            get_meta(&ltm, "embedding_dim").unwrap().as_deref(),
            Some("4")
        );

        // The stores now open with the new embedder.
        prepare_store(&stm, StoreKind::Stm, Some(&stm_path), &new).unwrap();
        prepare_store(&ltm, StoreKind::Ltm, Some(&ltm_path), &new).unwrap();
    }

    /// P2R-5: an embedder that fails only on LTM texts leaves BOTH stores
    /// unchanged — STM is not rewritten before LTM's vectors exist.
    #[tokio::test]
    async fn ltm_embed_failure_leaves_the_whole_workspace_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let (stm_path, ltm_path) = legacy_workspace(&dir);
        // Open once at the old identity so both stores carry meta + 8-dim vectors.
        let old = ident("old-model", 8);
        {
            let stm = init_db(Some(&stm_path)).unwrap();
            prepare_store(&stm, StoreKind::Stm, Some(&stm_path), &old).unwrap();
            stm.execute(
                "INSERT INTO vec_nodes (node_id, embedding) VALUES (1, ?1)",
                [to_bytes(&[1.0; 8])],
            )
            .unwrap();
            let ltm = init_db(Some(&ltm_path)).unwrap();
            prepare_store(&ltm, StoreKind::Ltm, Some(&ltm_path), &old).unwrap();
        }

        let err = reembed_workspace(
            &stm_path,
            &ltm_path,
            Arc::new(FailOnLtm { dim: 4 }),
            &ident("new-model", 4),
        )
        .await
        .unwrap_err();
        assert!(format!("{err:#}").contains("LTM"), "{err:#}");

        let stm = init_db(Some(&stm_path)).unwrap();
        let ltm = init_db(Some(&ltm_path)).unwrap();
        assert_eq!(
            vec_table_dim(&stm, "vec_nodes").unwrap(),
            Some(8),
            "STM untouched"
        );
        assert_eq!(count(&stm, "vec_nodes"), 1);
        assert_eq!(
            get_meta(&stm, "embedding_model").unwrap().as_deref(),
            Some("old-model")
        );
        assert_eq!(
            vec_table_dim(&ltm, "vec_ltm").unwrap(),
            Some(8),
            "LTM untouched"
        );
        assert_eq!(
            get_meta(&ltm, "embedding_model").unwrap().as_deref(),
            Some("old-model")
        );
        // And the workspace still opens with its original embedder.
        prepare_store(&stm, StoreKind::Stm, Some(&stm_path), &old).unwrap();
        prepare_store(&ltm, StoreKind::Ltm, Some(&ltm_path), &old).unwrap();
    }

    /// Embeds STM facts fine, fails on anything that looks like LTM text
    /// (concepts are "name: summary", the leaf is "a stored summary").
    struct FailOnLtm {
        dim: usize,
    }

    #[async_trait]
    impl LlmClient for FailOnLtm {
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
        async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            if text.contains(": ") || text.contains("summary") {
                bail!("embedder failed on an LTM text");
            }
            Ok(vec![1.0; self.dim])
        }
        async fn compress_context(&self, _m: &str) -> Result<String> {
            Ok(String::new())
        }
    }

    /// If the embedder fails, the store is left exactly as it was.
    #[tokio::test]
    async fn failed_reembed_leaves_store_untouched() {
        let conn = init_db(None as Option<&String>).unwrap();
        prepare_store(&conn, StoreKind::Stm, None, &ident("old", 8)).unwrap();
        conn.execute(
            "INSERT INTO nodes (tenant_id, payload) VALUES ('default', json_object('fact', 'x'))",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO vec_nodes (node_id, embedding) VALUES (1, ?1)",
            [to_bytes(&[1.0; 8])],
        )
        .unwrap();

        let err = reembed_stm(&conn, &DimEmbedder { dim: 4, fail: true }, &ident("new", 4))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("embedder down"));

        assert_eq!(vec_table_dim(&conn, "vec_nodes").unwrap(), Some(8));
        assert_eq!(count(&conn, "vec_nodes"), 1);
        assert_eq!(
            get_meta(&conn, "embedding_model").unwrap().as_deref(),
            Some("old")
        );
    }

    /// An embedder returning the wrong dimension is rejected before writing.
    #[tokio::test]
    async fn wrong_dimension_from_embedder_is_rejected() {
        let conn = init_db(None as Option<&String>).unwrap();
        prepare_store(&conn, StoreKind::Stm, None, &ident("old", 8)).unwrap();
        conn.execute(
            "INSERT INTO nodes (tenant_id, payload) VALUES ('default', json_object('fact', 'x'))",
            [],
        )
        .unwrap();
        let err = reembed_stm(
            &conn,
            &DimEmbedder {
                dim: 6,
                fail: false,
            },
            &ident("new", 4),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected 4"), "{err}");
        assert_eq!(vec_table_dim(&conn, "vec_nodes").unwrap(), Some(8));
    }
}
