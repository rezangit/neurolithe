//! Monitoring — builds the "brain CT scan" snapshot published to
//! `memory.metrics` (slice 9). Combines DB-derived stats from both stores with
//! runtime stats the daemon supplies (feeder lag/errors, session count).

use crate::domain::ltm::LtmRepository;
use crate::domain::ports::MemoryRepository;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Runtime stats that can't be read from the databases — supplied by the daemon
/// (slice 11). Defaults are safe placeholders for environments without a feeder.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeStats {
    /// Active STM session buffers (in-memory, from the SessionManager).
    pub sessions: u64,
    /// Feeder consumer lag in messages; `-1` when unknown.
    pub feeder_lag: i64,
    /// Cumulative feeder error count.
    pub feeder_errors: u64,
    /// Unix seconds of the last completed backfill, if any.
    pub last_backfill_unix: Option<i64>,
}

impl RuntimeStats {
    /// Lag unknown until the daemon wires consumer-lag reporting.
    pub fn unknown() -> Self {
        Self {
            feeder_lag: -1,
            ..Default::default()
        }
    }
}

/// Live feeder counters, shared (atomics) between the feeder loop that updates
/// them and the metrics scheduler that reads them. Lag stays unknown (`-1`)
/// until consumer-lag reporting is wired.
#[derive(Debug, Default)]
pub struct FeederStats {
    errors: std::sync::atomic::AtomicU64,
    documents: std::sync::atomic::AtomicU64,
    /// Unix seconds of the last successful ingest; 0 = none yet.
    last_ingest_unix: std::sync::atomic::AtomicI64,
}

impl FeederStats {
    pub fn record_error(&self) {
        self.errors
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a successful document ingest at `unix` seconds.
    pub fn record_document(&self, unix: i64) {
        self.documents
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.last_ingest_unix
            .store(unix, std::sync::atomic::Ordering::Relaxed);
    }

    /// Project into a [`RuntimeStats`] for the metrics snapshot.
    pub fn to_runtime(&self, sessions: u64) -> RuntimeStats {
        use std::sync::atomic::Ordering::Relaxed;
        let last = self.last_ingest_unix.load(Relaxed);
        RuntimeStats {
            sessions,
            feeder_lag: -1,
            feeder_errors: self.errors.load(Relaxed),
            last_backfill_unix: (last != 0).then_some(last),
        }
    }
}

/// The full CT-scan snapshot — the value published to `memory.metrics`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryMetrics {
    // STM (working memory)
    pub stm_active_nodes: i64,
    pub stm_archived_nodes: i64,
    pub stm_avg_relevance: f64,
    pub stm_decay_histogram: Vec<i64>,
    pub stm_db_bytes: i64,
    pub sessions: u64,
    // LTM (knowledge tree)
    pub ltm_tree_nodes: i64,
    pub ltm_leaves: i64,
    pub ltm_edges: i64,
    pub ltm_inbox_docs: i64,
    pub ltm_orphan_leaves: i64,
    pub ltm_max_depth: i64,
    pub ltm_db_bytes: i64,
    // Feeder (runtime)
    pub feeder_lag: i64,
    pub feeder_errors: u64,
    pub last_backfill_unix: Option<i64>,
}

/// Assembles [`MemoryMetrics`] from both stores plus runtime stats.
pub struct MonitoringService {
    stm: Arc<dyn MemoryRepository>,
    ltm: Arc<dyn LtmRepository>,
}

impl MonitoringService {
    pub fn new(stm: Arc<dyn MemoryRepository>, ltm: Arc<dyn LtmRepository>) -> Self {
        Self { stm, ltm }
    }

    /// Build one snapshot. DB stats are read live; runtime stats are merged in.
    pub fn snapshot(&self, runtime: &RuntimeStats) -> Result<MemoryMetrics> {
        let stm = self.stm.stm_stats()?;
        let ltm = self.ltm.ltm_stats()?;
        Ok(MemoryMetrics {
            stm_active_nodes: stm.active_nodes,
            stm_archived_nodes: stm.archived_nodes,
            stm_avg_relevance: stm.avg_relevance,
            stm_decay_histogram: stm.decay_histogram,
            stm_db_bytes: stm.db_size_bytes,
            sessions: runtime.sessions,
            ltm_tree_nodes: ltm.tree_nodes,
            ltm_leaves: ltm.leaves,
            ltm_edges: ltm.edges,
            ltm_inbox_docs: ltm.inbox_docs,
            ltm_orphan_leaves: ltm.orphan_leaves,
            ltm_max_depth: ltm.max_depth,
            ltm_db_bytes: ltm.db_size_bytes,
            feeder_lag: runtime.feeder_lag,
            feeder_errors: runtime.feeder_errors,
            last_backfill_unix: runtime.last_backfill_unix,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ltm::{Leaf, Provenance, TreeEdge, TreeNode, TreeNodeKind};
    use crate::domain::models::{MemoryNode, TenantId};
    use crate::infrastructure::database::init_db;
    use crate::infrastructure::ltm_repository::SqliteLtmRepository;
    use crate::infrastructure::repository::SqliteMemoryRepository;
    use crate::infrastructure::schema::{init_ltm_schema, init_schema};
    use serde_json::json;

    const DIM: usize = 4;

    fn node(score: f64, status: &str) -> MemoryNode {
        MemoryNode {
            id: None,
            tenant_id: TenantId("t".into()),
            source_episode_id: None,
            payload: json!({"fact": "f"}),
            status: status.into(),
            ccl: "reality".into(),
            is_explicit: true,
            support_count: 1,
            relevance_score: score,
            context_key: None,
        }
    }

    /// Snapshot fields are computed correctly on a seeded STM + LTM.
    #[test]
    fn test_snapshot_computes_fields() {
        // STM: two active nodes (0.9, 0.3) + one archived.
        let stm_conn = init_db(None as Option<&String>).unwrap();
        init_schema(&stm_conn, DIM).unwrap();
        let stm = Arc::new(SqliteMemoryRepository::new(stm_conn));
        stm.store_node(&node(0.9, "active"), &[0.0; DIM]).unwrap();
        stm.store_node(&node(0.3, "active"), &[0.1; DIM]).unwrap();
        stm.store_node(&node(0.05, "archived"), &[0.2; DIM])
            .unwrap();

        // LTM: spine (6 nodes) + one document leaf under the inbox.
        let ltm_conn = init_db(None as Option<&String>).unwrap();
        init_ltm_schema(&ltm_conn, DIM).unwrap();
        let ltm = Arc::new(SqliteLtmRepository::new(ltm_conn));
        ltm.seed_spine().unwrap();
        let inbox = ltm.get_inbox().unwrap().unwrap();
        let leaf = ltm
            .create_node(&TreeNode::new("doc", "a doc", TreeNodeKind::Leaf), None)
            .unwrap();
        ltm.create_leaf(&Leaf {
            tree_node_id: leaf,
            data_id: "doc_1".into(),
            provenance: Provenance {
                source: "t".into(),
                ingested_at: None,
                confidence: 1.0,
            },
        })
        .unwrap();
        ltm.add_edge(&TreeEdge::new(inbox.id.unwrap(), leaf))
            .unwrap();

        let monitor = MonitoringService::new(
            stm as Arc<dyn MemoryRepository>,
            ltm as Arc<dyn LtmRepository>,
        );
        let m = monitor
            .snapshot(&RuntimeStats {
                sessions: 2,
                feeder_lag: 5,
                feeder_errors: 1,
                last_backfill_unix: Some(1000),
            })
            .unwrap();

        // STM
        assert_eq!(m.stm_active_nodes, 2);
        assert_eq!(m.stm_archived_nodes, 1);
        assert!((m.stm_avg_relevance - 0.6).abs() < 1e-9); // (0.9+0.3)/2
        assert_eq!(
            m.stm_decay_histogram.iter().sum::<i64>(),
            2,
            "2 active binned"
        );
        assert_eq!(m.stm_decay_histogram[4], 1, "0.9 in top bin");
        assert_eq!(m.stm_decay_histogram[1], 1, "0.3 in bin 1");
        assert!(m.stm_db_bytes > 0);

        // LTM: 4 spine (root + notes + documents + inbox, the default spine)
        // + 1 leaf = 5 nodes; one inbox doc; tree height >= 2.
        assert_eq!(m.ltm_tree_nodes, 5);
        assert_eq!(m.ltm_leaves, 1);
        assert_eq!(m.ltm_inbox_docs, 1);
        assert_eq!(m.ltm_orphan_leaves, 0, "the leaf has a parent");
        assert!(m.ltm_max_depth >= 2);
        assert_eq!(m.ltm_edges, 4); // 3 spine edges + 1 leaf edge
        assert!(m.ltm_db_bytes > 0);

        // Runtime passed through.
        assert_eq!(m.sessions, 2);
        assert_eq!(m.feeder_lag, 5);
        assert_eq!(m.feeder_errors, 1);
        assert_eq!(m.last_backfill_unix, Some(1000));
    }
}
