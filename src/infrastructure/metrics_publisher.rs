//! Metrics publisher — sends the store metrics snapshot to the metrics topic
//! (`[kafka.topics] metrics`, default `memory.metrics`).
//!
//! The snapshot is published under a single fixed key, so on a compacted topic
//! it holds exactly one record (the latest state), never a growing log — handy
//! for a dashboard.
//!
//! `render` (key + JSON bytes) is pure and unit-tested; the rdkafka send needs
//! a live broker.

use crate::application::monitoring::MemoryMetrics;
use crate::infrastructure::config::KafkaConfig;
use crate::infrastructure::kafka_client::base_config;
use anyhow::{Context, Result};
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;

/// The single compaction key — one record on the topic, always the latest.
pub const METRICS_KEY: &str = "neurolithe";

/// The keyed message body for a snapshot: a fixed key + JSON bytes.
pub fn render(metrics: &MemoryMetrics) -> Result<(&'static str, Vec<u8>)> {
    let bytes = serde_json::to_vec(metrics).context("serializing metrics snapshot")?;
    Ok((METRICS_KEY, bytes))
}

pub struct MetricsPublisher {
    producer: FutureProducer,
    topic: String,
}

impl MetricsPublisher {
    pub fn new(kafka: &KafkaConfig) -> Result<Self> {
        let producer: FutureProducer = base_config(kafka)
            .create()
            .context("creating metrics producer")?;
        Ok(Self {
            producer,
            topic: kafka.topics.metrics.clone(),
        })
    }

    /// Publish one snapshot under the fixed compaction key.
    pub async fn publish(&self, metrics: &MemoryMetrics) -> Result<()> {
        let (key, payload) = render(metrics)?;
        let record = FutureRecord::to(&self.topic).key(key).payload(&payload);
        self.producer
            .send(record, Timeout::Never)
            .await
            .map_err(|(e, _)| e)
            .with_context(|| format!("publishing metrics to {}", self.topic))?;
        Ok(())
    }

    /// Deliver any queued snapshot before exit (graceful shutdown).
    pub fn flush(&self) {
        crate::infrastructure::kafka_client::flush_on_shutdown(&self.producer, "metrics publisher");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::monitoring::MemoryMetrics;

    fn metrics() -> MemoryMetrics {
        MemoryMetrics {
            stm_active_nodes: 3,
            stm_archived_nodes: 1,
            stm_avg_relevance: 0.6,
            stm_decay_histogram: vec![0, 1, 0, 0, 2],
            stm_db_bytes: 4096,
            sessions: 1,
            ltm_tree_nodes: 7,
            ltm_leaves: 1,
            ltm_edges: 6,
            ltm_inbox_docs: 1,
            ltm_orphan_leaves: 0,
            ltm_max_depth: 2,
            ltm_db_bytes: 8192,
            feeder_lag: 0,
            feeder_errors: 0,
            last_backfill_unix: Some(1000),
        }
    }

    /// render uses the single fixed key (so compaction keeps one record) and
    /// round-trips the snapshot.
    #[test]
    fn test_render_is_one_fixed_key_message() {
        let (key, payload) = render(&metrics()).unwrap();
        assert_eq!(key, METRICS_KEY);
        // Same key every time -> no churn on the compacted topic.
        let (key2, _) = render(&metrics()).unwrap();
        assert_eq!(key, key2);

        let decoded: MemoryMetrics = serde_json::from_slice(&payload).unwrap();
        assert_eq!(decoded, metrics());
    }

    /// The metrics topic comes from `[kafka.topics] metrics`.
    #[test]
    fn test_publisher_uses_configured_topic() {
        let mut kafka = crate::infrastructure::kafka_client::tests::test_kafka(&[]);
        assert_eq!(
            MetricsPublisher::new(&kafka).unwrap().topic,
            "memory.metrics"
        );
        kafka.topics.metrics = "ops.memory-metrics".into();
        assert_eq!(
            MetricsPublisher::new(&kafka).unwrap().topic,
            "ops.memory-metrics"
        );
    }
}
