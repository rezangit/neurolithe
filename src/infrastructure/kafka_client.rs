//! Shared rdkafka client settings for every Kafka client NeuroLithe creates
//! (the feeder consumer + its dead-letter producer, the command consumer, the
//! query consumer + its reply producer, and the metrics producer).
//!
//! `[kafka.client]` is a passthrough map of librdkafka properties applied to
//! all of them — the way to enable SASL/TLS without code changes:
//!
//! ```toml
//! [kafka.client]
//! "security.protocol" = "SASL_SSL"
//! "sasl.mechanism"    = "SCRAM-SHA-512"
//! "sasl.username"     = "neurolithe"
//! "sasl.password"     = "…"            # or NEUROLITHE__KAFKA__CLIENT__… / .env
//! "ssl.ca.location"   = "/etc/ssl/certs/ca.pem"
//! ```
//!
//! librdkafka property names never contain `_`, so an underscore in a key is
//! read as a dot: `sasl_password` = `sasl.password`. That lets secrets come from
//! the environment, where dotted names don't work:
//! `NEUROLITHE__KAFKA__CLIENT__SASL_PASSWORD=…`.
//!
//! Properties the clients depend on for correctness (brokers, group ids, offset
//! handling) are set by the code *after* the passthrough map, so the map can't
//! silently break offset management or merge the consumer groups.

use crate::infrastructure::config::KafkaConfig;
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{CommitMode, Consumer, StreamConsumer};
use rdkafka::error::{KafkaError, KafkaResult, RDKafkaErrorCode};
use rdkafka::message::BorrowedMessage;
use rdkafka::producer::{FutureProducer, Producer};
use rdkafka::util::Timeout;
use std::time::Duration;

/// Keys the code owns. A `[kafka.client]` entry for one of these is ignored
/// (with a warning) — use `kafka.brokers` / `kafka.group_id` instead.
///
/// Includes librdkafka aliases (`metadata.broker.list` = `bootstrap.servers`,
/// which would otherwise override `kafka.brokers` depending on map order) and
/// the consumer settings offset handling depends on (P2R-6).
pub const RESERVED_KEYS: &[&str] = &[
    "bootstrap.servers",
    "metadata.broker.list",
    "group.id",
    "group.instance.id",
    "enable.auto.commit",
    "enable.auto.offset.store",
    "enable.partition.eof",
    "auto.offset.reset",
];

/// A `[kafka.client]` key as a librdkafka property name: trimmed, lowercase,
/// `_` → `.` (see the module docs).
pub fn property_name(key: &str) -> String {
    key.trim().to_ascii_lowercase().replace('_', ".")
}

/// A client config with the passthrough properties + `bootstrap.servers`.
/// Producers use this directly; consumers go through [`consumer_config`].
pub fn base_config(kafka: &KafkaConfig) -> ClientConfig {
    let mut cfg = ClientConfig::new();
    for (raw_key, value) in &kafka.client {
        let key = property_name(raw_key);
        if RESERVED_KEYS.contains(&key.as_str()) {
            tracing::warn!("[kafka.client] '{key}' is managed by NeuroLithe and ignored");
            continue;
        }
        cfg.set(key, value);
    }
    cfg.set("bootstrap.servers", &kafka.brokers);
    cfg
}

/// A consumer config: [`base_config`] + the group id and manual offset commits
/// (`auto.offset.reset` = `earliest` for replay sources, `latest` for live
/// request traffic).
pub fn consumer_config(kafka: &KafkaConfig, group_id: &str, offset_reset: &str) -> ClientConfig {
    let mut cfg = base_config(kafka);
    cfg.set("group.id", group_id)
        .set("enable.auto.commit", "false")
        .set("auto.offset.reset", offset_reset);
    cfg
}

/// Consumer group of the command consumer.
pub fn command_group(kafka: &KafkaConfig) -> String {
    format!("{}-cmd", kafka.group_id)
}

/// Consumer group of the query consumer.
pub fn query_group(kafka: &KafkaConfig) -> String {
    format!("{}-query", kafka.group_id)
}

/// Shutdown signal shared by the daemon's Kafka loops: flips to `true` once.
pub type StopSignal = tokio::sync::watch::Receiver<bool>;

/// The next message, or `None` once `stop` fires (or its sender is gone).
/// Only the wait is raced, never a message being handled, so an in-flight
/// message is always finished and committed before a loop exits.
pub async fn recv_or_stop<'a>(
    consumer: &'a StreamConsumer,
    stop: &mut StopSignal,
) -> Option<KafkaResult<BorrowedMessage<'a>>> {
    if *stop.borrow() {
        return None;
    }
    tokio::select! {
        biased;
        _ = stop.changed() => None,
        next = consumer.recv() => Some(next),
    }
}

/// Graceful-shutdown tail for a consumer: synchronously commit its current
/// offsets (the per-message commits are async and may still be queued).
pub fn commit_on_shutdown(consumer: &StreamConsumer, name: &str) {
    match consumer.commit_consumer_state(CommitMode::Sync) {
        Ok(()) => tracing::debug!("{name}: offsets committed on shutdown"),
        // Nothing consumed since the last commit.
        Err(KafkaError::ConsumerCommit(RDKafkaErrorCode::NoOffset)) => {}
        Err(e) => tracing::warn!("{name}: final offset commit failed: {e}"),
    }
}

/// How long a producer may take to deliver its queue on shutdown.
pub const FLUSH_TIMEOUT: Duration = Duration::from_secs(5);

/// Graceful-shutdown tail for a producer: deliver whatever is still queued.
pub fn flush_on_shutdown(producer: &FutureProducer, name: &str) {
    match producer.flush(Timeout::After(FLUSH_TIMEOUT)) {
        Ok(()) => tracing::debug!("{name}: producer flushed"),
        Err(e) => tracing::warn!("{name}: producer flush failed: {e}"),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::infrastructure::config::KafkaTopics;

    /// A broker-less Kafka config for unit tests (clients are created but never
    /// connect).
    pub(crate) fn test_kafka(client: &[(&str, &str)]) -> KafkaConfig {
        KafkaConfig {
            brokers: "localhost:9092".into(),
            group_id: "nl-test".into(),
            topics: KafkaTopics::default(),
            client: client
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    /// SASL/TLS settings from `[kafka.client]` reach both producer and consumer
    /// configs.
    #[test]
    fn passthrough_reaches_every_client() {
        let kafka = test_kafka(&[
            ("security.protocol", "SASL_SSL"),
            ("sasl.mechanism", "SCRAM-SHA-512"),
            ("ssl.ca.location", "/tmp/ca.pem"),
        ]);
        for cfg in [
            base_config(&kafka),
            consumer_config(&kafka, "g", "earliest"),
        ] {
            assert_eq!(cfg.get("security.protocol"), Some("SASL_SSL"));
            assert_eq!(cfg.get("sasl.mechanism"), Some("SCRAM-SHA-512"));
            assert_eq!(cfg.get("ssl.ca.location"), Some("/tmp/ca.pem"));
            assert_eq!(cfg.get("bootstrap.servers"), Some("localhost:9092"));
        }
    }

    /// The passthrough map cannot override the settings offset management and
    /// group separation depend on.
    #[test]
    fn reserved_keys_cannot_be_overridden() {
        let kafka = test_kafka(&[
            ("bootstrap.servers", "evil:1"),
            ("group.id", "shared"),
            ("enable.auto.commit", "true"),
            ("auto.offset.reset", "latest"),
        ]);
        let cfg = consumer_config(&kafka, "nl-test-cmd", "earliest");
        assert_eq!(cfg.get("bootstrap.servers"), Some("localhost:9092"));
        assert_eq!(cfg.get("group.id"), Some("nl-test-cmd"));
        assert_eq!(cfg.get("enable.auto.commit"), Some("false"));
        assert_eq!(cfg.get("auto.offset.reset"), Some("earliest"));

        let producer = base_config(&kafka);
        assert_eq!(producer.get("group.id"), None);
    }

    /// Underscored keys (the only form env vars can carry) map to dotted
    /// librdkafka names, and reserved keys are caught in either spelling.
    #[test]
    fn underscore_keys_map_to_dotted_properties() {
        let kafka = test_kafka(&[
            ("sasl_password", "s3cret"),
            ("SECURITY_PROTOCOL", "SASL_SSL"),
            ("group_id", "shared"),
        ]);
        let cfg = consumer_config(&kafka, "nl-test", "earliest");
        assert_eq!(cfg.get("sasl.password"), Some("s3cret"));
        assert_eq!(cfg.get("security.protocol"), Some("SASL_SSL"));
        assert_eq!(cfg.get("group.id"), Some("nl-test"));
        assert_eq!(cfg.get("sasl_password"), None);
    }

    /// P2R-6: librdkafka aliases of the owned keys are reserved too, so
    /// `metadata.broker.list` can't silently replace `kafka.brokers`, and the
    /// offset/membership settings the consumers rely on stay put.
    #[test]
    fn reserved_aliases_cannot_be_overridden() {
        let kafka = test_kafka(&[
            ("metadata.broker.list", "evil:1"),
            ("metadata_broker_list", "evil:2"),
            ("group.instance.id", "static-1"),
            ("enable.auto.offset.store", "false"),
            ("enable.partition.eof", "true"),
        ]);
        for cfg in [
            base_config(&kafka),
            consumer_config(&kafka, "nl-test", "earliest"),
        ] {
            assert_eq!(cfg.get("metadata.broker.list"), None);
            assert_eq!(cfg.get("group.instance.id"), None);
            assert_eq!(cfg.get("enable.auto.offset.store"), None);
            assert_eq!(cfg.get("enable.partition.eof"), None);
            assert_eq!(cfg.get("bootstrap.servers"), Some("localhost:9092"));
        }
    }

    #[test]
    fn derived_group_ids() {
        let kafka = test_kafka(&[]);
        assert_eq!(command_group(&kafka), "nl-test-cmd");
        assert_eq!(query_group(&kafka), "nl-test-query");
    }
}
