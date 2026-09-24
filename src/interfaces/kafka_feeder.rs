//! Kafka feeder — consumes the documents topic (`[kafka.topics] documents`,
//! default `document.completed`) and drives ingestion.
//!
//! Each event carries its own text:
//! `{"data_id": "...", "title"?: "...", "text": "...", "tags"?: [...], "ts"?: "..."}`.
//!
//! Backfills from the earliest offset (if the topic is compacted, this replays
//! the latest state per document). Routing: a tombstone (null payload) forgets
//! the document named by the message key; a valid event is ingested; an event
//! without text is skipped with a warning; an event without a `data_id` goes to
//! the dead-letter topic; un-parseable bytes go to the parking topic. The offset
//! is committed only AFTER the write, so a crash re-processes rather than drops.
//!
//! The decision logic ([`decide`]) is pure and unit-tested; the rdkafka loop
//! itself needs a live broker.

use crate::application::ingestion::{DocumentEvent, IngestionService};
use crate::application::monitoring::FeederStats;
use crate::infrastructure::config::KafkaConfig;
use crate::infrastructure::kafka_client::{
    StopSignal, base_config, commit_on_shutdown, consumer_config, flush_on_shutdown, recv_or_stop,
};
use anyhow::{Context, Result};
use rdkafka::consumer::{CommitMode, Consumer, StreamConsumer};
use rdkafka::message::{Header, Message, OwnedHeaders};
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use std::sync::Arc;
use std::time::Duration;

/// How to handle one consumed message. Derived purely from its key + payload.
#[derive(Debug, PartialEq)]
pub enum FeedDecision {
    /// Tombstone (null payload): forget this document id (the message key).
    Forget(String),
    /// A valid event to ingest.
    Ingest(DocumentEvent),
    /// A well-formed event with no text: nothing to learn. Logged and skipped
    /// (not dead-lettered). Carries the document id.
    Skip(String),
    /// Parsed but structurally invalid (no `data_id`) — route to the DLQ topic.
    BadEvent(String),
    /// Un-parseable bytes — route to the parking topic.
    Park(String),
}

/// Current Unix time in seconds (for the last-ingest stat).
fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Classify a message by its key + payload. Pure.
pub fn decide(key: Option<&str>, payload: Option<&[u8]>) -> FeedDecision {
    match payload {
        // Null payload = compaction tombstone -> forget by key.
        None => match key {
            Some(k) if !k.is_empty() => FeedDecision::Forget(k.to_string()),
            _ => FeedDecision::Park("tombstone without a key".into()),
        },
        Some(bytes) => match serde_json::from_slice::<DocumentEvent>(bytes) {
            Ok(event) => match event.document_id() {
                None => FeedDecision::BadEvent("document event has no data_id".into()),
                Some(id) if event.body().is_none() => FeedDecision::Skip(id.to_string()),
                Some(_) => FeedDecision::Ingest(event),
            },
            Err(e) => FeedDecision::Park(format!("un-parseable document event: {e}")),
        },
    }
}

/// Number of ingest attempts before giving up to the DLQ topic (transient
/// retry).
const INGEST_ATTEMPTS: u32 = 3;
const RETRY_BACKOFF: Duration = Duration::from_millis(500);

pub struct KafkaFeeder {
    consumer: Arc<StreamConsumer>,
    producer: FutureProducer,
    ingestion: Arc<IngestionService>,
    stats: Arc<FeederStats>,
    source_topic: String,
    dlq_topic: String,
    parking_topic: String,
    group_id: String,
}

impl KafkaFeeder {
    pub fn new(
        kafka: &KafkaConfig,
        ingestion: Arc<IngestionService>,
        stats: Arc<FeederStats>,
    ) -> Result<Self> {
        // Backfill from the start; we manage offsets ourselves.
        let consumer: StreamConsumer = consumer_config(kafka, &kafka.group_id, "earliest")
            .create()
            .context("creating documents consumer")?;

        let producer: FutureProducer = base_config(kafka)
            .create()
            .context("creating dlq/parking producer")?;

        Ok(Self {
            consumer: Arc::new(consumer),
            producer,
            ingestion,
            stats,
            source_topic: kafka.topics.documents.clone(),
            dlq_topic: kafka.topics.dlq.clone(),
            parking_topic: kafka.topics.parking.clone(),
            group_id: kafka.group_id.clone(),
        })
    }

    /// Shared handle to the consumer, so the command consumer can rewind it to
    /// earliest after a hard reset.
    pub fn consumer(&self) -> Arc<StreamConsumer> {
        self.consumer.clone()
    }

    /// Consume until `stop` fires: classify -> act -> commit. Then commit
    /// offsets synchronously and flush the dead-letter producer.
    pub async fn run(&self, mut stop: StopSignal) -> Result<()> {
        self.consumer
            .subscribe(&[&self.source_topic])
            .with_context(|| format!("subscribing to {}", self.source_topic))?;

        while let Some(next) = recv_or_stop(&self.consumer, &mut stop).await {
            match next {
                Err(e) => tracing::warn!("consumer error: {e}"),
                Ok(msg) => {
                    self.process(&msg).await;
                    // Commit AFTER the write so a crash re-processes, not drops.
                    if let Err(e) = self.consumer.commit_message(&msg, CommitMode::Async) {
                        tracing::warn!("commit failed: {e}");
                    }
                }
            }
        }
        commit_on_shutdown(&self.consumer, "documents feeder");
        flush_on_shutdown(&self.producer, "documents feeder");
        Ok(())
    }

    async fn process(&self, msg: &rdkafka::message::BorrowedMessage<'_>) {
        let key = msg.key().and_then(|k| std::str::from_utf8(k).ok());
        match decide(key, msg.payload()) {
            FeedDecision::Forget(id) => {
                if let Err(e) = self.ingestion.forget(&id).await {
                    self.send_aside(msg, &self.dlq_topic, &format!("forget failed: {e}"))
                        .await;
                }
            }
            FeedDecision::Ingest(event) => match self.ingest_with_retry(&event).await {
                Ok(()) => self.stats.record_document(now_unix()),
                Err(e) => {
                    self.stats.record_error();
                    self.send_aside(msg, &self.dlq_topic, &format!("ingest failed: {e}"))
                        .await;
                }
            },
            FeedDecision::Skip(id) => {
                tracing::warn!("document event '{id}' has no text; skipped");
            }
            FeedDecision::BadEvent(reason) => {
                self.send_aside(msg, &self.dlq_topic, &reason).await;
            }
            FeedDecision::Park(reason) => {
                self.send_aside(msg, &self.parking_topic, &reason).await;
            }
        }
    }

    /// Retry transient ingest failures a bounded number of times before the
    /// caller dead-letters.
    async fn ingest_with_retry(&self, event: &DocumentEvent) -> Result<()> {
        let mut last_err = None;
        for attempt in 1..=INGEST_ATTEMPTS {
            match self.ingestion.ingest(event).await {
                Ok(_) => return Ok(()),
                Err(e) => {
                    tracing::warn!("ingest attempt {attempt} failed: {e}");
                    last_err = Some(e);
                    if attempt < INGEST_ATTEMPTS {
                        tokio::time::sleep(RETRY_BACKOFF).await;
                    }
                }
            }
        }
        Err(last_err.expect("loop ran at least once"))
    }

    /// Forward a message to a dead-letter / parking topic with context headers.
    /// Failures here are only logged — we still commit and move
    /// on, since the alternative is wedging the whole feeder.
    async fn send_aside(
        &self,
        msg: &rdkafka::message::BorrowedMessage<'_>,
        topic: &str,
        reason: &str,
    ) {
        let partition = msg.partition().to_string();
        let offset = msg.offset().to_string();
        let headers = OwnedHeaders::new()
            .insert(Header {
                key: "x-reason",
                value: Some(reason),
            })
            .insert(Header {
                key: "x-consumer",
                value: Some(self.group_id.as_str()),
            })
            .insert(Header {
                key: "x-source-topic",
                value: Some(self.source_topic.as_str()),
            })
            .insert(Header {
                key: "x-source-partition",
                value: Some(partition.as_str()),
            })
            .insert(Header {
                key: "x-source-offset",
                value: Some(offset.as_str()),
            });

        let key = msg.key().unwrap_or_default();
        let payload = msg.payload().unwrap_or_default();
        let record = FutureRecord::to(topic)
            .key(key)
            .payload(payload)
            .headers(headers);

        if let Err((e, _)) = self.producer.send(record, Timeout::Never).await {
            tracing::warn!("failed to forward to {topic}: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tombstone_with_key_is_forget() {
        assert_eq!(
            decide(Some("doc_1"), None),
            FeedDecision::Forget("doc_1".into())
        );
    }

    #[test]
    fn test_tombstone_without_key_is_parked() {
        assert!(matches!(decide(None, None), FeedDecision::Park(_)));
        assert!(matches!(decide(Some(""), None), FeedDecision::Park(_)));
    }

    /// The self-contained event shape is ingested; the text travels with it.
    #[test]
    fn test_valid_event_is_ingest() {
        let json = br#"{"data_id":"doc_1","title":"Lease","text":"The lease runs to 2027.","tags":["x"],"ts":"2026-09-01T00:00:00Z"}"#;
        match decide(Some("doc_1"), Some(json)) {
            FeedDecision::Ingest(e) => {
                assert_eq!(e.document_id(), Some("doc_1"));
                assert_eq!(e.body(), Some("The lease runs to 2027."));
            }
            other => panic!("expected Ingest, got {other:?}"),
        }
    }

    /// No text (missing or blank) → skipped with a warning, not dead-lettered.
    #[test]
    fn test_event_without_text_is_skipped() {
        for json in [
            &br#"{"data_id":"doc_2"}"#[..],
            &br#"{"data_id":"doc_2","text":"   "}"#[..],
        ] {
            assert_eq!(
                decide(Some("k"), Some(json)),
                FeedDecision::Skip("doc_2".into())
            );
        }
    }

    /// The old pointer-based event (text fetched from an archive service) has
    /// no text and no data_id → dead-lettered, never fetched.
    #[test]
    fn test_event_without_id_is_bad_event() {
        let legacy = br#"{"groupId":"grp_1","pages":[{"textUri":"pt://archive/p/text"}]}"#;
        assert!(matches!(
            decide(Some("k"), Some(legacy)),
            FeedDecision::BadEvent(_)
        ));
    }

    #[test]
    fn test_unparseable_is_parked() {
        assert!(matches!(
            decide(Some("k"), Some(b"\xff\x00 not json")),
            FeedDecision::Park(_)
        ));
    }
}
