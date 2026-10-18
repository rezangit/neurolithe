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
//! the dead-letter topic; un-parseable bytes go to the parking topic.
//!
//! **Delivery semantics: at-least-once, with idempotent writes.** A message's
//! offset is stored and committed only once the message is *dealt with*: its
//! write succeeded, it was skipped, or it was durably forwarded to the
//! dead-letter / parking topic. Automatic offset storing is off, so even the
//! shutdown commit covers only handled messages. A crash, rebalance or
//! shutdown before that point means the message is delivered again, which is
//! safe because ingest is an upsert by `dataId` (see
//! `application::ingestion`): a duplicate changes nothing.
//!
//! **Poison messages never block the partition.** Un-parseable and invalid
//! events are set aside at once; an ingest or forget that keeps failing is
//! retried a bounded number of times ([`RetryPolicy`]) and then dead-lettered.
//! Only forwarding to the dead-letter / parking topic itself is retried until
//! it succeeds (or shutdown), because committing past a message that is
//! neither stored nor set aside would lose it.
//!
//! **Rewinds.** A hard reset seeks the feeder back to the start while it may
//! still be handling a message. Every rewind advances a [`RewindGeneration`];
//! a message received before the latest rewind is not committed afterwards
//! ([`offset_action`]), so its stale offset can't overwrite the rewind
//! position that the rewind itself stores and commits.
//!
//! The decision logic ([`decide`]) and the per-message handling ([`handle`])
//! are broker-free and unit-tested; the rdkafka loop itself needs a live
//! broker.

use crate::application::ingestion::{DocumentEvent, IngestOutcome, IngestionService};
use crate::application::monitoring::FeederStats;
use crate::infrastructure::config::KafkaConfig;
use crate::infrastructure::kafka_client::{
    StopSignal, base_config, commit_on_shutdown, consumer_config, flush_on_shutdown, recv_or_stop,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use rdkafka::config::ClientConfig;
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

/// Retry and backoff settings for one message.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Ingest/forget attempts before the message is dead-lettered. Bounded, so
    /// a permanently failing (poison) event can't block the partition.
    pub write_attempts: u32,
    /// Pause between write attempts.
    pub write_backoff: Duration,
    /// First pause between attempts to forward a message to the dead-letter /
    /// parking topic; doubles up to [`Self::aside_backoff_max`]. These retries
    /// are unbounded (until success or shutdown).
    pub aside_backoff: Duration,
    pub aside_backoff_max: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            write_attempts: 3,
            write_backoff: Duration::from_millis(500),
            aside_backoff: Duration::from_secs(1),
            aside_backoff_max: Duration::from_secs(30),
        }
    }
}

/// Where a message that can't be written goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsideTopic {
    /// Well-formed but unusable, or its write kept failing.
    DeadLetter,
    /// Un-parseable bytes.
    Parking,
}

/// Forwards the message being handled to a side topic. Not `Send`: the
/// feeder runs on the daemon's single-threaded `LocalSet`.
#[async_trait(?Send)]
pub trait SetAside {
    async fn set_aside(&self, topic: AsideTopic, reason: &str) -> Result<()>;
}

/// Whether a message's offset may be committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Written, skipped, or durably set aside: commit.
    Done,
    /// Shutdown came first: don't commit; the message is redelivered.
    Interrupted,
}

/// Sleep for `pause`, or return `true` early once `stop` fires (or its sender
/// is gone).
async fn pause_or_stop(stop: &mut StopSignal, pause: Duration) -> bool {
    if *stop.borrow() {
        return true;
    }
    tokio::select! {
        biased;
        changed = stop.changed() => changed.is_err() || *stop.borrow(),
        _ = tokio::time::sleep(pause) => false,
    }
}

/// A bounded write retry's result.
enum Attempted<T> {
    Ok(T),
    Failed(anyhow::Error),
    Stopped,
}

/// Run `write` up to `policy.write_attempts` times (pausing in between,
/// interruptible by `stop`).
async fn retry_write<T, F, Fut>(
    policy: &RetryPolicy,
    stop: &mut StopSignal,
    write: F,
) -> Attempted<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let attempts = policy.write_attempts.max(1);
    let mut attempt = 1;
    loop {
        match write().await {
            Ok(v) => return Attempted::Ok(v),
            Err(e) => {
                tracing::warn!("write attempt {attempt}/{attempts} failed: {e:#}");
                if attempt >= attempts {
                    return Attempted::Failed(e);
                }
            }
        }
        if pause_or_stop(stop, policy.write_backoff).await {
            return Attempted::Stopped;
        }
        attempt += 1;
    }
}

/// Forward to a side topic, retrying with capped exponential backoff until it
/// succeeds or `stop` fires.
async fn set_aside_until_done(
    aside: &dyn SetAside,
    topic: AsideTopic,
    reason: &str,
    policy: &RetryPolicy,
    stop: &mut StopSignal,
) -> Disposition {
    let mut pause = policy.aside_backoff;
    loop {
        match aside.set_aside(topic, reason).await {
            Ok(()) => return Disposition::Done,
            Err(e) => tracing::error!(
                "forwarding a message to the {topic:?} topic failed ({e:#}); \
                 retrying in {pause:?}, the offset is not committed meanwhile"
            ),
        }
        if pause_or_stop(stop, pause).await {
            return Disposition::Interrupted;
        }
        pause = (pause * 2).min(policy.aside_backoff_max);
    }
}

/// Handle one classified message: write it (bounded retries) or set it aside.
/// Broker-free; the feeder loop commits the offset when this returns
/// [`Disposition::Done`].
pub async fn handle(
    decision: FeedDecision,
    ingestion: &IngestionService,
    stats: &FeederStats,
    policy: &RetryPolicy,
    aside: &dyn SetAside,
    stop: &mut StopSignal,
) -> Disposition {
    let (topic, reason) = match decision {
        FeedDecision::Forget(id) => match retry_write(policy, stop, || ingestion.forget(&id)).await
        {
            Attempted::Ok(()) => return Disposition::Done,
            Attempted::Stopped => return Disposition::Interrupted,
            Attempted::Failed(e) => (AsideTopic::DeadLetter, format!("forget failed: {e:#}")),
        },
        FeedDecision::Ingest(event) => {
            match retry_write(policy, stop, || ingestion.ingest(&event)).await {
                Attempted::Ok(IngestOutcome::Ingested { .. }) => {
                    stats.record_document(now_unix());
                    return Disposition::Done;
                }
                Attempted::Ok(_) => return Disposition::Done,
                Attempted::Stopped => return Disposition::Interrupted,
                Attempted::Failed(e) => {
                    stats.record_error();
                    (AsideTopic::DeadLetter, format!("ingest failed: {e:#}"))
                }
            }
        }
        FeedDecision::Skip(id) => {
            tracing::warn!("document event '{id}' has no text; skipped");
            return Disposition::Done;
        }
        FeedDecision::BadEvent(reason) => (AsideTopic::DeadLetter, reason),
        FeedDecision::Park(reason) => (AsideTopic::Parking, reason),
    };
    set_aside_until_done(aside, topic, &reason, policy, stop).await
}

/// The documents consumer's config: manual commits (see [`consumer_config`])
/// and no automatic offset storing, so an offset is stored only once its
/// message is handled and the shutdown commit can't skip an unhandled one.
pub fn feeder_consumer_config(kafka: &KafkaConfig) -> ClientConfig {
    let mut cfg = consumer_config(kafka, &kafka.group_id, "earliest");
    cfg.set("enable.auto.offset.store", "false");
    cfg
}

/// Counts feeder rewinds. Shared by the feeder loop (which notes the value
/// when it receives a message) and the hard-reset rewind (which advances it).
#[derive(Debug, Clone, Default)]
pub struct RewindGeneration(Arc<std::sync::atomic::AtomicU64>);

impl RewindGeneration {
    pub fn current(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Start a new generation; returns it.
    pub fn advance(&self) -> u64 {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1
    }
}

/// What the feeder loop does with a message's offset once [`handle`] returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetAction {
    /// Store and commit it.
    Commit,
    /// The feeder was rewound while the message was handled: leave the offset
    /// alone, or it would overwrite the rewind position.
    DropStale,
    /// Shutdown: don't commit; stop the loop.
    Stop,
}

/// The offset action for a message received in generation `received` and
/// handled with `disposition`, when the rewind generation is now `current`.
pub fn offset_action(disposition: Disposition, received: u64, current: u64) -> OffsetAction {
    match disposition {
        Disposition::Interrupted => OffsetAction::Stop,
        Disposition::Done if received != current => OffsetAction::DropStale,
        Disposition::Done => OffsetAction::Commit,
    }
}

/// What a rewind needs: the feeder's consumer and its rewind generation.
#[derive(Clone)]
pub struct FeederConsumer {
    pub consumer: Arc<StreamConsumer>,
    pub generation: RewindGeneration,
}

pub struct KafkaFeeder {
    consumer: Arc<StreamConsumer>,
    generation: RewindGeneration,
    producer: FutureProducer,
    ingestion: Arc<IngestionService>,
    stats: Arc<FeederStats>,
    policy: RetryPolicy,
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
        let consumer: StreamConsumer = feeder_consumer_config(kafka)
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
            generation: RewindGeneration::default(),
            policy: RetryPolicy::default(),
            source_topic: kafka.topics.documents.clone(),
            dlq_topic: kafka.topics.dlq.clone(),
            parking_topic: kafka.topics.parking.clone(),
            group_id: kafka.group_id.clone(),
        })
    }

    /// Shared handle to the consumer and its rewind generation, so the
    /// command consumer can rewind it to earliest after a hard reset.
    pub fn consumer(&self) -> FeederConsumer {
        FeederConsumer {
            consumer: self.consumer.clone(),
            generation: self.generation.clone(),
        }
    }

    /// Consume until `stop` fires: classify -> handle -> store + commit the
    /// offset. Then commit the stored offsets synchronously and flush the
    /// dead-letter producer.
    pub async fn run(&self, mut stop: StopSignal) -> Result<()> {
        self.consumer
            .subscribe(&[&self.source_topic])
            .with_context(|| format!("subscribing to {}", self.source_topic))?;

        while let Some(next) = recv_or_stop(&self.consumer, &mut stop).await {
            let msg = match next {
                Err(e) => {
                    tracing::warn!("consumer error: {e}");
                    continue;
                }
                Ok(msg) => msg,
            };
            let received = self.generation.current();
            let key = msg.key().and_then(|k| std::str::from_utf8(k).ok());
            let decision = decide(key, msg.payload());
            let aside = MessageAside {
                feeder: self,
                msg: &msg,
            };
            let disposition = handle(
                decision,
                &self.ingestion,
                &self.stats,
                &self.policy,
                &aside,
                &mut stop,
            )
            .await;
            match offset_action(disposition, received, self.generation.current()) {
                OffsetAction::Commit => {
                    // Store + commit only now, AFTER the write (or the
                    // set-aside), so a crash re-processes rather than drops.
                    if let Err(e) = self.consumer.store_offset_from_message(&msg) {
                        tracing::warn!("storing offset failed: {e}");
                    }
                    if let Err(e) = self.consumer.commit_message(&msg, CommitMode::Async) {
                        tracing::warn!("commit failed: {e}");
                    }
                }
                OffsetAction::DropStale => {
                    tracing::info!(
                        "feeder rewound while offset {} (partition {}) was handled; \
                         its commit is dropped",
                        msg.offset(),
                        msg.partition()
                    );
                }
                OffsetAction::Stop => {
                    tracing::info!(
                        "shutdown before offset {} (partition {}) was handled; \
                         it will be redelivered",
                        msg.offset(),
                        msg.partition()
                    );
                    break;
                }
            }
        }
        commit_on_shutdown(&self.consumer, "documents feeder");
        flush_on_shutdown(&self.producer, "documents feeder");
        Ok(())
    }

    /// Forward a message to a dead-letter / parking topic with context headers.
    async fn send_aside(
        &self,
        msg: &rdkafka::message::BorrowedMessage<'_>,
        topic: &str,
        reason: &str,
    ) -> Result<()> {
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

        self.producer
            .send(record, Timeout::After(SEND_TIMEOUT))
            .await
            .map(|_| ())
            .map_err(|(e, _)| anyhow::anyhow!("producing to {topic}: {e}"))
    }
}

/// How long one forward to a side topic may wait for delivery before it
/// counts as failed (and is retried).
const SEND_TIMEOUT: Duration = Duration::from_secs(30);

/// [`SetAside`] for the message the feeder loop is handling.
struct MessageAside<'a> {
    feeder: &'a KafkaFeeder,
    msg: &'a rdkafka::message::BorrowedMessage<'a>,
}

#[async_trait(?Send)]
impl SetAside for MessageAside<'_> {
    async fn set_aside(&self, topic: AsideTopic, reason: &str) -> Result<()> {
        let name = match topic {
            AsideTopic::DeadLetter => &self.feeder.dlq_topic,
            AsideTopic::Parking => &self.feeder.parking_topic,
        };
        self.feeder.send_aside(self.msg, name, reason).await
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

    // --- Per-message handling (broker-free) -----------------------------------

    use crate::application::ingestion::tests::{NoChatLlm, harness_with, text_event};
    use crate::domain::ltm::LtmRepository;
    use crate::domain::ports::{Embedder, MemoryRepository, Models};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Records every set-aside; the first `fail_first` calls fail (`usize::MAX`
    /// = always).
    #[derive(Default)]
    struct RecordingAside {
        fail_first: usize,
        calls: AtomicUsize,
        sent: Mutex<Vec<(AsideTopic, String)>>,
    }

    #[async_trait(?Send)]
    impl SetAside for RecordingAside {
        async fn set_aside(&self, topic: AsideTopic, reason: &str) -> Result<()> {
            if self.calls.fetch_add(1, Ordering::SeqCst) < self.fail_first {
                anyhow::bail!("broker unavailable");
            }
            self.sent.lock().unwrap().push((topic, reason.to_string()));
            Ok(())
        }
    }

    /// The embedder is down: every ingest fails.
    struct FailingLlm {
        embeds: AtomicUsize,
    }
    #[async_trait]
    impl Embedder for FailingLlm {
        async fn embed_text(&self, _t: &str) -> Result<Vec<f32>> {
            self.embeds.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("embedder unreachable")
        }
    }

    fn fast_policy() -> RetryPolicy {
        RetryPolicy {
            write_attempts: 3,
            write_backoff: Duration::from_millis(1),
            aside_backoff: Duration::from_millis(1),
            aside_backoff_max: Duration::from_millis(4),
        }
    }

    fn running() -> (tokio::sync::watch::Sender<bool>, StopSignal) {
        tokio::sync::watch::channel(false)
    }

    fn event_bytes(data_id: &str, text: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({"data_id": data_id, "text": text})).unwrap()
    }

    /// Un-parseable bytes are parked once and the offset may be committed: a
    /// poison message never blocks the partition.
    #[tokio::test]
    async fn test_poison_message_is_parked_and_committed() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        let stats = FeederStats::default();
        let aside = RecordingAside::default();
        let (_tx, mut stop) = running();

        let d = decide(Some("k"), Some(b"\xff\x00 not json"));
        let disposition = handle(d, &h.svc, &stats, &fast_policy(), &aside, &mut stop).await;

        assert_eq!(disposition, Disposition::Done);
        let sent = aside.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, AsideTopic::Parking);
    }

    /// An event without a data_id goes to the dead-letter topic.
    #[tokio::test]
    async fn test_bad_event_is_dead_lettered() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        let aside = RecordingAside::default();
        let (_tx, mut stop) = running();
        let d = decide(Some("k"), Some(br#"{"text":"orphan"}"#));
        let disposition = handle(
            d,
            &h.svc,
            &FeederStats::default(),
            &fast_policy(),
            &aside,
            &mut stop,
        )
        .await;
        assert_eq!(disposition, Disposition::Done);
        assert_eq!(aside.sent.lock().unwrap()[0].0, AsideTopic::DeadLetter);
    }

    /// A permanently failing ingest is retried a bounded number of times, then
    /// dead-lettered (with the reason) and committed.
    #[tokio::test]
    async fn test_failing_ingest_is_retried_then_dead_lettered() {
        let llm = Arc::new(FailingLlm {
            embeds: AtomicUsize::new(0),
        });
        let h = harness_with(Models::embed_only(llm.clone()));
        let stats = FeederStats::default();
        let aside = RecordingAside::default();
        let (_tx, mut stop) = running();

        let bytes = event_bytes("doc_1", "text");
        let d = decide(Some("doc_1"), Some(&bytes));
        let disposition = handle(d, &h.svc, &stats, &fast_policy(), &aside, &mut stop).await;

        assert_eq!(disposition, Disposition::Done);
        assert_eq!(llm.embeds.load(Ordering::SeqCst), 3, "three attempts");
        let sent = aside.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, AsideTopic::DeadLetter);
        assert!(sent[0].1.contains("embedder unreachable"), "{}", sent[0].1);
        assert_eq!(stats.to_runtime(0).feeder_errors, 1);
    }

    /// A valid event is written and committed; its redelivery is a no-op that
    /// is committed too and not counted as a new ingest.
    #[tokio::test]
    async fn test_valid_event_then_redelivery() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        let stats = FeederStats::default();
        let aside = RecordingAside::default();
        let (_tx, mut stop) = running();
        let bytes = event_bytes("doc_1", "The lease runs to 2027.");

        for _ in 0..2 {
            let d = decide(Some("doc_1"), Some(&bytes));
            let disposition = handle(d, &h.svc, &stats, &fast_policy(), &aside, &mut stop).await;
            assert_eq!(disposition, Disposition::Done);
        }
        assert!(aside.sent.lock().unwrap().is_empty());
        assert_eq!(h.ltm.get_nodes_by_data_id("doc_1").unwrap().len(), 1);
        assert_eq!(h.stm.count_by_data_id("doc_1").unwrap(), 1);
        assert!(stats.to_runtime(0).last_backfill_unix.is_some());
        assert_eq!(stats.to_runtime(0).feeder_errors, 0);
    }

    /// A tombstone forgets the document; a repeated tombstone is harmless.
    #[tokio::test]
    async fn test_tombstone_forgets_idempotently() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        h.svc.ingest(&text_event("doc_1")).await.unwrap();
        let aside = RecordingAside::default();
        let (_tx, mut stop) = running();
        for _ in 0..2 {
            let disposition = handle(
                decide(Some("doc_1"), None),
                &h.svc,
                &FeederStats::default(),
                &fast_policy(),
                &aside,
                &mut stop,
            )
            .await;
            assert_eq!(disposition, Disposition::Done);
        }
        assert!(h.ltm.get_nodes_by_data_id("doc_1").unwrap().is_empty());
        assert_eq!(h.stm.count_by_data_id("doc_1").unwrap(), 0);
        assert!(aside.sent.lock().unwrap().is_empty());
    }

    /// Forwarding to the side topic is retried until it succeeds: committing
    /// before that would lose the message.
    #[tokio::test]
    async fn test_set_aside_failure_is_retried_until_it_lands() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        let aside = RecordingAside {
            fail_first: 3,
            ..Default::default()
        };
        let (_tx, mut stop) = running();
        let disposition = handle(
            FeedDecision::Park("junk".into()),
            &h.svc,
            &FeederStats::default(),
            &fast_policy(),
            &aside,
            &mut stop,
        )
        .await;
        assert_eq!(disposition, Disposition::Done);
        assert_eq!(aside.calls.load(Ordering::SeqCst), 4);
        assert_eq!(aside.sent.lock().unwrap().len(), 1);
    }

    /// If the side topic stays unreachable, shutdown interrupts the retries
    /// and the offset is NOT committed (the message is redelivered).
    #[tokio::test]
    async fn test_shutdown_while_set_aside_fails_is_not_committed() {
        let h = harness_with(Models::embed_only(Arc::new(NoChatLlm)));
        let aside = RecordingAside {
            fail_first: usize::MAX,
            ..Default::default()
        };
        let (tx, mut stop) = running();
        let stats = FeederStats::default();
        let policy = fast_policy();
        let stopper = async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            tx.send(true).unwrap();
        };
        let (disposition, ()) = tokio::join!(
            handle(
                FeedDecision::Park("junk".into()),
                &h.svc,
                &stats,
                &policy,
                &aside,
                &mut stop,
            ),
            stopper
        );
        assert_eq!(disposition, Disposition::Interrupted);
        assert!(aside.calls.load(Ordering::SeqCst) >= 2, "it kept retrying");
        assert!(aside.sent.lock().unwrap().is_empty());
    }

    /// Shutdown during the ingest retries: not committed, not dead-lettered.
    #[tokio::test]
    async fn test_shutdown_during_ingest_retry_is_not_committed() {
        let h = harness_with(Models::embed_only(Arc::new(FailingLlm {
            embeds: AtomicUsize::new(0),
        })));
        let aside = RecordingAside::default();
        let (tx, mut stop) = running();
        tx.send(true).unwrap();
        let bytes = event_bytes("doc_1", "text");
        let disposition = handle(
            decide(Some("doc_1"), Some(&bytes)),
            &h.svc,
            &FeederStats::default(),
            &RetryPolicy {
                write_backoff: Duration::from_secs(60),
                ..fast_policy()
            },
            &aside,
            &mut stop,
        )
        .await;
        assert_eq!(disposition, Disposition::Interrupted);
        assert!(aside.sent.lock().unwrap().is_empty());
    }

    /// The feeder stores offsets itself (after handling), on top of manual
    /// commits, so the shutdown commit never covers an unhandled message.
    #[test]
    fn test_feeder_consumer_stores_offsets_manually() {
        let kafka = crate::infrastructure::kafka_client::tests::test_kafka(&[(
            "enable.auto.offset.store",
            "true",
        )]);
        let cfg = feeder_consumer_config(&kafka);
        assert_eq!(cfg.get("enable.auto.offset.store"), Some("false"));
        assert_eq!(cfg.get("enable.auto.commit"), Some("false"));
        assert_eq!(cfg.get("auto.offset.reset"), Some("earliest"));
        assert_eq!(cfg.get("group.id"), Some("nl-test"));
    }

    /// W2R-5: a message received before a rewind is not committed after it
    /// (its offset would overwrite the rewind position); shutdown never
    /// commits.
    #[test]
    fn test_offset_action_drops_commits_from_before_a_rewind() {
        let generation = RewindGeneration::default();
        let received = generation.current();
        assert_eq!(
            offset_action(Disposition::Done, received, generation.current()),
            OffsetAction::Commit
        );

        // A hard reset rewinds the feeder while the message is handled.
        let rewind_side = generation.clone();
        assert_eq!(rewind_side.advance(), received + 1);
        assert_eq!(
            offset_action(Disposition::Done, received, generation.current()),
            OffsetAction::DropStale
        );
        // The replay's messages (received after the rewind) commit again.
        let replayed = generation.current();
        assert_eq!(
            offset_action(Disposition::Done, replayed, generation.current()),
            OffsetAction::Commit
        );
        for current in [received, received + 1] {
            assert_eq!(
                offset_action(Disposition::Interrupted, received, current),
                OffsetAction::Stop
            );
        }
    }
}
