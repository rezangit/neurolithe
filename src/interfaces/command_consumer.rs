//! Command consumer — drives writes and soft/hard reset from the commands topic
//! (`[kafka.topics] commands`, default `memory.command`).
//!
//! Parses each command and dispatches to [`ResetService`] / [`WriteService`].
//! On a hard reset it then rewinds the feeder (via [`FeederRewind`]) so the
//! documents topic replays and the stores are re-derived. The command channel
//! is not a replay source: a new consumer group starts at the end of the topic
//! (see [`command_consumer_config`]), and each command is processed once.
//!
//! Security: anyone who can produce to this topic can write, forget and soft
//! reset. Restrict it with broker ACLs (and SASL/TLS via `[kafka.client]`).
//! Hard reset additionally needs `NEUROLITHE_RESET_TOKEN` (≥16 chars) and is
//! disabled without it.
//!
//! The rdkafka plumbing needs a live broker; the reset logic itself is
//! unit-tested in `reset_service`.

use crate::application::reset_service::{MemoryCommand, ResetKind, ResetService};
use crate::application::write_service::WriteService;
use crate::infrastructure::config::KafkaConfig;
use crate::infrastructure::kafka_client::{
    StopSignal, command_group, commit_on_shutdown, consumer_config, recv_or_stop,
};
use crate::interfaces::kafka_feeder::FeederConsumer;
use anyhow::{Context, Result};
use async_trait::async_trait;
use rdkafka::consumer::{CommitMode, Consumer, StreamConsumer};
use rdkafka::message::Message;
use rdkafka::util::Timeout;
use rdkafka::{Offset, TopicPartitionList};
use std::sync::Arc;
use std::time::Duration;

/// Rewinds the feeder's consumer group to the earliest offset, so a hard reset
/// replays the documents topic and re-derives LTM (+ STM). Implemented by the
/// daemon, which owns the feeder consumer.
#[async_trait]
pub trait FeederRewind: Send + Sync {
    async fn rewind_to_earliest(&self) -> Result<()>;
}

/// Rewinds a shared feeder consumer to the beginning of its assigned
/// partitions.
///
/// The rewind is made durable at once (W2R-5): it advances the feeder's
/// [`RewindGeneration`] first, so a message the feeder is still handling isn't
/// committed afterwards, then seeks to the beginning, and finally stores and
/// commits offset 0 for every assigned partition. A restart before the replay
/// commits anything therefore resumes from the start instead of from the old
/// position. Offset 0 below the log start (after retention) is out of range,
/// and the feeder's `auto.offset.reset = earliest` maps it to the earliest
/// offset.
pub struct ConsumerRewind {
    feeder: FeederConsumer,
}

impl ConsumerRewind {
    pub fn new(feeder: FeederConsumer) -> Self {
        Self { feeder }
    }
}

/// `offset` for every partition in `assignment`.
pub fn rewind_list(assignment: &TopicPartitionList, offset: Offset) -> Result<TopicPartitionList> {
    let mut tpl = TopicPartitionList::new();
    for elem in assignment.elements() {
        tpl.add_partition_offset(elem.topic(), elem.partition(), offset)
            .context("building rewind offsets")?;
    }
    Ok(tpl)
}

#[async_trait]
impl FeederRewind for ConsumerRewind {
    async fn rewind_to_earliest(&self) -> Result<()> {
        let consumer = &self.feeder.consumer;
        let assignment = consumer.assignment().context("reading feeder assignment")?;
        // Before anything else, so an in-flight message can't commit over the
        // rewind position (even if a step below fails).
        self.feeder.generation.advance();
        if assignment.count() == 0 {
            return Ok(());
        }
        consumer
            .seek_partitions(
                rewind_list(&assignment, Offset::Beginning)?,
                Timeout::After(Duration::from_secs(5)),
            )
            .context("seeking feeder to earliest")?;
        let start = rewind_list(&assignment, Offset::Offset(0))?;
        consumer
            .store_offsets(&start)
            .context("storing the rewind position")?;
        consumer
            .commit(&start, CommitMode::Sync)
            .context("committing the rewind position")?;
        Ok(())
    }
}

/// No-op rewind for when the feeder is disabled (hard reset still wipes stores).
pub struct NoopRewind;

#[async_trait]
impl FeederRewind for NoopRewind {
    async fn rewind_to_earliest(&self) -> Result<()> {
        Ok(())
    }
}

/// Re-embeds the curated spine after a hard reset. `hard_reset` re-seeds the
/// spine but seeding creates concept nodes *without* vectors; without this step
/// placement would be blind and the replay would file every document into the
/// inbox (re-introduced by every reset). Implemented by
/// the daemon, which owns the embedder + LTM store.
///
/// `?Send`: the concrete implementor holds the SQLite-backed LTM repo (`!Sync`),
/// so — like the daemon's other loops — it runs on the single-threaded `LocalSet`.
#[async_trait(?Send)]
pub trait SpineEmbedder {
    async fn embed_spine(&self) -> Result<()>;
}

/// No-op embedder for when the feeder is disabled (no ingestion → placement not
/// exercised) or in tests.
pub struct NoopSpineEmbedder;

#[async_trait(?Send)]
impl SpineEmbedder for NoopSpineEmbedder {
    async fn embed_spine(&self) -> Result<()> {
        Ok(())
    }
}

pub struct CommandConsumer {
    consumer: Arc<StreamConsumer>,
    reset: Arc<ResetService>,
    write: Arc<WriteService>,
    rewind: Arc<dyn FeederRewind>,
    spine_embedder: Arc<dyn SpineEmbedder>,
    topic: String,
    group: String,
}

/// The commands consumer's config. A group without a committed offset (first
/// start, a renamed group, or offsets the broker expired after
/// `offsets.retention.minutes`) starts at the **end** of the topic: commands
/// are imperative, not a replay source, so retained old `reset_soft`,
/// `forget` and `remember` commands are never re-run against the current
/// store. The cost: commands sent while the group has no offset yet are
/// skipped. A group with committed offsets resumes where it stopped, as before.
pub fn command_consumer_config(kafka: &KafkaConfig) -> rdkafka::config::ClientConfig {
    consumer_config(kafka, &command_group(kafka), "latest")
}

/// Where the command group starts reading one partition of the topic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartPosition {
    /// The group's committed offset.
    Resume(i64),
    /// No committed offset: the current end of the topic, committed once so
    /// that later restarts resume from it.
    PinAtEnd(i64),
}

/// Plan the start for each `(partition, committed offset, end offset)`.
pub fn plan_command_start(partitions: &[(i32, Option<i64>, i64)]) -> Vec<(i32, StartPosition)> {
    partitions
        .iter()
        .map(|&(p, committed, end)| {
            (
                p,
                match committed {
                    Some(offset) => StartPosition::Resume(offset),
                    None => StartPosition::PinAtEnd(end),
                },
            )
        })
        .collect()
}

/// The startup log line for a start plan.
pub fn describe_command_start(group: &str, topic: &str, plan: &[(i32, StartPosition)]) -> String {
    let parts: Vec<String> = plan
        .iter()
        .map(|(p, start)| match start {
            StartPosition::Resume(o) => format!("partition {p} resumes at offset {o}"),
            StartPosition::PinAtEnd(o) => format!(
                "partition {p} had no committed offset, starts at the end (offset {o}; \
                 older commands are skipped)"
            ),
        })
        .collect();
    format!(
        "command consumer: group '{group}' on '{topic}': {}",
        parts.join("; ")
    )
}

/// The topic's partitions, or an error when it has none (the topic doesn't
/// exist yet): the caller logs a warning and the daemon starts anyway.
fn require_partitions(topic: &str, partitions: Vec<i32>) -> Result<Vec<i32>> {
    if partitions.is_empty() {
        anyhow::bail!("topic '{topic}' not found (or it has no partitions)");
    }
    Ok(partitions)
}

/// Read each partition's committed offset and end offset for the command
/// group, commit the end for partitions without an offset, and log the plan.
/// Partitions that already have a committed offset are left untouched.
/// Blocking (broker round trips). It runs before this consumer subscribes, so
/// the commit can't race its own join; if another member already holds the
/// group, the broker rejects the commit and the caller logs a warning.
fn pin_command_start(consumer: &StreamConsumer, group: &str, topic: &str) -> Result<String> {
    let timeout = Timeout::After(START_PROBE_TIMEOUT);
    let metadata = consumer
        .fetch_metadata(Some(topic), timeout)
        .context("reading topic metadata")?;
    let partitions = require_partitions(
        topic,
        metadata
            .topics()
            .iter()
            .filter(|t| t.name() == topic && t.error().is_none())
            .flat_map(|t| t.partitions().iter().map(|p| p.id()))
            .collect(),
    )?;
    let mut tpl = TopicPartitionList::new();
    for p in &partitions {
        tpl.add_partition(topic, *p);
    }
    let committed = consumer
        .committed_offsets(tpl, timeout)
        .context("reading committed offsets")?;
    let mut probe = Vec::new();
    for p in partitions {
        let offset = committed
            .find_partition(topic, p)
            .and_then(|e| match e.offset() {
                Offset::Offset(o) => Some(o),
                _ => None,
            });
        let (_low, high) = consumer
            .fetch_watermarks(topic, p, timeout)
            .context("reading the topic's end offsets")?;
        probe.push((p, offset, high));
    }
    let plan = plan_command_start(&probe);
    let mut pins = TopicPartitionList::new();
    for (p, start) in &plan {
        if let StartPosition::PinAtEnd(o) = start {
            pins.add_partition_offset(topic, *p, Offset::Offset(*o))
                .context("building start offsets")?;
        }
    }
    if pins.count() > 0 {
        consumer
            .commit(&pins, CommitMode::Sync)
            .context("committing the start position")?;
    }
    Ok(describe_command_start(group, topic, &plan))
}

/// Per broker call while probing the command group's start position.
const START_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
/// Probe attempts, and the pause between them. A group's first request on a
/// fresh cluster can fail with a transient `NotCoordinator` while the broker
/// creates its offsets topic.
const START_PROBE_ATTEMPTS: u32 = 5;
const START_PROBE_PAUSE: Duration = Duration::from_secs(1);

/// Run `f` up to `attempts` times (blocking `pause` in between), returning its
/// first success or the last error.
fn retry_blocking<T>(
    attempts: u32,
    pause: Duration,
    mut f: impl FnMut() -> Result<T>,
) -> Result<T> {
    let mut attempt = 1;
    loop {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) if attempt >= attempts => return Err(e),
            Err(e) => {
                tracing::debug!("command start probe attempt {attempt}/{attempts} failed: {e:#}");
                std::thread::sleep(pause);
                attempt += 1;
            }
        }
    }
}

impl CommandConsumer {
    pub fn new(
        kafka: &KafkaConfig,
        reset: Arc<ResetService>,
        write: Arc<WriteService>,
        rewind: Arc<dyn FeederRewind>,
        spine_embedder: Arc<dyn SpineEmbedder>,
    ) -> Result<Self> {
        let consumer: StreamConsumer = command_consumer_config(kafka)
            .create()
            .context("creating commands consumer")?;
        let consumer = Arc::new(consumer);

        if !reset.hard_reset_enabled() {
            tracing::warn!(
                "hard reset is disabled (NEUROLITHE_RESET_TOKEN unset or shorter \
                 than 16 chars)"
            );
        }

        Ok(Self {
            consumer,
            reset,
            write,
            rewind,
            spine_embedder,
            topic: kafka.topics.commands.clone(),
            group: command_group(kafka),
        })
    }

    /// Consume until `stop` fires: parse -> apply -> (rewind on hard) -> commit,
    /// then commit offsets synchronously.
    pub async fn run(&self, mut stop: StopSignal) -> Result<()> {
        // Before joining: pin a group without a committed offset to the end of
        // the topic (W5-K1), and log where each partition starts. Off the
        // event loop, since it waits on the broker.
        let (consumer, group, topic) = (
            self.consumer.clone(),
            self.group.clone(),
            self.topic.clone(),
        );
        match tokio::task::spawn_blocking(move || {
            retry_blocking(START_PROBE_ATTEMPTS, START_PROBE_PAUSE, || {
                pin_command_start(&consumer, &group, &topic)
            })
        })
        .await
        {
            Ok(Ok(line)) => tracing::info!("{line}"),
            Ok(Err(e)) => tracing::warn!(
                "could not read the command group's start position ({e:#}); a partition \
                 without a committed offset starts at the end of {}",
                self.topic
            ),
            Err(e) => tracing::warn!("command start probe failed: {e}"),
        }
        self.consumer
            .subscribe(&[&self.topic])
            .with_context(|| format!("subscribing to {}", self.topic))?;

        while let Some(next) = recv_or_stop(&self.consumer, &mut stop).await {
            match next {
                Err(e) => tracing::warn!("command consumer error: {e}"),
                Ok(msg) => {
                    self.process(&msg).await;
                    if let Err(e) = self.consumer.commit_message(&msg, CommitMode::Async) {
                        tracing::warn!("command commit failed: {e}");
                    }
                }
            }
        }
        commit_on_shutdown(&self.consumer, "command consumer");
        Ok(())
    }

    async fn process(&self, msg: &rdkafka::message::BorrowedMessage<'_>) {
        let Some(payload) = msg.payload() else {
            return; // null command — nothing to do
        };
        let command = match MemoryCommand::parse(payload) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("invalid command on {}: {e}", self.topic);
                return;
            }
        };

        // Route by variant: resets go to ResetService (rewinding the feeder on a
        // hard reset), writes go to the idempotent WriteService.
        match &command {
            MemoryCommand::ResetSoft | MemoryCommand::ResetHard { .. } => {
                match self.reset.apply(&command) {
                    // Hard reset wiped both stores and re-seeded the spine —
                    // re-embed it (else placement is blind), THEN rewind so the
                    // replay files documents under concepts, not the inbox.
                    Ok(ResetKind::Hard) => {
                        if let Err(e) = self.spine_embedder.embed_spine().await {
                            tracing::warn!("spine re-embed after hard reset failed: {e}");
                        }
                        if let Err(e) = self.rewind.rewind_to_earliest().await {
                            tracing::warn!("feeder rewind after hard reset failed: {e}");
                        }
                    }
                    Ok(ResetKind::Soft) => {}
                    // e.g. confirmation token mismatch — refuse and keep running.
                    Err(e) => tracing::warn!("reset refused: {e}"),
                }
            }
            MemoryCommand::Remember(_) | MemoryCommand::Forget(_) => {
                // A write failure is logged, not retried in-loop; redelivery
                // re-applies (the commandId isn't marked until success).
                if let Err(e) = self.write.handle(&command).await {
                    tracing::warn!("memory write failed: {e}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::kafka_client::tests::test_kafka;
    use crate::interfaces::kafka_feeder::{RewindGeneration, feeder_consumer_config};

    /// The rewind targets every assigned partition, whatever its position.
    #[test]
    fn test_rewind_list_covers_every_assigned_partition() {
        let mut assignment = TopicPartitionList::new();
        assignment
            .add_partition_offset("docs", 0, Offset::Offset(41))
            .unwrap();
        assignment
            .add_partition_offset("docs", 3, Offset::Offset(7))
            .unwrap();

        let start = rewind_list(&assignment, Offset::Offset(0)).unwrap();
        let got: Vec<(String, i32, Offset)> = start
            .elements()
            .iter()
            .map(|e| (e.topic().to_string(), e.partition(), e.offset()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("docs".to_string(), 0, Offset::Offset(0)),
                ("docs".to_string(), 3, Offset::Offset(0)),
            ]
        );
        let seek = rewind_list(&assignment, Offset::Beginning).unwrap();
        assert!(
            seek.elements()
                .iter()
                .all(|e| e.offset() == Offset::Beginning)
        );
    }

    /// W2R-5: a rewind always advances the feeder's generation, even before
    /// the consumer has an assignment (no broker here), so an in-flight
    /// message's commit is dropped.
    #[tokio::test]
    async fn test_rewind_advances_the_generation() {
        let consumer: StreamConsumer = feeder_consumer_config(&test_kafka(&[])).create().unwrap();
        let generation = RewindGeneration::default();
        let rewind = ConsumerRewind::new(FeederConsumer {
            consumer: Arc::new(consumer),
            generation: generation.clone(),
        });
        let before = generation.current();
        rewind.rewind_to_earliest().await.unwrap();
        assert_eq!(generation.current(), before + 1);
    }

    /// W5-K1: a commands group without a committed offset (first deploy, a
    /// renamed group, or offsets expired after the broker's retention) must
    /// start at the end of the topic. Commands are imperative, not a replay
    /// source: replaying retained ones would re-run old `reset_soft`s (no token
    /// needed), `forget`s and `remember`s against the current store.
    #[test]
    fn test_command_consumer_starts_at_latest() {
        let cfg = command_consumer_config(&test_kafka(&[("auto.offset.reset", "earliest")]));
        assert_eq!(cfg.get("auto.offset.reset"), Some("latest"));
        assert_eq!(cfg.get("group.id"), Some("nl-test-cmd"));
        assert_eq!(cfg.get("enable.auto.commit"), Some("false"));
    }

    /// A partition with a committed offset resumes there; one without is
    /// pinned at the current end of the topic, so later restarts resume from
    /// it instead of skipping the commands sent while the daemon was down.
    #[test]
    fn test_plan_command_start_pins_partitions_without_an_offset() {
        let plan = plan_command_start(&[(0, Some(12), 20), (1, None, 3), (2, None, 0)]);
        assert_eq!(
            plan,
            vec![
                (0, StartPosition::Resume(12)),
                (1, StartPosition::PinAtEnd(3)),
                (2, StartPosition::PinAtEnd(0)),
            ]
        );
        assert!(plan_command_start(&[]).is_empty());
    }

    /// The startup log names the group, the topic and each partition's start.
    #[test]
    fn test_describe_command_start() {
        let line = describe_command_start(
            "nl-cmd",
            "memory.command",
            &[
                (0, StartPosition::Resume(12)),
                (1, StartPosition::PinAtEnd(3)),
            ],
        );
        assert_eq!(
            line,
            "command consumer: group 'nl-cmd' on 'memory.command': partition 0 \
             resumes at offset 12; partition 1 had no committed offset, starts at \
             the end (offset 3; older commands are skipped)"
        );
    }

    /// The start probe is retried on transient errors (e.g. `NotCoordinator`
    /// on a group's first request) and gives up after the last attempt.
    #[test]
    fn test_retry_blocking() {
        let mut calls = 0;
        let ok = retry_blocking(5, Duration::ZERO, || {
            calls += 1;
            if calls < 3 {
                anyhow::bail!("NotCoordinator")
            }
            Ok(calls)
        });
        assert_eq!(ok.unwrap(), 3);

        let mut calls = 0;
        let err = retry_blocking(4, Duration::ZERO, || -> Result<()> {
            calls += 1;
            anyhow::bail!("down {calls}")
        });
        assert_eq!(err.unwrap_err().to_string(), "down 4");
        assert_eq!(calls, 4);
    }

    /// A missing commands topic is an error for the probe (logged as a
    /// warning), not a silent empty plan.
    #[test]
    fn test_require_partitions() {
        assert_eq!(
            require_partitions("memory.command", vec![0, 1]).unwrap(),
            vec![0, 1]
        );
        let err = require_partitions("memory.command", vec![]).unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");
    }
}
