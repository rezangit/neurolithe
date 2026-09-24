//! Command consumer — drives writes and soft/hard reset from the commands topic
//! (`[kafka.topics] commands`, default `memory.command`).
//!
//! Parses each command and dispatches to [`ResetService`] / [`WriteService`].
//! On a hard reset it then rewinds the feeder (via [`FeederRewind`]) so the
//! documents topic replays and the stores are re-derived. The command channel
//! is short-retention (not a replay source); each command is processed once.
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

/// Rewinds a shared feeder consumer to the beginning of its assigned partitions.
pub struct ConsumerRewind {
    consumer: Arc<StreamConsumer>,
}

impl ConsumerRewind {
    pub fn new(consumer: Arc<StreamConsumer>) -> Self {
        Self { consumer }
    }
}

#[async_trait]
impl FeederRewind for ConsumerRewind {
    async fn rewind_to_earliest(&self) -> Result<()> {
        let assignment = self
            .consumer
            .assignment()
            .context("reading feeder assignment")?;
        let mut tpl = TopicPartitionList::new();
        for elem in assignment.elements() {
            tpl.add_partition_offset(elem.topic(), elem.partition(), Offset::Beginning)
                .context("building rewind offsets")?;
        }
        self.consumer
            .seek_partitions(tpl, Timeout::After(Duration::from_secs(5)))
            .context("seeking feeder to earliest")?;
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
    consumer: StreamConsumer,
    reset: Arc<ResetService>,
    write: Arc<WriteService>,
    rewind: Arc<dyn FeederRewind>,
    spine_embedder: Arc<dyn SpineEmbedder>,
    topic: String,
}

impl CommandConsumer {
    pub fn new(
        kafka: &KafkaConfig,
        reset: Arc<ResetService>,
        write: Arc<WriteService>,
        rewind: Arc<dyn FeederRewind>,
        spine_embedder: Arc<dyn SpineEmbedder>,
    ) -> Result<Self> {
        let consumer: StreamConsumer = consumer_config(kafka, &command_group(kafka), "earliest")
            .create()
            .context("creating commands consumer")?;

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
        })
    }

    /// Consume until `stop` fires: parse -> apply -> (rewind on hard) -> commit,
    /// then commit offsets synchronously.
    pub async fn run(&self, mut stop: StopSignal) -> Result<()> {
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
