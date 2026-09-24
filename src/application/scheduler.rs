//! Background scheduler — runs periodic maintenance tasks (STM decay sweep
//! now; metrics snapshot in slice 9) on fixed intervals.
//!
//! The loop is generic over [`PeriodicTask`] so the timing mechanism is unit
//! tested with a mock under a paused clock, independent of the real app. The
//! daemon (slice 11) spawns [`run_periodic`] with a [`SweepTask`]; it is not
//! wired into the current stdio server, whose single-threaded loop does not yet
//! serialize concurrent access to the SQLite connection.

use crate::application::app::NeurolitheApp;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

/// A unit of maintenance work the scheduler runs on a fixed interval.
///
/// Implementations own their error handling — a single failed run must log and
/// continue, never abort the schedule — so `run_once` cannot fail.
///
/// `?Send`: the app's SQLite/LLM internals are not `Sync`, so sweep futures are
/// not `Send`. The scheduler therefore runs on a single execution context; the
/// daemon's threading model (serialized connection access) lands in slice 11.
#[async_trait(?Send)]
pub trait PeriodicTask {
    async fn run_once(&self);
}

/// Smallest interval [`run_periodic`] will schedule.
pub const MIN_INTERVAL: Duration = Duration::from_secs(1);

/// Run `task` every `interval`, forever. The first tick fires immediately, then
/// every `interval` thereafter (Tokio's default `MissedTickBehavior::Burst`).
///
/// A zero `interval` would panic `tokio::time::interval`; config validation
/// rejects it, and this floors it to [`MIN_INTERVAL`] as a last line of defence
/// so a bad value can never take the daemon down (DEV-12).
pub async fn run_periodic(task: Arc<dyn PeriodicTask>, interval: Duration) {
    let mut ticker = tokio::time::interval(interval.max(MIN_INTERVAL));
    loop {
        ticker.tick().await;
        task.run_once().await;
    }
}

/// Adapts the STM decay sweep to [`PeriodicTask`]. Sweep failures are logged to
/// stderr and swallowed so a transient DB error never stops the schedule.
pub struct SweepTask {
    app: Arc<NeurolitheApp>,
}

impl SweepTask {
    pub fn new(app: Arc<NeurolitheApp>) -> Self {
        Self { app }
    }
}

/// How long a processed `commandId` is retained before the sweep drops it —
/// only needs to outlast Kafka's redelivery window, so two weeks is ample.
const COMMAND_ID_RETENTION_DAYS: i64 = 14;

#[async_trait(?Send)]
impl PeriodicTask for SweepTask {
    async fn run_once(&self) {
        if let Err(e) = self.app.run_decay_sweep().await {
            eprintln!("[neurolithe] decay sweep failed: {e}");
        }
        if let Err(e) = self.app.sweep_processed_commands(COMMAND_ID_RETENTION_DAYS) {
            eprintln!("[neurolithe] processed-command sweep failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingTask {
        runs: Arc<AtomicUsize>,
    }

    #[async_trait(?Send)]
    impl PeriodicTask for CountingTask {
        async fn run_once(&self) {
            self.runs.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Under a paused clock, the scheduler fires the task on each interval. We
    /// race it against a stopper that sleeps three intervals; Tokio
    /// auto-advances virtual time while both are idle, so by the time the
    /// stopper finishes the task has run on the immediate tick plus each
    /// elapsed interval.
    #[tokio::test(start_paused = true)]
    async fn test_run_periodic_fires_each_interval() {
        let runs = Arc::new(AtomicUsize::new(0));
        let task = Arc::new(CountingTask { runs: runs.clone() });
        let interval = Duration::from_secs(60);

        tokio::select! {
            _ = run_periodic(task, interval) => {},
            _ = async {
                for _ in 0..3 {
                    tokio::time::sleep(interval).await;
                }
            } => {},
        }

        // Immediate tick (t=0) + ticks at t=60,120,180 -> at least 3 runs.
        assert!(
            runs.load(Ordering::SeqCst) >= 3,
            "expected >=3 runs, got {}",
            runs.load(Ordering::SeqCst)
        );
    }

    /// DEV-12: a zero interval must not panic the scheduler (it used to panic
    /// inside `tokio::time::interval`); it is floored to `MIN_INTERVAL`.
    #[tokio::test(start_paused = true)]
    async fn test_zero_interval_does_not_panic() {
        let runs = Arc::new(AtomicUsize::new(0));
        let task = Arc::new(CountingTask { runs: runs.clone() });

        tokio::select! {
            _ = run_periodic(task, Duration::ZERO) => {},
            _ = tokio::time::sleep(MIN_INTERVAL * 2) => {},
        }

        let n = runs.load(Ordering::SeqCst);
        assert!(
            (1..=3).contains(&n),
            "expected a floored cadence, got {n} runs"
        );
    }
}
