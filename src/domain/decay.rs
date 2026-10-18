use crate::domain::models::WORKING_CCL;
use crate::domain::tuning::ARCHIVE_THRESHOLD;

/// Whether a decayed STM relevance score has faded enough to archive the node
/// (below [`ARCHIVE_THRESHOLD`]).
pub fn should_archive(score: f64) -> bool {
    score < ARCHIVE_THRESHOLD
}

/// The forgetting curve for STM. Exponential half-life decay, resolved *per
/// cognitive layer* (STM-WORKING-MEMORY slice 2): the `working` layer
/// (situational notes) fades on a much shorter half-life than the default fact
/// store, so short-term session context clears fast while durable facts persist.
/// LTM is never touched by this engine.
pub struct DecayEngine {
    /// Half-life (days) for every layer except `working` — the durable fact
    /// store (`reality`, `dream`, `simulation`, …).
    pub default_half_life_days: f64,
    /// Half-life (days) for the `working` layer. Typically far smaller than the
    /// default (minutes–hours vs. days).
    pub working_half_life_days: f64,
}

impl DecayEngine {
    /// Single half-life for every layer (the historical behaviour — `working`
    /// notes decay at the same rate as facts). Kept for callers/tests that don't
    /// care about the working regime.
    pub fn new(half_life_days: f64) -> Self {
        Self {
            default_half_life_days: half_life_days,
            working_half_life_days: half_life_days,
        }
    }

    /// Distinct half-lives for the default fact store vs. the `working` layer.
    pub fn with_half_lives(default_half_life_days: f64, working_half_life_days: f64) -> Self {
        Self {
            default_half_life_days,
            working_half_life_days,
        }
    }

    /// Resolve the half-life (days) that applies to a given CCL.
    pub fn half_life_for(&self, ccl: &str) -> f64 {
        if ccl == WORKING_CCL {
            self.working_half_life_days
        } else {
            self.default_half_life_days
        }
    }

    /// Decay a score over `days_elapsed` using the half-life for `ccl`.
    pub fn calculate_decay_for(&self, current_score: f64, days_elapsed: f64, ccl: &str) -> f64 {
        self.decay(current_score, days_elapsed, self.half_life_for(ccl))
    }

    fn decay(&self, current_score: f64, days_elapsed: f64, half_life_days: f64) -> f64 {
        // No time passed (or clock skew, or NaN) → no decay. Checked first so a
        // zero half-life can't produce 0/0 = NaN (DEV-12).
        if days_elapsed.is_nan() || days_elapsed <= 0.0 {
            return current_score;
        }
        // A non-positive / non-finite half-life is a misconfiguration (config
        // validation rejects it); degrade to "forget immediately" rather than
        // writing NaN/inf scores into the store.
        if !(half_life_days.is_finite() && half_life_days > 0.0) {
            return 0.0;
        }
        // score = current_score * (0.5 ^ (days_elapsed / half_life))
        current_score * 0.5f64.powf(days_elapsed / half_life_days)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_calculation() {
        let engine = DecayEngine::new(7.0); // 7 day half-life

        let score = engine.calculate_decay_for(1.0, 7.0, "reality");
        assert!((score - 0.5).abs() < 0.001);

        let score_14 = engine.calculate_decay_for(1.0, 14.0, "reality");
        assert!((score_14 - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_node_archiving() {
        let engine = DecayEngine::new(7.0);

        // 7 days later, 0.15 becomes 0.075, which is < 0.1
        let decayed = engine.calculate_decay_for(0.15, 7.0, "reality");
        assert!(decayed < 0.1);
        assert!(should_archive(decayed));
        assert!(!should_archive(
            engine.calculate_decay_for(1.0, 7.0, "reality")
        ));
    }

    /// The core of slice 2: at the *same age*, a `working` note decays below the
    /// 0.1 archive threshold while a `reality` fact barely moves — because they
    /// resolve to different half-lives.
    #[test]
    fn test_working_layer_decays_far_faster_than_reality() {
        // reality 7 days, working 30 minutes.
        let engine = DecayEngine::with_half_lives(7.0, 30.0 / 1440.0);

        // Two hours of inactivity.
        let two_hours_days = 2.0 / 24.0;

        let working = engine.calculate_decay_for(1.0, two_hours_days, WORKING_CCL);
        let reality = engine.calculate_decay_for(1.0, two_hours_days, "reality");

        assert!(
            working < 0.1,
            "a working note should be archivable after hours, got {working}"
        );
        assert!(
            reality > 0.9,
            "a reality fact should barely decay over hours, got {reality}"
        );
    }

    /// DEV-12: a zero half-life must never yield NaN/inf (which would poison
    /// ranking and the archive threshold).
    #[test]
    fn test_zero_half_life_never_nan() {
        let engine = DecayEngine::with_half_lives(0.0, 0.0);
        for elapsed in [0.0, 1e-9, 1.0, -1.0] {
            let s = engine.calculate_decay_for(0.8, elapsed, "reality");
            assert!(s.is_finite(), "elapsed {elapsed} gave {s}");
            assert!((0.0..=0.8).contains(&s));
        }
        // No elapsed time → unchanged, even with a broken half-life.
        assert_eq!(engine.calculate_decay_for(0.8, 0.0, WORKING_CCL), 0.8);
    }

    /// `half_life_for` routes `working` to the working curve and everything else
    /// (including unknown layers) to the default.
    #[test]
    fn test_half_life_resolution_by_ccl() {
        let engine = DecayEngine::with_half_lives(7.0, 0.5);
        assert_eq!(engine.half_life_for(WORKING_CCL), 0.5);
        assert_eq!(engine.half_life_for("reality"), 7.0);
        assert_eq!(engine.half_life_for("dream"), 7.0);
    }
}
