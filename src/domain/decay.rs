use crate::domain::models::{MemoryNode, WORKING_CCL};

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

    /// Decay a score over `days_elapsed` using the **default** half-life.
    /// Retained for back-compat; prefer [`Self::calculate_decay_for`] when the
    /// node's CCL is known.
    pub fn calculate_decay(&self, current_score: f64, days_elapsed: f64) -> f64 {
        self.decay(current_score, days_elapsed, self.default_half_life_days)
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

    /// Apply decay to a specific node using *its own* CCL's half-life, returning
    /// the modified node. If the score drops below 0.1 the node is `archived`.
    pub fn apply_to_node(&self, mut node: MemoryNode, days_elapsed: f64) -> MemoryNode {
        let new_score = self.calculate_decay_for(node.relevance_score, days_elapsed, &node.ccl);
        node.relevance_score = new_score;

        if new_score < 0.1 && node.status == "active" {
            node.status = "archived".into();
        }

        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::models::TenantId;
    use serde_json::json;

    #[test]
    fn test_decay_calculation() {
        let engine = DecayEngine::new(7.0); // 7 day half-life

        let score = engine.calculate_decay(1.0, 7.0);
        assert!((score - 0.5).abs() < 0.001);

        let score_14 = engine.calculate_decay(1.0, 14.0);
        assert!((score_14 - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_node_archiving() {
        let engine = DecayEngine::new(7.0);

        let node = MemoryNode {
            id: None,
            tenant_id: TenantId("t1".into()),
            source_episode_id: Some(1),
            payload: json!({}),
            status: "active".into(),
            ccl: "reality".into(),
            is_explicit: false,
            support_count: 1,
            relevance_score: 0.15,
            context_key: None,
        };

        // 7 days later, 0.15 becomes 0.075, which is < 0.1
        let decayed = engine.apply_to_node(node, 7.0);
        assert_eq!(decayed.status, "archived");
        assert!(decayed.relevance_score < 0.1);
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
