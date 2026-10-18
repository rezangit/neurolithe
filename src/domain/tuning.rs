//! Tuning knobs of the memory engine, in one place.
//!
//! The constants shape behaviour (how fast notes fade out of recall, how much
//! session history stays raw, how many hits a recall returns) but are not
//! exposed in the config file. [`Ranking`] is the STM search fusion, set from
//! the config file's `[retrieval]` section. Other configurable values
//! (half-lives, distance thresholds, intervals) live in `AppConfig`. Storage
//! mechanics, such as KNN over-fetch, stay next to the SQL that needs them.

/// STM relevance below which a decayed node is archived: it leaves recall and
/// the vector index. Nodes start at 1.0, so an untouched node is archived after
/// ~3.3 half-lives.
pub const ARCHIVE_THRESHOLD: f64 = 0.1;

/// Number of hits an STM recall returns when the caller names no breadth.
pub const STM_RECALL_K: usize = 10;

/// Number of related facts a session pulls from STM as context for a
/// dialogue turn.
pub const SESSION_CONTEXT_K: usize = 5;

/// Estimated session size (tokens) at which older messages are compressed
/// into the running summary.
pub const SESSION_TOKEN_THRESHOLD: usize = 4000;

/// Newest messages a session keeps verbatim when it compresses or trims.
pub const SESSION_KEEP_RECENT: usize = 10;

/// Rough characters per token for session size estimates (GPT tokenizer
/// heuristic).
pub const CHARS_PER_TOKEN: usize = 4;

/// How STM search fuses its two ranked legs, the vector (semantic) leg and
/// the FTS5 keyword leg, by reciprocal rank fusion: each leg ranks its own
/// top `leg_k_factor × k` candidates, and a node's score is
/// `Σ weight / (rrf_k + rank)` over the legs it appears in. A rank, unlike a
/// raw vector distance or BM25 value, means the same in both legs.
#[derive(Debug, Clone, PartialEq)]
pub struct Ranking {
    /// RRF damping constant: larger flattens the gap between top ranks.
    pub rrf_k: f64,
    /// Weight of the vector leg.
    pub vector_weight: f64,
    /// Weight of the keyword leg.
    pub keyword_weight: f64,
    /// Each leg's candidate list is `leg_k_factor × k` long.
    pub leg_k_factor: usize,
    /// Drop English stop-words from the keyword query (a query made only of
    /// stop-words keeps them, so it still searches).
    pub fts_stopwords: bool,
    /// Return each direct hit's fused score with the results.
    pub include_scores: bool,
}

/// Default RRF damping constant (the common choice from the RRF paper).
pub const DEFAULT_RRF_K: f64 = 60.0;
/// Default per-leg candidate list length, as a multiple of `k`.
pub const DEFAULT_LEG_K_FACTOR: usize = 4;
/// Largest accepted `leg_k_factor` (it multiplies the KNN breadth).
pub const MAX_LEG_K_FACTOR: usize = 50;
/// Largest accepted `rrf_k`.
pub const MAX_RRF_K: f64 = 10_000.0;

impl Default for Ranking {
    fn default() -> Self {
        Self {
            rrf_k: DEFAULT_RRF_K,
            vector_weight: 1.0,
            keyword_weight: 1.0,
            leg_k_factor: DEFAULT_LEG_K_FACTOR,
            fts_stopwords: true,
            include_scores: false,
        }
    }
}

impl Ranking {
    /// Every problem with these values (empty when usable), each naming its
    /// field as in the config file's `[retrieval]` section.
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if !(self.rrf_k.is_finite() && (0.0..=MAX_RRF_K).contains(&self.rrf_k)) {
            problems.push(format!(
                "rrf_k must be a number from 0 to {MAX_RRF_K} (got {})",
                self.rrf_k
            ));
        }
        for (name, w) in [
            ("vector_weight", self.vector_weight),
            ("keyword_weight", self.keyword_weight),
        ] {
            if !(w.is_finite() && w >= 0.0) {
                problems.push(format!("{name} must be a finite number >= 0 (got {w})"));
            }
        }
        if self.vector_weight == 0.0 && self.keyword_weight == 0.0 {
            problems.push("vector_weight and keyword_weight can't both be 0".into());
        }
        if !(1..=MAX_LEG_K_FACTOR).contains(&self.leg_k_factor) {
            problems.push(format!(
                "leg_k_factor must be from 1 to {MAX_LEG_K_FACTOR} (got {})",
                self.leg_k_factor
            ));
        }
        problems
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ranking_is_valid() {
        assert!(Ranking::default().problems().is_empty());
    }

    #[test]
    fn ranking_rejects_unusable_values() {
        let bad = |f: fn(&mut Ranking)| {
            let mut r = Ranking::default();
            f(&mut r);
            r.problems()
        };
        assert_eq!(bad(|r| r.rrf_k = -1.0).len(), 1);
        assert_eq!(bad(|r| r.rrf_k = f64::NAN).len(), 1);
        assert_eq!(bad(|r| r.rrf_k = 1e9).len(), 1);
        assert_eq!(bad(|r| r.vector_weight = -0.5).len(), 1);
        assert_eq!(bad(|r| r.keyword_weight = f64::INFINITY).len(), 1);
        assert_eq!(bad(|r| r.leg_k_factor = 0).len(), 1);
        assert_eq!(bad(|r| r.leg_k_factor = MAX_LEG_K_FACTOR + 1).len(), 1);
        let both_zero = bad(|r| {
            r.vector_weight = 0.0;
            r.keyword_weight = 0.0;
        });
        assert!(
            both_zero.iter().any(|p| p.contains("both be 0")),
            "{both_zero:?}"
        );
        // One leg switched off is fine (pure vector or pure keyword).
        assert!(bad(|r| r.keyword_weight = 0.0).is_empty());
        assert!(bad(|r| r.rrf_k = 0.0).is_empty());
    }
}
