//! Vector-distance thresholds, per embedding model.
//!
//! Three thresholds turn raw sqlite-vec (L2) distances into decisions, and the
//! right values depend on the embedding model's distance scale:
//! - **placement** (`[ltm] placement_max_distance`): a document is filed under
//!   its nearest concept only within this distance, otherwise into the inbox;
//! - **assimilation** (`[stm] assimilation_threshold`): a new fact at or below
//!   this distance from an existing one *is* that fact (reinforce it);
//! - **accommodation** (`[stm] accommodation_threshold`): at or below this, the
//!   new fact *refines* the existing one (replace its text); beyond it a new
//!   fact is created.
//!
//! Resolution, per value: explicit config → the model's entry in
//! [`MODEL_DEFAULTS`] → the fallback (the default local model's values, with a
//! startup warning suggesting calibration via `placement_debug`).

use anyhow::{Result, bail};
use serde::Serialize;

/// The three thresholds' values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdValues {
    pub placement_max_distance: f64,
    pub assimilation: f64,
    pub accommodation: f64,
}

/// Measured defaults for Google `text-embedding-004` (768-d, unit-normalised) —
/// the legacy hard-coded values. Placement: `placement_debug` over a real corpus
/// showed document→concept L2 distances in ~[0.91, 1.16] (median 1.05, cosine
/// ~0.33–0.59); 1.10 (cosine ≈ 0.40) files the confident majority and leaves the
/// ambiguous tail in the inbox (earlier guesses 0.5 and 0.85 filed everything
/// to the inbox).
pub const TEXT_EMBEDDING_004: ThresholdValues = ThresholdValues {
    placement_max_distance: 1.10,
    assimilation: 0.15,
    accommodation: 0.35,
};

/// Defaults for local `bge-small-en-v1.5` (384-d, unit-normalised), the default
/// embedder. Also the fallback for models without an entry.
///
/// Calibrated 2026-09-24 on a small hand-written English set (see calibration
/// notes); ±0.02; accommodation kept below the first distinct-fact overwrite
/// (0.585). Texts were embedded exactly as in production:
/// - placement 0.96: correct-branch distances 0.825–0.994 (median 0.909) on an
///   8-branch spine → 70% filed correctly, 17% inbox, 13% misfiled (the inbox is
///   preferred over the 1.00 error minimum because it is recoverable);
/// - assimilation 0.40: paraphrases 0.13–0.59 (median 0.30), updates ≥ 0.445 →
///   23/30 paraphrases merged, no update absorbed;
/// - accommodation 0.58: distinct facts about the same entity start at 0.585 →
///   12/30 updates refined in place, 0/30 distinct facts overwritten.
pub const BGE_SMALL_EN_V15: ThresholdValues = ThresholdValues {
    placement_max_distance: 0.96,
    assimilation: 0.40,
    accommodation: 0.58,
};

/// Per-model defaults, keyed by canonical embedding model id
/// ([`crate::domain::ports::LlmClient::embedding_model_id`]). An entry whose
/// key has no `provider:` prefix matches that model under any provider.
pub const MODEL_DEFAULTS: &[(&str, ThresholdValues)] = &[
    ("text-embedding-004", TEXT_EMBEDDING_004),
    ("local:bge-small-en-v1.5", BGE_SMALL_EN_V15),
    ("local:bge-small-en-v1.5-q", BGE_SMALL_EN_V15),
];

/// Values for models without an entry in [`MODEL_DEFAULTS`].
pub const FALLBACK: ThresholdValues = BGE_SMALL_EN_V15;

/// The table entry for `model_id` (`"provider:model"`), if any. Model names
/// compare case-insensitively.
pub fn model_defaults(model_id: &str) -> Option<ThresholdValues> {
    let id = model_id.trim().to_ascii_lowercase();
    let model_only = id.split_once(':').map(|(_, m)| m).unwrap_or(&id);
    MODEL_DEFAULTS.iter().find_map(|(key, values)| {
        let hit = if key.contains(':') {
            *key == id
        } else {
            *key == model_only
        };
        hit.then_some(*values)
    })
}

/// Where a resolved threshold came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdSource {
    /// Set explicitly in config.
    Config,
    /// The embedding model's entry in the defaults table.
    ModelDefault,
    /// No config and no table entry for the model: the generic fallback.
    Fallback,
}

/// One resolved threshold and its provenance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Threshold {
    pub value: f64,
    pub source: ThresholdSource,
}

/// Optional explicit values from config.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ThresholdOverrides {
    pub placement_max_distance: Option<f64>,
    pub assimilation: Option<f64>,
    pub accommodation: Option<f64>,
}

/// The effective thresholds for a process, resolved once at startup and
/// reported by `placement_debug`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Thresholds {
    /// The embedding model they were resolved for.
    pub embedding_model: String,
    pub placement_max_distance: Threshold,
    pub assimilation: Threshold,
    pub accommodation: Threshold,
}

impl Thresholds {
    /// Resolve against `model_id` with `overrides` taking precedence. Returns
    /// the thresholds and, when the fallback was used, a warning to log once.
    /// Errors if a value is not a finite positive number or if
    /// assimilation ≥ accommodation after resolution.
    pub fn resolve(
        model_id: &str,
        overrides: ThresholdOverrides,
    ) -> Result<(Thresholds, Option<String>)> {
        let (defaults, default_source) = match model_defaults(model_id) {
            Some(values) => (values, ThresholdSource::ModelDefault),
            None => (FALLBACK, ThresholdSource::Fallback),
        };
        let pick = |configured: Option<f64>, default: f64| match configured {
            Some(value) => Threshold {
                value,
                source: ThresholdSource::Config,
            },
            None => Threshold {
                value: default,
                source: default_source,
            },
        };
        let thresholds = Thresholds {
            embedding_model: model_id.to_string(),
            placement_max_distance: pick(
                overrides.placement_max_distance,
                defaults.placement_max_distance,
            ),
            assimilation: pick(overrides.assimilation, defaults.assimilation),
            accommodation: pick(overrides.accommodation, defaults.accommodation),
        };
        thresholds.validate()?;

        let uses_fallback = [
            thresholds.placement_max_distance,
            thresholds.assimilation,
            thresholds.accommodation,
        ]
        .iter()
        .any(|t| t.source == ThresholdSource::Fallback);
        let warning = uses_fallback.then(|| {
            format!(
                "no calibrated distance thresholds for embedding model '{model_id}'; using \
                 generic defaults (placement {}, assimilation {}, accommodation {}). Calibrate \
                 with the `placement_debug` tool and set `[ltm] placement_max_distance` / \
                 `[stm] assimilation_threshold` / `[stm] accommodation_threshold`.",
                thresholds.placement_max_distance.value,
                thresholds.assimilation.value,
                thresholds.accommodation.value
            )
        });
        Ok((thresholds, warning))
    }

    /// The legacy `text-embedding-004` values, all marked as model defaults —
    /// for tests and callers that need a concrete set without config.
    pub fn text_embedding_004() -> Thresholds {
        Self::resolve("vertex:text-embedding-004", ThresholdOverrides::default())
            .expect("built-in defaults are valid")
            .0
    }

    fn validate(&self) -> Result<()> {
        for (name, t) in [
            ("placement_max_distance", self.placement_max_distance),
            ("assimilation_threshold", self.assimilation),
            ("accommodation_threshold", self.accommodation),
        ] {
            if !(t.value.is_finite() && t.value > 0.0) {
                bail!("{name} must be a finite number > 0 (got {})", t.value);
            }
        }
        if self.assimilation.value >= self.accommodation.value {
            bail!(
                "assimilation_threshold ({}) must be smaller than accommodation_threshold ({}) \
                 for embedding model '{}'",
                self.assimilation.value,
                self.accommodation.value,
                self.embedding_model
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none() -> ThresholdOverrides {
        ThresholdOverrides::default()
    }

    #[test]
    fn model_table_matches_canonical_ids() {
        assert_eq!(
            model_defaults("vertex:text-embedding-004"),
            Some(TEXT_EMBEDDING_004)
        );
        assert_eq!(
            model_defaults("gemini:text-embedding-004"),
            Some(TEXT_EMBEDDING_004)
        );
        assert_eq!(
            model_defaults("openai:Text-Embedding-004"),
            Some(TEXT_EMBEDDING_004)
        );
        assert_eq!(
            model_defaults("local:bge-small-en-v1.5"),
            Some(BGE_SMALL_EN_V15)
        );
        assert_eq!(
            model_defaults("local:bge-small-en-v1.5-q"),
            Some(BGE_SMALL_EN_V15)
        );
        // A provider-qualified key does not match other providers.
        assert_eq!(model_defaults("custom:bge-small-en-v1.5"), None);
        assert_eq!(model_defaults("openai:text-embedding-3-small"), None);
    }

    /// Precedence, per value: config > model default > fallback.
    #[test]
    fn precedence_config_then_model_then_fallback() {
        // Known model, no config → all model defaults, no warning.
        let (t, warn) = Thresholds::resolve("vertex:text-embedding-004", none()).unwrap();
        assert_eq!(t.placement_max_distance.value, 1.10);
        assert_eq!(t.assimilation.value, 0.15);
        assert_eq!(t.accommodation.value, 0.35);
        assert!(
            [t.placement_max_distance, t.assimilation, t.accommodation]
                .iter()
                .all(|x| x.source == ThresholdSource::ModelDefault)
        );
        assert!(warn.is_none());

        // Config wins for the values it sets; the rest stay model defaults.
        let (t, _) = Thresholds::resolve(
            "vertex:text-embedding-004",
            ThresholdOverrides {
                placement_max_distance: Some(0.9),
                ..none()
            },
        )
        .unwrap();
        assert_eq!(t.placement_max_distance.value, 0.9);
        assert_eq!(t.placement_max_distance.source, ThresholdSource::Config);
        assert_eq!(t.assimilation.source, ThresholdSource::ModelDefault);

        // Unknown model → fallback values + one warning pointing at calibration.
        let (t, warn) = Thresholds::resolve("openai:text-embedding-3-small", none()).unwrap();
        assert_eq!(
            t.placement_max_distance.value,
            FALLBACK.placement_max_distance
        );
        assert_eq!(t.assimilation.source, ThresholdSource::Fallback);
        let warn = warn.expect("fallback warns");
        assert!(warn.contains("placement_debug") && warn.contains("placement_max_distance"));

        // Unknown model fully configured → no fallback, no warning.
        let (t, warn) = Thresholds::resolve(
            "openai:text-embedding-3-small",
            ThresholdOverrides {
                placement_max_distance: Some(1.0),
                assimilation: Some(0.2),
                accommodation: Some(0.4),
            },
        )
        .unwrap();
        assert!(warn.is_none());
        assert_eq!(t.accommodation.source, ThresholdSource::Config);
    }

    #[test]
    fn invalid_values_are_rejected() {
        for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            let err = Thresholds::resolve(
                "local:bge-small-en-v1.5",
                ThresholdOverrides {
                    placement_max_distance: Some(bad),
                    ..none()
                },
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("placement_max_distance"), "{err}");
        }
        // assimilation must stay below accommodation — also when only one side
        // is configured and the other comes from the table.
        let err = Thresholds::resolve(
            "vertex:text-embedding-004",
            ThresholdOverrides {
                assimilation: Some(0.5),
                ..none()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("must be smaller"), "{err}");
    }

    #[test]
    fn built_in_defaults_are_valid() {
        for (id, _) in MODEL_DEFAULTS {
            Thresholds::resolve(id, none()).unwrap();
        }
        Thresholds::resolve("unknown:model", none()).unwrap();
    }
}
