//! Task-specific fitted leaf payloads.

use crate::{ClassificationTargets, CoreError, CoreResult, RegressionTargets};

use super::{NodeMembership, ObservationIndex};

/// Classification summary retained by a fitted leaf.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClassificationLeaf {
    output_count: usize,
    /// Per-output sorted `(class_id, occurrence_count)` pairs.
    counts: Vec<Vec<(u32, usize)>>,
}

impl ClassificationLeaf {
    /// Builds weighted class counts from compact node membership.
    pub fn from_node(
        membership: &NodeMembership,
        observations: &[ObservationIndex],
        targets: &ClassificationTargets,
    ) -> CoreResult<Self> {
        validate_mapping(membership, observations, targets.sample_count())?;
        let mut counts = Vec::with_capacity(targets.output_count());
        for output in 0..targets.output_count() {
            let mut output_counts = std::collections::BTreeMap::<u32, usize>::new();
            for (distinct, observation) in observations.iter().enumerate() {
                let class = targets.get(observation.get(), output).ok_or_else(|| {
                    CoreError::invalid_data("classification target index was unavailable")
                })?;
                let count = output_counts.entry(class).or_default();
                *count = count
                    .checked_add(membership.multiplicities()[distinct])
                    .ok_or_else(|| CoreError::invalid_data("leaf class count overflowed usize"))?;
            }
            counts.push(output_counts.into_iter().collect());
        }
        Ok(Self {
            output_count: targets.output_count(),
            counts,
        })
    }

    /// Reconstructs a classification leaf from validated persisted counts.
    pub(crate) fn from_counts(counts: Vec<Vec<(u32, usize)>>) -> CoreResult<Self> {
        if counts.is_empty() {
            return Err(CoreError::invalid_data(
                "classification leaf requires at least one output",
            ));
        }
        for output_counts in &counts {
            if output_counts.is_empty() {
                return Err(CoreError::invalid_data(
                    "each classification output requires at least one represented class",
                ));
            }
            let mut previous = None;
            let mut total = 0usize;
            for &(class_id, occurrence_count) in output_counts {
                if occurrence_count == 0 {
                    return Err(CoreError::invalid_data(
                        "classification class counts must be positive",
                    ));
                }
                if previous.is_some_and(|value| class_id <= value) {
                    return Err(CoreError::invalid_data(
                        "classification class identifiers must be strictly increasing",
                    ));
                }
                total = total.checked_add(occurrence_count).ok_or_else(|| {
                    CoreError::invalid_data("classification class count overflowed usize")
                })?;
                previous = Some(class_id);
            }
            if total == 0 {
                return Err(CoreError::invalid_data(
                    "classification output requires at least one occurrence",
                ));
            }
        }
        Ok(Self {
            output_count: counts.len(),
            counts,
        })
    }

    /// Returns all sorted class-count collections in output order.
    pub(crate) fn all_counts(&self) -> &[Vec<(u32, usize)>] {
        &self.counts
    }

    /// Returns the number of outputs.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }

    /// Returns sorted class counts for one output.
    #[must_use]
    pub fn counts(&self, output: usize) -> Option<&[(u32, usize)]> {
        self.counts.get(output).map(Vec::as_slice)
    }
}

/// Regression summary retained by a fitted leaf.
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionLeaf {
    means: Vec<f64>,
}

impl RegressionLeaf {
    /// Builds multiplicity-weighted output means.
    pub fn from_node(
        membership: &NodeMembership,
        observations: &[ObservationIndex],
        targets: &RegressionTargets,
    ) -> CoreResult<Self> {
        validate_mapping(membership, observations, targets.sample_count())?;
        if membership.occurrence_count() == 0 {
            return Err(CoreError::invalid_data(
                "regression leaf requires at least one occurrence",
            ));
        }
        let mut means = Vec::with_capacity(targets.output_count());
        for output in 0..targets.output_count() {
            let mut weighted_sum = 0.0;
            for (distinct, observation) in observations.iter().enumerate() {
                let value = targets.get_f64(observation.get(), output).ok_or_else(|| {
                    CoreError::invalid_data("regression target index was unavailable")
                })?;
                #[allow(clippy::cast_precision_loss)]
                {
                    weighted_sum += value * membership.multiplicities()[distinct] as f64;
                }
            }
            #[allow(clippy::cast_precision_loss)]
            means.push(weighted_sum / membership.occurrence_count() as f64);
        }
        Ok(Self { means })
    }

    /// Reconstructs a regression leaf from validated persisted means.
    pub(crate) fn from_means(means: Vec<f64>) -> CoreResult<Self> {
        if means.is_empty() {
            return Err(CoreError::invalid_data(
                "regression leaf requires at least one output",
            ));
        }
        if means.iter().any(|value| !value.is_finite()) {
            return Err(CoreError::invalid_data(
                "regression leaf means must be finite",
            ));
        }
        Ok(Self { means })
    }

    /// Returns weighted output means.
    #[must_use]
    pub fn means(&self) -> &[f64] {
        &self.means
    }
}

/// Task-specific prediction payload stored at a fitted leaf.
#[derive(Clone, Debug, PartialEq)]
pub enum LeafPayload {
    /// No target payload, used by the task-independent builder skeleton.
    Empty,
    /// Classification class counts.
    Classification(ClassificationLeaf),
    /// Regression means.
    Regression(RegressionLeaf),
    /// Isolation leaf state used for path-length correction.
    Isolation {
        /// Number of bootstrap occurrences reaching the leaf.
        occurrence_count: usize,
    },
}

fn validate_mapping(
    membership: &NodeMembership,
    observations: &[ObservationIndex],
    target_count: usize,
) -> CoreResult<()> {
    if observations.len() != membership.distinct_count() {
        return Err(CoreError::shape_mismatch(
            "leaf observations and membership must have equal distinct counts",
        ));
    }
    if observations
        .iter()
        .any(|observation| observation.get() >= target_count)
    {
        return Err(CoreError::invalid_data(
            "a leaf observation is outside target storage",
        ));
    }
    Ok(())
}
