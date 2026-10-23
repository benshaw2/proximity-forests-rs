//! Training-only proximity-split candidates and score ordering.

use std::cmp::Ordering;

use crate::{CoreError, CoreResult, DistanceConfig};

use super::ObservationIndex;

/// Training-only candidate split.
///
/// Only a winning candidate is converted into a fitted `SplitNode`. Keeping
/// candidate state separate prevents temporary scores and partitions from
/// inflating the persisted tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateSplit {
    distance: DistanceConfig,
    selected_dimensions: Vec<usize>,
    exemplars: Vec<ObservationIndex>,
}

impl CandidateSplit {
    /// Creates a candidate split.
    ///
    /// An empty selected-dimension vector means all dimensions. Exemplars must
    /// be nonempty and distinct.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty exemplar set or duplicate exemplars.
    pub fn new(
        distance: DistanceConfig,
        selected_dimensions: Vec<usize>,
        exemplars: Vec<ObservationIndex>,
    ) -> CoreResult<Self> {
        if exemplars.is_empty() {
            return Err(CoreError::invalid_data(
                "a proximity split candidate requires at least one exemplar",
            ));
        }
        for (index, exemplar) in exemplars.iter().enumerate() {
            if exemplars[..index].contains(exemplar) {
                return Err(CoreError::invalid_data(format!(
                    "candidate exemplar {} appears more than once",
                    exemplar.get()
                )));
            }
        }
        Ok(Self {
            distance,
            selected_dimensions,
            exemplars,
        })
    }

    /// Returns the candidate distance configuration.
    #[must_use]
    pub const fn distance(&self) -> &DistanceConfig {
        &self.distance
    }

    /// Returns selected dimensions, or an empty slice when all apply.
    #[must_use]
    pub fn selected_dimensions(&self) -> &[usize] {
        &self.selected_dimensions
    }

    /// Returns candidate exemplars in branch order.
    #[must_use]
    pub fn exemplars(&self) -> &[ObservationIndex] {
        &self.exemplars
    }

    /// Returns the number of candidate branches.
    #[must_use]
    pub fn branch_count(&self) -> usize {
        self.exemplars.len()
    }

    /// Consumes the candidate into its persisted components.
    #[must_use]
    pub fn into_parts(self) -> (DistanceConfig, Vec<usize>, Vec<ObservationIndex>) {
        (self.distance, self.selected_dimensions, self.exemplars)
    }
}

/// Whether lower or higher criterion scores are preferred.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ScorePreference {
    /// Lower scores are better, as for weighted child impurity or MinChild.
    Minimize,
    /// Higher scores are better, as for impurity reduction.
    Maximize,
}

/// Valid finite score assigned to one candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateScore {
    value: f64,
    preference: ScorePreference,
}

impl CandidateScore {
    /// Creates a finite candidate score.
    ///
    /// # Errors
    ///
    /// Returns an error for NaN or infinity.
    pub fn new(value: f64, preference: ScorePreference) -> CoreResult<Self> {
        if !value.is_finite() {
            return Err(CoreError::invalid_data(
                "candidate split scores must be finite",
            ));
        }
        Ok(Self { value, preference })
    }

    /// Returns the numerical score.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }

    /// Returns the score preference.
    #[must_use]
    pub const fn preference(self) -> ScorePreference {
        self.preference
    }

    /// Compares two scores under their shared preference.
    ///
    /// # Errors
    ///
    /// Returns an error when the score directions differ.
    pub fn compare(self, other: Self) -> CoreResult<Ordering> {
        if self.preference != other.preference {
            return Err(CoreError::invalid_configuration(
                "candidate_score",
                "cannot compare candidate scores with different preferences",
            ));
        }
        let ordering = self
            .value
            .partial_cmp(&other.value)
            .expect("finite scores are always comparable");
        Ok(match self.preference {
            ScorePreference::Minimize => ordering.reverse(),
            ScorePreference::Maximize => ordering,
        })
    }

    /// Reports whether this score is strictly better than another.
    ///
    /// Exact ties remain ties for deterministic candidate-order handling.
    pub fn is_better_than(self, other: Self) -> CoreResult<bool> {
        Ok(self.compare(other)? == Ordering::Greater)
    }
}

/// Candidate together with its validated score.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredCandidate {
    candidate: CandidateSplit,
    score: CandidateScore,
}

impl ScoredCandidate {
    /// Creates a scored candidate.
    #[must_use]
    pub const fn new(candidate: CandidateSplit, score: CandidateScore) -> Self {
        Self { candidate, score }
    }

    /// Returns the candidate.
    #[must_use]
    pub const fn candidate(&self) -> &CandidateSplit {
        &self.candidate
    }

    /// Returns the score.
    #[must_use]
    pub const fn score(&self) -> CandidateScore {
        self.score
    }

    /// Consumes the pair.
    #[must_use]
    pub fn into_parts(self) -> (CandidateSplit, CandidateScore) {
        (self.candidate, self.score)
    }
}

#[cfg(test)]
mod tests {
    use super::ObservationIndex;
    use super::{CandidateScore, CandidateSplit, ScorePreference};
    use crate::DistanceConfig;

    #[test]
    fn duplicate_exemplars_are_rejected() {
        assert!(
            CandidateSplit::new(
                DistanceConfig::SquaredEuclidean,
                Vec::new(),
                vec![ObservationIndex::new(2), ObservationIndex::new(2)],
            )
            .is_err()
        );
    }

    #[test]
    fn score_direction_is_explicit() {
        let low = CandidateScore::new(1.0, ScorePreference::Minimize).expect("score");
        let high = CandidateScore::new(3.0, ScorePreference::Minimize).expect("score");
        assert!(low.is_better_than(high).expect("same preference"));
    }
}
