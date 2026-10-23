//! Best-candidate retention for iterative tree construction.

use crate::{CoreError, CoreResult};

use super::{CandidatePartition, CandidateScore, CandidateSplit, ScoredCandidate};

/// Winning candidate and its durable compact partition.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectedCandidate {
    scored: ScoredCandidate,
    partition: CandidatePartition,
}

impl SelectedCandidate {
    /// Creates a selected candidate.
    #[must_use]
    pub const fn new(
        candidate: CandidateSplit,
        score: CandidateScore,
        partition: CandidatePartition,
    ) -> Self {
        Self {
            scored: ScoredCandidate::new(candidate, score),
            partition,
        }
    }

    /// Returns the candidate.
    #[must_use]
    pub const fn candidate(&self) -> &CandidateSplit {
        self.scored.candidate()
    }

    /// Returns the score.
    #[must_use]
    pub const fn score(&self) -> CandidateScore {
        self.scored.score()
    }

    /// Returns the winning partition.
    #[must_use]
    pub const fn partition(&self) -> &CandidatePartition {
        &self.partition
    }

    /// Consumes the selection.
    #[must_use]
    pub fn into_parts(self) -> (CandidateSplit, CandidateScore, CandidatePartition) {
        let (candidate, score) = self.scored.into_parts();
        (candidate, score, self.partition)
    }
}

/// Retains the first strictly best valid candidate.
///
/// Exact score ties keep the earlier candidate, making candidate-order behavior
/// deterministic independently of branch-distance tie policy.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CandidateSelector {
    best: Option<SelectedCandidate>,
    valid_count: usize,
    rejected_count: usize,
}

impl CandidateSelector {
    /// Creates an empty selector.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            best: None,
            valid_count: 0,
            rejected_count: 0,
        }
    }

    /// Considers one valid scored candidate and durable partition.
    ///
    /// # Errors
    ///
    /// Returns an error if its score preference differs from the current best.
    pub fn consider(
        &mut self,
        candidate: CandidateSplit,
        score: CandidateScore,
        partition: CandidatePartition,
    ) -> CoreResult<bool> {
        if partition.branch_count() != candidate.branch_count() {
            return Err(CoreError::shape_mismatch(format!(
                "candidate has {} branches but partition has {}",
                candidate.branch_count(),
                partition.branch_count()
            )));
        }
        self.valid_count = self
            .valid_count
            .checked_add(1)
            .ok_or_else(|| CoreError::invalid_data("valid candidate count overflowed usize"))?;

        let replace = match &self.best {
            Some(best) => score.is_better_than(best.score())?,
            None => true,
        };
        if replace {
            self.best = Some(SelectedCandidate::new(candidate, score, partition));
        }
        Ok(replace)
    }

    /// Records one invalid or degenerate generated candidate.
    ///
    /// # Errors
    ///
    /// Returns an error if the rejection count overflows.
    pub fn reject(&mut self) -> CoreResult<()> {
        self.rejected_count = self
            .rejected_count
            .checked_add(1)
            .ok_or_else(|| CoreError::invalid_data("rejected candidate count overflowed usize"))?;
        Ok(())
    }

    /// Returns the current best candidate.
    #[must_use]
    pub const fn best(&self) -> Option<&SelectedCandidate> {
        self.best.as_ref()
    }

    /// Consumes and returns the current best candidate.
    #[must_use]
    pub fn into_best(self) -> Option<SelectedCandidate> {
        self.best
    }

    /// Returns the number of successfully scored candidates.
    #[must_use]
    pub const fn valid_count(&self) -> usize {
        self.valid_count
    }

    /// Returns the number of rejected candidates.
    #[must_use]
    pub const fn rejected_count(&self) -> usize {
        self.rejected_count
    }
}

#[cfg(test)]
mod tests {
    use super::CandidateSelector;
    use super::{CandidatePartition, CandidateScore, CandidateSplit};
    use crate::forest::{ObservationIndex, ScorePreference};
    use crate::{DistanceConfig, NodeMembership, SampleId};

    fn candidate(exemplar: usize) -> CandidateSplit {
        CandidateSplit::new(
            DistanceConfig::SquaredEuclidean,
            Vec::new(),
            vec![ObservationIndex::new(exemplar)],
        )
        .expect("candidate")
    }

    fn partition() -> CandidatePartition {
        let membership =
            NodeMembership::from_occurrences(&[Some(SampleId::new(1))]).expect("membership");
        CandidatePartition::from_assignments(&membership, 1, vec![0]).expect("partition")
    }

    #[test]
    fn lower_minimization_score_wins() {
        let mut selector = CandidateSelector::new();
        selector
            .consider(
                candidate(0),
                CandidateScore::new(2.0, ScorePreference::Minimize).expect("score"),
                partition(),
            )
            .expect("consider");
        selector
            .consider(
                candidate(1),
                CandidateScore::new(1.0, ScorePreference::Minimize).expect("score"),
                partition(),
            )
            .expect("consider");
        assert_eq!(
            selector.best().expect("best").candidate().exemplars()[0].get(),
            1
        );
    }

    #[test]
    fn exact_tie_keeps_first_candidate() {
        let mut selector = CandidateSelector::new();
        let score = CandidateScore::new(1.0, ScorePreference::Minimize).expect("score");
        selector
            .consider(candidate(0), score, partition())
            .expect("consider");
        selector
            .consider(candidate(1), score, partition())
            .expect("consider");
        assert_eq!(
            selector.best().expect("best").candidate().exemplars()[0].get(),
            0
        );
    }
}
