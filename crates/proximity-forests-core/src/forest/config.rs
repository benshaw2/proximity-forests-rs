//! Validated forest-level model configuration.

use std::num::NonZeroUsize;

use crate::{CoreError, CoreResult, DimensionSelection, IsolationCriterion, SupervisedCriterion};

/// Randomized tie behavior used by branch assignment and prediction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TiePolicy {
    /// Resolve ties reproducibly using an identity-derived random stream.
    Random,
    /// Resolve ties by choosing the lowest stable index.
    First,
}

/// Shared configuration for classifier, regressor, and isolation forests.
#[derive(Clone, Debug, PartialEq)]
pub struct ForestConfig {
    /// Number of trees.
    pub tree_count: NonZeroUsize,
    /// Number of split candidates evaluated at each eligible node.
    pub candidate_count: NonZeroUsize,
    /// Whether each tree trains on a size-preserving bootstrap sample.
    pub bootstrap: bool,
    /// Maximum depth, or no explicit depth limit.
    pub maximum_depth: Option<usize>,
    /// Minimum effective bootstrap occurrences required to split a node.
    pub minimum_split_size: NonZeroUsize,
    /// Node-level feature or channel selection.
    pub dimension_selection: DimensionSelection,
    /// Root seed for deterministic model construction.
    pub seed: u64,
    /// Whether distances may abandon after proving they cannot win.
    pub early_abandoning: bool,
    /// Policy used for equal minimum distances.
    pub branch_tie_policy: TiePolicy,
}

impl ForestConfig {
    /// Creates a configuration with explicit core size parameters.
    #[must_use]
    pub fn new(tree_count: NonZeroUsize, candidate_count: NonZeroUsize, seed: u64) -> Self {
        Self {
            tree_count,
            candidate_count,
            bootstrap: true,
            maximum_depth: None,
            minimum_split_size: NonZeroUsize::MIN,
            dimension_selection: DimensionSelection::All,
            seed,
            early_abandoning: true,
            branch_tie_policy: TiePolicy::Random,
        }
    }

    /// Validates relationships among configuration values.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid dimension-selection configuration or a
    /// maximum depth of zero.
    pub fn validate(&self) -> CoreResult<()> {
        self.dimension_selection.validate()?;
        if self.maximum_depth == Some(0) {
            return Err(CoreError::invalid_configuration(
                "maximum_depth",
                "maximum depth must be positive when specified",
            ));
        }
        Ok(())
    }
}

/// Task-specific classification settings.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ClassificationConfig {
    /// Classification impurity criterion.
    pub criterion: SupervisedCriterion,
    /// Majority-vote tie policy.
    pub vote_tie_policy: TiePolicy,
}

/// Task-specific regression settings.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RegressionConfig {
    /// Number of exemplars/branches considered by each candidate.
    pub branch_count: NonZeroUsize,
    /// Regression impurity criterion.
    pub criterion: SupervisedCriterion,
}

/// Task-specific isolation settings.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct IsolationConfig {
    /// Number of exemplars/branches considered by each candidate.
    pub branch_count: NonZeroUsize,
    /// Minimum child size accepted during recursive growth.
    pub minimum_leaf_size: NonZeroUsize,
    /// Isolation split criterion.
    pub criterion: IsolationCriterion,
}

#[cfg(test)]
mod tests {
    use super::ForestConfig;
    use std::num::NonZeroUsize;

    #[test]
    fn baseline_configuration_is_valid() {
        let config = ForestConfig::new(
            NonZeroUsize::new(100).expect("nonzero"),
            NonZeroUsize::new(5).expect("nonzero"),
            42,
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn zero_maximum_depth_is_rejected() {
        let mut config = ForestConfig::new(
            NonZeroUsize::new(10).expect("nonzero"),
            NonZeroUsize::new(2).expect("nonzero"),
            42,
        );
        config.maximum_depth = Some(0);
        assert!(config.validate().is_err());
    }
}
