//! Fitted forest containers owning all exemplar observations.

use crate::{ForestConfig, TiePolicy};

use super::{OwnedObservationStore, ProximityTree};

/// Shared fitted forest state.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedForest {
    config: ForestConfig,
    trees: Vec<ProximityTree>,
    observations: OwnedObservationStore,
}

impl FittedForest {
    /// Creates a complete fitted forest.
    #[must_use]
    pub fn new(
        config: ForestConfig,
        trees: Vec<ProximityTree>,
        observations: OwnedObservationStore,
    ) -> Self {
        Self {
            config,
            trees,
            observations,
        }
    }

    /// Returns fitted trees in stable tree-index order.
    #[must_use]
    pub fn trees(&self) -> &[ProximityTree] {
        &self.trees
    }

    /// Returns model-owned exemplar observations.
    #[must_use]
    pub const fn observations(&self) -> &OwnedObservationStore {
        &self.observations
    }

    /// Returns fitted configuration.
    #[must_use]
    pub const fn config(&self) -> &ForestConfig {
        &self.config
    }

    /// Returns prediction-time branch tie policy.
    #[must_use]
    pub const fn tie_policy(&self) -> TiePolicy {
        self.config.branch_tie_policy
    }
}

/// Fitted classification forest.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationForest {
    forest: FittedForest,
    output_count: usize,
}

impl ClassificationForest {
    /// Creates a fitted classification forest.
    #[must_use]
    pub const fn new(forest: FittedForest, output_count: usize) -> Self {
        Self {
            forest,
            output_count,
        }
    }

    /// Returns shared fitted forest state.
    #[must_use]
    pub const fn forest(&self) -> &FittedForest {
        &self.forest
    }

    /// Returns classification output count.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }
}

/// Fitted regression forest.
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionForest {
    forest: FittedForest,
    output_count: usize,
}

impl RegressionForest {
    /// Creates a fitted regression forest.
    #[must_use]
    pub const fn new(forest: FittedForest, output_count: usize) -> Self {
        Self {
            forest,
            output_count,
        }
    }

    /// Returns shared fitted forest state.
    #[must_use]
    pub const fn forest(&self) -> &FittedForest {
        &self.forest
    }

    /// Returns regression output count.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }
}

/// Fitted isolation forest.
#[derive(Clone, Debug, PartialEq)]
pub struct IsolationForest {
    forest: FittedForest,
    normalization_size: usize,
}

impl IsolationForest {
    /// Creates a fitted isolation forest.
    #[must_use]
    pub const fn new(forest: FittedForest, normalization_size: usize) -> Self {
        Self {
            forest,
            normalization_size,
        }
    }

    /// Returns shared fitted forest state.
    #[must_use]
    pub const fn forest(&self) -> &FittedForest {
        &self.forest
    }

    /// Returns path-length normalization sample size.
    #[must_use]
    pub const fn normalization_size(&self) -> usize {
        self.normalization_size
    }
}
