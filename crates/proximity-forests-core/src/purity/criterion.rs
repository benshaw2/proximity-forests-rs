//! Typed split-criterion configuration.

/// Supervised impurity criterion applied independently to each output.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum SupervisedCriterion {
    /// Gini impurity for classification outputs.
    Gini,
    /// Base-2 Shannon entropy for classification outputs.
    Entropy,
    /// Population variance for regression outputs.
    Variance,
    /// Mean absolute deviation from the arithmetic mean for regression outputs.
    MeanAbsoluteDeviation,
}

/// Isolation split criterion.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum IsolationCriterion {
    /// Prefer the candidate having the smallest nonempty child.
    MinChild,
}
