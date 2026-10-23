//! Split-quality criteria for supervised and isolation forests.

mod classification;
mod criterion;
mod isolation;
mod regression;

pub use classification::{weighted_entropy, weighted_gini};
pub use criterion::{IsolationCriterion, SupervisedCriterion};
pub use isolation::isolation_min_child;
pub use regression::{weighted_mean_absolute_deviation, weighted_variance};
