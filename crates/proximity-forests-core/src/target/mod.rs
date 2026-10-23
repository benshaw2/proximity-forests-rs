//! Typed supervised targets used by forest training and prediction.

mod classification;
mod regression;

pub use classification::{ClassId, ClassificationTargets};
pub use regression::RegressionTargets;
