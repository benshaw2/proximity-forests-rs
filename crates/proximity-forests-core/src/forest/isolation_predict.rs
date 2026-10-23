//! Isolation-tree path-length and anomaly-score prediction.

use crate::{
    CoreError, CoreResult, DistanceScratch, MultivariateSeriesView, NumericVectorView, TiePolicy,
};

use super::{LeafPayload, ProximityTree, traverse_multivariate, traverse_vector};

/// One isolation-tree prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsolationPrediction {
    path_length: f64,
    anomaly_score: f64,
}

impl IsolationPrediction {
    /// Creates a prediction from an aggregated path length and score.
    #[must_use]
    pub(crate) const fn from_parts(path_length: f64, anomaly_score: f64) -> Self {
        Self {
            path_length,
            anomaly_score,
        }
    }

    /// Returns depth plus terminal-size correction.
    #[must_use]
    pub const fn path_length(self) -> f64 {
        self.path_length
    }

    /// Returns the normalized anomaly score in `(0, 1]`.
    #[must_use]
    pub const fn anomaly_score(self) -> f64 {
        self.anomaly_score
    }
}

/// Predicts one vector observation.
pub fn predict_vector_isolation(
    tree: &ProximityTree,
    query: NumericVectorView<'_>,
    observations: &[NumericVectorView<'_>],
    normalization_size: usize,
    tie_policy: TiePolicy,
    prediction_seed: u64,
    scratch: &mut DistanceScratch,
) -> CoreResult<IsolationPrediction> {
    let visit = traverse_vector(
        tree,
        query,
        observations,
        tie_policy,
        prediction_seed,
        scratch,
    )?;
    isolation_prediction(visit.depth(), visit.leaf().payload(), normalization_size)
}

/// Predicts one multivariate observation.
pub fn predict_multivariate_isolation(
    tree: &ProximityTree,
    query: MultivariateSeriesView<'_>,
    observations: &[MultivariateSeriesView<'_>],
    normalization_size: usize,
    tie_policy: TiePolicy,
    prediction_seed: u64,
    scratch: &mut DistanceScratch,
) -> CoreResult<IsolationPrediction> {
    let visit = traverse_multivariate(
        tree,
        query,
        observations,
        tie_policy,
        prediction_seed,
        scratch,
    )?;
    isolation_prediction(visit.depth(), visit.leaf().payload(), normalization_size)
}

fn isolation_prediction(
    depth: usize,
    payload: &LeafPayload,
    normalization_size: usize,
) -> CoreResult<IsolationPrediction> {
    let LeafPayload::Isolation { occurrence_count } = payload else {
        return Err(CoreError::invalid_data(
            "isolation prediction reached a non-isolation leaf",
        ));
    };
    let normalizer = expected_path_length(normalization_size);
    if normalizer == 0.0 {
        return Err(CoreError::invalid_configuration(
            "normalization_size",
            "isolation scoring requires a normalization size of at least two",
        ));
    }
    #[allow(clippy::cast_precision_loss)]
    let path_length = depth as f64 + expected_path_length(*occurrence_count);
    let anomaly_score = 2.0_f64.powf(-path_length / normalizer);
    Ok(IsolationPrediction {
        path_length,
        anomaly_score,
    })
}

/// Isolation-forest expected unsuccessful search length.
#[must_use]
pub fn expected_path_length(size: usize) -> f64 {
    match size {
        0 | 1 => 0.0,
        2 => 1.0,
        _ => {
            #[allow(clippy::cast_precision_loss)]
            let n = size as f64;
            2.0 * harmonic_number(size - 1) - 2.0 * (n - 1.0) / n
        }
    }
}

fn harmonic_number(limit: usize) -> f64 {
    (1..=limit).map(|value| 1.0 / value as f64).sum()
}
