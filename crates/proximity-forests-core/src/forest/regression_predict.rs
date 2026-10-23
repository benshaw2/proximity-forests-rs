//! Regression prediction from fitted leaf means.

use crate::{
    CoreError, CoreResult, DistanceScratch, MultivariateSeriesView, NumericVectorView, TiePolicy,
};

use super::{LeafPayload, ProximityTree, traverse_multivariate, traverse_vector};

/// Predicts one vector observation's regression outputs.
pub fn predict_vector_regression(
    tree: &ProximityTree,
    query: NumericVectorView<'_>,
    observations: &[NumericVectorView<'_>],
    tie_policy: TiePolicy,
    prediction_seed: u64,
    scratch: &mut DistanceScratch,
) -> CoreResult<Vec<f64>> {
    let visit = traverse_vector(
        tree,
        query,
        observations,
        tie_policy,
        prediction_seed,
        scratch,
    )?;
    regression_prediction(visit.leaf().payload())
}

/// Predicts one multivariate observation's regression outputs.
pub fn predict_multivariate_regression(
    tree: &ProximityTree,
    query: MultivariateSeriesView<'_>,
    observations: &[MultivariateSeriesView<'_>],
    tie_policy: TiePolicy,
    prediction_seed: u64,
    scratch: &mut DistanceScratch,
) -> CoreResult<Vec<f64>> {
    let visit = traverse_multivariate(
        tree,
        query,
        observations,
        tie_policy,
        prediction_seed,
        scratch,
    )?;
    regression_prediction(visit.leaf().payload())
}

fn regression_prediction(payload: &LeafPayload) -> CoreResult<Vec<f64>> {
    let LeafPayload::Regression(leaf) = payload else {
        return Err(CoreError::invalid_data(
            "regression prediction reached a non-regression leaf",
        ));
    };
    Ok(leaf.means().to_vec())
}
