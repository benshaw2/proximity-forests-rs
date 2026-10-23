//! Query-parallel batch prediction for fitted forest models.

use crate::{CoreResult, MultivariateSeriesView, NumericVectorView, SeedMixer};

use super::{
    ClassificationForest, ForestClassificationPrediction, IsolationForest, IsolationPrediction,
    RegressionForest, parallel_indexed, predict_forest_multivariate_classification,
    predict_forest_multivariate_isolation, predict_forest_multivariate_regression,
    predict_forest_vector_classification, predict_forest_vector_isolation,
    predict_forest_vector_regression,
};

/// Predicts a batch with a vector-layout classification forest.
pub fn predict_vector_classification_many(
    model: &ClassificationForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<ForestClassificationPrediction>> {
    parallel_indexed(queries.len(), thread_count, |sample| {
        predict_forest_vector_classification(
            model,
            queries[sample],
            observation_seed(prediction_seed, sample),
        )
    })
}

/// Predicts a batch with a multivariate classification forest.
pub fn predict_multivariate_classification_many(
    model: &ClassificationForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<ForestClassificationPrediction>> {
    parallel_indexed(queries.len(), thread_count, |sample| {
        predict_forest_multivariate_classification(
            model,
            queries[sample],
            observation_seed(prediction_seed, sample),
        )
    })
}

/// Predicts a batch with a vector-layout regression forest.
pub fn predict_vector_regression_many(
    model: &RegressionForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<Vec<f64>>> {
    parallel_indexed(queries.len(), thread_count, |sample| {
        predict_forest_vector_regression(
            model,
            queries[sample],
            observation_seed(prediction_seed, sample),
        )
    })
}

/// Predicts a batch with a multivariate regression forest.
pub fn predict_multivariate_regression_many(
    model: &RegressionForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<Vec<f64>>> {
    parallel_indexed(queries.len(), thread_count, |sample| {
        predict_forest_multivariate_regression(
            model,
            queries[sample],
            observation_seed(prediction_seed, sample),
        )
    })
}

/// Scores a batch with a vector-layout isolation forest.
pub fn score_vector_isolation_many(
    model: &IsolationForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<IsolationPrediction>> {
    parallel_indexed(queries.len(), thread_count, |sample| {
        predict_forest_vector_isolation(
            model,
            queries[sample],
            observation_seed(prediction_seed, sample),
        )
    })
}

/// Scores a batch with a multivariate isolation forest.
pub fn score_multivariate_isolation_many(
    model: &IsolationForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<IsolationPrediction>> {
    parallel_indexed(queries.len(), thread_count, |sample| {
        predict_forest_multivariate_isolation(
            model,
            queries[sample],
            observation_seed(prediction_seed, sample),
        )
    })
}

/// Returns only anomaly scores for a vector-layout query batch.
pub fn anomaly_scores_vector_many(
    model: &IsolationForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<f64>> {
    Ok(
        score_vector_isolation_many(model, queries, prediction_seed, thread_count)?
            .into_iter()
            .map(IsolationPrediction::anomaly_score)
            .collect(),
    )
}

/// Returns only anomaly scores for a multivariate query batch.
pub fn anomaly_scores_multivariate_many(
    model: &IsolationForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<f64>> {
    Ok(
        score_multivariate_isolation_many(model, queries, prediction_seed, thread_count)?
            .into_iter()
            .map(IsolationPrediction::anomaly_score)
            .collect(),
    )
}

#[inline]
fn observation_seed(base_seed: u64, sample: usize) -> u64 {
    SeedMixer::observation(base_seed, sample as u64)
}
