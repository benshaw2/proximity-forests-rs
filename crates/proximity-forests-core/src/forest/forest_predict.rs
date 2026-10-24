//! Forest-level aggregation over model-owned exemplar observations.

use std::collections::BTreeMap;

use crate::{
    CoreError, CoreResult, DistanceScratch, MultivariateSeriesView, NumericVectorView, SeedMixer,
};

use super::{
    ClassificationForest, ClassificationPrediction, IsolationForest, IsolationPrediction,
    OwnedObservationStore, RegressionForest, expected_path_length,
    predict_multivariate_classification, predict_multivariate_isolation,
    predict_multivariate_regression, predict_vector_classification, predict_vector_isolation,
    predict_vector_regression,
};

/// Aggregated multi-output classification prediction.
#[derive(Clone, Debug, PartialEq)]
pub struct ForestClassificationPrediction {
    classes: Vec<u32>,
    probabilities: Vec<Vec<(u32, f64)>>,
}

impl ForestClassificationPrediction {
    /// Returns predicted class identifiers by output.
    #[must_use]
    pub fn classes(&self) -> &[u32] {
        &self.classes
    }

    /// Returns sorted class probabilities for one output.
    #[must_use]
    pub fn probabilities(&self, output: usize) -> Option<&[(u32, f64)]> {
        self.probabilities.get(output).map(Vec::as_slice)
    }
}

/// Aggregates all trees for one vector classification query.
pub fn predict_forest_vector_classification(
    model: &ClassificationForest,
    query: NumericVectorView<'_>,
    prediction_seed: u64,
) -> CoreResult<ForestClassificationPrediction> {
    let mut scratch = DistanceScratch::new();
    let mut predictions = Vec::with_capacity(model.forest().trees().len());
    for (tree_index, tree) in model.forest().trees().iter().enumerate() {
        let Some(OwnedObservationStore::Vector(store)) = tree.exemplars() else {
            return Err(layout_error("vector tree exemplars"));
        };
        let views = store.views();
        predictions.push(predict_vector_classification(
            tree,
            query,
            &views,
            model.forest().tie_policy(),
            SeedMixer::tree(prediction_seed, tree_index),
            &mut scratch,
        )?);
    }
    aggregate_classification(&predictions, model.output_count())
}

/// Aggregates all trees for one multivariate classification query.
pub fn predict_forest_multivariate_classification(
    model: &ClassificationForest,
    query: MultivariateSeriesView<'_>,
    prediction_seed: u64,
) -> CoreResult<ForestClassificationPrediction> {
    let mut scratch = DistanceScratch::new();
    let mut predictions = Vec::with_capacity(model.forest().trees().len());
    for (tree_index, tree) in model.forest().trees().iter().enumerate() {
        let Some(OwnedObservationStore::Multivariate(store)) = tree.exemplars() else {
            return Err(layout_error("multivariate tree exemplars"));
        };
        let views = store.views()?;
        predictions.push(predict_multivariate_classification(
            tree,
            query,
            &views,
            model.forest().tie_policy(),
            SeedMixer::tree(prediction_seed, tree_index),
            &mut scratch,
        )?);
    }
    aggregate_classification(&predictions, model.output_count())
}

/// Averages all trees for one vector regression query.
pub fn predict_forest_vector_regression(
    model: &RegressionForest,
    query: NumericVectorView<'_>,
    prediction_seed: u64,
) -> CoreResult<Vec<f64>> {
    let mut scratch = DistanceScratch::new();
    let mut sums = vec![0.0; model.output_count()];
    for (tree_index, tree) in model.forest().trees().iter().enumerate() {
        let Some(OwnedObservationStore::Vector(store)) = tree.exemplars() else {
            return Err(layout_error("vector tree exemplars"));
        };
        let views = store.views();
        let values = predict_vector_regression(
            tree,
            query,
            &views,
            model.forest().tie_policy(),
            SeedMixer::tree(prediction_seed, tree_index),
            &mut scratch,
        )?;
        add_regression(&mut sums, &values)?;
    }
    divide(&mut sums, model.forest().trees().len())?;
    Ok(sums)
}

/// Averages all trees for one multivariate regression query.
pub fn predict_forest_multivariate_regression(
    model: &RegressionForest,
    query: MultivariateSeriesView<'_>,
    prediction_seed: u64,
) -> CoreResult<Vec<f64>> {
    let mut scratch = DistanceScratch::new();
    let mut sums = vec![0.0; model.output_count()];
    for (tree_index, tree) in model.forest().trees().iter().enumerate() {
        let Some(OwnedObservationStore::Multivariate(store)) = tree.exemplars() else {
            return Err(layout_error("multivariate tree exemplars"));
        };
        let views = store.views()?;
        let values = predict_multivariate_regression(
            tree,
            query,
            &views,
            model.forest().tie_policy(),
            SeedMixer::tree(prediction_seed, tree_index),
            &mut scratch,
        )?;
        add_regression(&mut sums, &values)?;
    }
    divide(&mut sums, model.forest().trees().len())?;
    Ok(sums)
}

/// Aggregates mean path length and normalized score for one vector query.
pub fn predict_forest_vector_isolation(
    model: &IsolationForest,
    query: NumericVectorView<'_>,
    prediction_seed: u64,
) -> CoreResult<IsolationPrediction> {
    let mut scratch = DistanceScratch::new();
    let mut path_sum = 0.0;
    for (tree_index, tree) in model.forest().trees().iter().enumerate() {
        let Some(OwnedObservationStore::Vector(store)) = tree.exemplars() else {
            return Err(layout_error("vector tree exemplars"));
        };
        let views = store.views();
        path_sum += predict_vector_isolation(
            tree,
            query,
            &views,
            model.normalization_size(),
            model.forest().tie_policy(),
            SeedMixer::tree(prediction_seed, tree_index),
            &mut scratch,
        )?
        .path_length();
    }
    aggregate_isolation(path_sum, model)
}

/// Aggregates mean path length and normalized score for one multivariate query.
pub fn predict_forest_multivariate_isolation(
    model: &IsolationForest,
    query: MultivariateSeriesView<'_>,
    prediction_seed: u64,
) -> CoreResult<IsolationPrediction> {
    let mut scratch = DistanceScratch::new();
    let mut path_sum = 0.0;
    for (tree_index, tree) in model.forest().trees().iter().enumerate() {
        let Some(OwnedObservationStore::Multivariate(store)) = tree.exemplars() else {
            return Err(layout_error("multivariate tree exemplars"));
        };
        let views = store.views()?;
        path_sum += predict_multivariate_isolation(
            tree,
            query,
            &views,
            model.normalization_size(),
            model.forest().tie_policy(),
            SeedMixer::tree(prediction_seed, tree_index),
            &mut scratch,
        )?
        .path_length();
    }
    aggregate_isolation(path_sum, model)
}

fn aggregate_classification(
    predictions: &[ClassificationPrediction],
    output_count: usize,
) -> CoreResult<ForestClassificationPrediction> {
    if predictions.is_empty() {
        return Err(CoreError::invalid_data(
            "classification forest has no fitted trees",
        ));
    }
    let mut classes = Vec::with_capacity(output_count);
    let mut probabilities = Vec::with_capacity(output_count);
    for output in 0..output_count {
        let mut sums = BTreeMap::<u32, f64>::new();
        for prediction in predictions {
            let values = prediction.probabilities(output).ok_or_else(|| {
                CoreError::invalid_data("tree classification output was unavailable")
            })?;
            for &(class, probability) in values {
                *sums.entry(class).or_default() += probability;
            }
        }
        let tree_count = predictions.len() as f64;
        let mut best = None::<(u32, f64)>;
        let values: Vec<_> = sums
            .into_iter()
            .map(|(class, sum)| {
                let probability = sum / tree_count;
                if best.is_none_or(|(_, best_probability)| probability > best_probability) {
                    best = Some((class, probability));
                }
                (class, probability)
            })
            .collect();
        classes.push(
            best.ok_or_else(|| {
                CoreError::invalid_data("classification forest produced no class probabilities")
            })?
            .0,
        );
        probabilities.push(values);
    }
    Ok(ForestClassificationPrediction {
        classes,
        probabilities,
    })
}

fn add_regression(sums: &mut [f64], values: &[f64]) -> CoreResult<()> {
    if sums.len() != values.len() {
        return Err(CoreError::shape_mismatch(
            "regression tree output count mismatch",
        ));
    }
    for (sum, &value) in sums.iter_mut().zip(values) {
        *sum += value;
    }
    Ok(())
}

fn divide(values: &mut [f64], count: usize) -> CoreResult<()> {
    if count == 0 {
        return Err(CoreError::invalid_data("fitted forest has no trees"));
    }
    let denominator = count as f64;
    for value in values {
        *value /= denominator;
    }
    Ok(())
}

fn aggregate_isolation(path_sum: f64, model: &IsolationForest) -> CoreResult<IsolationPrediction> {
    let tree_count = model.forest().trees().len();
    if tree_count == 0 {
        return Err(CoreError::invalid_data(
            "isolation forest has no fitted trees",
        ));
    }
    let path_length = path_sum / tree_count as f64;
    let normalizer = expected_path_length(model.normalization_size());
    if normalizer == 0.0 {
        return Err(CoreError::invalid_configuration(
            "normalization_size",
            "isolation scoring requires at least two fitting observations",
        ));
    }
    Ok(IsolationPrediction::from_parts(
        path_length,
        2.0_f64.powf(-path_length / normalizer),
    ))
}

fn layout_error(expected: &str) -> CoreError {
    CoreError::invalid_data(format!(
        "fitted model does not contain {expected} observations"
    ))
}
