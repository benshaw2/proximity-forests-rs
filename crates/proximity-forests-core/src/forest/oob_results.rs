//! Out-of-bag predictions and scores for supervised proximity forests.
//!
//! This module intentionally consolidates classification, regression, and
//! zero-vote semantics. Parallel scheduling stays outside the numerical result
//! contract and is added only after the sequential reference path is locked.

use crate::{
    ClassificationTargets, CoreError, CoreResult, DistanceScratch, MultivariateSeriesView,
    NumericVectorView, RegressionTargets, SeedMixer,
};

use super::{
    ClassificationForest, ClassificationPrediction, OobMembership, OwnedObservationStore,
    RegressionForest, oob_execution_context, parallel_oob_rows_in,
    predict_multivariate_classification, predict_multivariate_regression,
    predict_vector_classification, predict_vector_regression,
};

/// Dense OOB classification results in row-major sample/output/class storage.
#[derive(Clone, Debug, PartialEq)]
pub struct OobClassificationResult {
    class_counts: Box<[usize]>,
    offsets: Box<[usize]>,
    probabilities: Box<[f64]>,
    predictions: Box<[Option<u32>]>,
    vote_counts: Box<[usize]>,
    score: f64,
}

impl OobClassificationResult {
    /// Returns one fitted class count per classification output.
    #[must_use]
    pub fn class_counts(&self) -> &[usize] {
        &self.class_counts
    }
    /// Returns the number of OOB tree votes received by each sample.
    #[must_use]
    pub fn vote_counts(&self) -> &[usize] {
        &self.vote_counts
    }
    /// Returns row-major encoded predictions, with `None` for zero-vote rows.
    #[must_use]
    pub fn predictions(&self) -> &[Option<u32>] {
        &self.predictions
    }
    /// Returns the OOB score computed only from sample-output pairs with votes.
    /// Returns the OOB score computed only from sample-output pairs with votes.
    #[must_use]
    pub const fn score(&self) -> f64 {
        self.score
    }
    /// Returns class probabilities for one sample and output.
    ///
    /// Zero-vote rows contain `NaN` values. Out-of-range indices return `None`.
    #[must_use]
    pub fn probabilities(&self, sample: usize, output: usize) -> Option<&[f64]> {
        let output_count = self.class_counts.len();
        if sample >= self.vote_counts.len() || output >= output_count {
            return None;
        }
        let row = sample.checked_mul(output_count)?.checked_add(output)?;
        let start = *self.offsets.get(row)?;
        let end = *self.offsets.get(row + 1)?;
        self.probabilities.get(start..end)
    }
    /// Returns the number of training samples that received no OOB votes.
    #[must_use]
    pub fn zero_vote_count(&self) -> usize {
        self.vote_counts.iter().filter(|&&count| count == 0).count()
    }
}

/// Dense row-major OOB regression results.
#[derive(Clone, Debug, PartialEq)]
pub struct OobRegressionResult {
    sample_count: usize,
    output_count: usize,
    predictions: Box<[f64]>,
    vote_counts: Box<[usize]>,
    score: f64,
}

impl OobRegressionResult {
    /// Returns the number of training samples represented by the result.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }
    /// Returns the number of regression outputs.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }
    /// Returns row-major OOB predictions, with `NaN` for zero-vote rows.
    #[must_use]
    pub fn predictions(&self) -> &[f64] {
        &self.predictions
    }
    /// Returns the number of OOB tree votes received by each sample.
    #[must_use]
    pub fn vote_counts(&self) -> &[usize] {
        &self.vote_counts
    }
    /// Returns uniformly averaged multi-output OOB R-squared.
    ///
    /// Samples with zero OOB votes are excluded. If no output has at
    /// least two voted samples, the result is `NaN`.
    #[must_use]
    pub const fn score(&self) -> f64 {
        self.score
    }
    /// Returns all regression outputs for one training sample.
    #[must_use]
    pub fn row(&self, sample: usize) -> Option<&[f64]> {
        if sample >= self.sample_count {
            return None;
        }
        let start = sample * self.output_count;
        self.predictions.get(start..start + self.output_count)
    }
    /// Returns the number of training samples that received no OOB votes.
    #[must_use]
    pub fn zero_vote_count(&self) -> usize {
        self.vote_counts.iter().filter(|&&count| count == 0).count()
    }
}

/// Computes sequential OOB classification results for vector observations.
pub fn oob_vector_classification(
    model: &ClassificationForest,
    observations: &[NumericVectorView<'_>],
    targets: &ClassificationTargets,
    class_counts: &[usize],
    prediction_seed: u64,
) -> CoreResult<OobClassificationResult> {
    validate_classification_inputs(model, observations.len(), targets, class_counts)?;
    let membership = model.forest().training_metadata().oob_membership()?;
    let mut scratch = DistanceScratch::new();
    build_classification_result(&membership, targets, class_counts, |sample, tree_index| {
        let tree = model.forest().trees().get(tree_index).ok_or_else(|| {
            CoreError::invalid_data("OOB membership references a missing classification tree")
        })?;
        let Some(OwnedObservationStore::Vector(store)) = tree.exemplars() else {
            return Err(CoreError::invalid_data(
                "classification tree does not contain vector exemplars",
            ));
        };
        predict_vector_classification(
            tree,
            observations[sample],
            &store.views(),
            model.forest().tie_policy(),
            SeedMixer::tree(prediction_seed, tree_index),
            &mut scratch,
        )
    })
}

/// Computes sequential OOB classification results for multivariate observations.
pub fn oob_multivariate_classification(
    model: &ClassificationForest,
    observations: &[MultivariateSeriesView<'_>],
    targets: &ClassificationTargets,
    class_counts: &[usize],
    prediction_seed: u64,
) -> CoreResult<OobClassificationResult> {
    validate_classification_inputs(model, observations.len(), targets, class_counts)?;
    let membership = model.forest().training_metadata().oob_membership()?;
    let mut scratch = DistanceScratch::new();
    build_classification_result(&membership, targets, class_counts, |sample, tree_index| {
        let tree = model.forest().trees().get(tree_index).ok_or_else(|| {
            CoreError::invalid_data("OOB membership references a missing classification tree")
        })?;
        let Some(OwnedObservationStore::Multivariate(store)) = tree.exemplars() else {
            return Err(CoreError::invalid_data(
                "classification tree does not contain multivariate exemplars",
            ));
        };
        predict_multivariate_classification(
            tree,
            observations[sample],
            &store.views()?,
            model.forest().tie_policy(),
            SeedMixer::tree(prediction_seed, tree_index),
            &mut scratch,
        )
    })
}

/// Computes sequential OOB regression results for vector observations.
pub fn oob_vector_regression(
    model: &RegressionForest,
    observations: &[NumericVectorView<'_>],
    targets: &RegressionTargets,
    prediction_seed: u64,
) -> CoreResult<OobRegressionResult> {
    validate_regression_inputs(model, observations.len(), targets)?;
    let membership = model.forest().training_metadata().oob_membership()?;
    let mut scratch = DistanceScratch::new();
    build_regression_result(&membership, targets, |sample, tree_index| {
        let tree = model.forest().trees().get(tree_index).ok_or_else(|| {
            CoreError::invalid_data("OOB membership references a missing regression tree")
        })?;
        let Some(OwnedObservationStore::Vector(store)) = tree.exemplars() else {
            return Err(CoreError::invalid_data(
                "regression tree does not contain vector exemplars",
            ));
        };
        predict_vector_regression(
            tree,
            observations[sample],
            &store.views(),
            model.forest().tie_policy(),
            SeedMixer::tree(prediction_seed, tree_index),
            &mut scratch,
        )
    })
}

/// Computes sequential OOB regression results for multivariate observations.
pub fn oob_multivariate_regression(
    model: &RegressionForest,
    observations: &[MultivariateSeriesView<'_>],
    targets: &RegressionTargets,
    prediction_seed: u64,
) -> CoreResult<OobRegressionResult> {
    validate_regression_inputs(model, observations.len(), targets)?;
    let membership = model.forest().training_metadata().oob_membership()?;
    let mut scratch = DistanceScratch::new();
    build_regression_result(&membership, targets, |sample, tree_index| {
        let tree = model.forest().trees().get(tree_index).ok_or_else(|| {
            CoreError::invalid_data("OOB membership references a missing regression tree")
        })?;
        let Some(OwnedObservationStore::Multivariate(store)) = tree.exemplars() else {
            return Err(CoreError::invalid_data(
                "regression tree does not contain multivariate exemplars",
            ));
        };
        predict_multivariate_regression(
            tree,
            observations[sample],
            &store.views()?,
            model.forest().tie_policy(),
            SeedMixer::tree(prediction_seed, tree_index),
            &mut scratch,
        )
    })
}

#[derive(Debug)]
struct ClassificationOobRow {
    probabilities: Vec<f64>,
    predictions: Vec<u32>,
}

#[derive(Debug)]
struct RegressionOobRow {
    predictions: Vec<f64>,
}

/// Computes bounded parallel OOB classification results for vector observations.
pub fn oob_vector_classification_in(
    model: &ClassificationForest,
    observations: &[NumericVectorView<'_>],
    targets: &ClassificationTargets,
    class_counts: &[usize],
    prediction_seed: u64,
    requested_workers: usize,
) -> CoreResult<OobClassificationResult> {
    validate_classification_inputs(model, observations.len(), targets, class_counts)?;
    let membership = model.forest().training_metadata().oob_membership()?;
    let context = oob_execution_context(&membership, requested_workers)?;
    let rows = parallel_oob_rows_in(
        &context,
        &membership,
        DistanceScratch::new,
        |scratch, sample, trees| {
            classification_row(class_counts, trees, |tree_index| {
                let tree = model.forest().trees().get(tree_index).ok_or_else(|| {
                    CoreError::invalid_data(
                        "OOB membership references a missing classification tree",
                    )
                })?;
                let Some(OwnedObservationStore::Vector(store)) = tree.exemplars() else {
                    return Err(CoreError::invalid_data(
                        "classification tree does not contain vector exemplars",
                    ));
                };
                predict_vector_classification(
                    tree,
                    observations[sample],
                    &store.views(),
                    model.forest().tie_policy(),
                    SeedMixer::tree(prediction_seed, tree_index),
                    scratch,
                )
            })
        },
    )?;
    assemble_classification_rows(&membership, targets, class_counts, rows)
}

/// Computes bounded parallel OOB classification results for multivariate observations.
pub fn oob_multivariate_classification_in(
    model: &ClassificationForest,
    observations: &[MultivariateSeriesView<'_>],
    targets: &ClassificationTargets,
    class_counts: &[usize],
    prediction_seed: u64,
    requested_workers: usize,
) -> CoreResult<OobClassificationResult> {
    validate_classification_inputs(model, observations.len(), targets, class_counts)?;
    let membership = model.forest().training_metadata().oob_membership()?;
    let context = oob_execution_context(&membership, requested_workers)?;
    let rows = parallel_oob_rows_in(
        &context,
        &membership,
        DistanceScratch::new,
        |scratch, sample, trees| {
            classification_row(class_counts, trees, |tree_index| {
                let tree = model.forest().trees().get(tree_index).ok_or_else(|| {
                    CoreError::invalid_data(
                        "OOB membership references a missing classification tree",
                    )
                })?;
                let Some(OwnedObservationStore::Multivariate(store)) = tree.exemplars() else {
                    return Err(CoreError::invalid_data(
                        "classification tree does not contain multivariate exemplars",
                    ));
                };
                predict_multivariate_classification(
                    tree,
                    observations[sample],
                    &store.views()?,
                    model.forest().tie_policy(),
                    SeedMixer::tree(prediction_seed, tree_index),
                    scratch,
                )
            })
        },
    )?;
    assemble_classification_rows(&membership, targets, class_counts, rows)
}

/// Computes bounded parallel OOB regression results for vector observations.
pub fn oob_vector_regression_in(
    model: &RegressionForest,
    observations: &[NumericVectorView<'_>],
    targets: &RegressionTargets,
    prediction_seed: u64,
    requested_workers: usize,
) -> CoreResult<OobRegressionResult> {
    validate_regression_inputs(model, observations.len(), targets)?;
    let membership = model.forest().training_metadata().oob_membership()?;
    let context = oob_execution_context(&membership, requested_workers)?;
    let rows = parallel_oob_rows_in(
        &context,
        &membership,
        DistanceScratch::new,
        |scratch, sample, trees| {
            regression_row(targets.output_count(), trees, |tree_index| {
                let tree = model.forest().trees().get(tree_index).ok_or_else(|| {
                    CoreError::invalid_data("OOB membership references a missing regression tree")
                })?;
                let Some(OwnedObservationStore::Vector(store)) = tree.exemplars() else {
                    return Err(CoreError::invalid_data(
                        "regression tree does not contain vector exemplars",
                    ));
                };
                predict_vector_regression(
                    tree,
                    observations[sample],
                    &store.views(),
                    model.forest().tie_policy(),
                    SeedMixer::tree(prediction_seed, tree_index),
                    scratch,
                )
            })
        },
    )?;
    assemble_regression_rows(&membership, targets, rows)
}

/// Computes bounded parallel OOB regression results for multivariate observations.
pub fn oob_multivariate_regression_in(
    model: &RegressionForest,
    observations: &[MultivariateSeriesView<'_>],
    targets: &RegressionTargets,
    prediction_seed: u64,
    requested_workers: usize,
) -> CoreResult<OobRegressionResult> {
    validate_regression_inputs(model, observations.len(), targets)?;
    let membership = model.forest().training_metadata().oob_membership()?;
    let context = oob_execution_context(&membership, requested_workers)?;
    let rows = parallel_oob_rows_in(
        &context,
        &membership,
        DistanceScratch::new,
        |scratch, sample, trees| {
            regression_row(targets.output_count(), trees, |tree_index| {
                let tree = model.forest().trees().get(tree_index).ok_or_else(|| {
                    CoreError::invalid_data("OOB membership references a missing regression tree")
                })?;
                let Some(OwnedObservationStore::Multivariate(store)) = tree.exemplars() else {
                    return Err(CoreError::invalid_data(
                        "regression tree does not contain multivariate exemplars",
                    ));
                };
                predict_multivariate_regression(
                    tree,
                    observations[sample],
                    &store.views()?,
                    model.forest().tie_policy(),
                    SeedMixer::tree(prediction_seed, tree_index),
                    scratch,
                )
            })
        },
    )?;
    assemble_regression_rows(&membership, targets, rows)
}

fn classification_row<F>(
    class_counts: &[usize],
    trees: &[usize],
    mut predict: F,
) -> CoreResult<ClassificationOobRow>
where
    F: FnMut(usize) -> CoreResult<ClassificationPrediction>,
{
    let row_width = class_counts.iter().sum::<usize>();
    if trees.is_empty() {
        return Ok(ClassificationOobRow {
            probabilities: vec![f64::NAN; row_width],
            predictions: Vec::new(),
        });
    }
    let stride = class_counts.iter().copied().max().unwrap_or(0);
    let mut sums = vec![0.0; class_counts.len() * stride];
    for &tree_index in trees {
        let prediction = predict(tree_index)?;
        for output in 0..class_counts.len() {
            for &(class, probability) in prediction.probabilities(output).ok_or_else(|| {
                CoreError::invalid_data("OOB classification probability output is missing")
            })? {
                let class = usize::try_from(class).map_err(|_| {
                    CoreError::invalid_data("OOB class identifier overflowed usize")
                })?;
                if class >= class_counts[output] {
                    return Err(CoreError::invalid_data(
                        "OOB class identifier is out of range",
                    ));
                }
                sums[output * stride + class] += probability;
            }
        }
    }
    let mut probabilities = Vec::with_capacity(row_width);
    let mut predictions = Vec::with_capacity(class_counts.len());
    for (output, &count) in class_counts.iter().enumerate() {
        let mut best = 0usize;
        let mut best_value = f64::NEG_INFINITY;
        for class in 0..count {
            let value = sums[output * stride + class] / trees.len() as f64;
            probabilities.push(value);
            if value > best_value {
                best_value = value;
                best = class;
            }
        }
        predictions.push(best as u32);
    }
    Ok(ClassificationOobRow {
        probabilities,
        predictions,
    })
}

fn regression_row<F>(
    output_count: usize,
    trees: &[usize],
    mut predict: F,
) -> CoreResult<RegressionOobRow>
where
    F: FnMut(usize) -> CoreResult<Vec<f64>>,
{
    if trees.is_empty() {
        return Ok(RegressionOobRow {
            predictions: vec![f64::NAN; output_count],
        });
    }
    let mut sums = vec![0.0; output_count];
    for &tree_index in trees {
        let values = predict(tree_index)?;
        if values.len() != output_count {
            return Err(CoreError::shape_mismatch(
                "OOB regression prediction width mismatch",
            ));
        }
        for (sum, value) in sums.iter_mut().zip(values) {
            *sum += value;
        }
    }
    for sum in &mut sums {
        *sum /= trees.len() as f64;
    }
    Ok(RegressionOobRow { predictions: sums })
}

fn assemble_classification_rows(
    membership: &OobMembership,
    targets: &ClassificationTargets,
    class_counts: &[usize],
    rows: Vec<ClassificationOobRow>,
) -> CoreResult<OobClassificationResult> {
    let output_count = class_counts.len();
    let mut probabilities = Vec::new();
    let mut predictions = Vec::with_capacity(rows.len() * output_count);
    let mut correct = 0usize;
    let mut scored = 0usize;
    for (sample, row) in rows.into_iter().enumerate() {
        probabilities.extend(row.probabilities);
        if row.predictions.is_empty() {
            predictions.extend((0..output_count).map(|_| None));
        } else {
            for (output, class) in row.predictions.into_iter().enumerate() {
                if targets.get(sample, output) == Some(class) {
                    correct += 1;
                }
                scored += 1;
                predictions.push(Some(class));
            }
        }
    }
    let mut offsets = Vec::with_capacity(membership.sample_count() * output_count + 1);
    offsets.push(0usize);
    for _ in 0..membership.sample_count() {
        for &count in class_counts {
            offsets.push(offsets.last().copied().unwrap_or(0) + count);
        }
    }
    Ok(OobClassificationResult {
        class_counts: class_counts.to_vec().into_boxed_slice(),
        offsets: offsets.into_boxed_slice(),
        probabilities: probabilities.into_boxed_slice(),
        predictions: predictions.into_boxed_slice(),
        vote_counts: membership.vote_counts().into_boxed_slice(),
        score: if scored == 0 {
            f64::NAN
        } else {
            correct as f64 / scored as f64
        },
    })
}

fn assemble_regression_rows(
    membership: &OobMembership,
    targets: &RegressionTargets,
    rows: Vec<RegressionOobRow>,
) -> CoreResult<OobRegressionResult> {
    let predictions = rows
        .into_iter()
        .flat_map(|row| row.predictions)
        .collect::<Vec<_>>();
    let vote_counts = membership.vote_counts();
    let score = regression_r2(targets, &predictions, &vote_counts)?;
    Ok(OobRegressionResult {
        sample_count: membership.sample_count(),
        output_count: targets.output_count(),
        predictions: predictions.into_boxed_slice(),
        vote_counts: vote_counts.into_boxed_slice(),
        score,
    })
}

fn validate_classification_inputs(
    model: &ClassificationForest,
    sample_count: usize,
    targets: &ClassificationTargets,
    class_counts: &[usize],
) -> CoreResult<()> {
    if sample_count != targets.sample_count()
        || sample_count != model.forest().training_metadata().sample_count()
    {
        return Err(CoreError::shape_mismatch(
            "OOB classification inputs disagree on sample count",
        ));
    }
    if class_counts.len() != targets.output_count() || class_counts.len() != model.output_count() {
        return Err(CoreError::shape_mismatch(
            "OOB classification output counts disagree",
        ));
    }
    if class_counts.iter().any(|&count| count == 0) {
        return Err(CoreError::invalid_data(
            "OOB classification class counts must be positive",
        ));
    }
    Ok(())
}

fn validate_regression_inputs(
    model: &RegressionForest,
    sample_count: usize,
    targets: &RegressionTargets,
) -> CoreResult<()> {
    if sample_count != targets.sample_count()
        || sample_count != model.forest().training_metadata().sample_count()
    {
        return Err(CoreError::shape_mismatch(
            "OOB regression inputs disagree on sample count",
        ));
    }
    if targets.output_count() != model.output_count() {
        return Err(CoreError::shape_mismatch(
            "OOB regression output counts disagree",
        ));
    }
    Ok(())
}

fn build_classification_result<F>(
    membership: &OobMembership,
    targets: &ClassificationTargets,
    class_counts: &[usize],
    mut predict: F,
) -> CoreResult<OobClassificationResult>
where
    F: FnMut(usize, usize) -> CoreResult<ClassificationPrediction>,
{
    let sample_count = membership.sample_count();
    let output_count = class_counts.len();
    let row_width: usize = class_counts.iter().sum();
    let mut probabilities = vec![f64::NAN; sample_count * row_width];
    let mut predictions = vec![None; sample_count * output_count];
    let vote_counts = membership.vote_counts();
    let mut correct = 0usize;
    let mut scored = 0usize;

    for (sample, trees) in membership.rows() {
        if trees.is_empty() {
            continue;
        }
        let mut sums = vec![0.0; output_count * class_counts.iter().copied().max().unwrap_or(0)];
        let stride = class_counts.iter().copied().max().unwrap_or(0);
        for &tree_index in trees {
            let prediction = predict(sample, tree_index)?;
            for output in 0..output_count {
                for &(class, probability) in prediction.probabilities(output).ok_or_else(|| {
                    CoreError::invalid_data("OOB classification probability output is missing")
                })? {
                    let class = usize::try_from(class).map_err(|_| {
                        CoreError::invalid_data("OOB class identifier overflowed usize")
                    })?;
                    if class >= class_counts[output] {
                        return Err(CoreError::invalid_data(
                            "OOB class identifier is out of range",
                        ));
                    }
                    sums[output * stride + class] += probability;
                }
            }
        }
        let mut probability_offset = sample * row_width;
        for output in 0..output_count {
            let mut best_class = 0usize;
            let mut best_probability = f64::NEG_INFINITY;
            for class in 0..class_counts[output] {
                let value = sums[output * stride + class] / trees.len() as f64;
                probabilities[probability_offset + class] = value;
                if value > best_probability {
                    best_probability = value;
                    best_class = class;
                }
            }
            predictions[sample * output_count + output] = Some(best_class as u32);
            if targets.get(sample, output) == Some(best_class as u32) {
                correct += 1;
            }
            scored += 1;
            probability_offset += class_counts[output];
        }
    }

    let mut offsets = Vec::with_capacity(sample_count * output_count + 1);
    offsets.push(0usize);
    for _sample in 0..sample_count {
        for &count in class_counts {
            offsets.push(offsets.last().copied().unwrap_or(0) + count);
        }
    }
    let score = if scored == 0 {
        f64::NAN
    } else {
        correct as f64 / scored as f64
    };
    Ok(OobClassificationResult {
        class_counts: class_counts.to_vec().into_boxed_slice(),
        offsets: offsets.into_boxed_slice(),
        probabilities: probabilities.into_boxed_slice(),
        predictions: predictions.into_boxed_slice(),
        vote_counts: vote_counts.into_boxed_slice(),
        score,
    })
}

fn build_regression_result<F>(
    membership: &OobMembership,
    targets: &RegressionTargets,
    mut predict: F,
) -> CoreResult<OobRegressionResult>
where
    F: FnMut(usize, usize) -> CoreResult<Vec<f64>>,
{
    let sample_count = membership.sample_count();
    let output_count = targets.output_count();
    let mut predictions = vec![f64::NAN; sample_count * output_count];
    let vote_counts = membership.vote_counts();
    for (sample, trees) in membership.rows() {
        if trees.is_empty() {
            continue;
        }
        let mut sums = vec![0.0; output_count];
        for &tree_index in trees {
            let values = predict(sample, tree_index)?;
            if values.len() != output_count {
                return Err(CoreError::shape_mismatch(
                    "OOB regression prediction width mismatch",
                ));
            }
            for (sum, value) in sums.iter_mut().zip(values) {
                *sum += value;
            }
        }
        for output in 0..output_count {
            predictions[sample * output_count + output] = sums[output] / trees.len() as f64;
        }
    }
    let score = regression_r2(targets, &predictions, &vote_counts)?;
    Ok(OobRegressionResult {
        sample_count,
        output_count,
        predictions: predictions.into_boxed_slice(),
        vote_counts: vote_counts.into_boxed_slice(),
        score,
    })
}

fn regression_r2(
    targets: &RegressionTargets,
    predictions: &[f64],
    vote_counts: &[usize],
) -> CoreResult<f64> {
    let mut output_scores = Vec::with_capacity(targets.output_count());
    for output in 0..targets.output_count() {
        let valid: Vec<_> = (0..targets.sample_count())
            .filter(|&sample| vote_counts[sample] > 0)
            .collect();
        if valid.len() < 2 {
            continue;
        }
        let mean = valid
            .iter()
            .map(|&sample| targets.get_f64(sample, output).unwrap())
            .sum::<f64>()
            / valid.len() as f64;
        let mut residual = 0.0;
        let mut total = 0.0;
        for sample in valid {
            let target = targets
                .get_f64(sample, output)
                .ok_or_else(|| CoreError::invalid_data("OOB regression target is unavailable"))?;
            let prediction = predictions[sample * targets.output_count() + output];
            residual += (target - prediction).powi(2);
            total += (target - mean).powi(2);
        }
        output_scores.push(if total == 0.0 {
            if residual == 0.0 { 1.0 } else { 0.0 }
        } else {
            1.0 - residual / total
        });
    }
    if output_scores.is_empty() {
        return Ok(f64::NAN);
    }
    Ok(output_scores.iter().sum::<f64>() / output_scores.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::{build_classification_result, build_regression_result};
    use crate::forest::{ClassificationPrediction, OobMembership, TreeSample};
    use crate::{ClassificationTargets, RegressionTargets};

    fn membership() -> OobMembership {
        OobMembership::from_tree_samples(
            3,
            &[
                TreeSample::from_parts(vec![0, 2], vec![1, 0, 1], vec![1]).unwrap(),
                TreeSample::from_parts(vec![1, 2], vec![0, 1, 1], vec![0]).unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn classification_preserves_nan_rows_and_scores_voted_rows() {
        let targets = ClassificationTargets::single_output(vec![0, 1, 1]).unwrap();
        let result = build_classification_result(&membership(), &targets, &[2], |_sample, tree| {
            Ok(ClassificationPrediction::from_parts_for_test(
                vec![tree as u32],
                vec![vec![
                    (0, if tree == 0 { 0.0 } else { 1.0 }),
                    (1, if tree == 0 { 1.0 } else { 0.0 }),
                ]],
            ))
        })
        .unwrap();
        assert_eq!(result.vote_counts(), &[1, 1, 0]);
        assert_eq!(result.zero_vote_count(), 1);
        assert!(
            result
                .probabilities(2, 0)
                .unwrap()
                .iter()
                .all(|value| value.is_nan())
        );
        assert!(result.score().is_finite());
    }

    #[test]
    fn regression_preserves_nan_rows_and_computes_score() {
        let targets = RegressionTargets::from_f64(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        let result = build_regression_result(&membership(), &targets, |sample, _tree| {
            Ok(vec![sample as f64 + 1.0])
        })
        .unwrap();
        assert_eq!(result.vote_counts(), &[1, 1, 0]);
        assert!(result.row(2).unwrap()[0].is_nan());
        assert_eq!(result.score(), 1.0);
    }
}
