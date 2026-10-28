//! Query-parallel batch prediction for fitted forest models.

use crate::{
    CoreError, CoreResult, DistanceScratch, MultivariateSeriesView, NumericVectorView, SeedMixer,
};

use super::{
    ClassificationForest, ClassificationPrediction, ExecutionContext,
    ForestClassificationPrediction, IsolationForest, IsolationPrediction, OwnedObservationStore,
    RegressionForest,
    forest_predict::{aggregate_classification, aggregate_isolation},
    parallel_indexed_in, predict_multivariate_classification, predict_multivariate_isolation,
    predict_multivariate_regression, predict_vector_classification, predict_vector_isolation,
    predict_vector_regression,
    work_estimate::{InferenceTilePlan, inference_thread_count},
};

/// Predicts a batch with a vector-layout classification forest.
pub fn predict_vector_classification_many(
    model: &ClassificationForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<ForestClassificationPrediction>> {
    let thread_count =
        inference_thread_count(queries.len(), model.forest().trees().len(), thread_count);
    let context = ExecutionContext::new(thread_count)?;
    predict_vector_classification_tiled(model, queries, prediction_seed, &context)
}

/// Predicts a batch with a multivariate classification forest.
pub fn predict_multivariate_classification_many(
    model: &ClassificationForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<ForestClassificationPrediction>> {
    let thread_count =
        inference_thread_count(queries.len(), model.forest().trees().len(), thread_count);
    let context = ExecutionContext::new(thread_count)?;
    predict_multivariate_classification_tiled(model, queries, prediction_seed, &context)
}

/// Predicts a batch with a vector-layout regression forest.
pub fn predict_vector_regression_many(
    model: &RegressionForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<Vec<f64>>> {
    let thread_count =
        inference_thread_count(queries.len(), model.forest().trees().len(), thread_count);
    let context = ExecutionContext::new(thread_count)?;
    predict_vector_regression_tiled(model, queries, prediction_seed, &context)
}

/// Predicts a batch with a multivariate regression forest.
pub fn predict_multivariate_regression_many(
    model: &RegressionForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<Vec<f64>>> {
    let thread_count =
        inference_thread_count(queries.len(), model.forest().trees().len(), thread_count);
    let context = ExecutionContext::new(thread_count)?;
    predict_multivariate_regression_tiled(model, queries, prediction_seed, &context)
}

/// Scores a batch with a vector-layout isolation forest.
pub fn score_vector_isolation_many(
    model: &IsolationForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<IsolationPrediction>> {
    let thread_count =
        inference_thread_count(queries.len(), model.forest().trees().len(), thread_count);
    let context = ExecutionContext::new(thread_count)?;
    predict_vector_isolation_tiled(model, queries, prediction_seed, &context)
}

/// Scores a batch with a multivariate isolation forest.
pub fn score_multivariate_isolation_many(
    model: &IsolationForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<Vec<IsolationPrediction>> {
    let thread_count =
        inference_thread_count(queries.len(), model.forest().trees().len(), thread_count);
    let context = ExecutionContext::new(thread_count)?;
    predict_multivariate_isolation_tiled(model, queries, prediction_seed, &context)
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

#[derive(Debug)]
struct ClassificationTile {
    query_start: usize,
    tree_start: usize,
    query_count: usize,
    tree_count: usize,
    values: Box<[ClassificationPrediction]>,
}

#[derive(Debug)]
struct IsolationTile {
    query_start: usize,
    tree_start: usize,
    query_count: usize,
    tree_count: usize,
    path_lengths: Box<[f64]>,
}

fn predict_vector_classification_tiled(
    model: &ClassificationForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    context: &ExecutionContext,
) -> CoreResult<Vec<ForestClassificationPrediction>> {
    let forest = model.forest();
    let plan = InferenceTilePlan::new(
        queries.len(),
        forest.trees().len(),
        context.budget().resolved(),
        2,
    );
    let tiles = parallel_indexed_in(context, plan.tile_count(), &|tile_index| {
        let (query_range, tree_range) = plan
            .tile(tile_index)
            .ok_or_else(|| CoreError::invalid_data("inference tile index exceeded its plan"))?;
        let (query_start, tree_start) = (query_range.start, tree_range.start);
        let (query_count, tree_count) = (query_range.len(), tree_range.len());
        let mut scratch = DistanceScratch::new();
        let mut values = Vec::with_capacity(query_count.saturating_mul(tree_count));
        for sample in query_range {
            let seed = observation_seed(prediction_seed, sample);
            for tree_index in tree_range.clone() {
                let tree = &forest.trees()[tree_index];
                let Some(OwnedObservationStore::Vector(store)) = tree.exemplars() else {
                    return Err(layout_error("vector tree exemplars"));
                };
                let exemplars = store.views();
                values.push(predict_vector_classification(
                    tree,
                    queries[sample],
                    &exemplars,
                    forest.tie_policy(),
                    SeedMixer::tree(seed, tree_index),
                    &mut scratch,
                )?);
            }
        }
        Ok(ClassificationTile {
            query_start,
            tree_start,
            query_count,
            tree_count,
            values: values.into_boxed_slice(),
        })
    })?;
    finish_classification_tiles(
        queries.len(),
        forest.trees().len(),
        model.output_count(),
        tiles,
    )
}

fn predict_multivariate_classification_tiled(
    model: &ClassificationForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    context: &ExecutionContext,
) -> CoreResult<Vec<ForestClassificationPrediction>> {
    let forest = model.forest();
    let plan = InferenceTilePlan::new(
        queries.len(),
        forest.trees().len(),
        context.budget().resolved(),
        2,
    );
    let tiles = parallel_indexed_in(context, plan.tile_count(), &|tile_index| {
        let (query_range, tree_range) = plan
            .tile(tile_index)
            .ok_or_else(|| CoreError::invalid_data("inference tile index exceeded its plan"))?;
        let (query_start, tree_start) = (query_range.start, tree_range.start);
        let (query_count, tree_count) = (query_range.len(), tree_range.len());
        let mut scratch = DistanceScratch::new();
        let mut values = Vec::with_capacity(query_count.saturating_mul(tree_count));
        for sample in query_range {
            let seed = observation_seed(prediction_seed, sample);
            for tree_index in tree_range.clone() {
                let tree = &forest.trees()[tree_index];
                let Some(OwnedObservationStore::Multivariate(store)) = tree.exemplars() else {
                    return Err(layout_error("multivariate tree exemplars"));
                };
                let exemplars = store.views()?;
                values.push(predict_multivariate_classification(
                    tree,
                    queries[sample],
                    &exemplars,
                    forest.tie_policy(),
                    SeedMixer::tree(seed, tree_index),
                    &mut scratch,
                )?);
            }
        }
        Ok(ClassificationTile {
            query_start,
            tree_start,
            query_count,
            tree_count,
            values: values.into_boxed_slice(),
        })
    })?;
    finish_classification_tiles(
        queries.len(),
        forest.trees().len(),
        model.output_count(),
        tiles,
    )
}

fn finish_classification_tiles(
    query_count: usize,
    tree_count: usize,
    output_count: usize,
    tiles: Vec<ClassificationTile>,
) -> CoreResult<Vec<ForestClassificationPrediction>> {
    let mut ordered = vec![None; checked_cells(query_count, tree_count, "classification")?];
    for tile in tiles {
        validate_tile_len(
            tile.values.len(),
            tile.query_count,
            tile.tree_count,
            "classification",
        )?;
        for local_query in 0..tile.query_count {
            for local_tree in 0..tile.tree_count {
                let source = local_query * tile.tree_count + local_tree;
                let destination =
                    (tile.query_start + local_query) * tree_count + tile.tree_start + local_tree;
                ordered[destination] = Some(tile.values[source].clone());
            }
        }
    }
    (0..query_count)
        .map(|sample| {
            let row = ordered[sample * tree_count..(sample + 1) * tree_count]
                .iter()
                .map(|value| {
                    value.clone().ok_or_else(|| {
                        CoreError::invalid_data("classification tile coverage was incomplete")
                    })
                })
                .collect::<CoreResult<Vec<_>>>()?;
            aggregate_classification(&row, output_count)
        })
        .collect()
}

fn predict_vector_isolation_tiled(
    model: &IsolationForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    context: &ExecutionContext,
) -> CoreResult<Vec<IsolationPrediction>> {
    let forest = model.forest();
    let plan = InferenceTilePlan::new(
        queries.len(),
        forest.trees().len(),
        context.budget().resolved(),
        2,
    );
    let tiles = parallel_indexed_in(context, plan.tile_count(), &|tile_index| {
        let (query_range, tree_range) = plan
            .tile(tile_index)
            .ok_or_else(|| CoreError::invalid_data("inference tile index exceeded its plan"))?;
        let (query_start, tree_start) = (query_range.start, tree_range.start);
        let (query_count, tree_count) = (query_range.len(), tree_range.len());
        let mut scratch = DistanceScratch::new();
        let mut path_lengths = Vec::with_capacity(query_count.saturating_mul(tree_count));
        for sample in query_range {
            let seed = observation_seed(prediction_seed, sample);
            for tree_index in tree_range.clone() {
                let tree = &forest.trees()[tree_index];
                let Some(OwnedObservationStore::Vector(store)) = tree.exemplars() else {
                    return Err(layout_error("vector tree exemplars"));
                };
                let exemplars = store.views();
                path_lengths.push(
                    predict_vector_isolation(
                        tree,
                        queries[sample],
                        &exemplars,
                        model.normalization_size(),
                        forest.tie_policy(),
                        SeedMixer::tree(seed, tree_index),
                        &mut scratch,
                    )?
                    .path_length(),
                );
            }
        }
        Ok(IsolationTile {
            query_start,
            tree_start,
            query_count,
            tree_count,
            path_lengths: path_lengths.into_boxed_slice(),
        })
    })?;
    finish_isolation_tiles(queries.len(), forest.trees().len(), model, tiles)
}

fn predict_multivariate_isolation_tiled(
    model: &IsolationForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    context: &ExecutionContext,
) -> CoreResult<Vec<IsolationPrediction>> {
    let forest = model.forest();
    let plan = InferenceTilePlan::new(
        queries.len(),
        forest.trees().len(),
        context.budget().resolved(),
        2,
    );
    let tiles = parallel_indexed_in(context, plan.tile_count(), &|tile_index| {
        let (query_range, tree_range) = plan
            .tile(tile_index)
            .ok_or_else(|| CoreError::invalid_data("inference tile index exceeded its plan"))?;
        let (query_start, tree_start) = (query_range.start, tree_range.start);
        let (query_count, tree_count) = (query_range.len(), tree_range.len());
        let mut scratch = DistanceScratch::new();
        let mut path_lengths = Vec::with_capacity(query_count.saturating_mul(tree_count));
        for sample in query_range {
            let seed = observation_seed(prediction_seed, sample);
            for tree_index in tree_range.clone() {
                let tree = &forest.trees()[tree_index];
                let Some(OwnedObservationStore::Multivariate(store)) = tree.exemplars() else {
                    return Err(layout_error("multivariate tree exemplars"));
                };
                let exemplars = store.views()?;
                path_lengths.push(
                    predict_multivariate_isolation(
                        tree,
                        queries[sample],
                        &exemplars,
                        model.normalization_size(),
                        forest.tie_policy(),
                        SeedMixer::tree(seed, tree_index),
                        &mut scratch,
                    )?
                    .path_length(),
                );
            }
        }
        Ok(IsolationTile {
            query_start,
            tree_start,
            query_count,
            tree_count,
            path_lengths: path_lengths.into_boxed_slice(),
        })
    })?;
    finish_isolation_tiles(queries.len(), forest.trees().len(), model, tiles)
}

fn finish_isolation_tiles(
    query_count: usize,
    tree_count: usize,
    model: &IsolationForest,
    tiles: Vec<IsolationTile>,
) -> CoreResult<Vec<IsolationPrediction>> {
    let mut ordered = vec![None; checked_cells(query_count, tree_count, "isolation")?];
    for tile in tiles {
        validate_tile_len(
            tile.path_lengths.len(),
            tile.query_count,
            tile.tree_count,
            "isolation",
        )?;
        for local_query in 0..tile.query_count {
            for local_tree in 0..tile.tree_count {
                let source = local_query * tile.tree_count + local_tree;
                let destination =
                    (tile.query_start + local_query) * tree_count + tile.tree_start + local_tree;
                ordered[destination] = Some(tile.path_lengths[source]);
            }
        }
    }
    (0..query_count)
        .map(|sample| {
            let mut sum = 0.0;
            for tree_index in 0..tree_count {
                sum += ordered[sample * tree_count + tree_index].ok_or_else(|| {
                    CoreError::invalid_data("isolation tile coverage was incomplete")
                })?;
            }
            aggregate_isolation(sum, model)
        })
        .collect()
}

fn checked_cells(query_count: usize, tree_count: usize, operation: &str) -> CoreResult<usize> {
    if tree_count == 0 {
        return Err(CoreError::invalid_data(format!(
            "{operation} forest has no fitted trees"
        )));
    }
    query_count.checked_mul(tree_count).ok_or_else(|| {
        CoreError::invalid_data(format!("{operation} prediction shape overflowed usize"))
    })
}

fn validate_tile_len(
    actual: usize,
    queries: usize,
    trees: usize,
    operation: &str,
) -> CoreResult<()> {
    let expected = queries.saturating_mul(trees);
    if actual == expected {
        Ok(())
    } else {
        Err(CoreError::shape_mismatch(format!(
            "{operation} tile has an invalid value count"
        )))
    }
}

#[derive(Debug)]
struct RegressionTile {
    query_start: usize,
    tree_start: usize,
    query_count: usize,
    tree_count: usize,
    values: Box<[Vec<f64>]>,
}

fn predict_vector_regression_tiled(
    model: &RegressionForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    context: &ExecutionContext,
) -> CoreResult<Vec<Vec<f64>>> {
    let forest = model.forest();
    let plan = InferenceTilePlan::new(
        queries.len(),
        forest.trees().len(),
        context.budget().resolved(),
        2,
    );
    let tiles = parallel_indexed_in(context, plan.tile_count(), &|tile_index| {
        let (query_range, tree_range) = plan
            .tile(tile_index)
            .ok_or_else(|| CoreError::invalid_data("inference tile index exceeded its plan"))?;
        let query_start = query_range.start;
        let tree_start = tree_range.start;
        let query_count = query_range.len();
        let tree_count = tree_range.len();
        let mut scratch = DistanceScratch::new();
        let mut values = Vec::with_capacity(query_count.saturating_mul(tree_count));
        for sample in query_range {
            let query_seed = observation_seed(prediction_seed, sample);
            for tree_index in tree_range.clone() {
                let tree = &forest.trees()[tree_index];
                let Some(OwnedObservationStore::Vector(store)) = tree.exemplars() else {
                    return Err(layout_error("vector tree exemplars"));
                };
                let exemplars = store.views();
                values.push(predict_vector_regression(
                    tree,
                    queries[sample],
                    &exemplars,
                    forest.tie_policy(),
                    SeedMixer::tree(query_seed, tree_index),
                    &mut scratch,
                )?);
            }
        }
        Ok(RegressionTile {
            query_start,
            tree_start,
            query_count,
            tree_count,
            values: values.into_boxed_slice(),
        })
    })?;
    finish_regression_tiles(
        queries.len(),
        forest.trees().len(),
        model.output_count(),
        tiles,
    )
}

fn predict_multivariate_regression_tiled(
    model: &RegressionForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    context: &ExecutionContext,
) -> CoreResult<Vec<Vec<f64>>> {
    let forest = model.forest();
    let plan = InferenceTilePlan::new(
        queries.len(),
        forest.trees().len(),
        context.budget().resolved(),
        2,
    );
    let tiles = parallel_indexed_in(context, plan.tile_count(), &|tile_index| {
        let (query_range, tree_range) = plan
            .tile(tile_index)
            .ok_or_else(|| CoreError::invalid_data("inference tile index exceeded its plan"))?;
        let query_start = query_range.start;
        let tree_start = tree_range.start;
        let query_count = query_range.len();
        let tree_count = tree_range.len();
        let mut scratch = DistanceScratch::new();
        let mut values = Vec::with_capacity(query_count.saturating_mul(tree_count));
        for sample in query_range {
            let query_seed = observation_seed(prediction_seed, sample);
            for tree_index in tree_range.clone() {
                let tree = &forest.trees()[tree_index];
                let Some(OwnedObservationStore::Multivariate(store)) = tree.exemplars() else {
                    return Err(layout_error("multivariate tree exemplars"));
                };
                let exemplars = store.views()?;
                values.push(predict_multivariate_regression(
                    tree,
                    queries[sample],
                    &exemplars,
                    forest.tie_policy(),
                    SeedMixer::tree(query_seed, tree_index),
                    &mut scratch,
                )?);
            }
        }
        Ok(RegressionTile {
            query_start,
            tree_start,
            query_count,
            tree_count,
            values: values.into_boxed_slice(),
        })
    })?;
    finish_regression_tiles(
        queries.len(),
        forest.trees().len(),
        model.output_count(),
        tiles,
    )
}

fn finish_regression_tiles(
    query_count: usize,
    tree_count: usize,
    output_count: usize,
    tiles: Vec<RegressionTile>,
) -> CoreResult<Vec<Vec<f64>>> {
    if tree_count == 0 {
        return Err(CoreError::invalid_data("fitted forest has no trees"));
    }
    let cell_count = query_count
        .checked_mul(tree_count)
        .ok_or_else(|| CoreError::invalid_data("regression prediction shape overflowed usize"))?;
    let mut ordered = vec![None; cell_count];
    for tile in tiles {
        let expected = tile.query_count.saturating_mul(tile.tree_count);
        if tile.values.len() != expected {
            return Err(CoreError::shape_mismatch(
                "regression tile has an invalid value count",
            ));
        }
        for local_query in 0..tile.query_count {
            for local_tree in 0..tile.tree_count {
                let source = local_query * tile.tree_count + local_tree;
                let destination =
                    (tile.query_start + local_query) * tree_count + tile.tree_start + local_tree;
                ordered[destination] = Some(tile.values[source].clone());
            }
        }
    }
    let mut results = Vec::with_capacity(query_count);
    for sample in 0..query_count {
        let mut sums = vec![0.0; output_count];
        for tree_index in 0..tree_count {
            let values = ordered[sample * tree_count + tree_index]
                .as_deref()
                .ok_or_else(|| {
                    CoreError::invalid_data("regression tile coverage was incomplete")
                })?;
            if values.len() != output_count {
                return Err(CoreError::shape_mismatch(
                    "regression tree output count mismatch",
                ));
            }
            for (sum, &value) in sums.iter_mut().zip(values) {
                *sum += value;
            }
        }
        let denominator = tree_count as f64;
        for value in &mut sums {
            *value /= denominator;
        }
        results.push(sums);
    }
    Ok(results)
}

fn layout_error(expected: &str) -> CoreError {
    CoreError::invalid_data(format!(
        "fitted model does not contain {expected} observations"
    ))
}

#[inline]
fn observation_seed(base_seed: u64, sample: usize) -> u64 {
    SeedMixer::observation(base_seed, sample as u64)
}
