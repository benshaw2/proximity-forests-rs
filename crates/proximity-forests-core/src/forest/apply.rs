//! Compact batch leaf assignment for fitted forests.

use crate::{
    CoreError, CoreResult, DistanceScratch, MultivariateSeriesView, NumericVectorView, SeedMixer,
};

use super::{
    ExecutionContext, FittedForest, OwnedObservationStore, parallel_indexed_in,
    traverse_multivariate, traverse_vector,
    work_estimate::{InferenceTilePlan, inference_thread_count},
};

/// Compact leaf identifier returned by forest application.
///
/// Tree node arenas are indexed internally by `usize`, but fitted trees are
/// checked and compacted to `u32` at the batch boundary. This halves assignment
/// storage on 64-bit platforms and maps directly to NumPy `uint32`.
pub type AppliedLeafId = u32;

/// Row-major sample-by-tree leaf assignments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForestApplyResult {
    sample_count: usize,
    tree_count: usize,
    leaf_indices: Box<[AppliedLeafId]>,
}

impl ForestApplyResult {
    /// Returns the number of query observations.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the number of fitted trees.
    #[must_use]
    pub const fn tree_count(&self) -> usize {
        self.tree_count
    }

    /// Returns row-major leaf indices with shape `sample_count x tree_count`.
    #[must_use]
    pub fn leaf_indices(&self) -> &[AppliedLeafId] {
        &self.leaf_indices
    }

    /// Consumes the result and returns compact row-major storage.
    #[must_use]
    pub fn into_leaf_indices(self) -> Box<[AppliedLeafId]> {
        self.leaf_indices
    }

    /// Returns one query row of leaf indices.
    #[must_use]
    pub fn leaf_row(&self, sample: usize) -> Option<&[AppliedLeafId]> {
        row(&self.leaf_indices, sample, self.tree_count)
    }
}

/// Row-major leaf assignments plus traversal depths for diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForestApplyWithDepths {
    assignments: ForestApplyResult,
    depths: Box<[u32]>,
}

impl ForestApplyWithDepths {
    /// Returns compact leaf assignments.
    #[must_use]
    pub const fn assignments(&self) -> &ForestApplyResult {
        &self.assignments
    }

    /// Returns row-major depths with the same shape as the assignments.
    #[must_use]
    pub fn depths(&self) -> &[u32] {
        &self.depths
    }

    /// Returns one query row of traversal depths.
    #[must_use]
    pub fn depth_row(&self, sample: usize) -> Option<&[u32]> {
        row(&self.depths, sample, self.assignments.tree_count)
    }
}

/// Applies a vector-layout forest without allocating a depth matrix.
pub fn apply_vector_many(
    forest: &FittedForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<ForestApplyResult> {
    let thread_count = inference_thread_count(queries.len(), forest.trees().len(), thread_count);
    let context = ExecutionContext::new(thread_count)?;
    apply_vector_tiled(forest, queries, prediction_seed, &context)
}

/// Applies a multivariate forest without allocating a depth matrix.
pub fn apply_multivariate_many(
    forest: &FittedForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<ForestApplyResult> {
    let thread_count = inference_thread_count(queries.len(), forest.trees().len(), thread_count);
    let context = ExecutionContext::new(thread_count)?;
    apply_multivariate_tiled(forest, queries, prediction_seed, &context)
}

/// Applies a vector-layout forest and also records traversal depths.
pub fn apply_vector_many_with_depths(
    forest: &FittedForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<ForestApplyWithDepths> {
    let thread_count = inference_thread_count(queries.len(), forest.trees().len(), thread_count);
    let context = ExecutionContext::new(thread_count)?;
    apply_vector_tiled_with_depths(forest, queries, prediction_seed, &context)
}

/// Applies a multivariate forest and also records traversal depths.
pub fn apply_multivariate_many_with_depths(
    forest: &FittedForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<ForestApplyWithDepths> {
    let thread_count = inference_thread_count(queries.len(), forest.trees().len(), thread_count);
    let context = ExecutionContext::new(thread_count)?;
    apply_multivariate_tiled_with_depths(forest, queries, prediction_seed, &context)
}

#[derive(Debug)]
struct ApplyTile {
    query_start: usize,
    tree_start: usize,
    query_count: usize,
    tree_count: usize,
    leaves: Box<[AppliedLeafId]>,
}

fn apply_vector_tiled(
    forest: &FittedForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    context: &ExecutionContext,
) -> CoreResult<ForestApplyResult> {
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
        let mut leaves = Vec::with_capacity(matrix_len(query_count, tree_count)?);
        for sample in query_range {
            let query_seed = SeedMixer::observation(prediction_seed, sample as u64);
            for tree_index in tree_range.clone() {
                let tree = &forest.trees()[tree_index];
                let Some(OwnedObservationStore::Vector(store)) = tree.exemplars() else {
                    return Err(layout_error("vector tree exemplars"));
                };
                let exemplars = store.views();
                let visit = traverse_vector(
                    tree,
                    queries[sample],
                    &exemplars,
                    forest.tie_policy(),
                    SeedMixer::tree(query_seed, tree_index),
                    &mut scratch,
                )?;
                leaves.push(compact_node_id(visit.node().get())?);
            }
        }
        Ok(ApplyTile {
            query_start,
            tree_start,
            query_count,
            tree_count,
            leaves: leaves.into_boxed_slice(),
        })
    })?;
    finish_apply_tiles(queries.len(), forest.trees().len(), tiles)
}

fn apply_multivariate_tiled(
    forest: &FittedForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    context: &ExecutionContext,
) -> CoreResult<ForestApplyResult> {
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
        let mut leaves = Vec::with_capacity(matrix_len(query_count, tree_count)?);
        for sample in query_range {
            let query_seed = SeedMixer::observation(prediction_seed, sample as u64);
            for tree_index in tree_range.clone() {
                let tree = &forest.trees()[tree_index];
                let Some(OwnedObservationStore::Multivariate(store)) = tree.exemplars() else {
                    return Err(layout_error("multivariate tree exemplars"));
                };
                let exemplars = store.views()?;
                let visit = traverse_multivariate(
                    tree,
                    queries[sample],
                    &exemplars,
                    forest.tie_policy(),
                    SeedMixer::tree(query_seed, tree_index),
                    &mut scratch,
                )?;
                leaves.push(compact_node_id(visit.node().get())?);
            }
        }
        Ok(ApplyTile {
            query_start,
            tree_start,
            query_count,
            tree_count,
            leaves: leaves.into_boxed_slice(),
        })
    })?;
    finish_apply_tiles(queries.len(), forest.trees().len(), tiles)
}

#[derive(Debug)]
struct ApplyDepthTile {
    query_start: usize,
    tree_start: usize,
    query_count: usize,
    tree_count: usize,
    leaves: Box<[AppliedLeafId]>,
    depths: Box<[u32]>,
}

fn apply_vector_tiled_with_depths(
    forest: &FittedForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    context: &ExecutionContext,
) -> CoreResult<ForestApplyWithDepths> {
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
        let capacity = matrix_len(query_count, tree_count)?;
        let mut scratch = DistanceScratch::new();
        let mut leaves = Vec::with_capacity(capacity);
        let mut depths = Vec::with_capacity(capacity);
        for sample in query_range {
            let query_seed = SeedMixer::observation(prediction_seed, sample as u64);
            for tree_index in tree_range.clone() {
                let tree = &forest.trees()[tree_index];
                let Some(OwnedObservationStore::Vector(store)) = tree.exemplars() else {
                    return Err(layout_error("vector tree exemplars"));
                };
                let exemplars = store.views();
                let visit = traverse_vector(
                    tree,
                    queries[sample],
                    &exemplars,
                    forest.tie_policy(),
                    SeedMixer::tree(query_seed, tree_index),
                    &mut scratch,
                )?;
                leaves.push(compact_node_id(visit.node().get())?);
                depths.push(compact_depth(visit.depth())?);
            }
        }
        Ok(ApplyDepthTile {
            query_start,
            tree_start,
            query_count,
            tree_count,
            leaves: leaves.into_boxed_slice(),
            depths: depths.into_boxed_slice(),
        })
    })?;
    finish_apply_depth_tiles(queries.len(), forest.trees().len(), tiles)
}

fn apply_multivariate_tiled_with_depths(
    forest: &FittedForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    context: &ExecutionContext,
) -> CoreResult<ForestApplyWithDepths> {
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
        let capacity = matrix_len(query_count, tree_count)?;
        let mut scratch = DistanceScratch::new();
        let mut leaves = Vec::with_capacity(capacity);
        let mut depths = Vec::with_capacity(capacity);
        for sample in query_range {
            let query_seed = SeedMixer::observation(prediction_seed, sample as u64);
            for tree_index in tree_range.clone() {
                let tree = &forest.trees()[tree_index];
                let Some(OwnedObservationStore::Multivariate(store)) = tree.exemplars() else {
                    return Err(layout_error("multivariate tree exemplars"));
                };
                let exemplars = store.views()?;
                let visit = traverse_multivariate(
                    tree,
                    queries[sample],
                    &exemplars,
                    forest.tie_policy(),
                    SeedMixer::tree(query_seed, tree_index),
                    &mut scratch,
                )?;
                leaves.push(compact_node_id(visit.node().get())?);
                depths.push(compact_depth(visit.depth())?);
            }
        }
        Ok(ApplyDepthTile {
            query_start,
            tree_start,
            query_count,
            tree_count,
            leaves: leaves.into_boxed_slice(),
            depths: depths.into_boxed_slice(),
        })
    })?;
    finish_apply_depth_tiles(queries.len(), forest.trees().len(), tiles)
}

fn finish_apply_depth_tiles(
    sample_count: usize,
    tree_count: usize,
    tiles: Vec<ApplyDepthTile>,
) -> CoreResult<ForestApplyWithDepths> {
    let capacity = matrix_len(sample_count, tree_count)?;
    let mut leaf_indices = vec![0; capacity];
    let mut depths = vec![0; capacity];
    for tile in tiles {
        let tile_capacity = matrix_len(tile.query_count, tile.tree_count)?;
        validate_row(tile.leaves.len(), tile_capacity)?;
        validate_row(tile.depths.len(), tile_capacity)?;
        for local_query in 0..tile.query_count {
            let source_start = local_query * tile.tree_count;
            let source = source_start..source_start + tile.tree_count;
            let destination_start = (tile.query_start + local_query) * tree_count + tile.tree_start;
            let destination = destination_start..destination_start + tile.tree_count;
            leaf_indices[destination.clone()].copy_from_slice(&tile.leaves[source.clone()]);
            depths[destination].copy_from_slice(&tile.depths[source]);
        }
    }
    Ok(ForestApplyWithDepths {
        assignments: ForestApplyResult {
            sample_count,
            tree_count,
            leaf_indices: leaf_indices.into_boxed_slice(),
        },
        depths: depths.into_boxed_slice(),
    })
}

fn finish_apply_tiles(
    sample_count: usize,
    tree_count: usize,
    tiles: Vec<ApplyTile>,
) -> CoreResult<ForestApplyResult> {
    let capacity = matrix_len(sample_count, tree_count)?;
    let mut leaf_indices = vec![0; capacity];
    for tile in tiles {
        validate_row(
            tile.leaves.len(),
            matrix_len(tile.query_count, tile.tree_count)?,
        )?;
        for local_query in 0..tile.query_count {
            let source_start = local_query * tile.tree_count;
            let destination_start = (tile.query_start + local_query) * tree_count + tile.tree_start;
            leaf_indices[destination_start..destination_start + tile.tree_count]
                .copy_from_slice(&tile.leaves[source_start..source_start + tile.tree_count]);
        }
    }
    Ok(ForestApplyResult {
        sample_count,
        tree_count,
        leaf_indices: leaf_indices.into_boxed_slice(),
    })
}

fn matrix_len(sample_count: usize, tree_count: usize) -> CoreResult<usize> {
    sample_count
        .checked_mul(tree_count)
        .ok_or_else(|| CoreError::invalid_data("apply result shape overflowed usize"))
}

fn validate_row(actual: usize, expected: usize) -> CoreResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(CoreError::shape_mismatch(
            "apply row does not match fitted tree count",
        ))
    }
}

fn compact_node_id(node: usize) -> CoreResult<AppliedLeafId> {
    AppliedLeafId::try_from(node).map_err(|_| {
        CoreError::invalid_data(
            "fitted tree contains more nodes than compact apply output can represent",
        )
    })
}

fn compact_depth(depth: usize) -> CoreResult<u32> {
    u32::try_from(depth)
        .map_err(|_| CoreError::invalid_data("tree depth exceeds compact u32 output"))
}

fn row<T>(values: &[T], sample: usize, width: usize) -> Option<&[T]> {
    let start = sample.checked_mul(width)?;
    values.get(start..start.checked_add(width)?)
}

fn layout_error(expected: &str) -> CoreError {
    CoreError::invalid_data(format!(
        "fitted model does not contain {expected} observations"
    ))
}
