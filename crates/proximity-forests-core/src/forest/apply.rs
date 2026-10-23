//! Compact batch leaf assignment for fitted forests.

use crate::{
    CoreError, CoreResult, DistanceScratch, MultivariateSeriesView, NumericVectorView, SeedMixer,
};

use super::{
    FittedForest, OwnedObservationStore, parallel_indexed, traverse_multivariate, traverse_vector,
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
    let OwnedObservationStore::Vector(store) = forest.observations() else {
        return Err(layout_error("vector"));
    };
    let exemplars = store.views();
    let rows = parallel_indexed(queries.len(), thread_count, |sample| {
        let query_seed = SeedMixer::observation(prediction_seed, sample as u64);
        let mut scratch = DistanceScratch::new();
        let mut leaves = Vec::with_capacity(forest.trees().len());
        for (tree_index, tree) in forest.trees().iter().enumerate() {
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
        Ok(leaves.into_boxed_slice())
    })?;
    finish_apply(rows, forest.trees().len())
}

/// Applies a multivariate forest without allocating a depth matrix.
pub fn apply_multivariate_many(
    forest: &FittedForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<ForestApplyResult> {
    let OwnedObservationStore::Multivariate(store) = forest.observations() else {
        return Err(layout_error("multivariate"));
    };
    let exemplars = store.views()?;
    let rows = parallel_indexed(queries.len(), thread_count, |sample| {
        let query_seed = SeedMixer::observation(prediction_seed, sample as u64);
        let mut scratch = DistanceScratch::new();
        let mut leaves = Vec::with_capacity(forest.trees().len());
        for (tree_index, tree) in forest.trees().iter().enumerate() {
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
        Ok(leaves.into_boxed_slice())
    })?;
    finish_apply(rows, forest.trees().len())
}

/// Applies a vector-layout forest and also records traversal depths.
pub fn apply_vector_many_with_depths(
    forest: &FittedForest,
    queries: &[NumericVectorView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<ForestApplyWithDepths> {
    let OwnedObservationStore::Vector(store) = forest.observations() else {
        return Err(layout_error("vector"));
    };
    let exemplars = store.views();
    let rows = parallel_indexed(queries.len(), thread_count, |sample| {
        let query_seed = SeedMixer::observation(prediction_seed, sample as u64);
        let mut scratch = DistanceScratch::new();
        let mut leaves = Vec::with_capacity(forest.trees().len());
        let mut depths = Vec::with_capacity(forest.trees().len());
        for (tree_index, tree) in forest.trees().iter().enumerate() {
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
        Ok((leaves.into_boxed_slice(), depths.into_boxed_slice()))
    })?;
    finish_apply_with_depths(rows, forest.trees().len())
}

/// Applies a multivariate forest and also records traversal depths.
pub fn apply_multivariate_many_with_depths(
    forest: &FittedForest,
    queries: &[MultivariateSeriesView<'_>],
    prediction_seed: u64,
    thread_count: usize,
) -> CoreResult<ForestApplyWithDepths> {
    let OwnedObservationStore::Multivariate(store) = forest.observations() else {
        return Err(layout_error("multivariate"));
    };
    let exemplars = store.views()?;
    let rows = parallel_indexed(queries.len(), thread_count, |sample| {
        let query_seed = SeedMixer::observation(prediction_seed, sample as u64);
        let mut scratch = DistanceScratch::new();
        let mut leaves = Vec::with_capacity(forest.trees().len());
        let mut depths = Vec::with_capacity(forest.trees().len());
        for (tree_index, tree) in forest.trees().iter().enumerate() {
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
        Ok((leaves.into_boxed_slice(), depths.into_boxed_slice()))
    })?;
    finish_apply_with_depths(rows, forest.trees().len())
}

fn finish_apply(
    rows: Vec<Box<[AppliedLeafId]>>,
    tree_count: usize,
) -> CoreResult<ForestApplyResult> {
    let sample_count = rows.len();
    let capacity = matrix_len(sample_count, tree_count)?;
    let mut leaf_indices = Vec::with_capacity(capacity);
    for leaves in rows {
        validate_row(leaves.len(), tree_count)?;
        leaf_indices.extend_from_slice(&leaves);
    }
    debug_assert_eq!(leaf_indices.len(), capacity);
    Ok(ForestApplyResult {
        sample_count,
        tree_count,
        leaf_indices: leaf_indices.into_boxed_slice(),
    })
}

fn finish_apply_with_depths(
    rows: Vec<(Box<[AppliedLeafId]>, Box<[u32]>)>,
    tree_count: usize,
) -> CoreResult<ForestApplyWithDepths> {
    let sample_count = rows.len();
    let capacity = matrix_len(sample_count, tree_count)?;
    let mut leaf_indices = Vec::with_capacity(capacity);
    let mut depths = Vec::with_capacity(capacity);
    for (leaves, row_depths) in rows {
        validate_row(leaves.len(), tree_count)?;
        validate_row(row_depths.len(), tree_count)?;
        leaf_indices.extend_from_slice(&leaves);
        depths.extend_from_slice(&row_depths);
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
