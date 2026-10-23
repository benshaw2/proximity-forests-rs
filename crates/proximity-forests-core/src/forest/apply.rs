//! Batch leaf assignment for fitted forests.

use crate::{
    CoreError, CoreResult, DistanceScratch, MultivariateSeriesView, NumericVectorView, SeedMixer,
};

use super::{
    FittedForest, OwnedObservationStore, parallel_indexed, traverse_multivariate, traverse_vector,
};

/// Row-major sample-by-tree leaf assignments and traversal depths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForestApplyResult {
    sample_count: usize,
    tree_count: usize,
    leaf_indices: Vec<usize>,
    depths: Vec<usize>,
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
    pub fn leaf_indices(&self) -> &[usize] {
        &self.leaf_indices
    }

    /// Returns row-major traversal depths with shape `sample_count x tree_count`.
    #[must_use]
    pub fn depths(&self) -> &[usize] {
        &self.depths
    }

    /// Returns one query row of leaf indices.
    #[must_use]
    pub fn leaf_row(&self, sample: usize) -> Option<&[usize]> {
        row(&self.leaf_indices, sample, self.tree_count)
    }

    /// Returns one query row of traversal depths.
    #[must_use]
    pub fn depth_row(&self, sample: usize) -> Option<&[usize]> {
        row(&self.depths, sample, self.tree_count)
    }
}

/// Applies a vector-layout forest to a batch of query observations.
///
/// The returned leaf identifiers are stable `NodeId` indices within each tree.
/// Query rows are parallelized while each row traverses trees in stable order.
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
            leaves.push(visit.node().get());
            depths.push(visit.depth());
        }
        Ok((leaves, depths))
    })?;
    finish_apply(rows, forest.trees().len())
}

/// Applies a multivariate forest to a batch of query observations.
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
            leaves.push(visit.node().get());
            depths.push(visit.depth());
        }
        Ok((leaves, depths))
    })?;
    finish_apply(rows, forest.trees().len())
}

fn finish_apply(
    rows: Vec<(Vec<usize>, Vec<usize>)>,
    tree_count: usize,
) -> CoreResult<ForestApplyResult> {
    let sample_count = rows.len();
    let capacity = sample_count
        .checked_mul(tree_count)
        .ok_or_else(|| CoreError::invalid_data("apply result shape overflowed usize"))?;
    let mut leaf_indices = Vec::with_capacity(capacity);
    let mut depths = Vec::with_capacity(capacity);
    for (leaves, row_depths) in rows {
        if leaves.len() != tree_count || row_depths.len() != tree_count {
            return Err(CoreError::shape_mismatch(
                "apply row does not match fitted tree count",
            ));
        }
        leaf_indices.extend(leaves);
        depths.extend(row_depths);
    }
    Ok(ForestApplyResult {
        sample_count,
        tree_count,
        leaf_indices,
        depths,
    })
}

fn row(values: &[usize], sample: usize, width: usize) -> Option<&[usize]> {
    let start = sample.checked_mul(width)?;
    values.get(start..start.checked_add(width)?)
}

fn layout_error(expected: &str) -> CoreError {
    CoreError::invalid_data(format!(
        "fitted model does not contain {expected} observations"
    ))
}
