//! Coarse forest-operation work estimates.
//!
//! These estimates combine distance metadata with counts already available at a
//! node or batch boundary. They do not scan observations and do not allocate.

use crate::{DistanceConfig, DistanceWorkShape, WorkUnits};

/// Structural metadata for one node's complete candidate batch.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CandidateBatchShape {
    /// Distinct observations routed once for each candidate.
    pub distinct_observations: usize,
    /// Exemplars evaluated for each routed observation.
    pub exemplars_per_candidate: usize,
    /// Candidates evaluated at the node.
    pub candidate_count: usize,
    /// Representative query/exemplar distance shape.
    pub distance_shape: DistanceWorkShape,
}

impl CandidateBatchShape {
    /// Estimates a node candidate batch after a concrete distance is prepared.
    #[must_use]
    pub fn estimate(self, distance: &DistanceConfig) -> WorkUnits {
        distance
            .estimate_work(self.distance_shape)
            .saturating_mul(self.distinct_observations)
            .saturating_mul(self.exemplars_per_candidate)
            .saturating_mul(self.candidate_count)
    }
}

/// Structural metadata for a rectangular inference tile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InferenceTileShape {
    /// Query rows in the tile.
    pub query_count: usize,
    /// Fitted trees in the tile.
    pub tree_count: usize,
    /// Conservative expected split-node visits per query/tree pair.
    pub expected_depth: usize,
    /// Conservative distance work at one visited split node.
    pub distance_work: WorkUnits,
}

impl InferenceTileShape {
    /// Estimates total traversal work for a query-tree tile.
    #[must_use]
    pub fn estimate(self) -> WorkUnits {
        self.distance_work
            .saturating_mul(self.expected_depth)
            .saturating_mul(self.query_count)
            .saturating_mul(self.tree_count)
    }
}

/// Minimum query-tree cells that justify constructing a parallel inference pool.
///
/// P5F showed useful scaling at 100 cells, including cheap Euclidean traversal,
/// while smaller jobs were not calibrated. The conservative initial boundary of
/// 64 keeps all benchmarked wins and leaves finer distance-aware calibration to
/// future work.
const MIN_PARALLEL_INFERENCE_CELLS: usize = 64;

/// Minimum routed OOB tree votes that justify entering a parallel pool.
///
/// OOB row work is irregular, but every routed vote requires one complete tree
/// traversal. A conservative boundary of 64 mirrors the calibrated inference
/// cell gate and preserves a genuinely pool-free path for tiny fits.
const MIN_PARALLEL_OOB_VOTES: usize = 64;

/// Resolves the worker request for one OOB result calculation.
///
/// `total_votes` is the sum of the compact membership row lengths. Returning
/// one prevents pool construction for explicit sequential requests and small
/// OOB workloads. Zero retains automatic hardware selection above the gate.
#[must_use]
pub(super) fn oob_thread_count(sample_count: usize, total_votes: usize, requested: usize) -> usize {
    if requested == 1 || sample_count < 2 || total_votes < MIN_PARALLEL_OOB_VOTES {
        1
    } else {
        requested
    }
}

/// Resolves the external worker request for one inference matrix.
///
/// Returning one preserves a genuinely pool-free path below the activation
/// boundary rather than constructing a parallel context with a single tile.
#[must_use]
pub(super) fn inference_thread_count(
    query_count: usize,
    tree_count: usize,
    requested: usize,
) -> usize {
    if requested == 1 || query_count.saturating_mul(tree_count) < MIN_PARALLEL_INFERENCE_CELLS {
        1
    } else {
        requested
    }
}

/// Stable rectangular tiling of a query-by-tree traversal matrix.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct InferenceTilePlan {
    query_count: usize,
    tree_count: usize,
    query_block: usize,
    tree_block: usize,
}

impl InferenceTilePlan {
    /// Plans bounded tiles with at least one cell on each nonempty axis.
    #[must_use]
    pub(super) fn new(
        query_count: usize,
        tree_count: usize,
        worker_count: usize,
        target_tiles_per_worker: usize,
    ) -> Self {
        if query_count == 0 || tree_count == 0 {
            return Self {
                query_count,
                tree_count,
                query_block: 1,
                tree_block: 1,
            };
        }
        let target_tiles = worker_count
            .max(1)
            .saturating_mul(target_tiles_per_worker.max(1));
        let tree_tiles = tree_count.min(target_tiles).max(1);
        let query_tiles = target_tiles.div_ceil(tree_tiles).min(query_count).max(1);
        Self {
            query_count,
            tree_count,
            query_block: query_count.div_ceil(query_tiles),
            tree_block: tree_count.div_ceil(tree_tiles),
        }
    }

    /// Number of stable row-major tiles.
    #[must_use]
    pub(super) fn tile_count(self) -> usize {
        if self.query_count == 0 || self.tree_count == 0 {
            0
        } else {
            self.query_count
                .div_ceil(self.query_block)
                .saturating_mul(self.tree_count.div_ceil(self.tree_block))
        }
    }

    /// Returns one row-major tile as query and tree half-open ranges.
    #[must_use]
    pub(super) fn tile(
        self,
        index: usize,
    ) -> Option<(core::ops::Range<usize>, core::ops::Range<usize>)> {
        let tree_tiles = self.tree_count.div_ceil(self.tree_block);
        if tree_tiles == 0 || index >= self.tile_count() {
            return None;
        }
        let query_tile = index / tree_tiles;
        let tree_tile = index % tree_tiles;
        let query_start = query_tile.saturating_mul(self.query_block);
        let tree_start = tree_tile.saturating_mul(self.tree_block);
        Some((
            query_start
                ..query_start
                    .saturating_add(self.query_block)
                    .min(self.query_count),
            tree_start
                ..tree_start
                    .saturating_add(self.tree_block)
                    .min(self.tree_count),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{CandidateBatchShape, InferenceTilePlan, InferenceTileShape};
    use crate::{DistanceConfig, DistanceWorkShape, WorkUnits};

    #[test]
    fn candidate_work_includes_node_size_exemplars_and_candidate_count() {
        let base = CandidateBatchShape {
            distinct_observations: 32,
            exemplars_per_candidate: 2,
            candidate_count: 5,
            distance_shape: DistanceWorkShape::new(64, 64, 1),
        };
        let estimate = base.estimate(&DistanceConfig::SquaredEuclidean);
        assert_eq!(estimate.get(), 32 * 2 * 5 * 64);
    }

    #[test]
    fn deeper_smaller_nodes_have_less_candidate_work() {
        let root = CandidateBatchShape {
            distinct_observations: 256,
            exemplars_per_candidate: 2,
            candidate_count: 5,
            distance_shape: DistanceWorkShape::new(96, 96, 1),
        };
        let deep = CandidateBatchShape {
            distinct_observations: 16,
            ..root
        };
        assert!(
            deep.estimate(&DistanceConfig::SquaredEuclidean)
                < root.estimate(&DistanceConfig::SquaredEuclidean)
        );
    }

    #[test]
    fn inference_tiles_scale_over_both_axes() {
        let small = InferenceTileShape {
            query_count: 1,
            tree_count: 10,
            expected_depth: 8,
            distance_work: WorkUnits::new(100),
        };
        let large = InferenceTileShape {
            query_count: 4,
            tree_count: 100,
            ..small
        };
        assert!(large.estimate() > small.estimate());
    }

    #[test]
    fn inference_tile_plan_is_stable_and_covers_both_axes() {
        let plan = InferenceTilePlan::new(7, 5, 4, 2);
        let tiles = (0..plan.tile_count())
            .map(|index| plan.tile(index).expect("tile"))
            .collect::<Vec<_>>();
        assert_eq!(tiles.first(), Some(&(0..4, 0..1)));
        assert_eq!(tiles.last(), Some(&(4..7, 4..5)));
        let covered = tiles
            .iter()
            .map(|(queries, trees)| queries.len() * trees.len())
            .sum::<usize>();
        assert_eq!(covered, 7 * 5);
    }

    #[test]
    fn inference_tile_plan_handles_empty_and_query_heavy_shapes() {
        assert_eq!(InferenceTilePlan::new(0, 10, 4, 2).tile_count(), 0);
        let plan = InferenceTilePlan::new(100, 2, 4, 2);
        assert!(plan.tile_count() >= 4);
        assert!(plan.tile(plan.tile_count()).is_none());
    }

    #[test]
    fn oob_gate_preserves_pool_free_small_work() {
        assert_eq!(super::oob_thread_count(128, 63, 8), 1);
        assert_eq!(super::oob_thread_count(128, 64, 8), 8);
        assert_eq!(super::oob_thread_count(128, 64, 0), 0);
        assert_eq!(super::oob_thread_count(1, 1_000, 8), 1);
        assert_eq!(super::oob_thread_count(128, 1_000, 1), 1);
    }

    #[test]
    fn inference_gate_preserves_pool_free_small_work() {
        assert_eq!(super::inference_thread_count(1, 63, 8), 1);
        assert_eq!(super::inference_thread_count(1, 64, 8), 8);
        assert_eq!(super::inference_thread_count(64, 1, 0), 0);
        assert_eq!(super::inference_thread_count(64, 1, 1), 1);
    }
}
