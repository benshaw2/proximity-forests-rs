//! Iterative fitted-tree traversal for vector and multivariate observations.

use crate::{
    CoreError, CoreResult, DistanceCutoff, DistanceEvaluator, DistanceScratch, ModelRng,
    MultivariateSeriesView, NumericVectorView, RandomPurpose, SeedMixer, TiePolicy,
};

use super::{LeafNode, NodeId, ProximityTree, SplitNode, TreeNode};

/// Terminal leaf reached by one iterative traversal.
#[derive(Clone, Copy, Debug)]
pub struct LeafVisit<'a> {
    leaf: &'a LeafNode,
    depth: usize,
    node: NodeId,
}

impl<'a> LeafVisit<'a> {
    /// Returns the fitted terminal leaf.
    #[must_use]
    pub const fn leaf(self) -> &'a LeafNode {
        self.leaf
    }

    /// Returns the number of internal nodes traversed.
    #[must_use]
    pub const fn depth(self) -> usize {
        self.depth
    }

    /// Returns the terminal node index.
    #[must_use]
    pub const fn node(self) -> NodeId {
        self.node
    }
}

/// Traverses a fitted tree for one vector observation.
pub fn traverse_vector<'a>(
    tree: &'a ProximityTree,
    query: NumericVectorView<'_>,
    observations: &[NumericVectorView<'_>],
    tie_policy: TiePolicy,
    prediction_seed: u64,
    scratch: &mut DistanceScratch,
) -> CoreResult<LeafVisit<'a>> {
    traverse(
        tree,
        tie_policy,
        prediction_seed,
        |split, node, scratch| {
            route_split_vector(
                split,
                node,
                query,
                observations,
                tie_policy,
                prediction_seed,
                scratch,
            )
        },
        scratch,
    )
}

/// Traverses a fitted tree for one multivariate observation.
pub fn traverse_multivariate<'a>(
    tree: &'a ProximityTree,
    query: MultivariateSeriesView<'_>,
    observations: &[MultivariateSeriesView<'_>],
    tie_policy: TiePolicy,
    prediction_seed: u64,
    scratch: &mut DistanceScratch,
) -> CoreResult<LeafVisit<'a>> {
    traverse(
        tree,
        tie_policy,
        prediction_seed,
        |split, node, scratch| {
            route_split_multivariate(
                split,
                node,
                query,
                observations,
                tie_policy,
                prediction_seed,
                scratch,
            )
        },
        scratch,
    )
}

fn traverse<'a, Route>(
    tree: &'a ProximityTree,
    _tie_policy: TiePolicy,
    _prediction_seed: u64,
    mut route: Route,
    scratch: &mut DistanceScratch,
) -> CoreResult<LeafVisit<'a>>
where
    Route: FnMut(&SplitNode, NodeId, &mut DistanceScratch) -> CoreResult<NodeId>,
{
    let mut node = tree
        .root()
        .ok_or_else(|| CoreError::invalid_data("cannot traverse an empty fitted tree"))?;
    let mut depth = 0usize;
    loop {
        match tree.node(node).ok_or_else(|| {
            CoreError::invalid_data(format!("tree node {} was unavailable", node.get()))
        })? {
            TreeNode::Leaf(leaf) => return Ok(LeafVisit { leaf, depth, node }),
            TreeNode::Split(split) => {
                node = route(split, node, scratch)?;
                depth = depth
                    .checked_add(1)
                    .ok_or_else(|| CoreError::invalid_data("prediction depth overflowed usize"))?;
            }
        }
    }
}

fn route_split_vector(
    split: &SplitNode,
    node: NodeId,
    query: NumericVectorView<'_>,
    observations: &[NumericVectorView<'_>],
    tie_policy: TiePolicy,
    prediction_seed: u64,
    scratch: &mut DistanceScratch,
) -> CoreResult<NodeId> {
    let evaluator = DistanceEvaluator::new(split.distance().clone());
    let selected = selected(split);
    route_branches(
        split,
        node,
        tie_policy,
        prediction_seed,
        |branch, cutoff| {
            let exemplar_index = split.branches()[branch].exemplar().index();
            let exemplar = observations.get(exemplar_index).copied().ok_or_else(|| {
                CoreError::invalid_data(format!(
                    "tree exemplar {exemplar_index} is outside vector exemplar count {}",
                    observations.len()
                ))
            })?;
            evaluator.evaluate_vector(query, exemplar, selected, cutoff, scratch)
        },
    )
}

fn route_split_multivariate(
    split: &SplitNode,
    node: NodeId,
    query: MultivariateSeriesView<'_>,
    observations: &[MultivariateSeriesView<'_>],
    tie_policy: TiePolicy,
    prediction_seed: u64,
    scratch: &mut DistanceScratch,
) -> CoreResult<NodeId> {
    let evaluator = DistanceEvaluator::new(split.distance().clone());
    let selected = selected(split);
    route_branches(
        split,
        node,
        tie_policy,
        prediction_seed,
        |branch, cutoff| {
            let exemplar_index = split.branches()[branch].exemplar().index();
            let exemplar = observations.get(exemplar_index).copied().ok_or_else(|| {
                CoreError::invalid_data(format!(
                    "tree exemplar {exemplar_index} is outside multivariate exemplar count {}",
                    observations.len()
                ))
            })?;
            evaluator.evaluate_multivariate(query, exemplar, selected, cutoff, scratch)
        },
    )
}

fn route_branches<Evaluate>(
    split: &SplitNode,
    node: NodeId,
    tie_policy: TiePolicy,
    prediction_seed: u64,
    mut evaluate: Evaluate,
) -> CoreResult<NodeId>
where
    Evaluate: FnMut(usize, DistanceCutoff) -> CoreResult<f64>,
{
    if split.branches().is_empty() {
        return Err(CoreError::invalid_data(
            "a fitted split has no exemplar branches",
        ));
    }
    let seed = SeedMixer::node_purpose(
        prediction_seed,
        node.get() as u64,
        RandomPurpose::PredictionTie,
    );
    let mut rng = ModelRng::from_seed(seed);
    let mut best = f64::INFINITY;
    let mut winner = 0usize;
    let mut tie_count = 0usize;

    for branch in 0..split.branches().len() {
        let cutoff = if best.is_finite() {
            DistanceCutoff::finite(best)?
        } else {
            DistanceCutoff::None
        };
        let distance = evaluate(branch, cutoff)?;
        if distance.is_nan() || distance < 0.0 {
            return Err(CoreError::invalid_data(
                "prediction routing distance must be nonnegative and not NaN",
            ));
        }
        if distance < best {
            best = distance;
            winner = branch;
            tie_count = 1;
        } else if distance == best {
            tie_count += 1;
            if tie_policy == TiePolicy::Random && rng.index(tie_count)? == 0 {
                winner = branch;
            }
        }
    }
    Ok(split.branches()[winner].child())
}

#[inline]
fn selected(split: &SplitNode) -> Option<&[usize]> {
    (!split.selected_dimensions().is_empty()).then_some(split.selected_dimensions())
}
