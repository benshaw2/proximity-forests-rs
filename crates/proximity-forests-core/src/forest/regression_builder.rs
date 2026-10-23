//! Iterative regression-tree entry points.

use super::{
    CandidateSelectionContext, LeafPayload, NodeMembership, ObservationIndex, ProximityTree,
    RegressionLeaf, TreeWorkspace, build_tree_iterative, select_multivariate_regression,
    select_vector_regression,
};
use crate::{
    CoreResult, DistanceChoice, DistanceSamplingContext, ForestConfig, MultivariateSeriesView,
    NumericVectorView, RegressionTargets, SupervisedCriterion,
};

/// Builds one vector-layout regression tree iteratively.
pub fn build_vector_regression_tree(
    config: &ForestConfig,
    tree_index: usize,
    observations: &[NumericVectorView<'_>],
    targets: &RegressionTargets,
    root_membership: NodeMembership,
    root_observations: Vec<ObservationIndex>,
    distance_catalog: &[DistanceChoice],
    distance_sampling: DistanceSamplingContext,
    available_dimensions: usize,
    exemplar_count: usize,
    criterion: SupervisedCriterion,
    workspace: &mut TreeWorkspace,
) -> CoreResult<ProximityTree> {
    build_tree_iterative(
        config,
        tree_index,
        root_membership,
        root_observations,
        workspace,
        |tree_seed, _depth, node_path, membership, node_observations, workspace| {
            select_vector_regression(
                observations,
                node_observations,
                membership,
                targets,
                criterion,
                CandidateSelectionContext {
                    distance_catalog,
                    distance_sampling,
                    available_dimensions,
                    dimension_selection: config.dimension_selection,
                    exemplar_count,
                    candidate_count: config.candidate_count.get(),
                    tie_policy: config.branch_tie_policy,
                    tree_seed,
                    node_path,
                },
                workspace,
            )
        },
        |membership, node_observations| {
            Ok(LeafPayload::Regression(RegressionLeaf::from_node(
                membership,
                node_observations,
                targets,
            )?))
        },
    )
}

/// Builds one multivariate-layout regression tree iteratively.
pub fn build_multivariate_regression_tree(
    config: &ForestConfig,
    tree_index: usize,
    observations: &[MultivariateSeriesView<'_>],
    targets: &RegressionTargets,
    root_membership: NodeMembership,
    root_observations: Vec<ObservationIndex>,
    distance_catalog: &[DistanceChoice],
    distance_sampling: DistanceSamplingContext,
    available_dimensions: usize,
    exemplar_count: usize,
    criterion: SupervisedCriterion,
    workspace: &mut TreeWorkspace,
) -> CoreResult<ProximityTree> {
    build_tree_iterative(
        config,
        tree_index,
        root_membership,
        root_observations,
        workspace,
        |tree_seed, _depth, node_path, membership, node_observations, workspace| {
            select_multivariate_regression(
                observations,
                node_observations,
                membership,
                targets,
                criterion,
                CandidateSelectionContext {
                    distance_catalog,
                    distance_sampling,
                    available_dimensions,
                    dimension_selection: config.dimension_selection,
                    exemplar_count,
                    candidate_count: config.candidate_count.get(),
                    tie_policy: config.branch_tie_policy,
                    tree_seed,
                    node_path,
                },
                workspace,
            )
        },
        |membership, node_observations| {
            Ok(LeafPayload::Regression(RegressionLeaf::from_node(
                membership,
                node_observations,
                targets,
            )?))
        },
    )
}
