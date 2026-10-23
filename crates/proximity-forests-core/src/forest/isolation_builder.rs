//! Iterative isolation-tree entry points.

use crate::{
    CoreResult, DistanceConfig, ForestConfig, IsolationCriterion, MultivariateSeriesView,
    NumericVectorView,
};

use super::{
    CandidateSelectionContext, LeafPayload, NodeMembership, ObservationIndex, ProximityTree,
    TreeWorkspace, build_tree_iterative, select_multivariate_isolation, select_vector_isolation,
};

/// Builds one vector-layout isolation tree iteratively.
pub fn build_vector_isolation_tree(
    config: &ForestConfig,
    tree_index: usize,
    observations: &[NumericVectorView<'_>],
    root_membership: NodeMembership,
    root_observations: Vec<ObservationIndex>,
    distance_catalog: &[DistanceConfig],
    available_dimensions: usize,
    exemplar_count: usize,
    criterion: IsolationCriterion,
    workspace: &mut TreeWorkspace,
) -> CoreResult<ProximityTree> {
    build_tree_iterative(
        config,
        tree_index,
        root_membership,
        root_observations,
        workspace,
        |tree_seed, _depth, node_path, membership, node_observations, workspace| {
            select_vector_isolation(
                observations,
                node_observations,
                membership,
                criterion,
                CandidateSelectionContext {
                    distance_catalog,
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
        |membership, _node_observations| {
            Ok(LeafPayload::Isolation {
                occurrence_count: membership.occurrence_count(),
            })
        },
    )
}

/// Builds one multivariate isolation tree iteratively.
pub fn build_multivariate_isolation_tree(
    config: &ForestConfig,
    tree_index: usize,
    observations: &[MultivariateSeriesView<'_>],
    root_membership: NodeMembership,
    root_observations: Vec<ObservationIndex>,
    distance_catalog: &[DistanceConfig],
    available_dimensions: usize,
    exemplar_count: usize,
    criterion: IsolationCriterion,
    workspace: &mut TreeWorkspace,
) -> CoreResult<ProximityTree> {
    build_tree_iterative(
        config,
        tree_index,
        root_membership,
        root_observations,
        workspace,
        |tree_seed, _depth, node_path, membership, node_observations, workspace| {
            select_multivariate_isolation(
                observations,
                node_observations,
                membership,
                criterion,
                CandidateSelectionContext {
                    distance_catalog,
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
        |membership, _node_observations| {
            Ok(LeafPayload::Isolation {
                occurrence_count: membership.occurrence_count(),
            })
        },
    )
}
