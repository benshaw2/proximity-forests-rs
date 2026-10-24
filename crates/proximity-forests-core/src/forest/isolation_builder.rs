//! Iterative isolation-tree entry points.

use super::{
    CandidateSelectionContext, LeafPayload, MultivariateExemplarBuilder, NodeMembership,
    ObservationIndex, OwnedObservationStore, ProximityTree, TreeWorkspace, VectorExemplarBuilder,
    build_tree_iterative, select_multivariate_isolation, select_vector_isolation,
};
use crate::{
    CoreResult, DistanceChoice, DistanceSamplingContext, ForestConfig, IsolationCriterion,
    MultivariateSeriesView, NumericVectorView,
};

/// Builds one vector-layout isolation tree iteratively.
pub fn build_vector_isolation_tree(
    config: &ForestConfig,
    tree_index: usize,
    observations: &[NumericVectorView<'_>],
    root_membership: NodeMembership,
    root_observations: Vec<ObservationIndex>,
    distance_catalog: &[DistanceChoice],
    distance_sampling: DistanceSamplingContext,
    available_dimensions: usize,
    exemplar_count: usize,
    criterion: IsolationCriterion,
    workspace: &mut TreeWorkspace,
) -> CoreResult<ProximityTree> {
    let mut exemplar_builder = VectorExemplarBuilder::new(observations[0].scalar_kind());
    let mut tree = build_tree_iterative(
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
        |selected| {
            selected
                .iter()
                .map(|index| {
                    let observation = observations.get(index.get()).copied().ok_or_else(|| {
                        crate::CoreError::invalid_data(
                            "selected exemplar observation was unavailable",
                        )
                    })?;
                    exemplar_builder.push(observation)
                })
                .collect()
        },
        |membership, _node_observations| {
            Ok(LeafPayload::Isolation {
                occurrence_count: membership.occurrence_count(),
            })
        },
    )?;
    tree.set_exemplars(OwnedObservationStore::Vector(exemplar_builder.finish()));
    Ok(tree)
}

/// Builds one multivariate-layout isolation tree iteratively.
pub fn build_multivariate_isolation_tree(
    config: &ForestConfig,
    tree_index: usize,
    observations: &[MultivariateSeriesView<'_>],
    root_membership: NodeMembership,
    root_observations: Vec<ObservationIndex>,
    distance_catalog: &[DistanceChoice],
    distance_sampling: DistanceSamplingContext,
    available_dimensions: usize,
    exemplar_count: usize,
    criterion: IsolationCriterion,
    workspace: &mut TreeWorkspace,
) -> CoreResult<ProximityTree> {
    let mut exemplar_builder = MultivariateExemplarBuilder::new(
        observations[0].scalar_kind(),
        observations[0].dimension_count(),
    );
    let mut tree = build_tree_iterative(
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
        |selected| {
            selected
                .iter()
                .map(|index| {
                    let observation = observations.get(index.get()).copied().ok_or_else(|| {
                        crate::CoreError::invalid_data(
                            "selected exemplar observation was unavailable",
                        )
                    })?;
                    exemplar_builder.push(observation)
                })
                .collect()
        },
        |membership, _node_observations| {
            Ok(LeafPayload::Isolation {
                occurrence_count: membership.occurrence_count(),
            })
        },
    )?;
    tree.set_exemplars(OwnedObservationStore::Multivariate(
        exemplar_builder.finish(),
    ));
    Ok(tree)
}
