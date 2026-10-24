//! Iterative classification-tree entry points.

use crate::{
    ClassificationTargets, CoreResult, DistanceChoice, DistanceSamplingContext, ForestConfig,
    MultivariateSeriesView, NumericVectorView, SupervisedCriterion,
};

use super::{
    CandidateSelectionContext, ClassificationLeaf, LeafPayload, MultivariateExemplarBuilder,
    NodeMembership, ObservationIndex, OwnedObservationStore, ProximityTree, TreeWorkspace,
    VectorExemplarBuilder, build_tree_iterative, classification_node_is_pure,
    select_multivariate_classification, select_vector_classification,
};

/// Builds one vector-layout classification tree iteratively.
pub fn build_vector_classification_tree(
    config: &ForestConfig,
    tree_index: usize,
    observations: &[NumericVectorView<'_>],
    targets: &ClassificationTargets,
    root_membership: NodeMembership,
    root_observations: Vec<ObservationIndex>,
    distance_catalog: &[DistanceChoice],
    distance_sampling: DistanceSamplingContext,
    available_dimensions: usize,
    criterion: SupervisedCriterion,
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
            if classification_node_is_pure(node_observations, targets)? {
                return Ok(None);
            }
            select_vector_classification(
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
                    exemplar_count: 1,
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
        |membership, node_observations| {
            Ok(LeafPayload::Classification(ClassificationLeaf::from_node(
                membership,
                node_observations,
                targets,
            )?))
        },
    )?;
    tree.set_exemplars(OwnedObservationStore::Vector(exemplar_builder.finish()));
    Ok(tree)
}

/// Builds one multivariate classification tree iteratively.
pub fn build_multivariate_classification_tree(
    config: &ForestConfig,
    tree_index: usize,
    observations: &[MultivariateSeriesView<'_>],
    targets: &ClassificationTargets,
    root_membership: NodeMembership,
    root_observations: Vec<ObservationIndex>,
    distance_catalog: &[DistanceChoice],
    distance_sampling: DistanceSamplingContext,
    available_dimensions: usize,
    criterion: SupervisedCriterion,
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
            if classification_node_is_pure(node_observations, targets)? {
                return Ok(None);
            }
            select_multivariate_classification(
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
                    exemplar_count: 1,
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
        |membership, node_observations| {
            Ok(LeafPayload::Classification(ClassificationLeaf::from_node(
                membership,
                node_observations,
                targets,
            )?))
        },
    )?;
    tree.set_exemplars(OwnedObservationStore::Multivariate(
        exemplar_builder.finish(),
    ));
    Ok(tree)
}
