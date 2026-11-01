//! Iterative regression-tree entry points.

use super::{
    CandidateSelectionContext, ExecutionContext, LeafPayload, MultivariateExemplarBuilder,
    NodeMembership, ObservationIndex, ProximityTree, RegressionLeaf, TreeWorkspace,
    VectorExemplarBuilder, build_tree_iterative, select_multivariate_regression,
    select_vector_regression,
};
use crate::{
    CoreResult, DistanceChoice, DistanceSamplingContext, DistanceWorkShape, ForestConfig,
    MultivariateSeriesView, NumericVectorView, RegressionTargets, SupervisedCriterion,
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
    execution: &ExecutionContext,
    workspace: &mut TreeWorkspace,
) -> CoreResult<ProximityTree> {
    let distance_work = distance_catalog
        .iter()
        .map(|choice| {
            choice.estimate_work(DistanceWorkShape::new(
                distance_sampling.maximum_length,
                distance_sampling.maximum_length,
                1,
            ))
        })
        .max()
        .unwrap_or_default();
    let mut exemplar_builder = VectorExemplarBuilder::new(observations[0].scalar_kind());
    let mut tree = build_tree_iterative(
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
                    execution,
                    distance_catalog,
                    distance_sampling,
                    distance_work,
                    available_dimensions,
                    dimension_selection: config.dimension_selection,
                    exemplar_count: exemplar_count.min(membership.distinct_count()),
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
                    exemplar_builder.intern(index.get(), observation)
                })
                .collect()
        },
        |membership, node_observations| {
            Ok(LeafPayload::Regression(RegressionLeaf::from_node(
                membership,
                node_observations,
                targets,
            )?))
        },
    )?;
    tree.set_exemplars(exemplar_builder.finish()?);
    Ok(tree)
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
    execution: &ExecutionContext,
    workspace: &mut TreeWorkspace,
) -> CoreResult<ProximityTree> {
    let distance_work = distance_catalog
        .iter()
        .map(|choice| {
            choice.estimate_work(DistanceWorkShape::new(
                distance_sampling.maximum_length,
                distance_sampling.maximum_length,
                available_dimensions,
            ))
        })
        .max()
        .unwrap_or_default();
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
            select_multivariate_regression(
                observations,
                node_observations,
                membership,
                targets,
                criterion,
                CandidateSelectionContext {
                    execution,
                    distance_catalog,
                    distance_sampling,
                    distance_work,
                    available_dimensions,
                    dimension_selection: config.dimension_selection,
                    exemplar_count: exemplar_count.min(membership.distinct_count()),
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
                    exemplar_builder.intern(index.get(), observation)
                })
                .collect()
        },
        |membership, node_observations| {
            Ok(LeafPayload::Regression(RegressionLeaf::from_node(
                membership,
                node_observations,
                targets,
            )?))
        },
    )?;
    tree.set_exemplars(exemplar_builder.finish()?);
    Ok(tree)
}
