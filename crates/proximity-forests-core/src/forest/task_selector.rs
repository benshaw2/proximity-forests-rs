//! Task-specific candidate generation, routing, scoring, and retention.

use crate::{
    ClassificationTargets, CoreError, CoreResult, DimensionSelection, DistanceConfig,
    IsolationCriterion, ModelRng, MultivariateSeriesView, NumericVectorView, RandomPurpose,
    RegressionTargets, SeedMixer, SupervisedCriterion, TiePolicy,
};

use super::{
    CandidateIdentity, CandidatePartition, CandidateSelector, CandidateSplit, NodeMembership,
    ObservationIndex, SelectedCandidate, TreeWorkspace, generate_candidate, partition_multivariate,
    partition_vectors, sample_class_exemplars, score_classification, score_isolation,
    score_regression,
};

/// Immutable controls shared by task-specific candidate selectors.
#[derive(Clone, Copy, Debug)]
pub struct CandidateSelectionContext<'a> {
    /// Candidate distance catalog.
    pub distance_catalog: &'a [DistanceConfig],
    /// Number of available dimensions or channels.
    pub available_dimensions: usize,
    /// Node-level dimension selection rule.
    pub dimension_selection: DimensionSelection,
    /// Number of exemplars per candidate.
    pub exemplar_count: usize,
    /// Number of candidates evaluated at the node.
    pub candidate_count: usize,
    /// Equal-distance branch policy.
    pub tie_policy: TiePolicy,
    /// Stable tree seed.
    pub tree_seed: u64,
    /// Stable node-path identity.
    pub node_path: u64,
}

impl CandidateSelectionContext<'_> {
    fn validate(self, membership: &NodeMembership) -> CoreResult<()> {
        if self.candidate_count == 0 {
            return Err(CoreError::invalid_configuration(
                "candidate_count",
                "candidate selection requires at least one candidate",
            ));
        }
        if self.exemplar_count == 0 || self.exemplar_count > membership.distinct_count() {
            return Err(CoreError::invalid_configuration(
                "exemplar_count",
                "exemplar count must be between one and the node distinct count",
            ));
        }
        Ok(())
    }
}

/// Selects the best vector-layout classification candidate.
pub fn select_vector_classification(
    observations: &[NumericVectorView<'_>],
    node_observations: &[ObservationIndex],
    membership: &NodeMembership,
    targets: &ClassificationTargets,
    criterion: SupervisedCriterion,
    context: CandidateSelectionContext<'_>,
    workspace: &mut TreeWorkspace,
) -> CoreResult<Option<SelectedCandidate>> {
    select_classification_candidates(
        targets,
        membership,
        node_observations,
        context,
        workspace,
        |candidate, rng, workspace| {
            partition_vectors(
                observations,
                node_observations,
                membership,
                candidate,
                context.tie_policy,
                Some(rng),
                workspace,
            )
        },
        |partition| {
            score_classification(partition, membership, node_observations, targets, criterion)
        },
    )
}

/// Selects the best multivariate classification candidate.
pub fn select_multivariate_classification(
    observations: &[MultivariateSeriesView<'_>],
    node_observations: &[ObservationIndex],
    membership: &NodeMembership,
    targets: &ClassificationTargets,
    criterion: SupervisedCriterion,
    context: CandidateSelectionContext<'_>,
    workspace: &mut TreeWorkspace,
) -> CoreResult<Option<SelectedCandidate>> {
    select_classification_candidates(
        targets,
        membership,
        node_observations,
        context,
        workspace,
        |candidate, rng, workspace| {
            partition_multivariate(
                observations,
                node_observations,
                membership,
                candidate,
                context.tie_policy,
                Some(rng),
                workspace,
            )
        },
        |partition| {
            score_classification(partition, membership, node_observations, targets, criterion)
        },
    )
}

/// Selects the best vector-layout regression candidate.
pub fn select_vector_regression(
    observations: &[NumericVectorView<'_>],
    node_observations: &[ObservationIndex],
    membership: &NodeMembership,
    targets: &RegressionTargets,
    criterion: SupervisedCriterion,
    context: CandidateSelectionContext<'_>,
    workspace: &mut TreeWorkspace,
) -> CoreResult<Option<SelectedCandidate>> {
    select_candidates(
        membership,
        node_observations,
        context,
        workspace,
        |candidate, rng, workspace| {
            partition_vectors(
                observations,
                node_observations,
                membership,
                candidate,
                context.tie_policy,
                Some(rng),
                workspace,
            )
        },
        |partition| score_regression(partition, membership, node_observations, targets, criterion),
    )
}

/// Selects the best multivariate regression candidate.
pub fn select_multivariate_regression(
    observations: &[MultivariateSeriesView<'_>],
    node_observations: &[ObservationIndex],
    membership: &NodeMembership,
    targets: &RegressionTargets,
    criterion: SupervisedCriterion,
    context: CandidateSelectionContext<'_>,
    workspace: &mut TreeWorkspace,
) -> CoreResult<Option<SelectedCandidate>> {
    select_candidates(
        membership,
        node_observations,
        context,
        workspace,
        |candidate, rng, workspace| {
            partition_multivariate(
                observations,
                node_observations,
                membership,
                candidate,
                context.tie_policy,
                Some(rng),
                workspace,
            )
        },
        |partition| score_regression(partition, membership, node_observations, targets, criterion),
    )
}

/// Selects the best vector-layout isolation candidate.
pub fn select_vector_isolation(
    observations: &[NumericVectorView<'_>],
    node_observations: &[ObservationIndex],
    membership: &NodeMembership,
    criterion: IsolationCriterion,
    context: CandidateSelectionContext<'_>,
    workspace: &mut TreeWorkspace,
) -> CoreResult<Option<SelectedCandidate>> {
    select_candidates(
        membership,
        node_observations,
        context,
        workspace,
        |candidate, rng, workspace| {
            partition_vectors(
                observations,
                node_observations,
                membership,
                candidate,
                context.tie_policy,
                Some(rng),
                workspace,
            )
        },
        |partition| score_isolation(partition, criterion),
    )
}

/// Selects the best multivariate isolation candidate.
pub fn select_multivariate_isolation(
    observations: &[MultivariateSeriesView<'_>],
    node_observations: &[ObservationIndex],
    membership: &NodeMembership,
    criterion: IsolationCriterion,
    context: CandidateSelectionContext<'_>,
    workspace: &mut TreeWorkspace,
) -> CoreResult<Option<SelectedCandidate>> {
    select_candidates(
        membership,
        node_observations,
        context,
        workspace,
        |candidate, rng, workspace| {
            partition_multivariate(
                observations,
                node_observations,
                membership,
                candidate,
                context.tie_policy,
                Some(rng),
                workspace,
            )
        },
        |partition| score_isolation(partition, criterion),
    )
}

fn select_classification_candidates<Route, Score>(
    targets: &ClassificationTargets,
    membership: &NodeMembership,
    node_observations: &[ObservationIndex],
    context: CandidateSelectionContext<'_>,
    workspace: &mut TreeWorkspace,
    mut route: Route,
    mut score: Score,
) -> CoreResult<Option<SelectedCandidate>>
where
    Route:
        FnMut(&CandidateSplit, &mut ModelRng, &mut TreeWorkspace) -> CoreResult<CandidatePartition>,
    Score: FnMut(&CandidatePartition) -> CoreResult<super::CandidateScore>,
{
    context.validate(membership)?;
    let mut selector = CandidateSelector::new();
    for candidate_index in 0..context.candidate_count {
        let generic = generate_candidate(
            context.distance_catalog,
            context.available_dimensions,
            context.dimension_selection,
            1,
            membership,
            node_observations,
            CandidateIdentity {
                tree_seed: context.tree_seed,
                node_path: context.node_path,
                candidate_index,
            },
        )?;
        let (distance, dimensions, _ignored) = generic.into_parts();
        let exemplars = match sample_class_exemplars(
            membership,
            node_observations,
            targets,
            context.tree_seed,
            context.node_path,
            candidate_index,
        ) {
            Ok((_output, exemplars)) => exemplars,
            Err(error)
                if error
                    .to_string()
                    .contains("at least two represented classes") =>
            {
                selector.reject()?;
                continue;
            }
            Err(error) => return Err(error),
        };
        let candidate = CandidateSplit::new(distance, dimensions, exemplars)?;
        let mut branch_rng = branch_rng(context, candidate_index);
        let partition = match route(&candidate, &mut branch_rng, workspace) {
            Ok(partition) => partition,
            Err(error) if is_degenerate_candidate(&error) => {
                selector.reject()?;
                continue;
            }
            Err(error) => return Err(error),
        };
        let candidate_score = score(&partition)?;
        selector.consider(candidate, candidate_score, partition)?;
    }
    Ok(selector.into_best())
}

fn select_candidates<Route, Score>(
    membership: &NodeMembership,
    node_observations: &[ObservationIndex],
    context: CandidateSelectionContext<'_>,
    workspace: &mut TreeWorkspace,
    mut route: Route,
    mut score: Score,
) -> CoreResult<Option<SelectedCandidate>>
where
    Route: FnMut(
        &super::CandidateSplit,
        &mut ModelRng,
        &mut TreeWorkspace,
    ) -> CoreResult<CandidatePartition>,
    Score: FnMut(&CandidatePartition) -> CoreResult<super::CandidateScore>,
{
    context.validate(membership)?;
    let mut selector = CandidateSelector::new();

    for candidate_index in 0..context.candidate_count {
        let candidate = generate_candidate(
            context.distance_catalog,
            context.available_dimensions,
            context.dimension_selection,
            context.exemplar_count,
            membership,
            node_observations,
            CandidateIdentity {
                tree_seed: context.tree_seed,
                node_path: context.node_path,
                candidate_index,
            },
        )?;
        let mut branch_rng = branch_rng(context, candidate_index);

        let partition = match route(&candidate, &mut branch_rng, workspace) {
            Ok(partition) => partition,
            Err(error) if is_degenerate_candidate(&error) => {
                selector.reject()?;
                continue;
            }
            Err(error) => return Err(error),
        };
        let candidate_score = score(&partition)?;
        selector.consider(candidate, candidate_score, partition)?;
    }

    Ok(selector.into_best())
}

fn branch_rng(context: CandidateSelectionContext<'_>, candidate_index: usize) -> ModelRng {
    let node_seed = SeedMixer::node_purpose(
        context.tree_seed,
        context.node_path,
        RandomPurpose::BranchAssignment,
    );
    ModelRng::from_seed(SeedMixer::candidate(node_seed, candidate_index))
}

fn is_degenerate_candidate(error: &CoreError) -> bool {
    let message = error.to_string();
    message.contains("empty exemplar branch")
}
