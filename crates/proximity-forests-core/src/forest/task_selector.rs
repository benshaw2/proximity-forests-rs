//! Task-specific candidate generation, routing, scoring, and retention.

use super::{
    CandidateEvaluation, CandidateIdentity, CandidatePartition, CandidateSelector, CandidateSplit,
    ExecutionContext, NodeMembership, ObservationIndex, SelectedCandidate, TreeWorkspace,
    generate_candidate, parallel_indexed_init_in, partition_multivariate, partition_vectors,
    sample_class_exemplars, score_classification, score_isolation, score_regression,
};
use crate::{
    ClassificationTargets, CoreError, CoreResult, DimensionSelection, DistanceChoice,
    IsolationCriterion, ModelRng, MultivariateSeriesView, NumericVectorView, RandomPurpose,
    RegressionTargets, SeedMixer, SupervisedCriterion, TiePolicy,
};
use crate::{DistanceSamplingContext, WorkUnits};

/// Immutable controls shared by task-specific candidate selectors.
#[derive(Clone, Copy, Debug)]
pub struct CandidateSelectionContext<'a> {
    /// Operation-scoped bounded executor shared by trees and candidate work.
    pub execution: &'a ExecutionContext,
    /// Fixed configurations and sampled families available to each candidate.
    ///
    /// Sampled entries are resolved once before routing and scoring.
    pub distance_catalog: &'a [DistanceChoice],
    /// Dataset-level statistics reused during candidate-local parameter sampling.
    ///
    /// The context is computed once at the fitting boundary; candidate selection
    /// performs no dataset scan.
    pub distance_sampling: DistanceSamplingContext,
    /// Representative work for one routed observation/exemplar comparison.
    pub distance_work: WorkUnits,
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
    route: Route,
    score: Score,
) -> CoreResult<Option<SelectedCandidate>>
where
    Route: Fn(&CandidateSplit, &mut ModelRng, &mut TreeWorkspace) -> CoreResult<CandidatePartition>
        + Sync,
    Score: Fn(&CandidatePartition) -> CoreResult<super::CandidateScore> + Sync,
{
    context.validate(membership)?;
    let mut selector = CandidateSelector::new();
    if should_parallelize_candidates(context, membership) {
        let evaluations = parallel_indexed_init_in(
            context.execution,
            context.candidate_count,
            TreeWorkspace::new,
            |candidate_workspace, candidate_index| {
                evaluate_classification_candidate(
                    candidate_index,
                    targets,
                    membership,
                    node_observations,
                    context,
                    candidate_workspace,
                    &route,
                    &score,
                )
            },
        )?;
        for evaluation in evaluations {
            selector.consider_evaluation(evaluation)?;
        }
    } else {
        for candidate_index in 0..context.candidate_count {
            let evaluation = evaluate_classification_candidate(
                candidate_index,
                targets,
                membership,
                node_observations,
                context,
                workspace,
                &route,
                &score,
            )?;
            selector.consider_evaluation(evaluation)?;
        }
    }

    Ok(selector.into_best())
}

// Calibrated relative-work threshold, evaluated once per node. P4C2 showed
// negligible or negative scaling for linear-distance workloads through 786,432
// units, while dynamic-programming work at 884,736 units improved with 4-8
// workers. The representative cost is cached before tree construction, so this
// gate performs only saturating integer arithmetic and no distance dispatch.
const MIN_PARALLEL_CANDIDATE_WORK: u64 = 800_000;

#[inline]
fn should_parallelize_candidates(
    context: CandidateSelectionContext<'_>,
    membership: &NodeMembership,
) -> bool {
    context.execution.is_parallel()
        && context.execution.policy().candidate_splits()
        && context.candidate_count > 1
        && context
            .distance_work
            .saturating_mul(membership.distinct_count())
            .saturating_mul(context.exemplar_count)
            .saturating_mul(context.candidate_count)
            .get()
            >= MIN_PARALLEL_CANDIDATE_WORK
}

fn evaluate_classification_candidate<Route, Score>(
    candidate_index: usize,
    targets: &ClassificationTargets,
    membership: &NodeMembership,
    node_observations: &[ObservationIndex],
    context: CandidateSelectionContext<'_>,
    workspace: &mut TreeWorkspace,
    route: &Route,
    score: &Score,
) -> CoreResult<CandidateEvaluation>
where
    Route: Fn(&CandidateSplit, &mut ModelRng, &mut TreeWorkspace) -> CoreResult<CandidatePartition>
        + Sync,
    Score: Fn(&CandidatePartition) -> CoreResult<super::CandidateScore> + Sync,
{
    let generic = generate_candidate(
        context.distance_catalog,
        context.distance_sampling,
        context.available_dimensions,
        context.dimension_selection,
        1,
        membership,
        node_observations,
        candidate_identity(context, candidate_index),
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
        Err(error) if is_unrepresented_class_candidate(&error) => {
            return Ok(CandidateEvaluation::Rejected);
        }
        Err(error) => return Err(error),
    };
    let candidate = CandidateSplit::new(distance, dimensions, exemplars)?;
    evaluate_generated_candidate(candidate, candidate_index, context, workspace, route, score)
}

fn select_candidates<Route, Score>(
    membership: &NodeMembership,
    node_observations: &[ObservationIndex],
    context: CandidateSelectionContext<'_>,
    workspace: &mut TreeWorkspace,
    route: Route,
    score: Score,
) -> CoreResult<Option<SelectedCandidate>>
where
    Route: Fn(&CandidateSplit, &mut ModelRng, &mut TreeWorkspace) -> CoreResult<CandidatePartition>
        + Sync,
    Score: Fn(&CandidatePartition) -> CoreResult<super::CandidateScore> + Sync,
{
    context.validate(membership)?;
    let mut selector = CandidateSelector::new();
    if should_parallelize_candidates(context, membership) {
        let evaluations = parallel_indexed_init_in(
            context.execution,
            context.candidate_count,
            TreeWorkspace::new,
            |candidate_workspace, candidate_index| {
                evaluate_candidate(
                    candidate_index,
                    membership,
                    node_observations,
                    context,
                    candidate_workspace,
                    &route,
                    &score,
                )
            },
        )?;
        for evaluation in evaluations {
            selector.consider_evaluation(evaluation)?;
        }
    } else {
        for candidate_index in 0..context.candidate_count {
            let evaluation = evaluate_candidate(
                candidate_index,
                membership,
                node_observations,
                context,
                workspace,
                &route,
                &score,
            )?;
            selector.consider_evaluation(evaluation)?;
        }
    }

    Ok(selector.into_best())
}

fn evaluate_candidate<Route, Score>(
    candidate_index: usize,
    membership: &NodeMembership,
    node_observations: &[ObservationIndex],
    context: CandidateSelectionContext<'_>,
    workspace: &mut TreeWorkspace,
    route: &Route,
    score: &Score,
) -> CoreResult<CandidateEvaluation>
where
    Route: Fn(&CandidateSplit, &mut ModelRng, &mut TreeWorkspace) -> CoreResult<CandidatePartition>
        + Sync,
    Score: Fn(&CandidatePartition) -> CoreResult<super::CandidateScore> + Sync,
{
    let candidate = generate_candidate(
        context.distance_catalog,
        context.distance_sampling,
        context.available_dimensions,
        context.dimension_selection,
        context.exemplar_count,
        membership,
        node_observations,
        candidate_identity(context, candidate_index),
    )?;
    evaluate_generated_candidate(candidate, candidate_index, context, workspace, route, score)
}

fn evaluate_generated_candidate<Route, Score>(
    candidate: CandidateSplit,
    candidate_index: usize,
    context: CandidateSelectionContext<'_>,
    workspace: &mut TreeWorkspace,
    route: &Route,
    score: &Score,
) -> CoreResult<CandidateEvaluation>
where
    Route: Fn(&CandidateSplit, &mut ModelRng, &mut TreeWorkspace) -> CoreResult<CandidatePartition>
        + Sync,
    Score: Fn(&CandidatePartition) -> CoreResult<super::CandidateScore> + Sync,
{
    let mut branch_rng = branch_rng(context, candidate_index);
    let partition = match route(&candidate, &mut branch_rng, workspace) {
        Ok(partition) => partition,
        Err(error) if is_degenerate_candidate(&error) => {
            return Ok(CandidateEvaluation::Rejected);
        }
        Err(error) => return Err(error),
    };
    let candidate_score = score(&partition)?;
    Ok(CandidateEvaluation::valid(
        candidate,
        candidate_score,
        partition,
    ))
}

#[inline]
const fn candidate_identity(
    context: CandidateSelectionContext<'_>,
    candidate_index: usize,
) -> CandidateIdentity {
    CandidateIdentity {
        tree_seed: context.tree_seed,
        node_path: context.node_path,
        candidate_index,
    }
}

fn branch_rng(context: CandidateSelectionContext<'_>, candidate_index: usize) -> ModelRng {
    let node_seed = SeedMixer::node_purpose(
        context.tree_seed,
        context.node_path,
        RandomPurpose::BranchAssignment,
    );
    ModelRng::from_seed(SeedMixer::candidate(node_seed, candidate_index))
}

fn is_unrepresented_class_candidate(error: &CoreError) -> bool {
    error
        .to_string()
        .contains("at least two represented classes")
}

fn is_degenerate_candidate(error: &CoreError) -> bool {
    error.to_string().contains("empty exemplar branch")
}

#[cfg(test)]
mod tests {
    use super::ExecutionContext;
    use super::{
        CandidatePartition, CandidateSelectionContext, CandidateSplit, ModelRng, TreeWorkspace,
        candidate_identity, should_parallelize_candidates,
    };
    use crate::{
        CoreError, CoreResult, DimensionSelection, DistanceChoice, DistanceConfig,
        DistanceSamplingContext, NodeMembership, SampleId, TiePolicy, WorkUnits,
    };

    fn context<'a>(catalog: &'a [DistanceChoice]) -> CandidateSelectionContext<'a> {
        let execution = Box::leak(Box::new(ExecutionContext::new(1).expect("execution")));
        CandidateSelectionContext {
            execution,
            distance_catalog: catalog,
            distance_sampling: DistanceSamplingContext::new(8, 1.0).expect("sampling"),
            distance_work: WorkUnits::new(8),
            available_dimensions: 3,
            dimension_selection: DimensionSelection::All,
            exemplar_count: 2,
            candidate_count: 5,
            tie_policy: TiePolicy::First,
            tree_seed: 17,
            node_path: 9,
        }
    }

    fn assert_sync<T: Sync>(_value: &T) {}

    fn membership(count: usize) -> NodeMembership {
        let occurrences = (0..count)
            .map(|index| Some(SampleId::new(index as u64)))
            .collect::<Vec<_>>();
        NodeMembership::from_occurrences(&occurrences).expect("membership")
    }

    #[test]
    fn candidate_parallel_gate_requires_parallel_context_and_sufficient_grain() {
        let catalog = [DistanceChoice::Fixed(DistanceConfig::SquaredEuclidean)];
        let sequential = context(&catalog);
        assert!(!should_parallelize_candidates(sequential, &membership(512)));

        let execution = Box::leak(Box::new(ExecutionContext::new(4).expect("execution")));
        let parallel = CandidateSelectionContext {
            execution,
            candidate_count: 8,
            exemplar_count: 2,
            ..context(&catalog)
        };
        assert!(!should_parallelize_candidates(parallel, &membership(6_249)));
        assert!(should_parallelize_candidates(parallel, &membership(6_250)));
    }

    #[test]
    fn candidate_callbacks_can_be_shared_between_workers() {
        let route = |_candidate: &CandidateSplit,
                     _rng: &mut ModelRng,
                     _workspace: &mut TreeWorkspace|
         -> CoreResult<CandidatePartition> {
            Err(CoreError::invalid_data("compile-time callback probe"))
        };
        let score = |_partition: &CandidatePartition| -> CoreResult<crate::forest::CandidateScore> {
            Err(CoreError::invalid_data("compile-time score probe"))
        };
        assert_sync(&route);
        assert_sync(&score);
    }

    #[test]
    fn candidate_identity_is_stable_and_indexed() {
        let catalog = [DistanceChoice::Fixed(DistanceConfig::SquaredEuclidean)];
        let context = context(&catalog);
        let first = candidate_identity(context, 0);
        let fourth = candidate_identity(context, 3);
        assert_eq!(first.tree_seed, fourth.tree_seed);
        assert_eq!(first.node_path, fourth.node_path);
        assert_eq!(first.candidate_index, 0);
        assert_eq!(fourth.candidate_index, 3);
    }
}
