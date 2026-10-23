//! Whole-node candidate routing over distinct observations.

use crate::{
    CoreError, CoreResult, ModelRng, MultivariateSeriesView, NumericVectorView, TiePolicy,
};

use super::{
    CandidatePartition, CandidateSplit, NodeMembership, ObservationIndex, TreeWorkspace,
    route_multivariate, route_vector,
};

/// Routes every distinct vector observation at a node exactly once.
///
/// # Errors
///
/// Returns an error for mapping mismatches, unavailable observations, invalid
/// exemplar references, distance failures, or empty candidate branches.
pub fn partition_vectors(
    observations: &[NumericVectorView<'_>],
    node_observations: &[ObservationIndex],
    membership: &NodeMembership,
    candidate: &CandidateSplit,
    tie_policy: TiePolicy,
    mut rng: Option<&mut ModelRng>,
    workspace: &mut TreeWorkspace,
) -> CoreResult<CandidatePartition> {
    validate_mapping(node_observations, membership, observations.len())?;
    workspace.assignments(membership.distinct_count());

    for (distinct, &observation) in node_observations.iter().enumerate() {
        let query = observations
            .get(observation.get())
            .copied()
            .ok_or_else(|| {
                CoreError::invalid_data(format!(
                    "node observation {} is outside vector observation count {}",
                    observation.get(),
                    observations.len()
                ))
            })?;
        let result = route_vector(
            query,
            observations,
            candidate,
            tie_policy,
            rng.as_deref_mut(),
            workspace.distance_scratch(),
        )?;
        workspace.set_assignment(distinct, result.branch())?;
    }

    finish_partition(membership, candidate, workspace.current_assignments())
}

/// Routes every distinct multivariate observation at a node exactly once.
///
/// # Errors
///
/// Returns an error for mapping mismatches, unavailable observations, invalid
/// exemplar references, distance failures, or empty candidate branches.
pub fn partition_multivariate(
    observations: &[MultivariateSeriesView<'_>],
    node_observations: &[ObservationIndex],
    membership: &NodeMembership,
    candidate: &CandidateSplit,
    tie_policy: TiePolicy,
    mut rng: Option<&mut ModelRng>,
    workspace: &mut TreeWorkspace,
) -> CoreResult<CandidatePartition> {
    validate_mapping(node_observations, membership, observations.len())?;
    workspace.assignments(membership.distinct_count());

    for (distinct, &observation) in node_observations.iter().enumerate() {
        let query = observations
            .get(observation.get())
            .copied()
            .ok_or_else(|| {
                CoreError::invalid_data(format!(
                    "node observation {} is outside multivariate observation count {}",
                    observation.get(),
                    observations.len()
                ))
            })?;
        let result = route_multivariate(
            query,
            observations,
            candidate,
            tie_policy,
            rng.as_deref_mut(),
            workspace.distance_scratch(),
        )?;
        workspace.set_assignment(distinct, result.branch())?;
    }

    finish_partition(membership, candidate, workspace.current_assignments())
}

fn finish_partition(
    membership: &NodeMembership,
    candidate: &CandidateSplit,
    assignments: &[usize],
) -> CoreResult<CandidatePartition> {
    let partition = CandidatePartition::from_assignments(
        membership,
        candidate.branch_count(),
        assignments.to_vec(),
    )?;
    if partition.has_no_empty_branches() {
        Ok(partition)
    } else {
        Err(CoreError::invalid_data(
            "candidate routing produced an empty exemplar branch",
        ))
    }
}

fn validate_mapping(
    node_observations: &[ObservationIndex],
    membership: &NodeMembership,
    observation_count: usize,
) -> CoreResult<()> {
    if node_observations.len() != membership.distinct_count() {
        return Err(CoreError::shape_mismatch(format!(
            "received {} node observations for {} distinct membership entries",
            node_observations.len(),
            membership.distinct_count()
        )));
    }
    if node_observations
        .iter()
        .any(|observation| observation.get() >= observation_count)
    {
        return Err(CoreError::invalid_data(
            "a node observation is outside dataset storage",
        ));
    }
    Ok(())
}
