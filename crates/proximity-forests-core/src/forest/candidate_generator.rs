//! Deterministic split-candidate generation.

use crate::{
    CoreError, CoreResult, DimensionSelection, DistanceConfig, ModelRng, RandomPurpose, SeedMixer,
    sample_dimensions, sample_weighted_without_replacement,
};

use super::{CandidateSplit, NodeMembership, ObservationIndex};

/// Inputs controlling one node's candidate-generation stream.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CandidateIdentity {
    /// Deterministic seed assigned to the current tree.
    pub tree_seed: u64,
    /// Stable path identity of the current node.
    pub node_path: u64,
    /// Zero-based candidate index within the node.
    pub candidate_index: usize,
}

/// Generates one fully specified candidate from a supplied distance catalog.
///
/// Distance, dimension, and exemplar choices use separate purpose-specific
/// streams. Candidate generation is consequently independent of worker
/// scheduling and of how many random draws another choice requires.
///
/// Exemplar sampling is weighted by bootstrap multiplicity and performed over
/// distinct observations without replacement.
///
/// # Errors
///
/// Returns an error for an empty distance catalog, incompatible node mappings,
/// an invalid dimension selection, or an excessive exemplar count.
pub fn generate_candidate(
    distance_catalog: &[DistanceConfig],
    available_dimensions: usize,
    dimension_selection: DimensionSelection,
    exemplar_count: usize,
    membership: &NodeMembership,
    node_observations: &[ObservationIndex],
    identity: CandidateIdentity,
) -> CoreResult<CandidateSplit> {
    if distance_catalog.is_empty() {
        return Err(CoreError::invalid_configuration(
            "distance_catalog",
            "candidate generation requires at least one distance configuration",
        ));
    }
    if exemplar_count == 0 {
        return Err(CoreError::invalid_configuration(
            "exemplar_count",
            "candidate generation requires at least one exemplar",
        ));
    }
    if node_observations.len() != membership.distinct_count() {
        return Err(CoreError::shape_mismatch(format!(
            "received {} node observations for {} distinct membership entries",
            node_observations.len(),
            membership.distinct_count()
        )));
    }

    let distance_seed = candidate_seed(identity, RandomPurpose::CandidateDistance);
    let dimension_seed = candidate_seed(identity, RandomPurpose::Dimensions);
    let exemplar_seed = candidate_seed(identity, RandomPurpose::CandidateExemplars);

    let mut distance_rng = ModelRng::from_seed(distance_seed);
    let distance = distance_catalog[distance_rng.index(distance_catalog.len())?].clone();

    let mut dimension_rng = ModelRng::from_seed(dimension_seed);
    let selected_dimensions = sample_dimensions(
        available_dimensions,
        dimension_selection,
        &mut dimension_rng,
    )?;

    let mut exemplar_rng = ModelRng::from_seed(exemplar_seed);
    let selected_distinct = sample_weighted_without_replacement(
        membership.multiplicities(),
        exemplar_count,
        &mut exemplar_rng,
    )?;
    let exemplars = selected_distinct
        .into_iter()
        .map(|distinct| node_observations[distinct])
        .collect();

    CandidateSplit::new(distance, selected_dimensions, exemplars)
}

#[inline]
fn candidate_seed(identity: CandidateIdentity, purpose: RandomPurpose) -> u64 {
    let node_seed = SeedMixer::node_purpose(identity.tree_seed, identity.node_path, purpose);
    SeedMixer::candidate(node_seed, identity.candidate_index)
}

#[cfg(test)]
mod tests {
    use super::ObservationIndex;
    use super::{CandidateIdentity, generate_candidate};
    use crate::{DimensionSelection, DistanceConfig, NodeMembership, SampleId};

    #[test]
    fn generation_is_repeatable_and_uses_distinct_exemplars() {
        let membership = NodeMembership::from_occurrences(&[
            Some(SampleId::new(1)),
            Some(SampleId::new(1)),
            Some(SampleId::new(2)),
            Some(SampleId::new(3)),
        ])
        .expect("membership");
        let observations = [
            ObservationIndex::new(4),
            ObservationIndex::new(7),
            ObservationIndex::new(9),
        ];
        let identity = CandidateIdentity {
            tree_seed: 13,
            node_path: 2,
            candidate_index: 1,
        };
        let first = generate_candidate(
            &[DistanceConfig::SquaredEuclidean, DistanceConfig::Manhattan],
            5,
            DimensionSelection::FixedCount(2),
            2,
            &membership,
            &observations,
            identity,
        )
        .expect("candidate");
        let second = generate_candidate(
            &[DistanceConfig::SquaredEuclidean, DistanceConfig::Manhattan],
            5,
            DimensionSelection::FixedCount(2),
            2,
            &membership,
            &observations,
            identity,
        )
        .expect("candidate");
        assert_eq!(first, second);
        assert_ne!(first.exemplars()[0], first.exemplars()[1]);
    }
}
