//! Class-aware exemplar sampling for supervised proximity splits.

use std::collections::BTreeMap;

use crate::{
    ClassificationTargets, CoreError, CoreResult, ModelRng, RandomPurpose, SeedMixer,
    sample_weighted_without_replacement,
};

use super::{NodeMembership, ObservationIndex};

/// Samples one multiplicity-weighted exemplar from each represented class.
///
/// The classification output is selected from its own deterministic random
/// stream. Class groups are traversed in ascending `ClassId` order, making
/// branch ordering independent of hash iteration order.
///
/// # Errors
///
/// Returns an error for mapping mismatches, invalid target indices, or fewer
/// than two represented classes.
pub fn sample_class_exemplars(
    membership: &NodeMembership,
    node_observations: &[ObservationIndex],
    targets: &ClassificationTargets,
    tree_seed: u64,
    node_path: u64,
    candidate_index: usize,
) -> CoreResult<(usize, Vec<ObservationIndex>)> {
    if node_observations.len() != membership.distinct_count() {
        return Err(CoreError::shape_mismatch(format!(
            "received {} node observations for {} distinct membership entries",
            node_observations.len(),
            membership.distinct_count()
        )));
    }

    let output_node_seed =
        SeedMixer::node_purpose(tree_seed, node_path, RandomPurpose::ClassificationOutput);
    let mut output_rng =
        ModelRng::from_seed(SeedMixer::candidate(output_node_seed, candidate_index));
    let output = output_rng.index(targets.output_count())?;

    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (distinct, observation) in node_observations.iter().enumerate() {
        let class = targets.get(observation.get(), output).ok_or_else(|| {
            CoreError::invalid_data("classification target index was unavailable")
        })?;
        groups.entry(class).or_default().push(distinct);
    }
    if groups.len() < 2 {
        return Err(CoreError::invalid_data(
            "classification candidate requires at least two represented classes",
        ));
    }

    let exemplar_node_seed =
        SeedMixer::node_purpose(tree_seed, node_path, RandomPurpose::CandidateExemplars);
    let candidate_seed = SeedMixer::candidate(exemplar_node_seed, candidate_index);
    let mut exemplars = Vec::with_capacity(groups.len());
    for (&class, distinct_indices) in &groups {
        let class_seed = SeedMixer::derive(candidate_seed, &[u64::from(class)]);
        let mut rng = ModelRng::from_seed(class_seed);
        let weights: Vec<usize> = distinct_indices
            .iter()
            .map(|&distinct| membership.multiplicities()[distinct])
            .collect();
        let selected = sample_weighted_without_replacement(&weights, 1, &mut rng)?[0];
        exemplars.push(node_observations[distinct_indices[selected]]);
    }
    Ok((output, exemplars))
}

/// Reports whether one classification output is pure at the node.
///
/// # Errors
///
/// Returns an error for invalid target indices.
pub fn classification_output_is_pure(
    node_observations: &[ObservationIndex],
    targets: &ClassificationTargets,
    output: usize,
) -> CoreResult<bool> {
    let Some(first_observation) = node_observations.first() else {
        return Ok(true);
    };
    let first = targets
        .get(first_observation.get(), output)
        .ok_or_else(|| CoreError::invalid_data("classification target index was unavailable"))?;
    for observation in &node_observations[1..] {
        let value = targets.get(observation.get(), output).ok_or_else(|| {
            CoreError::invalid_data("classification target index was unavailable")
        })?;
        if value != first {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Reports whether every classification output is pure at the node.
pub fn classification_node_is_pure(
    node_observations: &[ObservationIndex],
    targets: &ClassificationTargets,
) -> CoreResult<bool> {
    for output in 0..targets.output_count() {
        if !classification_output_is_pure(node_observations, targets, output)? {
            return Ok(false);
        }
    }
    Ok(true)
}
