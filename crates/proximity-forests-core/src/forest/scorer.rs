//! Candidate scoring bridges compact partitions to purity primitives.

use crate::purity::{
    isolation_min_child, weighted_entropy, weighted_gini, weighted_mean_absolute_deviation,
    weighted_variance,
};
use crate::{
    ClassificationTargets, CoreError, CoreResult, IsolationCriterion, RegressionTargets,
    SupervisedCriterion,
};

use super::{
    CandidatePartition, CandidateScore, NodeMembership, ObservationIndex, ScorePreference,
};

/// Scores an isolation partition.
///
/// Lower scores are preferred because `MinChild` rewards a candidate that
/// isolates at least one small nonempty group.
///
/// # Errors
///
/// Returns an error for empty children.
pub fn score_isolation(
    partition: &CandidatePartition,
    criterion: IsolationCriterion,
) -> CoreResult<CandidateScore> {
    let value = match criterion {
        IsolationCriterion::MinChild => isolation_min_child(partition.occurrence_counts())?,
    };
    CandidateScore::new(value, ScorePreference::Minimize)
}

/// Scores a classification partition by mean weighted child impurity.
///
/// This follows the Java convention: lower supervised scores are preferred.
/// Each output is scored independently, child impurities are weighted by
/// bootstrap occurrence count, and the output scores are averaged.
///
/// # Errors
///
/// Returns an error for incompatible mappings, an invalid criterion, invalid
/// target indices, empty branches, or arithmetic overflow.
pub fn score_classification(
    partition: &CandidatePartition,
    membership: &NodeMembership,
    observations: &[ObservationIndex],
    targets: &ClassificationTargets,
    criterion: SupervisedCriterion,
) -> CoreResult<CandidateScore> {
    validate_mapping(partition, membership, observations, targets.sample_count())?;
    validate_nonempty(partition, "classification")?;

    let mut output_sum = 0.0;
    for output in 0..targets.output_count() {
        let labels = classification_values(observations, targets, output)?;
        let mut weighted_children = 0.0;
        for branch in 0..partition.branch_count() {
            let (child_labels, child_weights) =
                classification_child(partition, membership, &labels, branch);
            let impurity = classification_impurity(&child_labels, &child_weights, criterion)?;
            #[allow(clippy::cast_precision_loss)]
            {
                weighted_children += partition.occurrence_counts()[branch] as f64 * impurity;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        {
            output_sum += weighted_children / membership.occurrence_count() as f64;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let mean_child_impurity = output_sum / targets.output_count() as f64;
    CandidateScore::new(mean_child_impurity, ScorePreference::Minimize)
}

/// Scores a regression partition by mean weighted child impurity.
///
/// This follows the Java convention: lower supervised scores are preferred.
///
/// # Errors
///
/// Returns an error for incompatible mappings, an invalid criterion, invalid
/// target indices, empty branches, or arithmetic overflow.
pub fn score_regression(
    partition: &CandidatePartition,
    membership: &NodeMembership,
    observations: &[ObservationIndex],
    targets: &RegressionTargets,
    criterion: SupervisedCriterion,
) -> CoreResult<CandidateScore> {
    validate_mapping(partition, membership, observations, targets.sample_count())?;
    validate_nonempty(partition, "regression")?;

    let mut output_sum = 0.0;
    for output in 0..targets.output_count() {
        let values = regression_values(observations, targets, output)?;
        let mut weighted_children = 0.0;
        for branch in 0..partition.branch_count() {
            let (child_values, child_weights) =
                regression_child(partition, membership, &values, branch);
            let impurity = regression_impurity(&child_values, &child_weights, criterion)?;
            #[allow(clippy::cast_precision_loss)]
            {
                weighted_children += partition.occurrence_counts()[branch] as f64 * impurity;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        {
            output_sum += weighted_children / membership.occurrence_count() as f64;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let mean_child_impurity = output_sum / targets.output_count() as f64;
    CandidateScore::new(mean_child_impurity, ScorePreference::Minimize)
}

fn classification_values(
    observations: &[ObservationIndex],
    targets: &ClassificationTargets,
    output: usize,
) -> CoreResult<Vec<u32>> {
    observations
        .iter()
        .map(|observation| {
            targets.get(observation.get(), output).ok_or_else(|| {
                CoreError::invalid_data("classification target index was unavailable")
            })
        })
        .collect()
}

fn regression_values(
    observations: &[ObservationIndex],
    targets: &RegressionTargets,
    output: usize,
) -> CoreResult<Vec<f64>> {
    observations
        .iter()
        .map(|observation| {
            targets
                .get_f64(observation.get(), output)
                .ok_or_else(|| CoreError::invalid_data("regression target index was unavailable"))
        })
        .collect()
}

fn classification_child(
    partition: &CandidatePartition,
    membership: &NodeMembership,
    labels: &[u32],
    branch: usize,
) -> (Vec<u32>, Vec<usize>) {
    let capacity = partition.distinct_counts()[branch];
    let mut child_labels = Vec::with_capacity(capacity);
    let mut child_weights = Vec::with_capacity(capacity);
    for distinct in 0..membership.distinct_count() {
        if partition.branch_of(distinct) == Some(branch) {
            child_labels.push(labels[distinct]);
            child_weights.push(membership.multiplicities()[distinct]);
        }
    }
    (child_labels, child_weights)
}

fn regression_child(
    partition: &CandidatePartition,
    membership: &NodeMembership,
    values: &[f64],
    branch: usize,
) -> (Vec<f64>, Vec<usize>) {
    let capacity = partition.distinct_counts()[branch];
    let mut child_values = Vec::with_capacity(capacity);
    let mut child_weights = Vec::with_capacity(capacity);
    for distinct in 0..membership.distinct_count() {
        if partition.branch_of(distinct) == Some(branch) {
            child_values.push(values[distinct]);
            child_weights.push(membership.multiplicities()[distinct]);
        }
    }
    (child_values, child_weights)
}

fn classification_impurity(
    labels: &[u32],
    weights: &[usize],
    criterion: SupervisedCriterion,
) -> CoreResult<f64> {
    match criterion {
        SupervisedCriterion::Gini => weighted_gini(labels, weights),
        SupervisedCriterion::Entropy => weighted_entropy(labels, weights),
        _ => Err(CoreError::invalid_configuration(
            "criterion",
            "classification requires Gini or Entropy",
        )),
    }
}

fn regression_impurity(
    values: &[f64],
    weights: &[usize],
    criterion: SupervisedCriterion,
) -> CoreResult<f64> {
    match criterion {
        SupervisedCriterion::Variance => weighted_variance(values, weights),
        SupervisedCriterion::MeanAbsoluteDeviation => {
            weighted_mean_absolute_deviation(values, weights)
        }
        _ => Err(CoreError::invalid_configuration(
            "criterion",
            "regression requires Variance or MeanAbsoluteDeviation",
        )),
    }
}

fn validate_mapping(
    partition: &CandidatePartition,
    membership: &NodeMembership,
    observations: &[ObservationIndex],
    target_count: usize,
) -> CoreResult<()> {
    if observations.len() != membership.distinct_count()
        || partition.assignments().len() != membership.distinct_count()
    {
        return Err(CoreError::shape_mismatch(
            "partition, membership, and observation mappings must have equal distinct counts",
        ));
    }
    if observations
        .iter()
        .any(|observation| observation.get() >= target_count)
    {
        return Err(CoreError::invalid_data(
            "an observation index is outside target storage",
        ));
    }
    if membership.occurrence_count() == 0 {
        return Err(CoreError::invalid_data(
            "supervised scoring requires at least one occurrence",
        ));
    }
    Ok(())
}

fn validate_nonempty(partition: &CandidatePartition, task: &str) -> CoreResult<()> {
    if partition.has_no_empty_branches() {
        Ok(())
    } else {
        Err(CoreError::invalid_data(format!(
            "{task} scoring does not accept empty branches"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::score_isolation;
    use super::{CandidatePartition, ScorePreference};
    use crate::{IsolationCriterion, NodeMembership, SampleId};

    #[test]
    fn isolation_scores_are_minimized() {
        let membership =
            NodeMembership::from_occurrences(&[Some(SampleId::new(1)), Some(SampleId::new(2))])
                .expect("membership");
        let partition =
            CandidatePartition::from_assignments(&membership, 2, vec![0, 1]).expect("partition");
        let score = score_isolation(&partition, IsolationCriterion::MinChild).expect("score");
        assert_eq!(score.preference(), ScorePreference::Minimize);
        assert_eq!(score.value(), 1.0);
    }
}
