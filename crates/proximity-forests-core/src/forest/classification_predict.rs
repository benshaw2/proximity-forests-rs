//! Classification prediction from fitted leaf class counts.

use crate::{
    CoreError, CoreResult, DistanceScratch, MultivariateSeriesView, NumericVectorView, TiePolicy,
};

use super::{
    ClassificationLeaf, LeafPayload, ProximityTree, traverse_multivariate, traverse_vector,
};

/// Per-output classification prediction and probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationPrediction {
    classes: Vec<u32>,
    probabilities: Vec<Vec<(u32, f64)>>,
}

impl ClassificationPrediction {
    /// Returns predicted class IDs by output.
    #[must_use]
    pub fn classes(&self) -> &[u32] {
        &self.classes
    }

    /// Returns sorted `(class_id, probability)` pairs for one output.
    #[must_use]
    pub fn probabilities(&self, output: usize) -> Option<&[(u32, f64)]> {
        self.probabilities.get(output).map(Vec::as_slice)
    }

    #[cfg(test)]
    pub(crate) fn from_parts_for_test(
        classes: Vec<u32>,
        probabilities: Vec<Vec<(u32, f64)>>,
    ) -> Self {
        Self {
            classes,
            probabilities,
        }
    }
}

/// Predicts one vector observation.
pub fn predict_vector_classification(
    tree: &ProximityTree,
    query: NumericVectorView<'_>,
    observations: &[NumericVectorView<'_>],
    tie_policy: TiePolicy,
    prediction_seed: u64,
    scratch: &mut DistanceScratch,
) -> CoreResult<ClassificationPrediction> {
    let visit = traverse_vector(
        tree,
        query,
        observations,
        tie_policy,
        prediction_seed,
        scratch,
    )?;
    classification_prediction(visit.leaf().payload())
}

/// Predicts one multivariate observation.
pub fn predict_multivariate_classification(
    tree: &ProximityTree,
    query: MultivariateSeriesView<'_>,
    observations: &[MultivariateSeriesView<'_>],
    tie_policy: TiePolicy,
    prediction_seed: u64,
    scratch: &mut DistanceScratch,
) -> CoreResult<ClassificationPrediction> {
    let visit = traverse_multivariate(
        tree,
        query,
        observations,
        tie_policy,
        prediction_seed,
        scratch,
    )?;
    classification_prediction(visit.leaf().payload())
}

fn classification_prediction(payload: &LeafPayload) -> CoreResult<ClassificationPrediction> {
    let LeafPayload::Classification(leaf) = payload else {
        return Err(CoreError::invalid_data(
            "classification prediction reached a non-classification leaf",
        ));
    };
    from_counts(leaf)
}

fn from_counts(leaf: &ClassificationLeaf) -> CoreResult<ClassificationPrediction> {
    let mut classes = Vec::with_capacity(leaf.output_count());
    let mut probabilities = Vec::with_capacity(leaf.output_count());
    for output in 0..leaf.output_count() {
        let counts = leaf
            .counts(output)
            .ok_or_else(|| CoreError::invalid_data("classification leaf output was unavailable"))?;
        let total = counts.iter().try_fold(0usize, |total, &(_, count)| {
            total.checked_add(count).ok_or_else(|| {
                CoreError::invalid_data("classification leaf count overflowed usize")
            })
        })?;
        if total == 0 || counts.is_empty() {
            return Err(CoreError::invalid_data(
                "classification leaf contains no class occurrences",
            ));
        }
        let mut best_class = counts[0].0;
        let mut best_count = counts[0].1;
        let mut output_probabilities = Vec::with_capacity(counts.len());
        for &(class, count) in counts {
            if count > best_count {
                best_class = class;
                best_count = count;
            }
            #[allow(clippy::cast_precision_loss)]
            output_probabilities.push((class, count as f64 / total as f64));
        }
        classes.push(best_class);
        probabilities.push(output_probabilities);
    }
    Ok(ClassificationPrediction {
        classes,
        probabilities,
    })
}
