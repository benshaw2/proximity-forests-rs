//! Classification impurity primitives.

use std::collections::HashMap;
use std::hash::Hash;

use crate::{CoreError, CoreResult};

/// Computes weighted Gini impurity for one classification output.
pub fn weighted_gini<L>(labels: &[L], weights: &[usize]) -> CoreResult<f64>
where
    L: Eq + Hash,
{
    let (counts, total) = weighted_counts(labels, weights)?;
    if total == 0 {
        return Ok(0.0);
    }

    let total = total as f64;
    let squared_sum = counts
        .values()
        .map(|&count| {
            let probability = count as f64 / total;
            probability * probability
        })
        .sum::<f64>();

    Ok(1.0 - squared_sum)
}

/// Computes weighted base-2 Shannon entropy for one classification output.
pub fn weighted_entropy<L>(labels: &[L], weights: &[usize]) -> CoreResult<f64>
where
    L: Eq + Hash,
{
    let (counts, total) = weighted_counts(labels, weights)?;
    if total == 0 {
        return Ok(0.0);
    }

    let total = total as f64;
    let entropy = counts
        .values()
        .map(|&count| {
            let probability = count as f64 / total;
            -probability * probability.log2()
        })
        .sum();

    Ok(entropy)
}

fn weighted_counts<'a, L>(
    labels: &'a [L],
    weights: &[usize],
) -> CoreResult<(HashMap<&'a L, usize>, usize)>
where
    L: Eq + Hash,
{
    validate_lengths(labels.len(), weights.len())?;

    let mut counts = HashMap::new();
    let mut total = 0usize;
    for (label, &weight) in labels.iter().zip(weights) {
        if weight == 0 {
            continue;
        }
        total = total.checked_add(weight).ok_or_else(|| {
            CoreError::invalid_data("classification multiplicity sum overflowed usize")
        })?;
        let count = counts.entry(label).or_insert(0usize);
        *count = count.checked_add(weight).ok_or_else(|| {
            CoreError::invalid_data("classification class count overflowed usize")
        })?;
    }
    Ok((counts, total))
}

fn validate_lengths(values: usize, weights: usize) -> CoreResult<()> {
    if values == weights {
        Ok(())
    } else {
        Err(CoreError::shape_mismatch(format!(
            "received {values} labels but {weights} multiplicities"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::{weighted_entropy, weighted_gini};

    #[test]
    fn weighted_scores_match_explicit_duplicates() {
        let compact_labels = ["a", "b"];
        let compact_weights = [3, 1];
        let expanded_labels = ["a", "a", "a", "b"];
        let expanded_weights = [1, 1, 1, 1];

        assert_eq!(
            weighted_gini(&compact_labels, &compact_weights),
            weighted_gini(&expanded_labels, &expanded_weights)
        );
        assert_eq!(
            weighted_entropy(&compact_labels, &compact_weights),
            weighted_entropy(&expanded_labels, &expanded_weights)
        );
    }

    #[test]
    fn zero_total_weight_has_zero_impurity() {
        assert_eq!(weighted_gini(&[0, 1], &[0, 0]), Ok(0.0));
        assert_eq!(weighted_entropy(&[0, 1], &[0, 0]), Ok(0.0));
    }
}
