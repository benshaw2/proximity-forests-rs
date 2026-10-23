//! Regression impurity primitives.

use crate::{CoreError, CoreResult};

/// Computes weighted population variance for one regression output.
pub fn weighted_variance(values: &[f64], weights: &[usize]) -> CoreResult<f64> {
    let (mean, total_weight) = weighted_mean(values, weights)?;
    if total_weight == 0 {
        return Ok(0.0);
    }

    let squared_deviation = values
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| {
            let difference = value - mean;
            weight as f64 * difference * difference
        })
        .sum::<f64>();

    Ok(squared_deviation / total_weight as f64)
}

/// Computes weighted mean absolute deviation from the weighted mean.
pub fn weighted_mean_absolute_deviation(values: &[f64], weights: &[usize]) -> CoreResult<f64> {
    let (mean, total_weight) = weighted_mean(values, weights)?;
    if total_weight == 0 {
        return Ok(0.0);
    }

    let absolute_deviation = values
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| weight as f64 * (value - mean).abs())
        .sum::<f64>();

    Ok(absolute_deviation / total_weight as f64)
}

fn weighted_mean(values: &[f64], weights: &[usize]) -> CoreResult<(f64, usize)> {
    if values.len() != weights.len() {
        return Err(CoreError::shape_mismatch(format!(
            "received {} regression values but {} multiplicities",
            values.len(),
            weights.len()
        )));
    }

    let mut weighted_sum = 0.0;
    let mut total_weight = 0usize;
    for (&value, &weight) in values.iter().zip(weights) {
        if !value.is_finite() {
            return Err(CoreError::invalid_data("regression targets must be finite"));
        }
        total_weight = total_weight.checked_add(weight).ok_or_else(|| {
            CoreError::invalid_data("regression multiplicity sum overflowed usize")
        })?;
        weighted_sum += weight as f64 * value;
    }

    if total_weight == 0 {
        Ok((0.0, 0))
    } else {
        Ok((weighted_sum / total_weight as f64, total_weight))
    }
}

#[cfg(test)]
mod tests {
    use super::{weighted_mean_absolute_deviation, weighted_variance};

    #[test]
    fn weighted_scores_match_explicit_duplicates() {
        let compact_values = [1.0, 5.0];
        let compact_weights = [3, 1];
        let expanded_values = [1.0, 1.0, 1.0, 5.0];
        let expanded_weights = [1, 1, 1, 1];

        assert_eq!(
            weighted_variance(&compact_values, &compact_weights),
            weighted_variance(&expanded_values, &expanded_weights)
        );
        assert_eq!(
            weighted_mean_absolute_deviation(&compact_values, &compact_weights),
            weighted_mean_absolute_deviation(&expanded_values, &expanded_weights)
        );
    }

    #[test]
    fn rejects_non_finite_targets() {
        assert!(weighted_variance(&[1.0, f64::NAN], &[1, 1]).is_err());
    }
}
