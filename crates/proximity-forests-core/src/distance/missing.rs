//! Optimized missing-value-aware numeric vector distances.

use crate::{CoreError, CoreResult, DistanceCutoff, NumericVectorView};

/// Behavior when two observations have no jointly observed positions.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NoOverlapPolicy {
    /// Return positive infinity so the pair cannot win a finite-distance route.
    Infinite,
    /// Return zero, treating no comparable values as no evidence of difference.
    Zero,
}

/// Computes NaN-aware squared Euclidean distance over all vector positions.
///
/// Positions for which either value is NaN are ignored. The observed squared
/// sum is scaled by `total_positions / observed_positions`. The implementation
/// may abandon safely when the unscaled partial sum already exceeds the final
/// cutoff because the eventual normalization factor is at least one.
///
/// # Errors
///
/// Returns an error for unequal lengths, mismatched dtypes, or infinite
/// nonmissing values.
pub fn nan_squared_euclidean(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    cutoff: DistanceCutoff,
    no_overlap: NoOverlapPolicy,
) -> CoreResult<f64> {
    validate_vectors(first, second)?;
    let compared = first.len();
    let accumulation = match (first, second) {
        (NumericVectorView::F32(first), NumericVectorView::F32(second)) => {
            accumulate_f32(first, second, cutoff)?
        }
        (NumericVectorView::F64(first), NumericVectorView::F64(second)) => {
            accumulate_f64(first, second, cutoff)?
        }
        _ => {
            return Err(CoreError::shape_mismatch(
                "NaN-aware squared Euclidean distance requires matching storage dtypes",
            ));
        }
    };
    finish(accumulation, compared, cutoff, no_overlap)
}

/// Computes NaN-aware squared Euclidean distance over selected positions.
///
/// The normalization numerator is the number of selected positions. Selected
/// positions are borrowed without creating a compact vector.
///
/// # Errors
///
/// Returns an error for unequal lengths, mismatched dtypes, an empty or
/// out-of-range selection, or infinite nonmissing values.
pub fn nan_squared_euclidean_selected(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    selected: &[usize],
    cutoff: DistanceCutoff,
    no_overlap: NoOverlapPolicy,
) -> CoreResult<f64> {
    validate_vectors(first, second)?;
    validate_selected(first.len(), selected)?;
    let accumulation = match (first, second) {
        (NumericVectorView::F32(first), NumericVectorView::F32(second)) => {
            accumulate_selected_f32(first, second, selected, cutoff)?
        }
        (NumericVectorView::F64(first), NumericVectorView::F64(second)) => {
            accumulate_selected_f64(first, second, selected, cutoff)?
        }
        _ => {
            return Err(CoreError::shape_mismatch(
                "NaN-aware squared Euclidean distance requires matching storage dtypes",
            ));
        }
    };
    finish(accumulation, selected.len(), cutoff, no_overlap)
}

#[derive(Clone, Copy)]
pub(crate) struct NaNAccumulation {
    pub(crate) sum: f64,
    pub(crate) observed: usize,
    pub(crate) abandoned: bool,
}

#[inline]
pub(crate) fn accumulate_f32(
    first: &[f32],
    second: &[f32],
    cutoff: DistanceCutoff,
) -> CoreResult<NaNAccumulation> {
    let limit = cutoff_limit(cutoff);
    let mut sum = 0.0;
    let mut observed = 0usize;
    for index in 0..first.len() {
        let left = first[index];
        let right = second[index];
        if left.is_nan() || right.is_nan() {
            continue;
        }
        if !left.is_finite() || !right.is_finite() {
            return Err(nonfinite_error());
        }
        let difference = f64::from(left) - f64::from(right);
        sum = difference.mul_add(difference, sum);
        observed += 1;
        if sum > limit {
            return Ok(NaNAccumulation {
                sum,
                observed,
                abandoned: true,
            });
        }
    }
    Ok(NaNAccumulation {
        sum,
        observed,
        abandoned: false,
    })
}

#[inline]
pub(crate) fn accumulate_f64(
    first: &[f64],
    second: &[f64],
    cutoff: DistanceCutoff,
) -> CoreResult<NaNAccumulation> {
    let limit = cutoff_limit(cutoff);
    let mut sum = 0.0;
    let mut observed = 0usize;
    for index in 0..first.len() {
        let left = first[index];
        let right = second[index];
        if left.is_nan() || right.is_nan() {
            continue;
        }
        if !left.is_finite() || !right.is_finite() {
            return Err(nonfinite_error());
        }
        let difference = left - right;
        sum = difference.mul_add(difference, sum);
        observed += 1;
        if sum > limit {
            return Ok(NaNAccumulation {
                sum,
                observed,
                abandoned: true,
            });
        }
    }
    Ok(NaNAccumulation {
        sum,
        observed,
        abandoned: false,
    })
}

fn accumulate_selected_f32(
    first: &[f32],
    second: &[f32],
    selected: &[usize],
    cutoff: DistanceCutoff,
) -> CoreResult<NaNAccumulation> {
    let limit = cutoff_limit(cutoff);
    let mut sum = 0.0;
    let mut observed = 0usize;
    for &index in selected {
        let left = first[index];
        let right = second[index];
        if left.is_nan() || right.is_nan() {
            continue;
        }
        if !left.is_finite() || !right.is_finite() {
            return Err(nonfinite_error());
        }
        let difference = f64::from(left) - f64::from(right);
        sum = difference.mul_add(difference, sum);
        observed += 1;
        if sum > limit {
            return Ok(NaNAccumulation {
                sum,
                observed,
                abandoned: true,
            });
        }
    }
    Ok(NaNAccumulation {
        sum,
        observed,
        abandoned: false,
    })
}

fn accumulate_selected_f64(
    first: &[f64],
    second: &[f64],
    selected: &[usize],
    cutoff: DistanceCutoff,
) -> CoreResult<NaNAccumulation> {
    let limit = cutoff_limit(cutoff);
    let mut sum = 0.0;
    let mut observed = 0usize;
    for &index in selected {
        let left = first[index];
        let right = second[index];
        if left.is_nan() || right.is_nan() {
            continue;
        }
        if !left.is_finite() || !right.is_finite() {
            return Err(nonfinite_error());
        }
        let difference = left - right;
        sum = difference.mul_add(difference, sum);
        observed += 1;
        if sum > limit {
            return Ok(NaNAccumulation {
                sum,
                observed,
                abandoned: true,
            });
        }
    }
    Ok(NaNAccumulation {
        sum,
        observed,
        abandoned: false,
    })
}

#[inline]
pub(crate) fn finish(
    accumulation: NaNAccumulation,
    compared: usize,
    cutoff: DistanceCutoff,
    no_overlap: NoOverlapPolicy,
) -> CoreResult<f64> {
    if accumulation.abandoned {
        return Ok(f64::INFINITY);
    }
    if accumulation.observed == 0 {
        return Ok(no_overlap_value(no_overlap));
    }
    #[allow(clippy::cast_precision_loss)]
    let scaled = accumulation.sum * compared as f64 / accumulation.observed as f64;
    if cutoff.exceeded_by(scaled) {
        Ok(f64::INFINITY)
    } else {
        Ok(scaled)
    }
}

#[inline]
const fn cutoff_limit(cutoff: DistanceCutoff) -> f64 {
    match cutoff {
        DistanceCutoff::None => f64::INFINITY,
        DistanceCutoff::Finite(limit) => limit,
    }
}

fn validate_vectors(first: NumericVectorView<'_>, second: NumericVectorView<'_>) -> CoreResult<()> {
    if first.len() != second.len() {
        return Err(CoreError::shape_mismatch(format!(
            "NaN-aware squared Euclidean distance requires equal lengths; received {} and {}",
            first.len(),
            second.len()
        )));
    }
    Ok(())
}

fn validate_selected(length: usize, selected: &[usize]) -> CoreResult<()> {
    if selected.is_empty() {
        return Err(CoreError::invalid_data(
            "NaN-aware Euclidean distance requires at least one selected position",
        ));
    }
    if let Some(&invalid) = selected.iter().find(|&&index| index >= length) {
        return Err(CoreError::invalid_data(format!(
            "selected position {invalid} is outside vector length {length}"
        )));
    }
    Ok(())
}

#[inline]
fn nonfinite_error() -> CoreError {
    CoreError::invalid_data("NaN-aware Euclidean inputs may contain NaN but not infinity")
}

pub(crate) const fn no_overlap_value(policy: NoOverlapPolicy) -> f64 {
    match policy {
        NoOverlapPolicy::Infinite => f64::INFINITY,
        NoOverlapPolicy::Zero => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::{NoOverlapPolicy, nan_squared_euclidean, nan_squared_euclidean_selected};
    use crate::{DistanceCutoff, NumericVectorView};

    #[test]
    fn complete_data_matches_squared_euclidean() {
        assert_eq!(
            nan_squared_euclidean(
                NumericVectorView::from_f64(&[1.0, 2.0, 3.0]),
                NumericVectorView::from_f64(&[2.0, 4.0, 6.0]),
                DistanceCutoff::None,
                NoOverlapPolicy::Infinite,
            ),
            Ok(14.0)
        );
    }

    #[test]
    fn scales_by_observed_overlap() {
        assert_eq!(
            nan_squared_euclidean(
                NumericVectorView::from_f32(&[1.0, f32::NAN, 3.0, 4.0]),
                NumericVectorView::from_f32(&[2.0, 8.0, 5.0, f32::NAN]),
                DistanceCutoff::None,
                NoOverlapPolicy::Infinite,
            ),
            Ok(10.0)
        );
    }

    #[test]
    fn partial_sum_abandonment_is_safe() {
        assert_eq!(
            nan_squared_euclidean(
                NumericVectorView::from_f64(&[10.0, f64::NAN, 0.0]),
                NumericVectorView::from_f64(&[0.0, 2.0, 0.0]),
                DistanceCutoff::finite(50.0).expect("valid cutoff"),
                NoOverlapPolicy::Infinite,
            ),
            Ok(f64::INFINITY)
        );
    }

    #[test]
    fn selected_normalization_uses_selected_count() {
        assert_eq!(
            nan_squared_euclidean_selected(
                NumericVectorView::from_f64(&[1.0, f64::NAN, 3.0, 100.0]),
                NumericVectorView::from_f64(&[2.0, 8.0, 5.0, -100.0]),
                &[0, 1, 2],
                DistanceCutoff::None,
                NoOverlapPolicy::Infinite,
            ),
            Ok(7.5)
        );
    }

    #[test]
    fn no_overlap_policy_is_explicit() {
        let first = NumericVectorView::from_f64(&[f64::NAN, 1.0]);
        let second = NumericVectorView::from_f64(&[2.0, f64::NAN]);
        assert_eq!(
            nan_squared_euclidean(
                first,
                second,
                DistanceCutoff::None,
                NoOverlapPolicy::Infinite,
            ),
            Ok(f64::INFINITY)
        );
        assert_eq!(
            nan_squared_euclidean(first, second, DistanceCutoff::None, NoOverlapPolicy::Zero,),
            Ok(0.0)
        );
    }
}
