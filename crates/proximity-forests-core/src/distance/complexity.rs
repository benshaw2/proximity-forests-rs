//! Optimized complexity-invariant distance for univariate and independent series.

use crate::{CoreError, CoreResult, DistanceCutoff, MultivariateSeriesView, NumericVectorView};

/// Computes squared complexity-invariant distance.
///
/// Complexity estimates for both vectors are computed in one traversal. The
/// correction is then used to convert the final CID cutoff into a squared
/// Euclidean cutoff before the pointwise distance pass.
///
/// # Errors
///
/// Returns an error for unequal lengths, mismatched storage dtypes, or
/// non-finite values.
pub fn cid(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    validate_vectors(first, second)?;

    match (first, second) {
        (NumericVectorView::F32(first), NumericVectorView::F32(second)) => {
            cid_f32(first, second, cutoff)
        }
        (NumericVectorView::F64(first), NumericVectorView::F64(second)) => {
            cid_f64(first, second, cutoff)
        }
        _ => Err(CoreError::shape_mismatch(
            "CID requires matching storage dtypes",
        )),
    }
}

/// Computes independent multivariate CID by summing selected-channel costs.
///
/// Each selected channel receives only the remaining global cutoff. Contiguous
/// channels dispatch directly to typed kernels; strided channels use a
/// zero-allocation fallback.
///
/// # Errors
///
/// Returns an error for incompatible shapes or dtypes, invalid selected
/// dimensions, or non-finite values.
pub fn cid_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    validate_multivariate(first, second, selected)?;
    let mut total = 0.0;

    match selected {
        Some(selected) => {
            for &dimension in selected {
                let component =
                    cid_channel(first, second, dimension, cutoff.remaining_after(total))?;
                if component.is_infinite() {
                    return Ok(f64::INFINITY);
                }
                total += component;
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                let component =
                    cid_channel(first, second, dimension, cutoff.remaining_after(total))?;
                if component.is_infinite() {
                    return Ok(f64::INFINITY);
                }
                total += component;
            }
        }
    }

    if cutoff.exceeded_by(total) {
        Ok(f64::INFINITY)
    } else {
        Ok(total)
    }
}

#[inline]
fn cid_f32(first: &[f32], second: &[f32], cutoff: DistanceCutoff) -> CoreResult<f64> {
    let (first_complexity, second_complexity) = paired_complexity_f32(first, second)?;
    let correction = squared_correction(first_complexity, second_complexity);
    if correction.is_infinite() {
        return Ok(f64::INFINITY);
    }
    let euclidean_limit = converted_cutoff(cutoff, correction);
    let squared = match cutoff {
        DistanceCutoff::None => squared_complete_f32(first, second)?,
        DistanceCutoff::Finite(_) => squared_cutoff_f32(first, second, euclidean_limit)?,
    };
    finish_cid(squared, correction, cutoff)
}

#[inline]
fn cid_f64(first: &[f64], second: &[f64], cutoff: DistanceCutoff) -> CoreResult<f64> {
    let (first_complexity, second_complexity) = paired_complexity_f64(first, second)?;
    let correction = squared_correction(first_complexity, second_complexity);
    if correction.is_infinite() {
        return Ok(f64::INFINITY);
    }
    let euclidean_limit = converted_cutoff(cutoff, correction);
    let squared = match cutoff {
        DistanceCutoff::None => squared_complete_f64(first, second)?,
        DistanceCutoff::Finite(_) => squared_cutoff_f64(first, second, euclidean_limit)?,
    };
    finish_cid(squared, correction, cutoff)
}

fn cid_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    match (
        first.contiguous_dimension(dimension),
        second.contiguous_dimension(dimension),
    ) {
        (Some(NumericVectorView::F32(first)), Some(NumericVectorView::F32(second))) => {
            cid_f32(first, second, cutoff)
        }
        (Some(NumericVectorView::F64(first)), Some(NumericVectorView::F64(second))) => {
            cid_f64(first, second, cutoff)
        }
        (Some(_), Some(_)) => Err(CoreError::shape_mismatch(
            "independent CID requires matching storage dtypes",
        )),
        _ => cid_strided(first, second, dimension, cutoff),
    }
}

fn cid_strided(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    let (first_complexity, second_complexity) =
        paired_complexity_strided(first, second, dimension)?;
    let correction = squared_correction(first_complexity, second_complexity);
    if correction.is_infinite() {
        return Ok(f64::INFINITY);
    }
    let euclidean_limit = converted_cutoff(cutoff, correction);
    let mut squared = 0.0;
    for time in 0..first.time_count() {
        let left = series_value(first, dimension, time)?;
        let right = series_value(second, dimension, time)?;
        ensure_finite(left)?;
        ensure_finite(right)?;
        let difference = left - right;
        squared = difference.mul_add(difference, squared);
        if squared > euclidean_limit {
            return Ok(f64::INFINITY);
        }
    }
    finish_cid(squared, correction, cutoff)
}

#[inline]
fn paired_complexity_f32(first: &[f32], second: &[f32]) -> CoreResult<(f64, f64)> {
    ensure_all_finite_f32(first)?;
    ensure_all_finite_f32(second)?;
    if first.len() < 2 {
        return Ok((0.0, 0.0));
    }

    let mut first_total = 0.0;
    let mut second_total = 0.0;
    let mut prior_first = f64::from(first[0]);
    let mut prior_second = f64::from(second[0]);
    for index in 1..first.len() {
        let current_first = f64::from(first[index]);
        let current_second = f64::from(second[index]);
        let first_difference = current_first - prior_first;
        let second_difference = current_second - prior_second;
        first_total = first_difference.mul_add(first_difference, first_total);
        second_total = second_difference.mul_add(second_difference, second_total);
        prior_first = current_first;
        prior_second = current_second;
    }
    Ok((first_total, second_total))
}

#[inline]
fn paired_complexity_f64(first: &[f64], second: &[f64]) -> CoreResult<(f64, f64)> {
    ensure_all_finite_f64(first)?;
    ensure_all_finite_f64(second)?;
    if first.len() < 2 {
        return Ok((0.0, 0.0));
    }

    let mut first_total = 0.0;
    let mut second_total = 0.0;
    let mut prior_first = first[0];
    let mut prior_second = second[0];
    for index in 1..first.len() {
        let current_first = first[index];
        let current_second = second[index];
        let first_difference = current_first - prior_first;
        let second_difference = current_second - prior_second;
        first_total = first_difference.mul_add(first_difference, first_total);
        second_total = second_difference.mul_add(second_difference, second_total);
        prior_first = current_first;
        prior_second = current_second;
    }
    Ok((first_total, second_total))
}

fn paired_complexity_strided(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
) -> CoreResult<(f64, f64)> {
    if first.time_count() < 2 {
        let first_value = series_value(first, dimension, 0)?;
        let second_value = series_value(second, dimension, 0)?;
        ensure_finite(first_value)?;
        ensure_finite(second_value)?;
        return Ok((0.0, 0.0));
    }

    let mut prior_first = series_value(first, dimension, 0)?;
    let mut prior_second = series_value(second, dimension, 0)?;
    ensure_finite(prior_first)?;
    ensure_finite(prior_second)?;
    let mut first_total = 0.0;
    let mut second_total = 0.0;

    for time in 1..first.time_count() {
        let current_first = series_value(first, dimension, time)?;
        let current_second = series_value(second, dimension, time)?;
        ensure_finite(current_first)?;
        ensure_finite(current_second)?;
        let first_difference = current_first - prior_first;
        let second_difference = current_second - prior_second;
        first_total = first_difference.mul_add(first_difference, first_total);
        second_total = second_difference.mul_add(second_difference, second_total);
        prior_first = current_first;
        prior_second = current_second;
    }
    Ok((first_total, second_total))
}

#[inline]
fn squared_complete_f32(first: &[f32], second: &[f32]) -> CoreResult<f64> {
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut index = 0usize;
    let block_end = first.len() - first.len() % 4;

    while index < block_end {
        let difference0 = f64::from(first[index]) - f64::from(second[index]);
        let difference1 = f64::from(first[index + 1]) - f64::from(second[index + 1]);
        let difference2 = f64::from(first[index + 2]) - f64::from(second[index + 2]);
        let difference3 = f64::from(first[index + 3]) - f64::from(second[index + 3]);
        sum0 = difference0.mul_add(difference0, sum0);
        sum1 = difference1.mul_add(difference1, sum1);
        sum2 = difference2.mul_add(difference2, sum2);
        sum3 = difference3.mul_add(difference3, sum3);
        index += 4;
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);
    while index < first.len() {
        let difference = f64::from(first[index]) - f64::from(second[index]);
        total = difference.mul_add(difference, total);
        index += 1;
    }
    Ok(total)
}

#[inline]
fn squared_complete_f64(first: &[f64], second: &[f64]) -> CoreResult<f64> {
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut index = 0usize;
    let block_end = first.len() - first.len() % 4;

    while index < block_end {
        let difference0 = first[index] - second[index];
        let difference1 = first[index + 1] - second[index + 1];
        let difference2 = first[index + 2] - second[index + 2];
        let difference3 = first[index + 3] - second[index + 3];
        sum0 = difference0.mul_add(difference0, sum0);
        sum1 = difference1.mul_add(difference1, sum1);
        sum2 = difference2.mul_add(difference2, sum2);
        sum3 = difference3.mul_add(difference3, sum3);
        index += 4;
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);
    while index < first.len() {
        let difference = first[index] - second[index];
        total = difference.mul_add(difference, total);
        index += 1;
    }
    Ok(total)
}

#[inline]
fn squared_cutoff_f32(first: &[f32], second: &[f32], limit: f64) -> CoreResult<f64> {
    let mut total = 0.0;
    for index in 0..first.len() {
        let difference = f64::from(first[index]) - f64::from(second[index]);
        total = difference.mul_add(difference, total);
        if total > limit {
            return Ok(f64::INFINITY);
        }
    }
    Ok(total)
}

#[inline]
fn squared_cutoff_f64(first: &[f64], second: &[f64], limit: f64) -> CoreResult<f64> {
    let mut total = 0.0;
    for index in 0..first.len() {
        let difference = first[index] - second[index];
        total = difference.mul_add(difference, total);
        if total > limit {
            return Ok(f64::INFINITY);
        }
    }
    Ok(total)
}

#[inline]
fn finish_cid(squared: f64, correction: f64, cutoff: DistanceCutoff) -> CoreResult<f64> {
    if squared.is_infinite() {
        return Ok(f64::INFINITY);
    }
    let value = squared * correction;
    if cutoff.exceeded_by(value) {
        Ok(f64::INFINITY)
    } else {
        Ok(value)
    }
}

#[inline]
fn converted_cutoff(cutoff: DistanceCutoff, correction: f64) -> f64 {
    match cutoff {
        DistanceCutoff::None => f64::INFINITY,
        DistanceCutoff::Finite(limit) => limit / correction,
    }
}

#[inline]
fn squared_correction(first: f64, second: f64) -> f64 {
    if first == 0.0 {
        return if second == 0.0 { 1.0 } else { f64::INFINITY };
    }
    if second == 0.0 {
        return f64::INFINITY;
    }
    first.max(second) / first.min(second)
}

fn ensure_all_finite_f32(values: &[f32]) -> CoreResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(nonfinite_error())
    } else {
        Ok(())
    }
}

fn ensure_all_finite_f64(values: &[f64]) -> CoreResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(nonfinite_error())
    } else {
        Ok(())
    }
}

#[inline]
fn ensure_finite(value: f64) -> CoreResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(nonfinite_error())
    }
}

#[inline]
fn nonfinite_error() -> CoreError {
    CoreError::invalid_data("CID requires finite numeric values")
}

fn validate_vectors(first: NumericVectorView<'_>, second: NumericVectorView<'_>) -> CoreResult<()> {
    if first.len() != second.len() {
        return Err(CoreError::shape_mismatch(format!(
            "CID requires equal lengths; received {} and {}",
            first.len(),
            second.len()
        )));
    }
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "CID requires matching storage dtypes",
        ));
    }
    Ok(())
}

fn validate_multivariate(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
) -> CoreResult<()> {
    if first.scalar_kind() != second.scalar_kind()
        || first.dimension_count() != second.dimension_count()
        || first.time_count() != second.time_count()
    {
        return Err(CoreError::shape_mismatch(
            "independent CID requires matching rectangular shapes and dtypes",
        ));
    }
    if let Some(selected) = selected {
        if selected.is_empty() {
            return Err(CoreError::invalid_data(
                "CID requires at least one selected dimension",
            ));
        }
        if let Some(&invalid) = selected
            .iter()
            .find(|&&dimension| dimension >= first.dimension_count())
        {
            return Err(CoreError::invalid_data(format!(
                "selected dimension {invalid} is outside dimension count {}",
                first.dimension_count()
            )));
        }
    }
    Ok(())
}

fn series_value(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    time: usize,
) -> CoreResult<f64> {
    series
        .get_f64(dimension, time)
        .ok_or_else(|| CoreError::invalid_data("validated multivariate CID index was unavailable"))
}

#[cfg(test)]
mod tests {
    use super::{cid, cid_independent};
    use crate::{DistanceCutoff, MultivariateSeriesView, NumericVectorView};

    #[test]
    fn equal_complexities_reduce_to_squared_euclidean() {
        assert_eq!(
            cid(
                NumericVectorView::from_f64(&[1.0, 2.0, 3.0]),
                NumericVectorView::from_f64(&[2.0, 3.0, 4.0]),
                DistanceCutoff::None,
            ),
            Ok(3.0)
        );
    }

    #[test]
    fn one_constant_series_has_infinite_distance() {
        assert_eq!(
            cid(
                NumericVectorView::from_f32(&[1.0, 1.0, 1.0]),
                NumericVectorView::from_f32(&[1.0, 2.0, 3.0]),
                DistanceCutoff::None,
            ),
            Ok(f64::INFINITY)
        );
    }

    #[test]
    fn two_singleton_series_use_unit_correction() {
        assert_eq!(
            cid(
                NumericVectorView::from_f64(&[1.0]),
                NumericVectorView::from_f64(&[3.0]),
                DistanceCutoff::None,
            ),
            Ok(4.0)
        );
    }

    #[test]
    fn cutoff_is_converted_by_complexity_correction() {
        let first = NumericVectorView::from_f64(&[0.0, 1.0, 2.0]);
        let second = NumericVectorView::from_f64(&[0.0, 2.0, 4.0]);
        assert_eq!(
            cid(
                first,
                second,
                DistanceCutoff::finite(19.0).expect("valid cutoff"),
            ),
            Ok(f64::INFINITY)
        );
        assert_eq!(
            cid(
                first,
                second,
                DistanceCutoff::finite(20.0).expect("valid cutoff"),
            ),
            Ok(20.0)
        );
    }

    #[test]
    fn independent_cid_sums_channels() {
        let values =
            MultivariateSeriesView::contiguous_f64(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 2, 3)
                .expect("valid series");
        assert_eq!(
            cid_independent(values, values, None, DistanceCutoff::None),
            Ok(0.0)
        );
    }

    #[test]
    fn strided_independent_path_matches_contiguous_semantics() {
        let first_values = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let second_values = [2.0, 11.0, 3.0, 21.0, 4.0, 31.0];
        let first = MultivariateSeriesView::strided_f64(&first_values, 2, 3, 1, 2)
            .expect("valid first series");
        let second = MultivariateSeriesView::strided_f64(&second_values, 2, 3, 1, 2)
            .expect("valid second series");
        assert_eq!(
            cid_independent(first, second, None, DistanceCutoff::None),
            Ok(6.0)
        );
    }
}
