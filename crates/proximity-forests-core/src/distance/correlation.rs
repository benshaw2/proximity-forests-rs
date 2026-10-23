//! Optimized shape-based distance using normalized cross-correlation.

use crate::{CoreError, CoreResult, DistanceCutoff, MultivariateSeriesView, NumericVectorView};

/// Computes shape-based distance as one minus the maximum normalized linear
/// cross-correlation over every shift.
///
/// This dependency-free short-series kernel is quadratic. It avoids temporary
/// centered vectors, signed-index checks, and repeated enum dispatch. An FFT
/// backend can later replace the correlation kernels without changing callers.
///
/// # Errors
///
/// Returns an error for empty or unequal-length inputs, mismatched dtypes, or
/// non-finite values.
pub fn sbd(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    _cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    validate_vectors(first, second)?;
    match (first, second) {
        (NumericVectorView::F32(first), NumericVectorView::F32(second)) => sbd_f32(first, second),
        (NumericVectorView::F64(first), NumericVectorView::F64(second)) => sbd_f64(first, second),
        _ => Err(CoreError::shape_mismatch(
            "SBD requires matching storage dtypes",
        )),
    }
}

/// Computes independent multivariate SBD by summing selected-channel distances.
///
/// # Errors
///
/// Returns an error for incompatible shapes or dtypes, invalid selected
/// dimensions, or non-finite values.
pub fn sbd_independent(
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
                total += sbd_channel(first, second, dimension)?;
                if cutoff.exceeded_by(total) {
                    return Ok(f64::INFINITY);
                }
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                total += sbd_channel(first, second, dimension)?;
                if cutoff.exceeded_by(total) {
                    return Ok(f64::INFINITY);
                }
            }
        }
    }
    Ok(total)
}

#[inline]
fn sbd_f32(first: &[f32], second: &[f32]) -> CoreResult<f64> {
    let (first_mean, first_norm) = statistics_f32(first)?;
    let (second_mean, second_norm) = statistics_f32(second)?;
    finish_sbd(first_norm, second_norm, || {
        maximum_correlation_f32(first, second, first_mean, second_mean)
    })
}

#[inline]
fn sbd_f64(first: &[f64], second: &[f64]) -> CoreResult<f64> {
    let (first_mean, first_norm) = statistics_f64(first)?;
    let (second_mean, second_norm) = statistics_f64(second)?;
    finish_sbd(first_norm, second_norm, || {
        maximum_correlation_f64(first, second, first_mean, second_mean)
    })
}

fn sbd_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
) -> CoreResult<f64> {
    match (
        first.contiguous_dimension(dimension),
        second.contiguous_dimension(dimension),
    ) {
        (Some(NumericVectorView::F32(first)), Some(NumericVectorView::F32(second))) => {
            sbd_f32(first, second)
        }
        (Some(NumericVectorView::F64(first)), Some(NumericVectorView::F64(second))) => {
            sbd_f64(first, second)
        }
        (Some(_), Some(_)) => Err(CoreError::shape_mismatch(
            "independent SBD requires matching storage dtypes",
        )),
        _ => sbd_strided(first, second, dimension),
    }
}

fn sbd_strided(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
) -> CoreResult<f64> {
    let (first_mean, first_norm) = strided_statistics(first, dimension)?;
    let (second_mean, second_norm) = strided_statistics(second, dimension)?;
    finish_sbd(first_norm, second_norm, || {
        maximum_correlation_strided(first, second, dimension, first_mean, second_mean)
    })
}

fn finish_sbd<Maximum>(first_norm: f64, second_norm: f64, maximum: Maximum) -> CoreResult<f64>
where
    Maximum: FnOnce() -> CoreResult<f64>,
{
    if first_norm == 0.0 {
        return Ok(if second_norm == 0.0 { 0.0 } else { 1.0 });
    }
    if second_norm == 0.0 {
        return Ok(1.0);
    }
    let correlation = maximum()?;
    let normalized = (correlation / (first_norm * second_norm).sqrt()).clamp(-1.0, 1.0);
    Ok(1.0 - normalized)
}

/// Calculates mean and centered squared norm in two stable, linear passes.
fn statistics_f32(values: &[f32]) -> CoreResult<(f64, f64)> {
    let mut sum = 0.0;
    for &value in values {
        if !value.is_finite() {
            return Err(nonfinite_error());
        }
        sum += f64::from(value);
    }
    #[allow(clippy::cast_precision_loss)]
    let mean = sum / values.len() as f64;

    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut index = 0usize;
    let block_end = values.len() - values.len() % 4;
    while index < block_end {
        let centered0 = f64::from(values[index]) - mean;
        let centered1 = f64::from(values[index + 1]) - mean;
        let centered2 = f64::from(values[index + 2]) - mean;
        let centered3 = f64::from(values[index + 3]) - mean;
        sum0 = centered0.mul_add(centered0, sum0);
        sum1 = centered1.mul_add(centered1, sum1);
        sum2 = centered2.mul_add(centered2, sum2);
        sum3 = centered3.mul_add(centered3, sum3);
        index += 4;
    }
    let mut norm = (sum0 + sum1) + (sum2 + sum3);
    while index < values.len() {
        let centered = f64::from(values[index]) - mean;
        norm = centered.mul_add(centered, norm);
        index += 1;
    }
    Ok((mean, norm))
}

fn statistics_f64(values: &[f64]) -> CoreResult<(f64, f64)> {
    let mut sum = 0.0;
    for &value in values {
        if !value.is_finite() {
            return Err(nonfinite_error());
        }
        sum += value;
    }
    #[allow(clippy::cast_precision_loss)]
    let mean = sum / values.len() as f64;

    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut index = 0usize;
    let block_end = values.len() - values.len() % 4;
    while index < block_end {
        let centered0 = values[index] - mean;
        let centered1 = values[index + 1] - mean;
        let centered2 = values[index + 2] - mean;
        let centered3 = values[index + 3] - mean;
        sum0 = centered0.mul_add(centered0, sum0);
        sum1 = centered1.mul_add(centered1, sum1);
        sum2 = centered2.mul_add(centered2, sum2);
        sum3 = centered3.mul_add(centered3, sum3);
        index += 4;
    }
    let mut norm = (sum0 + sum1) + (sum2 + sum3);
    while index < values.len() {
        let centered = values[index] - mean;
        norm = centered.mul_add(centered, norm);
        index += 1;
    }
    Ok((mean, norm))
}

/// Evaluates zero lag, then positive and negative lags using direct overlap
/// bounds. No signed indices or per-sample range checks are required.
fn maximum_correlation_f32(
    first: &[f32],
    second: &[f32],
    first_mean: f64,
    second_mean: f64,
) -> CoreResult<f64> {
    let mut maximum = correlation_f32(first, second, 0, 0, first.len(), first_mean, second_mean);
    for shift in 1..first.len() {
        maximum = maximum.max(correlation_f32(
            first,
            second,
            shift,
            0,
            first.len() - shift,
            first_mean,
            second_mean,
        ));
        maximum = maximum.max(correlation_f32(
            first,
            second,
            0,
            shift,
            first.len() - shift,
            first_mean,
            second_mean,
        ));
    }
    Ok(maximum)
}

fn maximum_correlation_f64(
    first: &[f64],
    second: &[f64],
    first_mean: f64,
    second_mean: f64,
) -> CoreResult<f64> {
    let mut maximum = correlation_f64(first, second, 0, 0, first.len(), first_mean, second_mean);
    for shift in 1..first.len() {
        maximum = maximum.max(correlation_f64(
            first,
            second,
            shift,
            0,
            first.len() - shift,
            first_mean,
            second_mean,
        ));
        maximum = maximum.max(correlation_f64(
            first,
            second,
            0,
            shift,
            first.len() - shift,
            first_mean,
            second_mean,
        ));
    }
    Ok(maximum)
}

#[inline]
fn correlation_f32(
    first: &[f32],
    second: &[f32],
    first_start: usize,
    second_start: usize,
    overlap: usize,
    first_mean: f64,
    second_mean: f64,
) -> f64 {
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut offset = 0usize;
    let block_end = overlap - overlap % 4;
    while offset < block_end {
        let left0 = f64::from(first[first_start + offset]) - first_mean;
        let right0 = f64::from(second[second_start + offset]) - second_mean;
        let left1 = f64::from(first[first_start + offset + 1]) - first_mean;
        let right1 = f64::from(second[second_start + offset + 1]) - second_mean;
        let left2 = f64::from(first[first_start + offset + 2]) - first_mean;
        let right2 = f64::from(second[second_start + offset + 2]) - second_mean;
        let left3 = f64::from(first[first_start + offset + 3]) - first_mean;
        let right3 = f64::from(second[second_start + offset + 3]) - second_mean;
        sum0 = left0.mul_add(right0, sum0);
        sum1 = left1.mul_add(right1, sum1);
        sum2 = left2.mul_add(right2, sum2);
        sum3 = left3.mul_add(right3, sum3);
        offset += 4;
    }
    let mut total = (sum0 + sum1) + (sum2 + sum3);
    while offset < overlap {
        let left = f64::from(first[first_start + offset]) - first_mean;
        let right = f64::from(second[second_start + offset]) - second_mean;
        total = left.mul_add(right, total);
        offset += 1;
    }
    total
}

#[inline]
fn correlation_f64(
    first: &[f64],
    second: &[f64],
    first_start: usize,
    second_start: usize,
    overlap: usize,
    first_mean: f64,
    second_mean: f64,
) -> f64 {
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut offset = 0usize;
    let block_end = overlap - overlap % 4;
    while offset < block_end {
        let left0 = first[first_start + offset] - first_mean;
        let right0 = second[second_start + offset] - second_mean;
        let left1 = first[first_start + offset + 1] - first_mean;
        let right1 = second[second_start + offset + 1] - second_mean;
        let left2 = first[first_start + offset + 2] - first_mean;
        let right2 = second[second_start + offset + 2] - second_mean;
        let left3 = first[first_start + offset + 3] - first_mean;
        let right3 = second[second_start + offset + 3] - second_mean;
        sum0 = left0.mul_add(right0, sum0);
        sum1 = left1.mul_add(right1, sum1);
        sum2 = left2.mul_add(right2, sum2);
        sum3 = left3.mul_add(right3, sum3);
        offset += 4;
    }
    let mut total = (sum0 + sum1) + (sum2 + sum3);
    while offset < overlap {
        let left = first[first_start + offset] - first_mean;
        let right = second[second_start + offset] - second_mean;
        total = left.mul_add(right, total);
        offset += 1;
    }
    total
}

fn strided_statistics(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
) -> CoreResult<(f64, f64)> {
    let mut sum = 0.0;
    for time in 0..series.time_count() {
        sum += series_value(series, dimension, time)?;
    }
    #[allow(clippy::cast_precision_loss)]
    let mean = sum / series.time_count() as f64;
    let mut norm = 0.0;
    for time in 0..series.time_count() {
        let centered = series_value(series, dimension, time)? - mean;
        norm = centered.mul_add(centered, norm);
    }
    Ok((mean, norm))
}

fn maximum_correlation_strided(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    first_mean: f64,
    second_mean: f64,
) -> CoreResult<f64> {
    let length = first.time_count();
    let mut maximum = correlation_strided(
        first,
        second,
        dimension,
        0,
        0,
        length,
        first_mean,
        second_mean,
    )?;
    for shift in 1..length {
        maximum = maximum.max(correlation_strided(
            first,
            second,
            dimension,
            shift,
            0,
            length - shift,
            first_mean,
            second_mean,
        )?);
        maximum = maximum.max(correlation_strided(
            first,
            second,
            dimension,
            0,
            shift,
            length - shift,
            first_mean,
            second_mean,
        )?);
    }
    Ok(maximum)
}

fn correlation_strided(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    first_start: usize,
    second_start: usize,
    overlap: usize,
    first_mean: f64,
    second_mean: f64,
) -> CoreResult<f64> {
    let mut total = 0.0;
    for offset in 0..overlap {
        let left = series_value(first, dimension, first_start + offset)? - first_mean;
        let right = series_value(second, dimension, second_start + offset)? - second_mean;
        total = left.mul_add(right, total);
    }
    Ok(total)
}

fn validate_vectors(first: NumericVectorView<'_>, second: NumericVectorView<'_>) -> CoreResult<()> {
    if first.is_empty() || second.is_empty() {
        return Err(CoreError::invalid_data("SBD requires nonempty series"));
    }
    if first.len() != second.len() {
        return Err(CoreError::shape_mismatch(format!(
            "SBD requires equal lengths; received {} and {}",
            first.len(),
            second.len()
        )));
    }
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "SBD requires matching storage dtypes",
        ));
    }
    Ok(())
}

fn validate_multivariate(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
) -> CoreResult<()> {
    if first.time_count() == 0
        || first.scalar_kind() != second.scalar_kind()
        || first.dimension_count() != second.dimension_count()
        || first.time_count() != second.time_count()
    {
        return Err(CoreError::shape_mismatch(
            "independent SBD requires matching nonempty rectangular shapes and dtypes",
        ));
    }
    if let Some(selected) = selected {
        if selected.is_empty() {
            return Err(CoreError::invalid_data(
                "SBD requires at least one selected dimension",
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
    let value = series.get_f64(dimension, time).ok_or_else(|| {
        CoreError::invalid_data("validated multivariate SBD index was unavailable")
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(nonfinite_error())
    }
}

#[inline]
fn nonfinite_error() -> CoreError {
    CoreError::invalid_data("SBD requires finite numeric values")
}

#[cfg(test)]
mod tests {
    use super::{sbd, sbd_independent};
    use crate::{DistanceCutoff, MultivariateSeriesView, NumericVectorView};

    #[test]
    fn identical_series_have_zero_distance() {
        let values = [1.0_f64, 2.0, 3.0, 4.0];
        assert!(
            sbd(
                NumericVectorView::from_f64(&values),
                NumericVectorView::from_f64(&values),
                DistanceCutoff::None,
            )
            .expect("valid distance")
            .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn constant_series_conventions_are_finite() {
        assert_eq!(
            sbd(
                NumericVectorView::from_f32(&[1.0, 1.0]),
                NumericVectorView::from_f32(&[1.0, 1.0]),
                DistanceCutoff::None,
            ),
            Ok(0.0)
        );
        assert_eq!(
            sbd(
                NumericVectorView::from_f64(&[1.0, 1.0]),
                NumericVectorView::from_f64(&[1.0, 2.0]),
                DistanceCutoff::None,
            ),
            Ok(1.0)
        );
    }

    #[test]
    fn shifted_shape_can_match_at_nonzero_lag() {
        let first = [0.0_f64, 1.0, 0.0, 0.0];
        let second = [0.0_f64, 0.0, 1.0, 0.0];
        let distance = sbd(
            NumericVectorView::from_f64(&first),
            NumericVectorView::from_f64(&second),
            DistanceCutoff::None,
        )
        .expect("valid distance");
        assert!(distance < 1.0);
    }

    #[test]
    fn independent_sbd_sums_channels() {
        let values =
            MultivariateSeriesView::contiguous_f64(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 2, 3)
                .expect("valid series");
        assert!(
            sbd_independent(values, values, None, DistanceCutoff::None)
                .expect("valid distance")
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn nonfinite_values_are_rejected() {
        assert!(
            sbd(
                NumericVectorView::from_f64(&[1.0, f64::NAN]),
                NumericVectorView::from_f64(&[1.0, 2.0]),
                DistanceCutoff::None,
            )
            .is_err()
        );
    }
}
