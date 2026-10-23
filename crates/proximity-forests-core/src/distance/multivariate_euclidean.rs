//! Optimized independent multivariate squared Euclidean distance.

use crate::{CoreError, CoreResult, DistanceCutoff, MultivariateSeriesView, NumericVectorView};

/// Sums per-dimension squared Euclidean distances over selected channels.
///
/// Complete and cutoff-aware paths are separated. Contiguous channels dispatch
/// once to concrete `f32` or `f64` kernels; strided views use a zero-allocation
/// scalar fallback.
///
/// # Errors
///
/// Returns an error for mismatched dtypes, dimensions, time lengths, an empty
/// selection, or an out-of-range selected dimension.
pub fn squared_euclidean_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected_dimensions: Option<&[usize]>,
    cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    validate_inputs(first, second)?;
    if let Some(selected) = selected_dimensions {
        validate_selected(first.dimension_count(), selected)?;
    }

    match cutoff {
        DistanceCutoff::None => accumulate_complete(first, second, selected_dimensions),
        DistanceCutoff::Finite(limit) => {
            accumulate_cutoff(first, second, selected_dimensions, limit)
        }
    }
}

fn accumulate_complete(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
) -> CoreResult<f64> {
    let mut total = 0.0;
    match selected {
        Some(selected) => {
            for &dimension in selected {
                total += complete_dimension(first, second, dimension)?;
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                total += complete_dimension(first, second, dimension)?;
            }
        }
    }
    Ok(total)
}

fn accumulate_cutoff(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    limit: f64,
) -> CoreResult<f64> {
    let mut total = 0.0;
    match selected {
        Some(selected) => {
            for &dimension in selected {
                total = cutoff_dimension(first, second, dimension, total, limit)?;
                if total.is_infinite() {
                    return Ok(total);
                }
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                total = cutoff_dimension(first, second, dimension, total, limit)?;
                if total.is_infinite() {
                    return Ok(total);
                }
            }
        }
    }
    Ok(total)
}

fn complete_dimension(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
) -> CoreResult<f64> {
    match (
        first.contiguous_dimension(dimension),
        second.contiguous_dimension(dimension),
    ) {
        (Some(NumericVectorView::F32(first)), Some(NumericVectorView::F32(second))) => {
            Ok(complete_f32(first, second))
        }
        (Some(NumericVectorView::F64(first)), Some(NumericVectorView::F64(second))) => {
            Ok(complete_f64(first, second))
        }
        (Some(_), Some(_)) => Err(CoreError::shape_mismatch(
            "independent multivariate Euclidean distance requires matching dtypes",
        )),
        _ => complete_strided(first, second, dimension, 0.0),
    }
}

fn cutoff_dimension(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    initial: f64,
    limit: f64,
) -> CoreResult<f64> {
    match (
        first.contiguous_dimension(dimension),
        second.contiguous_dimension(dimension),
    ) {
        (Some(NumericVectorView::F32(first)), Some(NumericVectorView::F32(second))) => {
            Ok(cutoff_f32(first, second, initial, limit))
        }
        (Some(NumericVectorView::F64(first)), Some(NumericVectorView::F64(second))) => {
            Ok(cutoff_f64(first, second, initial, limit))
        }
        (Some(_), Some(_)) => Err(CoreError::shape_mismatch(
            "independent multivariate Euclidean distance requires matching dtypes",
        )),
        _ => cutoff_strided(first, second, dimension, initial, limit),
    }
}

#[inline]
fn complete_f32(first: &[f32], second: &[f32]) -> f64 {
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
    total
}

#[inline]
fn complete_f64(first: &[f64], second: &[f64]) -> f64 {
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
    total
}

#[inline]
fn cutoff_f32(first: &[f32], second: &[f32], initial: f64, limit: f64) -> f64 {
    let mut total = initial;
    for index in 0..first.len() {
        let difference = f64::from(first[index]) - f64::from(second[index]);
        total = difference.mul_add(difference, total);
        if total > limit {
            return f64::INFINITY;
        }
    }
    total
}

#[inline]
fn cutoff_f64(first: &[f64], second: &[f64], initial: f64, limit: f64) -> f64 {
    let mut total = initial;
    for index in 0..first.len() {
        let difference = first[index] - second[index];
        total = difference.mul_add(difference, total);
        if total > limit {
            return f64::INFINITY;
        }
    }
    total
}

fn complete_strided(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    mut total: f64,
) -> CoreResult<f64> {
    for time in 0..first.time_count() {
        let left = first.get_f64(dimension, time).ok_or_else(|| {
            CoreError::invalid_data("validated first-series index was unavailable")
        })?;
        let right = second.get_f64(dimension, time).ok_or_else(|| {
            CoreError::invalid_data("validated second-series index was unavailable")
        })?;
        let difference = left - right;
        total = difference.mul_add(difference, total);
    }
    Ok(total)
}

fn cutoff_strided(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    mut total: f64,
    limit: f64,
) -> CoreResult<f64> {
    for time in 0..first.time_count() {
        let left = first.get_f64(dimension, time).ok_or_else(|| {
            CoreError::invalid_data("validated first-series index was unavailable")
        })?;
        let right = second.get_f64(dimension, time).ok_or_else(|| {
            CoreError::invalid_data("validated second-series index was unavailable")
        })?;
        let difference = left - right;
        total = difference.mul_add(difference, total);
        if total > limit {
            return Ok(f64::INFINITY);
        }
    }
    Ok(total)
}

fn validate_inputs(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
) -> CoreResult<()> {
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "independent multivariate Euclidean distance requires matching dtypes",
        ));
    }
    if first.dimension_count() != second.dimension_count() {
        return Err(CoreError::shape_mismatch(format!(
            "dimension counts differ: {} and {}",
            first.dimension_count(),
            second.dimension_count()
        )));
    }
    if first.time_count() != second.time_count() {
        return Err(CoreError::shape_mismatch(format!(
            "independent squared Euclidean distance requires equal time lengths; received {} and {}",
            first.time_count(),
            second.time_count()
        )));
    }
    Ok(())
}

fn validate_selected(dimension_count: usize, selected: &[usize]) -> CoreResult<()> {
    if selected.is_empty() {
        return Err(CoreError::invalid_data(
            "independent squared Euclidean distance requires at least one selected dimension",
        ));
    }
    if let Some(&invalid) = selected
        .iter()
        .find(|&&dimension| dimension >= dimension_count)
    {
        return Err(CoreError::invalid_data(format!(
            "selected dimension {invalid} is outside dimension count {dimension_count}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::squared_euclidean_independent;
    use crate::{DistanceCutoff, MultivariateSeriesView};

    #[test]
    fn sums_selected_contiguous_dimensions() {
        let first =
            MultivariateSeriesView::contiguous_f32(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 2, 3)
                .expect("valid first series");
        let second =
            MultivariateSeriesView::contiguous_f32(&[2.0, 4.0, 6.0, 11.0, 20.0, 29.0], 2, 3)
                .expect("valid second series");
        assert_eq!(
            squared_euclidean_independent(first, second, Some(&[0]), DistanceCutoff::None),
            Ok(14.0)
        );
        assert_eq!(
            squared_euclidean_independent(first, second, None, DistanceCutoff::None),
            Ok(16.0)
        );
    }

    #[test]
    fn complete_path_handles_non_multiple_of_four_length() {
        let first = MultivariateSeriesView::contiguous_f64(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            2,
            5,
        )
        .expect("valid first series");
        let second = MultivariateSeriesView::contiguous_f64(
            &[2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 21.0, 31.0, 41.0, 51.0],
            2,
            5,
        )
        .expect("valid second series");
        assert_eq!(
            squared_euclidean_independent(first, second, None, DistanceCutoff::None),
            Ok(10.0)
        );
    }

    #[test]
    fn cutoff_is_global_and_preserves_equality() {
        let first = MultivariateSeriesView::contiguous_f64(&[0.0, 0.0, 0.0, 0.0], 2, 2)
            .expect("valid first series");
        let second = MultivariateSeriesView::contiguous_f64(&[0.0, 1.0, 0.0, 2.0], 2, 2)
            .expect("valid second series");
        assert_eq!(
            squared_euclidean_independent(
                first,
                second,
                None,
                DistanceCutoff::finite(5.0).expect("valid cutoff"),
            ),
            Ok(5.0)
        );
        assert_eq!(
            squared_euclidean_independent(
                first,
                second,
                None,
                DistanceCutoff::finite(4.0).expect("valid cutoff"),
            ),
            Ok(f64::INFINITY)
        );
    }

    #[test]
    fn strided_fallback_matches_contiguous_result() {
        let first_values = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let second_values = [2.0, 11.0, 4.0, 20.0, 6.0, 29.0];
        let first = MultivariateSeriesView::strided_f64(&first_values, 2, 3, 1, 2)
            .expect("valid first series");
        let second = MultivariateSeriesView::strided_f64(&second_values, 2, 3, 1, 2)
            .expect("valid second series");
        assert_eq!(
            squared_euclidean_independent(first, second, None, DistanceCutoff::None),
            Ok(16.0)
        );
    }

    #[test]
    fn invalid_shapes_and_empty_selection_are_rejected() {
        let first =
            MultivariateSeriesView::contiguous_f64(&[1.0, 2.0], 1, 2).expect("valid first series");
        let second = MultivariateSeriesView::contiguous_f64(&[1.0, 2.0, 3.0], 1, 3)
            .expect("valid second series");
        assert!(squared_euclidean_independent(first, second, None, DistanceCutoff::None).is_err());

        let values =
            MultivariateSeriesView::contiguous_f64(&[1.0, 2.0], 1, 2).expect("valid series");
        assert!(
            squared_euclidean_independent(values, values, Some(&[]), DistanceCutoff::None,)
                .is_err()
        );
    }
}
