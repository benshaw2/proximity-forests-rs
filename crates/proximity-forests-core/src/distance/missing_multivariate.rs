//! Optimized missing-value-aware independent multivariate distances.

use crate::distance::missing::{NaNAccumulation, accumulate_f32, accumulate_f64, finish};
use crate::{
    CoreError, CoreResult, DistanceCutoff, MultivariateSeriesView, NoOverlapPolicy,
    NumericVectorView,
};

/// Sums NaN-aware squared Euclidean distances over selected channels.
///
/// Each channel is normalized independently by its own observed overlap. A
/// channel with no overlap follows `no_overlap`. Contiguous channels use the
/// typed vector accumulators directly; strided channels remain allocation-free.
///
/// # Errors
///
/// Returns an error for incompatible shapes or dtypes, an empty or out-of-range
/// selection, or infinite nonmissing values.
pub fn nan_squared_euclidean_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected_dimensions: Option<&[usize]>,
    cutoff: DistanceCutoff,
    no_overlap: NoOverlapPolicy,
) -> CoreResult<f64> {
    validate_inputs(first, second)?;
    if let Some(selected) = selected_dimensions {
        validate_selected(first.dimension_count(), selected)?;
    }

    let mut total = 0.0;
    match selected_dimensions {
        Some(selected) => {
            for &dimension in selected {
                total = accumulate_channel(first, second, dimension, total, cutoff, no_overlap)?;
                if total.is_infinite() {
                    return Ok(total);
                }
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                total = accumulate_channel(first, second, dimension, total, cutoff, no_overlap)?;
                if total.is_infinite() {
                    return Ok(total);
                }
            }
        }
    }
    Ok(total)
}

fn accumulate_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    total: f64,
    cutoff: DistanceCutoff,
    no_overlap: NoOverlapPolicy,
) -> CoreResult<f64> {
    let remaining = cutoff.remaining_after(total);
    let component = match (
        first.contiguous_dimension(dimension),
        second.contiguous_dimension(dimension),
    ) {
        (Some(NumericVectorView::F32(first)), Some(NumericVectorView::F32(second))) => finish(
            accumulate_f32(first, second, remaining)?,
            first.len(),
            remaining,
            no_overlap,
        )?,
        (Some(NumericVectorView::F64(first)), Some(NumericVectorView::F64(second))) => finish(
            accumulate_f64(first, second, remaining)?,
            first.len(),
            remaining,
            no_overlap,
        )?,
        (Some(_), Some(_)) => {
            return Err(CoreError::shape_mismatch(
                "independent NaN-Euclidean distance requires matching dtypes",
            ));
        }
        _ => strided_channel(first, second, dimension, remaining, no_overlap)?,
    };

    if component.is_infinite() {
        return Ok(f64::INFINITY);
    }
    let accumulated = total + component;
    if cutoff.exceeded_by(accumulated) {
        Ok(f64::INFINITY)
    } else {
        Ok(accumulated)
    }
}

fn strided_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    cutoff: DistanceCutoff,
    no_overlap: NoOverlapPolicy,
) -> CoreResult<f64> {
    let limit = match cutoff {
        DistanceCutoff::None => f64::INFINITY,
        DistanceCutoff::Finite(limit) => limit,
    };
    let mut sum = 0.0;
    let mut observed = 0usize;

    for time in 0..first.time_count() {
        let left = first.get_f64(dimension, time).ok_or_else(|| {
            CoreError::invalid_data("validated first-series index was unavailable")
        })?;
        let right = second.get_f64(dimension, time).ok_or_else(|| {
            CoreError::invalid_data("validated second-series index was unavailable")
        })?;
        if left.is_nan() || right.is_nan() {
            continue;
        }
        if !left.is_finite() || !right.is_finite() {
            return Err(CoreError::invalid_data(
                "NaN-aware Euclidean inputs may contain NaN but not infinity",
            ));
        }
        let difference = left - right;
        sum = difference.mul_add(difference, sum);
        observed += 1;
        if sum > limit {
            return Ok(f64::INFINITY);
        }
    }

    finish(
        NaNAccumulation {
            sum,
            observed,
            abandoned: false,
        },
        first.time_count(),
        cutoff,
        no_overlap,
    )
}

fn validate_inputs(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
) -> CoreResult<()> {
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "independent NaN-Euclidean distance requires matching dtypes",
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
            "independent NaN-Euclidean distance requires equal time lengths; received {} and {}",
            first.time_count(),
            second.time_count()
        )));
    }
    Ok(())
}

fn validate_selected(dimension_count: usize, selected: &[usize]) -> CoreResult<()> {
    if selected.is_empty() {
        return Err(CoreError::invalid_data(
            "independent NaN-Euclidean distance requires at least one selected dimension",
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
    use super::nan_squared_euclidean_independent;
    use crate::{DistanceCutoff, MultivariateSeriesView, NoOverlapPolicy};

    #[test]
    fn supports_selected_channels_with_missing_values() {
        let first =
            MultivariateSeriesView::contiguous_f64(&[1.0, f64::NAN, 3.0, 10.0, 20.0, 30.0], 2, 3)
                .expect("valid first series");
        let second =
            MultivariateSeriesView::contiguous_f64(&[2.0, 8.0, 5.0, 11.0, 20.0, 29.0], 2, 3)
                .expect("valid second series");
        assert_eq!(
            nan_squared_euclidean_independent(
                first,
                second,
                Some(&[0]),
                DistanceCutoff::None,
                NoOverlapPolicy::Infinite,
            ),
            Ok(7.5)
        );
    }

    #[test]
    fn cutoff_is_reduced_between_channels() {
        let first = MultivariateSeriesView::contiguous_f32(&[0.0, 0.0, 0.0, 0.0], 2, 2)
            .expect("valid first series");
        let second = MultivariateSeriesView::contiguous_f32(&[0.0, 1.0, 0.0, 2.0], 2, 2)
            .expect("valid second series");
        assert_eq!(
            nan_squared_euclidean_independent(
                first,
                second,
                None,
                DistanceCutoff::finite(4.0).expect("valid cutoff"),
                NoOverlapPolicy::Infinite,
            ),
            Ok(f64::INFINITY)
        );
    }

    #[test]
    fn strided_path_matches_contiguous_semantics() {
        let first_values = [1.0, 10.0, f64::NAN, 20.0, 3.0, 30.0];
        let second_values = [2.0, 11.0, 8.0, 20.0, 5.0, 29.0];
        let first = MultivariateSeriesView::strided_f64(&first_values, 2, 3, 1, 2)
            .expect("valid first series");
        let second = MultivariateSeriesView::strided_f64(&second_values, 2, 3, 1, 2)
            .expect("valid second series");
        assert_eq!(
            nan_squared_euclidean_independent(
                first,
                second,
                Some(&[0]),
                DistanceCutoff::None,
                NoOverlapPolicy::Infinite,
            ),
            Ok(7.5)
        );
    }
}
