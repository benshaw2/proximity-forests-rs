//! DTW-AROW distances for incomplete univariate and multivariate series.

use crate::{
    CoreError, CoreResult, DistanceCutoff, DistanceScratch, DtwArowParameters,
    MultivariateSeriesView, NumericVectorView,
};

/// Computes univariate DTW-AROW.
///
/// Missing comparisons have zero local cost. Horizontal and vertical moves are
/// prohibited when either endpoint comparison touches missingness. Diagonal
/// moves remain available. The returned distance applies the availability
/// correction and square root defined by DTW-AROW.
///
/// # Errors
///
/// Returns an error for empty inputs, mismatched dtypes, or infinite observed
/// values.
pub fn dtw_arow(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    parameters: DtwArowParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    if first.is_empty() || second.is_empty() {
        return Err(CoreError::invalid_data("DTW-AROW requires nonempty series"));
    }
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "DTW-AROW requires matching storage dtypes",
        ));
    }

    let available_first = count_available_vector(first)?;
    let available_second = count_available_vector(second)?;
    let total_available = available_first + available_second;
    if total_available == 0 {
        return Ok(f64::INFINITY);
    }

    let Some(window) = resolve_window(first.len(), second.len(), parameters.window) else {
        return Ok(f64::INFINITY);
    };
    #[allow(clippy::cast_precision_loss)]
    let gamma = (first.len() + second.len()) as f64 / total_available as f64;
    let raw_cutoff = raw_cutoff(cutoff, gamma);

    let (mut previous, mut current, mut previous_comparable, mut current_comparable) =
        scratch.arow_rows(second.len() + 1);
    previous.fill(0.0);

    for first_dp in 1..=first.len() {
        current.fill(f64::INFINITY);
        current_comparable.fill(false);
        current[0] = 0.0;
        let start = 1.max(first_dp.saturating_sub(window));
        let stop = second.len().min(first_dp.saturating_add(window));
        let first_index = first_dp - 1;
        let first_value = vector_value(first, first_index)?;
        validate_observed(first_value)?;
        let first_missing = first_value.is_nan();

        for second_dp in start..=stop {
            let second_index = second_dp - 1;
            let second_value = vector_value(second, second_index)?;
            validate_observed(second_value)?;
            let comparable = !first_missing && !second_value.is_nan();
            current_comparable[second_dp] = comparable;

            let diagonal = previous[second_dp - 1];
            let horizontal =
                if comparable && (second_index == 0 || current_comparable[second_dp - 1]) {
                    current[second_dp - 1]
                } else {
                    f64::INFINITY
                };
            let vertical = if comparable && (first_index == 0 || previous_comparable[second_dp]) {
                previous[second_dp]
            } else {
                f64::INFINITY
            };

            let predecessor = diagonal.min(horizontal).min(vertical);
            if predecessor.is_infinite() || predecessor > raw_cutoff {
                continue;
            }
            let local = if comparable {
                let difference = first_value - second_value;
                difference * difference
            } else {
                0.0
            };
            let cumulative = predecessor + local;
            if cumulative <= raw_cutoff {
                current[second_dp] = cumulative;
            }
        }

        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut previous_comparable, &mut current_comparable);
    }

    finish_arow(previous[second.len()], gamma)
}

/// Computes independent multivariate DTW-AROW by summing channel distances.
///
/// Each channel receives the remaining ordinary-distance cutoff and computes
/// its own availability correction.
///
/// # Errors
///
/// Returns an error for incompatible dimensions or dtypes, or invalid selected
/// dimensions.
pub fn dtw_arow_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: DtwArowParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_multivariate(first, second, selected)?;
    let mut total = 0.0;
    match selected {
        Some(selected) => {
            for &dimension in selected {
                let cost = dtw_arow_channel(
                    first,
                    second,
                    dimension,
                    parameters,
                    cutoff.remaining_after(total),
                    scratch,
                )?;
                if cost.is_infinite() {
                    return Ok(f64::INFINITY);
                }
                total += cost;
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                let cost = dtw_arow_channel(
                    first,
                    second,
                    dimension,
                    parameters,
                    cutoff.remaining_after(total),
                    scratch,
                )?;
                if cost.is_infinite() {
                    return Ok(f64::INFINITY);
                }
                total += cost;
            }
        }
    }
    if cutoff.exceeded_by(total) {
        Ok(f64::INFINITY)
    } else {
        Ok(total)
    }
}

/// Computes dependent multivariate DTW-AROW using one shared warping path.
///
/// A time pair is comparable when at least one selected channel is jointly
/// observed. Its local cost is selected-count-scaled squared NaN-Euclidean
/// cost. Availability correction counts time points having at least one
/// observed selected channel. Horizontal and vertical moves require both
/// endpoint comparisons to be comparable; diagonal moves always remain
/// available.
///
/// # Errors
///
/// Returns an error for incompatible dimensions or dtypes, invalid selected
/// dimensions, or infinite observed values.
pub fn dtw_arow_dependent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: DtwArowParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_multivariate(first, second, selected)?;
    let selected_count = selected.map_or(first.dimension_count(), <[usize]>::len);
    let available_first = count_available_time_points(first, selected)?;
    let available_second = count_available_time_points(second, selected)?;
    let total_available = available_first + available_second;
    if total_available == 0 {
        return Ok(f64::INFINITY);
    }

    let Some(window) = resolve_window(first.time_count(), second.time_count(), parameters.window)
    else {
        return Ok(f64::INFINITY);
    };
    #[allow(clippy::cast_precision_loss)]
    let gamma = (first.time_count() + second.time_count()) as f64 / total_available as f64;
    let raw_cutoff = raw_cutoff(cutoff, gamma);
    let (mut previous, mut current, mut previous_comparable, mut current_comparable) =
        scratch.arow_rows(second.time_count() + 1);
    previous.fill(0.0);

    for first_dp in 1..=first.time_count() {
        current.fill(f64::INFINITY);
        current_comparable.fill(false);
        current[0] = 0.0;
        let start = 1.max(first_dp.saturating_sub(window));
        let stop = second.time_count().min(first_dp.saturating_add(window));
        let first_time = first_dp - 1;

        for second_dp in start..=stop {
            let second_time = second_dp - 1;
            let local = dependent_local(
                first,
                first_time,
                second,
                second_time,
                selected,
                selected_count,
            )?;
            current_comparable[second_dp] = local.comparable;

            let diagonal = previous[second_dp - 1];
            let horizontal =
                if local.comparable && (second_time == 0 || current_comparable[second_dp - 1]) {
                    current[second_dp - 1]
                } else {
                    f64::INFINITY
                };
            let vertical =
                if local.comparable && (first_time == 0 || previous_comparable[second_dp]) {
                    previous[second_dp]
                } else {
                    f64::INFINITY
                };
            let predecessor = diagonal.min(horizontal).min(vertical);
            if predecessor.is_infinite() || predecessor > raw_cutoff {
                continue;
            }
            if local.cost <= raw_cutoff - predecessor {
                current[second_dp] = predecessor + local.cost;
            }
        }

        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut previous_comparable, &mut current_comparable);
    }

    finish_arow(previous[second.time_count()], gamma)
}

fn dtw_arow_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    parameters: DtwArowParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    match (
        first.contiguous_dimension(dimension),
        second.contiguous_dimension(dimension),
    ) {
        (Some(first), Some(second)) => dtw_arow(first, second, parameters, cutoff, scratch),
        _ => dtw_arow_strided(first, second, dimension, parameters, cutoff, scratch),
    }
}

fn dtw_arow_strided(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    parameters: DtwArowParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    let available_first = count_available_channel(first, dimension)?;
    let available_second = count_available_channel(second, dimension)?;
    let total_available = available_first + available_second;
    if total_available == 0 {
        return Ok(f64::INFINITY);
    }
    let Some(window) = resolve_window(first.time_count(), second.time_count(), parameters.window)
    else {
        return Ok(f64::INFINITY);
    };
    #[allow(clippy::cast_precision_loss)]
    let gamma = (first.time_count() + second.time_count()) as f64 / total_available as f64;
    let raw_cutoff = raw_cutoff(cutoff, gamma);
    let (mut previous, mut current, mut previous_comparable, mut current_comparable) =
        scratch.arow_rows(second.time_count() + 1);
    previous.fill(0.0);

    for first_dp in 1..=first.time_count() {
        current.fill(f64::INFINITY);
        current_comparable.fill(false);
        current[0] = 0.0;
        let start = 1.max(first_dp.saturating_sub(window));
        let stop = second.time_count().min(first_dp.saturating_add(window));
        let first_time = first_dp - 1;
        let first_value = series_value(first, dimension, first_time)?;
        validate_observed(first_value)?;
        let first_missing = first_value.is_nan();

        for second_dp in start..=stop {
            let second_time = second_dp - 1;
            let second_value = series_value(second, dimension, second_time)?;
            validate_observed(second_value)?;
            let comparable = !first_missing && !second_value.is_nan();
            current_comparable[second_dp] = comparable;
            let diagonal = previous[second_dp - 1];
            let horizontal =
                if comparable && (second_time == 0 || current_comparable[second_dp - 1]) {
                    current[second_dp - 1]
                } else {
                    f64::INFINITY
                };
            let vertical = if comparable && (first_time == 0 || previous_comparable[second_dp]) {
                previous[second_dp]
            } else {
                f64::INFINITY
            };
            let predecessor = diagonal.min(horizontal).min(vertical);
            if predecessor.is_infinite() || predecessor > raw_cutoff {
                continue;
            }
            let local = if comparable {
                let difference = first_value - second_value;
                difference * difference
            } else {
                0.0
            };
            let cumulative = predecessor + local;
            if cumulative <= raw_cutoff {
                current[second_dp] = cumulative;
            }
        }
        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut previous_comparable, &mut current_comparable);
    }
    finish_arow(previous[second.time_count()], gamma)
}

#[derive(Clone, Copy)]
struct LocalComparison {
    comparable: bool,
    cost: f64,
}

fn dependent_local(
    first: MultivariateSeriesView<'_>,
    first_time: usize,
    second: MultivariateSeriesView<'_>,
    second_time: usize,
    selected: Option<&[usize]>,
    selected_count: usize,
) -> CoreResult<LocalComparison> {
    let mut squared_sum = 0.0;
    let mut jointly_observed = 0usize;
    match selected {
        Some(selected) => {
            for &dimension in selected {
                accumulate_joint(
                    first,
                    first_time,
                    second,
                    second_time,
                    dimension,
                    &mut squared_sum,
                    &mut jointly_observed,
                )?;
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                accumulate_joint(
                    first,
                    first_time,
                    second,
                    second_time,
                    dimension,
                    &mut squared_sum,
                    &mut jointly_observed,
                )?;
            }
        }
    }
    if jointly_observed == 0 {
        return Ok(LocalComparison {
            comparable: false,
            cost: 0.0,
        });
    }
    #[allow(clippy::cast_precision_loss)]
    let cost = selected_count as f64 / jointly_observed as f64 * squared_sum;
    Ok(LocalComparison {
        comparable: true,
        cost,
    })
}

fn accumulate_joint(
    first: MultivariateSeriesView<'_>,
    first_time: usize,
    second: MultivariateSeriesView<'_>,
    second_time: usize,
    dimension: usize,
    squared_sum: &mut f64,
    jointly_observed: &mut usize,
) -> CoreResult<()> {
    let first_value = series_value(first, dimension, first_time)?;
    let second_value = series_value(second, dimension, second_time)?;
    if first_value.is_nan() || second_value.is_nan() {
        return Ok(());
    }
    validate_observed(first_value)?;
    validate_observed(second_value)?;
    let difference = first_value - second_value;
    *squared_sum = difference.mul_add(difference, *squared_sum);
    *jointly_observed += 1;
    Ok(())
}

fn count_available_vector(series: NumericVectorView<'_>) -> CoreResult<usize> {
    match series {
        NumericVectorView::F32(values) => count_available_f32(values),
        NumericVectorView::F64(values) => count_available_f64(values),
    }
}

fn count_available_f32(values: &[f32]) -> CoreResult<usize> {
    let mut count = 0usize;
    for &value in values {
        if value.is_nan() {
            continue;
        }
        if !value.is_finite() {
            return Err(nonfinite_error());
        }
        count += 1;
    }
    Ok(count)
}

fn count_available_f64(values: &[f64]) -> CoreResult<usize> {
    let mut count = 0usize;
    for &value in values {
        if value.is_nan() {
            continue;
        }
        if !value.is_finite() {
            return Err(nonfinite_error());
        }
        count += 1;
    }
    Ok(count)
}

fn count_available_channel(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
) -> CoreResult<usize> {
    let mut count = 0usize;
    for time in 0..series.time_count() {
        let value = series_value(series, dimension, time)?;
        if !value.is_nan() {
            validate_observed(value)?;
            count += 1;
        }
    }
    Ok(count)
}

fn count_available_time_points(
    series: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
) -> CoreResult<usize> {
    let mut count = 0usize;
    for time in 0..series.time_count() {
        let mut available = false;
        match selected {
            Some(selected) => {
                for &dimension in selected {
                    let value = series_value(series, dimension, time)?;
                    if !value.is_nan() {
                        validate_observed(value)?;
                        available = true;
                        break;
                    }
                }
            }
            None => {
                for dimension in 0..series.dimension_count() {
                    let value = series_value(series, dimension, time)?;
                    if !value.is_nan() {
                        validate_observed(value)?;
                        available = true;
                        break;
                    }
                }
            }
        }
        count += usize::from(available);
    }
    Ok(count)
}

fn resolve_window(
    first_length: usize,
    second_length: usize,
    window: Option<usize>,
) -> Option<usize> {
    match window {
        Some(window) if first_length.abs_diff(second_length) > window => None,
        Some(window) => Some(window),
        None => Some(first_length.max(second_length)),
    }
}

fn raw_cutoff(cutoff: DistanceCutoff, gamma: f64) -> f64 {
    match cutoff {
        DistanceCutoff::None => f64::INFINITY,
        DistanceCutoff::Finite(limit) => limit * limit / gamma,
    }
}

fn finish_arow(raw: f64, gamma: f64) -> CoreResult<f64> {
    if raw.is_infinite() {
        Ok(f64::INFINITY)
    } else {
        Ok((gamma * raw).sqrt())
    }
}

fn vector_value(series: NumericVectorView<'_>, index: usize) -> CoreResult<f64> {
    series
        .get_f64(index)
        .ok_or_else(|| CoreError::invalid_data("validated DTW-AROW vector index was unavailable"))
}

fn series_value(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    time: usize,
) -> CoreResult<f64> {
    series
        .get_f64(dimension, time)
        .ok_or_else(|| CoreError::invalid_data("validated DTW-AROW series index was unavailable"))
}

#[inline]
fn validate_observed(value: f64) -> CoreResult<()> {
    if value.is_infinite() {
        Err(nonfinite_error())
    } else {
        Ok(())
    }
}

#[inline]
fn nonfinite_error() -> CoreError {
    CoreError::invalid_data("DTW-AROW inputs may contain NaN but not infinity")
}

fn validate_multivariate(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
) -> CoreResult<()> {
    if first.time_count() == 0 || second.time_count() == 0 {
        return Err(CoreError::invalid_data(
            "multivariate DTW-AROW requires nonempty time axes",
        ));
    }
    if first.scalar_kind() != second.scalar_kind()
        || first.dimension_count() != second.dimension_count()
    {
        return Err(CoreError::shape_mismatch(
            "multivariate DTW-AROW requires matching dimensions and dtypes",
        ));
    }
    if let Some(selected) = selected {
        if selected.is_empty() {
            return Err(CoreError::invalid_data(
                "DTW-AROW requires at least one selected dimension",
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

#[cfg(test)]
mod tests {
    use super::{dtw_arow, dtw_arow_dependent, dtw_arow_independent};
    use crate::{
        DistanceCutoff, DistanceScratch, DtwArowParameters, MultivariateSeriesView,
        NumericVectorView,
    };

    #[test]
    fn identical_complete_series_have_zero_distance() {
        let values = [1.0_f64, 2.0, 3.0];
        assert_eq!(
            dtw_arow(
                NumericVectorView::from_f64(&values),
                NumericVectorView::from_f64(&values),
                DtwArowParameters::default(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }

    #[test]
    fn all_missing_pair_is_unavailable() {
        assert_eq!(
            dtw_arow(
                NumericVectorView::from_f32(&[f32::NAN, f32::NAN]),
                NumericVectorView::from_f32(&[f32::NAN, f32::NAN]),
                DtwArowParameters::default(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(f64::INFINITY)
        );
    }

    #[test]
    fn finite_window_is_not_widened() {
        assert_eq!(
            dtw_arow(
                NumericVectorView::from_f64(&[1.0]),
                NumericVectorView::from_f64(&[1.0, 1.0, 1.0]),
                DtwArowParameters { window: Some(0) },
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(f64::INFINITY)
        );
    }

    #[test]
    fn multivariate_variants_handle_partial_comparability() {
        let first =
            MultivariateSeriesView::contiguous_f64(&[1.0, f64::NAN, 3.0, f64::NAN, 2.0, 3.0], 2, 3)
                .expect("valid first series");
        assert_eq!(
            dtw_arow_independent(
                first,
                first,
                None,
                DtwArowParameters::default(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
        assert_eq!(
            dtw_arow_dependent(
                first,
                first,
                None,
                DtwArowParameters::default(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }

    #[test]
    fn cutoff_equality_is_preserved() {
        assert_eq!(
            dtw_arow(
                NumericVectorView::from_f64(&[0.0]),
                NumericVectorView::from_f64(&[2.0]),
                DtwArowParameters::default(),
                DistanceCutoff::finite(2.0).expect("valid cutoff"),
                &mut DistanceScratch::new(),
            ),
            Ok(2.0)
        );
    }

    #[test]
    fn infinite_observed_values_are_rejected() {
        assert!(
            dtw_arow(
                NumericVectorView::from_f64(&[f64::INFINITY]),
                NumericVectorView::from_f64(&[0.0]),
                DtwArowParameters::default(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            )
            .is_err()
        );
    }
}
