//! Ordinary univariate and independent multivariate dynamic time warping.

use crate::{
    CoreError, CoreResult, DistanceCutoff, DistanceScratch, DtwParameters, MultivariateSeriesView,
    NumericVectorView,
};

/// Computes univariate DTW with squared Euclidean local costs.
///
/// Unequal lengths are supported. The optional window is widened to at least
/// the absolute length difference. Only two rows are retained in reusable
/// worker-local scratch storage.
///
/// # Errors
///
/// Returns an error for empty inputs, mismatched storage dtypes, or invalid
/// internal access.
pub fn dtw(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    parameters: DtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    if first.is_empty() || second.is_empty() {
        return Err(CoreError::invalid_data("DTW requires two nonempty series"));
    }

    match (first, second) {
        (NumericVectorView::F32(first), NumericVectorView::F32(second)) => {
            dtw_slices(first, second, parameters.window, cutoff, scratch)
        }
        (NumericVectorView::F64(first), NumericVectorView::F64(second)) => {
            dtw_slices(first, second, parameters.window, cutoff, scratch)
        }
        _ => Err(CoreError::shape_mismatch(
            "DTW requires matching storage dtypes",
        )),
    }
}

/// Computes independent multivariate DTW by aligning each selected channel
/// separately and summing the resulting squared costs.
///
/// A completed channel reduces the cutoff budget available to later channels.
/// Contiguous dimension-major channels use the typed univariate kernel. Strided
/// channels use a zero-allocation scalar-access kernel.
///
/// # Errors
///
/// Returns an error for incompatible dimensions or dtypes, an empty or invalid
/// dimension selection, empty time axes, or invalid internal access.
pub fn dtw_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected_dimensions: Option<&[usize]>,
    parameters: DtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_independent(first, second)?;

    match selected_dimensions {
        Some(selected) => {
            validate_selected_dimensions(first.dimension_count(), selected)?;
            accumulate_channels(
                first,
                second,
                selected.iter().copied(),
                parameters,
                cutoff,
                scratch,
            )
        }
        None => accumulate_channels(
            first,
            second,
            0..first.dimension_count(),
            parameters,
            cutoff,
            scratch,
        ),
    }
}

fn dtw_slices<T>(
    first: &[T],
    second: &[T],
    window: Option<usize>,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64>
where
    T: Copy + Into<f64>,
{
    let (rows, columns) = if first.len() >= second.len() {
        (first, second)
    } else {
        (second, first)
    };

    dtw_core(
        rows.len(),
        columns.len(),
        window,
        cutoff,
        scratch,
        |row, column, allowance| {
            let left = rows[row].into();
            let right = columns[column].into();
            if !left.is_finite() || !right.is_finite() {
                return Err(CoreError::invalid_data(
                    "DTW requires finite numeric values",
                ));
            }
            let difference = left - right;
            let cost = difference * difference;
            Ok(if cost > allowance {
                f64::INFINITY
            } else {
                cost
            })
        },
    )
}

fn dtw_strided_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    window: Option<usize>,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    let (rows, columns) = if first.time_count() >= second.time_count() {
        (first, second)
    } else {
        (second, first)
    };

    dtw_core(
        rows.time_count(),
        columns.time_count(),
        window,
        cutoff,
        scratch,
        |row, column, allowance| {
            let left = rows.get_f64(dimension, row).ok_or_else(|| {
                CoreError::invalid_data("validated first DTW index was unavailable")
            })?;
            let right = columns.get_f64(dimension, column).ok_or_else(|| {
                CoreError::invalid_data("validated second DTW index was unavailable")
            })?;
            if !left.is_finite() || !right.is_finite() {
                return Err(CoreError::invalid_data(
                    "DTW requires finite numeric values",
                ));
            }
            let difference = left - right;
            let cost = difference * difference;
            Ok(if cost > allowance {
                f64::INFINITY
            } else {
                cost
            })
        },
    )
}

fn dtw_core<F>(
    row_count: usize,
    column_count: usize,
    window: Option<usize>,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
    mut local_cost: F,
) -> CoreResult<f64>
where
    F: FnMut(usize, usize, f64) -> CoreResult<f64>,
{
    let length_difference = row_count - column_count;
    let effective_window = window
        .unwrap_or(row_count.max(column_count))
        .max(length_difference);

    let (mut previous, mut current) = scratch.rolling_rows(column_count + 1, f64::INFINITY);
    previous[0] = 0.0;

    for row in 1..=row_count {
        current.fill(f64::INFINITY);
        let start = 1.max(row.saturating_sub(effective_window));
        let end = column_count.min(row.saturating_add(effective_window));
        let mut row_minimum = f64::INFINITY;

        for column in start..=end {
            let predecessor = previous[column]
                .min(current[column - 1])
                .min(previous[column - 1]);
            let limit = match cutoff {
                DistanceCutoff::None => f64::INFINITY,
                DistanceCutoff::Finite(limit) => limit,
            };
            if predecessor.is_infinite() || predecessor > limit {
                continue;
            }
            let local = local_cost(row - 1, column - 1, limit - predecessor)?;
            if local.is_infinite() {
                continue;
            }
            let value = local + predecessor;
            current[column] = value;
            row_minimum = row_minimum.min(value);
        }

        if cutoff.exceeded_by(row_minimum) {
            return Ok(f64::INFINITY);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    let result = previous[column_count];
    if cutoff.exceeded_by(result) {
        Ok(f64::INFINITY)
    } else {
        Ok(result)
    }
}

fn validate_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
) -> CoreResult<()> {
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "independent DTW requires matching storage dtypes",
        ));
    }
    if first.dimension_count() != second.dimension_count() {
        return Err(CoreError::shape_mismatch(format!(
            "independent DTW requires matching dimension counts; received {} and {}",
            first.dimension_count(),
            second.dimension_count()
        )));
    }
    Ok(())
}

fn validate_selected_dimensions(dimension_count: usize, selected: &[usize]) -> CoreResult<()> {
    if selected.is_empty() {
        return Err(CoreError::invalid_data(
            "independent DTW requires at least one selected dimension",
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

fn accumulate_channels<I>(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimensions: I,
    parameters: DtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64>
where
    I: IntoIterator<Item = usize>,
{
    let mut total = 0.0;
    for dimension in dimensions {
        let remaining = cutoff.remaining_after(total);
        let channel_cost = match (
            first.contiguous_dimension(dimension),
            second.contiguous_dimension(dimension),
        ) {
            (Some(left), Some(right)) => dtw(left, right, parameters, remaining, scratch)?,
            _ => dtw_strided_channel(
                first,
                second,
                dimension,
                parameters.window,
                remaining,
                scratch,
            )?,
        };

        if channel_cost.is_infinite() {
            return Ok(f64::INFINITY);
        }
        total += channel_cost;
        if cutoff.exceeded_by(total) {
            return Ok(f64::INFINITY);
        }
    }
    Ok(total)
}

/// Computes dependent multivariate DTW using one shared warping path.
pub fn dtw_dependent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    window: Option<usize>,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_independent(first, second)?;
    if let Some(selected) = selected {
        validate_selected_dimensions(first.dimension_count(), selected)?;
    }
    let (rows, columns) = if first.time_count() >= second.time_count() {
        (first, second)
    } else {
        (second, first)
    };
    let row_count = rows.time_count();
    let column_count = columns.time_count();
    let effective_window = window.unwrap_or(row_count).max(row_count - column_count);
    let limit = match cutoff {
        DistanceCutoff::None => f64::INFINITY,
        DistanceCutoff::Finite(limit) => limit,
    };
    let (mut previous, mut current) = scratch.rolling_rows(column_count + 1, f64::INFINITY);
    previous[0] = 0.0;

    for row in 1..=row_count {
        current.fill(f64::INFINITY);
        let start = 1.max(row.saturating_sub(effective_window));
        let end = column_count.min(row.saturating_add(effective_window));
        let mut row_minimum = f64::INFINITY;
        for column in start..=end {
            let predecessor = previous[column]
                .min(current[column - 1])
                .min(previous[column - 1]);
            if predecessor.is_infinite() || predecessor > limit {
                continue;
            }
            let allowance = limit - predecessor;
            let local =
                dependent_local_cost(rows, row - 1, columns, column - 1, selected, allowance)?;
            if local > allowance {
                continue;
            }
            let value = predecessor + local;
            current[column] = value;
            row_minimum = row_minimum.min(value);
        }
        if row_minimum.is_infinite() || row_minimum > limit {
            return Ok(f64::INFINITY);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let result = previous[column_count];
    Ok(if result > limit {
        f64::INFINITY
    } else {
        result
    })
}

fn dependent_local_cost(
    first: MultivariateSeriesView<'_>,
    first_time: usize,
    second: MultivariateSeriesView<'_>,
    second_time: usize,
    selected: Option<&[usize]>,
    allowance: f64,
) -> CoreResult<f64> {
    let mut total = 0.0;
    match selected {
        Some(selected) => {
            for &dimension in selected {
                total = dependent_dimension_cost(
                    first,
                    first_time,
                    second,
                    second_time,
                    dimension,
                    total,
                )?;
                if total > allowance {
                    break;
                }
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                total = dependent_dimension_cost(
                    first,
                    first_time,
                    second,
                    second_time,
                    dimension,
                    total,
                )?;
                if total > allowance {
                    break;
                }
            }
        }
    }
    Ok(total)
}

#[inline(always)]
fn dependent_dimension_cost(
    first: MultivariateSeriesView<'_>,
    first_time: usize,
    second: MultivariateSeriesView<'_>,
    second_time: usize,
    dimension: usize,
    total: f64,
) -> CoreResult<f64> {
    let left = first.get_f64(dimension, first_time).ok_or_else(|| {
        CoreError::invalid_data("validated first dependent-DTW index was unavailable")
    })?;
    let right = second.get_f64(dimension, second_time).ok_or_else(|| {
        CoreError::invalid_data("validated second dependent-DTW index was unavailable")
    })?;
    if !left.is_finite() || !right.is_finite() {
        return Err(CoreError::invalid_data(
            "dependent DTW requires finite numeric values",
        ));
    }
    let difference = left - right;
    Ok(difference.mul_add(difference, total))
}

#[cfg(test)]
mod tests {
    use super::{dtw, dtw_dependent, dtw_independent};
    use crate::{
        DistanceCutoff, DistanceScratch, DtwParameters, MultivariateSeriesView, NumericVectorView,
    };

    #[test]
    fn identical_univariate_series_have_zero_cost() {
        let values = [1.0_f32, 2.0, 3.0];
        assert_eq!(
            dtw(
                NumericVectorView::from_f32(&values),
                NumericVectorView::from_f32(&values),
                DtwParameters::default(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }

    #[test]
    fn univariate_dtw_supports_unequal_lengths() {
        assert_eq!(
            dtw(
                NumericVectorView::from_f64(&[1.0, 2.0]),
                NumericVectorView::from_f64(&[1.0, 1.0, 2.0]),
                DtwParameters { window: Some(0) },
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }

    #[test]
    fn independent_dtw_sums_selected_channel_costs() {
        let first = MultivariateSeriesView::contiguous_f64(&[1.0, 2.0, 10.0, 20.0], 2, 2)
            .expect("valid first series");
        let second =
            MultivariateSeriesView::contiguous_f64(&[1.0, 1.0, 2.0, 10.0, 10.0, 20.0], 2, 3)
                .expect("valid second series");
        assert_eq!(
            dtw_independent(
                first,
                second,
                Some(&[0, 1]),
                DtwParameters { window: Some(0) },
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }

    #[test]
    fn dependent_dtw_uses_one_shared_path() {
        let values =
            MultivariateSeriesView::contiguous_f64(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 2, 3)
                .expect("valid series");
        assert_eq!(
            dtw_dependent(
                values,
                values,
                None,
                None,
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }
}
