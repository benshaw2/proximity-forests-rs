//! Derivative transformation and derivative DTW variants.

use crate::{
    CoreError, CoreResult, DistanceCutoff, DistanceScratch, DtwParameters, MultivariateSeriesView,
    NumericVectorView,
};

/// Writes the Keogh-Pazzani derivative into caller-owned `f64` output.
///
/// Boundary behavior matches PFGAP: one point maps to zero, two points both
/// receive the observed slope, and longer series copy the nearest interior
/// derivative to each endpoint.
///
/// # Errors
///
/// Returns an error for empty input or an output length mismatch.
pub fn derivative_transform(input: NumericVectorView<'_>, output: &mut [f64]) -> CoreResult<()> {
    if input.is_empty() {
        return Err(CoreError::invalid_data(
            "derivative transformation requires a nonempty series",
        ));
    }
    if input.len() != output.len() {
        return Err(CoreError::shape_mismatch(format!(
            "derivative output length {} does not match input length {}",
            output.len(),
            input.len()
        )));
    }

    if input.len() == 1 {
        output[0] = 0.0;
        return Ok(());
    }
    if input.len() == 2 {
        let slope = value(input, 1)? - value(input, 0)?;
        output.fill(slope);
        return Ok(());
    }

    for index in 1..input.len() - 1 {
        let previous = value(input, index - 1)?;
        output[index] =
            0.5 * (value(input, index)? - previous + 0.5 * (value(input, index + 1)? - previous));
    }
    output[0] = output[1];
    output[input.len() - 1] = output[input.len() - 2];
    Ok(())
}

/// Computes derivative DTW for two univariate series.
///
/// # Errors
///
/// Returns an error for empty inputs, mismatched dtypes, or invalid access.
pub fn ddtw(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    parameters: DtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "DDTW requires matching storage dtypes",
        ));
    }

    let row_length = first.len().min(second.len()) + 1;
    let (first_derivative, second_derivative, previous, current) =
        scratch.transform_and_rows(first.len(), second.len(), row_length);
    derivative_transform(first, first_derivative)?;
    derivative_transform(second, second_derivative)?;
    dtw_f64_rows(
        first_derivative,
        second_derivative,
        parameters.window,
        cutoff,
        previous,
        current,
    )
}

/// Computes independent multivariate DDTW by transforming and aligning each
/// selected channel separately.
///
/// # Errors
///
/// Returns an error for incompatible dtypes or dimensions, or an invalid
/// selected dimension.
pub fn ddtw_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected_dimensions: Option<&[usize]>,
    parameters: DtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_multivariate(first, second)?;
    match selected_dimensions {
        Some(selected) => {
            validate_selected(first.dimension_count(), selected)?;
            sum_independent(
                first,
                second,
                selected.iter().copied(),
                parameters,
                cutoff,
                scratch,
            )
        }
        None => sum_independent(
            first,
            second,
            0..first.dimension_count(),
            parameters,
            cutoff,
            scratch,
        ),
    }
}

/// Computes dependent multivariate DDTW using one shared warping path.
///
/// Derivatives are evaluated on demand from the original series, avoiding a
/// per-call compact matrix allocation. The local channel sum abandons once it
/// exceeds the remaining cell allowance.
///
/// # Errors
///
/// Returns an error for incompatible dtypes or dimensions, or an invalid
/// selected dimension.
pub fn ddtw_dependent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected_dimensions: Option<&[usize]>,
    parameters: DtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_multivariate(first, second)?;
    if let Some(selected) = selected_dimensions {
        validate_selected(first.dimension_count(), selected)?;
    }

    let (rows, columns) = if first.time_count() >= second.time_count() {
        (first, second)
    } else {
        (second, first)
    };
    let row_count = rows.time_count();
    let column_count = columns.time_count();
    let window = parameters
        .window
        .unwrap_or(row_count)
        .max(row_count - column_count);
    let limit = match cutoff {
        DistanceCutoff::None => f64::INFINITY,
        DistanceCutoff::Finite(limit) => limit,
    };
    let (mut previous, mut current) = scratch.rolling_rows(column_count + 1, f64::INFINITY);
    previous[0] = 0.0;

    for row in 1..=row_count {
        current.fill(f64::INFINITY);
        let start = 1.max(row.saturating_sub(window));
        let end = column_count.min(row.saturating_add(window));
        let mut row_minimum = f64::INFINITY;
        for column in start..=end {
            let predecessor = previous[column]
                .min(current[column - 1])
                .min(previous[column - 1]);
            if predecessor.is_infinite() || predecessor > limit {
                continue;
            }
            let allowance = limit - predecessor;
            let local = dependent_derivative_cost(
                rows,
                row - 1,
                columns,
                column - 1,
                selected_dimensions,
                allowance,
            )?;
            if local > allowance {
                continue;
            }
            let accumulated = predecessor + local;
            if cutoff.exceeded_by(accumulated) {
                continue;
            }
            current[column] = accumulated;
            row_minimum = row_minimum.min(accumulated);
        }
        if row_minimum.is_infinite() {
            return Ok(f64::INFINITY);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    Ok(previous[column_count])
}

fn sum_independent<I>(
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
        let first_channel = first.contiguous_dimension(dimension).ok_or_else(|| {
            CoreError::unsupported("independent DDTW currently requires contiguous channel storage")
        })?;
        let second_channel = second.contiguous_dimension(dimension).ok_or_else(|| {
            CoreError::unsupported("independent DDTW currently requires contiguous channel storage")
        })?;
        let remaining = cutoff.remaining_after(total);
        let cost = ddtw(
            first_channel,
            second_channel,
            parameters,
            remaining,
            scratch,
        )?;
        if cost.is_infinite() {
            return Ok(f64::INFINITY);
        }
        total += cost;
        if cutoff.exceeded_by(total) {
            return Ok(f64::INFINITY);
        }
    }
    Ok(total)
}

fn dtw_f64_rows<'a>(
    first: &[f64],
    second: &[f64],
    window: Option<usize>,
    cutoff: DistanceCutoff,
    mut previous: &'a mut [f64],
    mut current: &'a mut [f64],
) -> CoreResult<f64> {
    let (rows, columns) = if first.len() >= second.len() {
        (first, second)
    } else {
        (second, first)
    };
    if previous.len() != columns.len() + 1 || current.len() != columns.len() + 1 {
        return Err(CoreError::shape_mismatch(
            "DDTW rolling-row scratch has an invalid length",
        ));
    }
    previous.fill(f64::INFINITY);
    current.fill(f64::INFINITY);
    previous[0] = 0.0;
    let limit = match cutoff {
        DistanceCutoff::None => f64::INFINITY,
        DistanceCutoff::Finite(limit) => limit,
    };
    let effective_window = window.unwrap_or(rows.len()).max(rows.len() - columns.len());

    for row in 1..=rows.len() {
        current.fill(f64::INFINITY);
        let start = 1.max(row.saturating_sub(effective_window));
        let end = columns.len().min(row.saturating_add(effective_window));
        let mut row_minimum = f64::INFINITY;
        for column in start..=end {
            let predecessor = previous[column]
                .min(current[column - 1])
                .min(previous[column - 1]);
            if predecessor.is_infinite() || predecessor > limit {
                continue;
            }
            let difference = rows[row - 1] - columns[column - 1];
            let local = difference * difference;
            if local > limit - predecessor {
                continue;
            }
            let accumulated = predecessor + local;
            current[column] = accumulated;
            row_minimum = row_minimum.min(accumulated);
        }
        if row_minimum.is_infinite() {
            return Ok(f64::INFINITY);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    Ok(previous[columns.len()])
}

fn dependent_derivative_cost(
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
                let difference = derivative_at(first, dimension, first_time)?
                    - derivative_at(second, dimension, second_time)?;
                total = difference.mul_add(difference, total);
                if total > allowance {
                    break;
                }
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                let difference = derivative_at(first, dimension, first_time)?
                    - derivative_at(second, dimension, second_time)?;
                total = difference.mul_add(difference, total);
                if total > allowance {
                    break;
                }
            }
        }
    }
    Ok(total)
}

fn derivative_at(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    time: usize,
) -> CoreResult<f64> {
    let length = series.time_count();
    if length == 1 {
        return Ok(0.0);
    }
    if length == 2 {
        return Ok(series_value(series, dimension, 1)? - series_value(series, dimension, 0)?);
    }
    let interior = time.clamp(1, length - 2);
    let previous = series_value(series, dimension, interior - 1)?;
    Ok(0.5
        * (series_value(series, dimension, interior)? - previous
            + 0.5 * (series_value(series, dimension, interior + 1)? - previous)))
}

fn value(series: NumericVectorView<'_>, index: usize) -> CoreResult<f64> {
    let value = series
        .get_f64(index)
        .ok_or_else(|| CoreError::invalid_data("validated derivative index was unavailable"))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CoreError::invalid_data(
            "derivative distances require finite numeric values",
        ))
    }
}

fn series_value(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    time: usize,
) -> CoreResult<f64> {
    let value = series.get_f64(dimension, time).ok_or_else(|| {
        CoreError::invalid_data("validated derivative-series index was unavailable")
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CoreError::invalid_data(
            "derivative distances require finite numeric values",
        ))
    }
}

fn validate_multivariate(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
) -> CoreResult<()> {
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "multivariate DDTW requires matching storage dtypes",
        ));
    }
    if first.dimension_count() != second.dimension_count() {
        return Err(CoreError::shape_mismatch(
            "multivariate DDTW requires matching dimension counts",
        ));
    }
    Ok(())
}

fn validate_selected(dimension_count: usize, selected: &[usize]) -> CoreResult<()> {
    if selected.is_empty() {
        return Err(CoreError::invalid_data(
            "multivariate DDTW requires at least one selected dimension",
        ));
    }
    if selected
        .iter()
        .any(|&dimension| dimension >= dimension_count)
    {
        return Err(CoreError::invalid_data(
            "a selected DDTW dimension is outside the available range",
        ));
    }
    Ok(())
}
