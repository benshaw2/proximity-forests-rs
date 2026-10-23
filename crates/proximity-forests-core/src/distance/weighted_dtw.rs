//! Weighted DTW and weighted derivative DTW variants.

use crate::{
    CoreError, CoreResult, DistanceCutoff, DistanceScratch, MultivariateSeriesView,
    NumericVectorView, WeightedDtwParameters,
};

/// Computes weighted DTW for two univariate series.
///
/// # Errors
///
/// Returns an error for empty inputs, mismatched dtypes, or invalid access.
pub fn wdtw(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    parameters: WeightedDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    if first.is_empty() || second.is_empty() {
        return Err(CoreError::invalid_data("WDTW requires nonempty series"));
    }
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch("WDTW requires matching dtypes"));
    }
    weighted_core(
        first.len(),
        second.len(),
        parameters,
        cutoff,
        scratch,
        |row, column, _allowance| {
            let difference = vector_value(first, row)? - vector_value(second, column)?;
            Ok(difference * difference)
        },
    )
}

/// Computes independent multivariate WDTW.
///
/// # Errors
///
/// Returns an error for incompatible inputs or selected dimensions.
pub fn wdtw_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: WeightedDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_multivariate(first, second, selected)?;
    sum_channels(first, second, selected, cutoff, |dimension, remaining| {
        weighted_channel(
            first, second, dimension, parameters, remaining, scratch, false,
        )
    })
}

/// Computes dependent multivariate WDTW using one shared path.
///
/// # Errors
///
/// Returns an error for incompatible inputs or selected dimensions.
pub fn wdtw_dependent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: WeightedDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_multivariate(first, second, selected)?;
    weighted_dependent(first, second, selected, parameters, cutoff, scratch, false)
}

/// Computes univariate weighted derivative DTW.
///
/// # Errors
///
/// Returns an error for empty inputs, mismatched dtypes, or invalid access.
pub fn wddtw(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    parameters: WeightedDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    if first.is_empty() || second.is_empty() {
        return Err(CoreError::invalid_data("WDDTW requires nonempty series"));
    }
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch("WDDTW requires matching dtypes"));
    }
    weighted_core(
        first.len(),
        second.len(),
        parameters,
        cutoff,
        scratch,
        |row, column, _allowance| {
            let difference = vector_derivative(first, row)? - vector_derivative(second, column)?;
            Ok(difference * difference)
        },
    )
}

/// Computes independent multivariate WDDTW.
///
/// # Errors
///
/// Returns an error for incompatible inputs or selected dimensions.
pub fn wddtw_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: WeightedDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_multivariate(first, second, selected)?;
    sum_channels(first, second, selected, cutoff, |dimension, remaining| {
        weighted_channel(
            first, second, dimension, parameters, remaining, scratch, true,
        )
    })
}

/// Computes dependent multivariate WDDTW using one shared path.
///
/// # Errors
///
/// Returns an error for incompatible inputs or selected dimensions.
pub fn wddtw_dependent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: WeightedDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_multivariate(first, second, selected)?;
    weighted_dependent(first, second, selected, parameters, cutoff, scratch, true)
}

fn weighted_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    parameters: WeightedDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
    derivative: bool,
) -> CoreResult<f64> {
    weighted_core(
        first.time_count(),
        second.time_count(),
        parameters,
        cutoff,
        scratch,
        |row, column, _allowance| {
            let left = if derivative {
                series_derivative(first, dimension, row)?
            } else {
                series_value(first, dimension, row)?
            };
            let right = if derivative {
                series_derivative(second, dimension, column)?
            } else {
                series_value(second, dimension, column)?
            };
            let difference = left - right;
            Ok(difference * difference)
        },
    )
}

fn weighted_dependent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: WeightedDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
    derivative: bool,
) -> CoreResult<f64> {
    weighted_core(
        first.time_count(),
        second.time_count(),
        parameters,
        cutoff,
        scratch,
        |row, column, allowance| {
            let unweighted_allowance = allowance;
            let mut total = 0.0;
            match selected {
                Some(selected) => {
                    for &dimension in selected {
                        let difference = if derivative {
                            series_derivative(first, dimension, row)?
                                - series_derivative(second, dimension, column)?
                        } else {
                            series_value(first, dimension, row)?
                                - series_value(second, dimension, column)?
                        };
                        total = difference.mul_add(difference, total);
                        if total > unweighted_allowance {
                            break;
                        }
                    }
                }
                None => {
                    for dimension in 0..first.dimension_count() {
                        let difference = if derivative {
                            series_derivative(first, dimension, row)?
                                - series_derivative(second, dimension, column)?
                        } else {
                            series_value(first, dimension, row)?
                                - series_value(second, dimension, column)?
                        };
                        total = difference.mul_add(difference, total);
                        if total > unweighted_allowance {
                            break;
                        }
                    }
                }
            }
            Ok(total)
        },
    )
}

fn weighted_core<F>(
    first_length: usize,
    second_length: usize,
    parameters: WeightedDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
    mut unweighted_local: F,
) -> CoreResult<f64>
where
    F: FnMut(usize, usize, f64) -> CoreResult<f64>,
{
    let (row_count, column_count, swapped) = if first_length >= second_length {
        (first_length, second_length, false)
    } else {
        (second_length, first_length, true)
    };
    let window = parameters
        .window
        .unwrap_or(row_count)
        .max(row_count - column_count);
    let limit = match cutoff {
        DistanceCutoff::None => f64::INFINITY,
        DistanceCutoff::Finite(limit) => limit,
    };
    let (weights, mut previous, mut current) = scratch.weighted_rows(row_count, column_count + 1);
    for (displacement, weight) in weights.iter_mut().enumerate() {
        *weight = logistic_weight(displacement, row_count, parameters.stiffness.get());
    }
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
            let weight = weights[row.abs_diff(column)];
            let weighted_allowance = limit - predecessor;
            let allowance = if weight == 0.0 {
                f64::INFINITY
            } else {
                weighted_allowance / weight
            };
            let (first_index, second_index) = if swapped {
                (column - 1, row - 1)
            } else {
                (row - 1, column - 1)
            };
            let unweighted = unweighted_local(first_index, second_index, allowance)?;
            if unweighted > allowance {
                continue;
            }
            let local = weight * unweighted;
            let accumulated = predecessor + local;
            if accumulated > limit {
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

fn sum_channels<F>(
    first: MultivariateSeriesView<'_>,
    _second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    cutoff: DistanceCutoff,
    mut channel_cost: F,
) -> CoreResult<f64>
where
    F: FnMut(usize, DistanceCutoff) -> CoreResult<f64>,
{
    let mut total = 0.0;
    match selected {
        Some(selected) => {
            for &dimension in selected {
                let cost = channel_cost(dimension, cutoff.remaining_after(total))?;
                if cost.is_infinite() {
                    return Ok(f64::INFINITY);
                }
                total += cost;
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                let cost = channel_cost(dimension, cutoff.remaining_after(total))?;
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

fn logistic_weight(displacement: usize, maximum_length: usize, stiffness: f64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let centered = displacement as f64 - maximum_length as f64 / 2.0;
    1.0 / (1.0 + (-stiffness * centered).exp())
}

fn validate_multivariate(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
) -> CoreResult<()> {
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "multivariate weighted DTW requires matching dtypes",
        ));
    }
    if first.dimension_count() != second.dimension_count() {
        return Err(CoreError::shape_mismatch(
            "multivariate weighted DTW requires matching dimension counts",
        ));
    }
    if let Some(selected) = selected {
        if selected.is_empty() || selected.iter().any(|&d| d >= first.dimension_count()) {
            return Err(CoreError::invalid_data(
                "weighted DTW selected dimensions are empty or out of range",
            ));
        }
    }
    Ok(())
}

fn vector_value(series: NumericVectorView<'_>, index: usize) -> CoreResult<f64> {
    let value = series
        .get_f64(index)
        .ok_or_else(|| CoreError::invalid_data("validated WDTW index was unavailable"))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CoreError::invalid_data(
            "weighted DTW requires finite numeric values",
        ))
    }
}

fn vector_derivative(series: NumericVectorView<'_>, time: usize) -> CoreResult<f64> {
    derivative_from_length(series.len(), time, |index| vector_value(series, index))
}

fn series_value(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    time: usize,
) -> CoreResult<f64> {
    let value = series.get_f64(dimension, time).ok_or_else(|| {
        CoreError::invalid_data("validated multivariate WDTW index was unavailable")
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CoreError::invalid_data(
            "weighted DTW requires finite numeric values",
        ))
    }
}

fn series_derivative(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    time: usize,
) -> CoreResult<f64> {
    derivative_from_length(series.time_count(), time, |index| {
        series_value(series, dimension, index)
    })
}

fn derivative_from_length<F>(length: usize, time: usize, mut value: F) -> CoreResult<f64>
where
    F: FnMut(usize) -> CoreResult<f64>,
{
    if length == 1 {
        return Ok(0.0);
    }
    if length == 2 {
        return Ok(value(1)? - value(0)?);
    }
    let interior = time.clamp(1, length - 2);
    let previous = value(interior - 1)?;
    Ok(0.5 * (value(interior)? - previous + 0.5 * (value(interior + 1)? - previous)))
}

#[cfg(test)]
mod tests {
    use super::{wddtw, wdtw};
    use crate::{
        DistanceCutoff, DistanceScratch, FiniteF64, NumericVectorView, WeightedDtwParameters,
    };

    #[test]
    fn identical_series_have_zero_weighted_cost() {
        let values = [1.0_f64, 2.0, 3.0];
        let parameters = WeightedDtwParameters {
            window: None,
            stiffness: FiniteF64::new(0.05).expect("finite"),
        };
        assert_eq!(
            wdtw(
                NumericVectorView::from_f64(&values),
                NumericVectorView::from_f64(&values),
                parameters,
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
        assert_eq!(
            wddtw(
                NumericVectorView::from_f64(&values),
                NumericVectorView::from_f64(&values),
                parameters,
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }
}
