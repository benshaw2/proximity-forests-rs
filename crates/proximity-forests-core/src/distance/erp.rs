//! Edit distance with real penalty for univariate and independent series.

use crate::{
    CoreError, CoreResult, DistanceCutoff, DistanceScratch, ErpParameters, MultivariateSeriesView,
    NumericVectorView,
};

/// Computes squared-cost edit distance with real penalty.
///
/// # Errors
///
/// Returns an error for empty inputs, mismatched dtypes, or invalid access.
pub fn erp(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    parameters: ErpParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    if first.is_empty() || second.is_empty() {
        return Err(CoreError::invalid_data("ERP requires two nonempty series"));
    }
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "ERP requires matching storage dtypes",
        ));
    }
    erp_core(
        first.len(),
        second.len(),
        parameters,
        cutoff,
        scratch,
        |index| vector_value(first, index),
        |index| vector_value(second, index),
    )
}

/// Computes independent multivariate ERP by summing selected-channel costs.
///
/// # Errors
///
/// Returns an error for incompatible inputs or selected dimensions.
pub fn erp_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: ErpParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_multivariate(first, second, selected)?;
    let mut total = 0.0;
    match selected {
        Some(selected) => {
            for &dimension in selected {
                let cost = erp_channel(
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
                let cost = erp_channel(
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

fn erp_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    parameters: ErpParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    erp_core(
        first.time_count(),
        second.time_count(),
        parameters,
        cutoff,
        scratch,
        |index| series_value(first, dimension, index),
        |index| series_value(second, dimension, index),
    )
}

fn erp_core<First, Second>(
    first_length: usize,
    second_length: usize,
    parameters: ErpParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
    mut first_value: First,
    mut second_value: Second,
) -> CoreResult<f64>
where
    First: FnMut(usize) -> CoreResult<f64>,
    Second: FnMut(usize) -> CoreResult<f64>,
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
    let gap = parameters.gap.get();
    let limit = match cutoff {
        DistanceCutoff::None => f64::INFINITY,
        DistanceCutoff::Finite(limit) => limit,
    };
    let (mut previous, mut current) = scratch.rolling_rows(column_count + 1, f64::INFINITY);
    previous[0] = 0.0;

    let initial_end = column_count.min(window);
    for column in 1..=initial_end {
        let value = if swapped {
            first_value(column - 1)?
        } else {
            second_value(column - 1)?
        };
        let difference = value - gap;
        let accumulated = difference.mul_add(difference, previous[column - 1]);
        previous[column] = if accumulated > limit {
            f64::INFINITY
        } else {
            accumulated
        };
    }

    for row in 1..=row_count {
        current.fill(f64::INFINITY);
        let row_value = if swapped {
            second_value(row - 1)?
        } else {
            first_value(row - 1)?
        };
        if row <= window && previous[0].is_finite() {
            let difference = row_value - gap;
            let accumulated = difference.mul_add(difference, previous[0]);
            if accumulated <= limit {
                current[0] = accumulated;
            }
        }

        let start = 1.max(row.saturating_sub(window));
        let end = column_count.min(row.saturating_add(window));
        let row_gap_difference = row_value - gap;
        let row_gap_cost = row_gap_difference * row_gap_difference;
        let mut row_minimum = current[0];

        for column in start..=end {
            let column_value = if swapped {
                first_value(column - 1)?
            } else {
                second_value(column - 1)?
            };
            let diagonal = previous[column - 1];
            let above = previous[column];
            let left = current[column - 1];
            if diagonal.is_infinite() && above.is_infinite() && left.is_infinite() {
                continue;
            }

            let match_difference = row_value - column_value;
            let match_local = match_difference * match_difference;
            let match_cost = if diagonal <= limit && match_local <= limit - diagonal {
                diagonal + match_local
            } else {
                f64::INFINITY
            };
            let delete_row = if above <= limit && row_gap_cost <= limit - above {
                above + row_gap_cost
            } else {
                f64::INFINITY
            };
            let column_gap_difference = column_value - gap;
            let column_gap_cost = column_gap_difference * column_gap_difference;
            let delete_column = if left <= limit && column_gap_cost <= limit - left {
                left + column_gap_cost
            } else {
                f64::INFINITY
            };
            let value = match_cost.min(delete_row).min(delete_column);
            if value.is_finite() {
                current[column] = value;
                row_minimum = row_minimum.min(value);
            }
        }

        if row_minimum.is_infinite() {
            return Ok(f64::INFINITY);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    Ok(previous[column_count])
}

fn validate_multivariate(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
) -> CoreResult<()> {
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "independent ERP requires matching storage dtypes",
        ));
    }
    if first.dimension_count() != second.dimension_count() {
        return Err(CoreError::shape_mismatch(
            "independent ERP requires matching dimension counts",
        ));
    }
    if let Some(selected) = selected {
        if selected.is_empty() || selected.iter().any(|&d| d >= first.dimension_count()) {
            return Err(CoreError::invalid_data(
                "ERP selected dimensions are empty or out of range",
            ));
        }
    }
    Ok(())
}

fn vector_value(series: NumericVectorView<'_>, index: usize) -> CoreResult<f64> {
    let value = series
        .get_f64(index)
        .ok_or_else(|| CoreError::invalid_data("validated ERP index was unavailable"))?;
    ensure_finite(value)
}

fn series_value(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    time: usize,
) -> CoreResult<f64> {
    let value = series.get_f64(dimension, time).ok_or_else(|| {
        CoreError::invalid_data("validated multivariate ERP index was unavailable")
    })?;
    ensure_finite(value)
}

#[inline]
fn ensure_finite(value: f64) -> CoreResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CoreError::invalid_data(
            "ERP requires finite numeric values",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{erp, erp_independent};
    use crate::{
        DistanceCutoff, DistanceScratch, ErpParameters, FiniteF64, MultivariateSeriesView,
        NumericVectorView,
    };

    fn parameters() -> ErpParameters {
        ErpParameters {
            window: None,
            gap: FiniteF64::new(0.0).expect("finite gap"),
        }
    }

    #[test]
    fn identical_series_have_zero_cost() {
        let values = [1.0_f64, 2.0, 3.0];
        assert_eq!(
            erp(
                NumericVectorView::from_f64(&values),
                NumericVectorView::from_f64(&values),
                parameters(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }

    #[test]
    fn independent_erp_sums_channels() {
        let first = MultivariateSeriesView::contiguous_f32(&[1.0, 2.0, 10.0, 20.0], 2, 2)
            .expect("valid series");
        assert_eq!(
            erp_independent(
                first,
                first,
                None,
                parameters(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }
}
