//! Move-split-merge distance for univariate and independent series.

use crate::{
    CoreError, CoreResult, DistanceCutoff, DistanceScratch, MsmParameters, MultivariateSeriesView,
    NumericVectorView,
};

/// Computes the ordinary move-split-merge metric cost.
///
/// # Errors
///
/// Returns an error for empty inputs, mismatched dtypes, a nonpositive operation
/// cost, or invalid internal access.
pub fn msm(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    parameters: MsmParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_cost(parameters)?;
    if first.is_empty() || second.is_empty() {
        return Err(CoreError::invalid_data("MSM requires two nonempty series"));
    }
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "MSM requires matching storage dtypes",
        ));
    }
    msm_core(
        first.len(),
        second.len(),
        parameters.cost.get(),
        cutoff,
        scratch,
        |index| vector_value(first, index),
        |index| vector_value(second, index),
    )
}

/// Computes independent multivariate MSM by summing selected-channel costs.
///
/// # Errors
///
/// Returns an error for incompatible inputs, invalid selected dimensions, or
/// a nonpositive operation cost.
pub fn msm_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: MsmParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_cost(parameters)?;
    validate_multivariate(first, second, selected)?;
    let mut total = 0.0;
    match selected {
        Some(selected) => {
            for &dimension in selected {
                let value = msm_channel(
                    first,
                    second,
                    dimension,
                    parameters,
                    cutoff.remaining_after(total),
                    scratch,
                )?;
                if value.is_infinite() {
                    return Ok(f64::INFINITY);
                }
                total += value;
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                let value = msm_channel(
                    first,
                    second,
                    dimension,
                    parameters,
                    cutoff.remaining_after(total),
                    scratch,
                )?;
                if value.is_infinite() {
                    return Ok(f64::INFINITY);
                }
                total += value;
            }
        }
    }
    if cutoff.exceeded_by(total) {
        Ok(f64::INFINITY)
    } else {
        Ok(total)
    }
}

fn msm_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    parameters: MsmParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    msm_core(
        first.time_count(),
        second.time_count(),
        parameters.cost.get(),
        cutoff,
        scratch,
        |index| series_value(first, dimension, index),
        |index| series_value(second, dimension, index),
    )
}

fn msm_core<First, Second>(
    first_length: usize,
    second_length: usize,
    operation_cost: f64,
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
    let limit = match cutoff {
        DistanceCutoff::None => f64::INFINITY,
        DistanceCutoff::Finite(limit) => limit,
    };
    let (mut previous, mut current) = scratch.rolling_rows(column_count, f64::INFINITY);
    let row_zero = if swapped {
        second_value(0)?
    } else {
        first_value(0)?
    };
    let column_zero = if swapped {
        first_value(0)?
    } else {
        second_value(0)?
    };
    previous[0] = (row_zero - column_zero).abs();

    for column in 1..column_count {
        let value = if swapped {
            first_value(column)?
        } else {
            second_value(column)?
        };
        let prior = if swapped {
            first_value(column - 1)?
        } else {
            second_value(column - 1)?
        };
        let accumulated =
            previous[column - 1] + split_merge_cost(value, row_zero, prior, operation_cost);
        previous[column] = if accumulated > limit {
            f64::INFINITY
        } else {
            accumulated
        };
    }
    if previous.iter().all(|value| value.is_infinite()) {
        return Ok(f64::INFINITY);
    }

    for row in 1..row_count {
        let row_value = if swapped {
            second_value(row)?
        } else {
            first_value(row)?
        };
        let prior_row = if swapped {
            second_value(row - 1)?
        } else {
            first_value(row - 1)?
        };
        let first_column = if swapped {
            first_value(0)?
        } else {
            second_value(0)?
        };
        let edge =
            previous[0] + split_merge_cost(row_value, prior_row, first_column, operation_cost);
        current[0] = if edge > limit { f64::INFINITY } else { edge };
        let mut row_minimum = current[0];

        for column in 1..column_count {
            let column_value = if swapped {
                first_value(column)?
            } else {
                second_value(column)?
            };
            let prior_column = if swapped {
                first_value(column - 1)?
            } else {
                second_value(column - 1)?
            };
            let diagonal = previous[column - 1];
            let above = previous[column];
            let left = current[column - 1];
            if diagonal.is_infinite() && above.is_infinite() && left.is_infinite() {
                current[column] = f64::INFINITY;
                continue;
            }

            let movement_local = (row_value - column_value).abs();
            let movement = if diagonal <= limit && movement_local <= limit - diagonal {
                diagonal + movement_local
            } else {
                f64::INFINITY
            };
            let merge_local = split_merge_cost(row_value, prior_row, column_value, operation_cost);
            let merge = if above <= limit && merge_local <= limit - above {
                above + merge_local
            } else {
                f64::INFINITY
            };
            let split_local =
                split_merge_cost(column_value, row_value, prior_column, operation_cost);
            let split = if left <= limit && split_local <= limit - left {
                left + split_local
            } else {
                f64::INFINITY
            };
            current[column] = movement.min(merge).min(split);
            row_minimum = row_minimum.min(current[column]);
        }
        if row_minimum.is_infinite() {
            return Ok(f64::INFINITY);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    Ok(previous[column_count - 1])
}

fn split_merge_cost(new_point: f64, first: f64, second: f64, cost: f64) -> f64 {
    if (first <= new_point && new_point <= second) || (second <= new_point && new_point <= first) {
        cost
    } else {
        cost + (new_point - first).abs().min((new_point - second).abs())
    }
}

fn validate_cost(parameters: MsmParameters) -> CoreResult<()> {
    if parameters.cost.get() <= 0.0 {
        return Err(CoreError::invalid_configuration(
            "msm_cost",
            "MSM operation cost must be positive",
        ));
    }
    Ok(())
}

fn validate_multivariate(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
) -> CoreResult<()> {
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "independent MSM requires matching storage dtypes",
        ));
    }
    if first.dimension_count() != second.dimension_count() {
        return Err(CoreError::shape_mismatch(
            "independent MSM requires matching dimension counts",
        ));
    }
    if let Some(selected) = selected {
        if selected.is_empty() || selected.iter().any(|&d| d >= first.dimension_count()) {
            return Err(CoreError::invalid_data(
                "MSM selected dimensions are empty or out of range",
            ));
        }
    }
    Ok(())
}

fn vector_value(series: NumericVectorView<'_>, index: usize) -> CoreResult<f64> {
    let value = series
        .get_f64(index)
        .ok_or_else(|| CoreError::invalid_data("validated MSM index was unavailable"))?;
    finite_value(value)
}

fn series_value(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    time: usize,
) -> CoreResult<f64> {
    let value = series.get_f64(dimension, time).ok_or_else(|| {
        CoreError::invalid_data("validated multivariate MSM index was unavailable")
    })?;
    finite_value(value)
}

#[inline]
fn finite_value(value: f64) -> CoreResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CoreError::invalid_data(
            "MSM requires finite numeric values",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{msm, msm_independent};
    use crate::{
        DistanceCutoff, DistanceScratch, FiniteF64, MsmParameters, MultivariateSeriesView,
        NumericVectorView,
    };

    fn parameters() -> MsmParameters {
        MsmParameters {
            cost: FiniteF64::new(0.1).expect("finite cost"),
        }
    }

    #[test]
    fn identical_series_have_zero_cost() {
        let values = [1.0_f64, 2.0, 3.0];
        assert_eq!(
            msm(
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
    fn independent_msm_sums_channels() {
        let values = MultivariateSeriesView::contiguous_f32(&[1.0, 2.0, 10.0, 20.0], 2, 2)
            .expect("valid series");
        assert_eq!(
            msm_independent(
                values,
                values,
                None,
                parameters(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }
}
