//! Time warp edit distance for univariate and independent series.

use crate::{
    CoreError, CoreResult, DistanceCutoff, DistanceScratch, MultivariateSeriesView,
    NumericVectorView, TweParameters,
};

/// Computes time warp edit distance using implicit unit-spaced timestamps.
///
/// # Errors
///
/// Returns an error for empty inputs, mismatched dtypes, negative parameters,
/// or invalid internal access.
pub fn twe(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    parameters: TweParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_parameters(parameters)?;
    if first.is_empty() || second.is_empty() {
        return Err(CoreError::invalid_data("TWE requires two nonempty series"));
    }
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "TWE requires matching storage dtypes",
        ));
    }
    twe_core(
        first.len(),
        second.len(),
        parameters,
        cutoff,
        scratch,
        |index| vector_value(first, index),
        |index| vector_value(second, index),
    )
}

/// Computes independent multivariate TWE by summing selected-channel costs.
///
/// # Errors
///
/// Returns an error for incompatible inputs, invalid selected dimensions, or
/// invalid TWE parameters.
pub fn twe_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: TweParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_parameters(parameters)?;
    validate_multivariate(first, second, selected)?;
    let mut total = 0.0;
    match selected {
        Some(selected) => {
            for &dimension in selected {
                let value = twe_channel(
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
                let value = twe_channel(
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

fn twe_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    parameters: TweParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    twe_core(
        first.time_count(),
        second.time_count(),
        parameters,
        cutoff,
        scratch,
        |index| series_value(first, dimension, index),
        |index| series_value(second, dimension, index),
    )
}

fn twe_core<First, Second>(
    first_length: usize,
    second_length: usize,
    parameters: TweParameters,
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
    let stiffness = parameters.stiffness.get();
    let penalty = parameters.penalty.get();
    let deletion_base = penalty + stiffness;
    let limit = match cutoff {
        DistanceCutoff::None => f64::INFINITY,
        DistanceCutoff::Finite(limit) => limit,
    };
    let (mut previous, mut current) = scratch.rolling_rows(column_count + 1, f64::INFINITY);
    previous[0] = 0.0;

    for row in 1..=row_count {
        current.fill(f64::INFINITY);
        let row_value = if swapped {
            second_value(row - 1)?
        } else {
            first_value(row - 1)?
        };
        let prior_row = if row > 1 {
            if swapped {
                second_value(row - 2)?
            } else {
                first_value(row - 2)?
            }
        } else {
            0.0
        };
        let row_difference = row_value - prior_row;
        let row_deletion = row_difference.mul_add(row_difference, deletion_base);
        let mut row_minimum = f64::INFINITY;

        for column in 1..=column_count {
            let column_value = if swapped {
                first_value(column - 1)?
            } else {
                second_value(column - 1)?
            };
            let prior_column = if column > 1 {
                if swapped {
                    first_value(column - 2)?
                } else {
                    second_value(column - 2)?
                }
            } else {
                0.0
            };
            let diagonal = previous[column - 1];
            let above = previous[column];
            let left = current[column - 1];
            if diagonal.is_infinite() && above.is_infinite() && left.is_infinite() {
                continue;
            }

            let value_difference = row_value - column_value;
            let prior_difference = prior_row - prior_column;
            #[allow(clippy::cast_precision_loss)]
            let temporal =
                stiffness * (row.abs_diff(column) as f64 + (row - 1).abs_diff(column - 1) as f64);
            let matching_local = value_difference.mul_add(
                value_difference,
                prior_difference.mul_add(prior_difference, temporal),
            );
            let matching = if diagonal <= limit && matching_local <= limit - diagonal {
                diagonal + matching_local
            } else {
                f64::INFINITY
            };
            let delete_row = if above <= limit && row_deletion <= limit - above {
                above + row_deletion
            } else {
                f64::INFINITY
            };
            let column_difference = column_value - prior_column;
            let column_deletion = column_difference.mul_add(column_difference, deletion_base);
            let delete_column = if left <= limit && column_deletion <= limit - left {
                left + column_deletion
            } else {
                f64::INFINITY
            };
            let best = matching.min(delete_row).min(delete_column);
            if best.is_finite() {
                current[column] = best;
                row_minimum = row_minimum.min(best);
            }
        }
        if row_minimum.is_infinite() {
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

fn validate_parameters(parameters: TweParameters) -> CoreResult<()> {
    if parameters.stiffness.get() < 0.0 || parameters.penalty.get() < 0.0 {
        return Err(CoreError::invalid_configuration(
            "twe_parameters",
            "TWE stiffness and penalty must be nonnegative",
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
            "independent TWE requires matching storage dtypes",
        ));
    }
    if first.dimension_count() != second.dimension_count() {
        return Err(CoreError::shape_mismatch(
            "independent TWE requires matching dimension counts",
        ));
    }
    if let Some(selected) = selected {
        if selected.is_empty() || selected.iter().any(|&d| d >= first.dimension_count()) {
            return Err(CoreError::invalid_data(
                "TWE selected dimensions are empty or out of range",
            ));
        }
    }
    Ok(())
}

fn vector_value(series: NumericVectorView<'_>, index: usize) -> CoreResult<f64> {
    let value = series
        .get_f64(index)
        .ok_or_else(|| CoreError::invalid_data("validated TWE index was unavailable"))?;
    finite_value(value)
}

fn series_value(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    time: usize,
) -> CoreResult<f64> {
    let value = series.get_f64(dimension, time).ok_or_else(|| {
        CoreError::invalid_data("validated multivariate TWE index was unavailable")
    })?;
    finite_value(value)
}

#[inline]
fn finite_value(value: f64) -> CoreResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CoreError::invalid_data(
            "TWE requires finite numeric values",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{twe, twe_independent};
    use crate::{
        DistanceCutoff, DistanceScratch, FiniteF64, MultivariateSeriesView, NumericVectorView,
        TweParameters,
    };

    fn parameters() -> TweParameters {
        TweParameters {
            stiffness: FiniteF64::new(0.001).expect("finite stiffness"),
            penalty: FiniteF64::new(0.01).expect("finite penalty"),
        }
    }

    #[test]
    fn identical_series_have_zero_cost() {
        let values = [1.0_f64, 2.0, 3.0];
        assert_eq!(
            twe(
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
    fn independent_twe_sums_channels() {
        let values = MultivariateSeriesView::contiguous_f32(&[1.0, 2.0, 10.0, 20.0], 2, 2)
            .expect("valid series");
        assert_eq!(
            twe_independent(
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
