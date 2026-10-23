//! Longest common subsequence distance for univariate and independent series.

use crate::{
    CoreError, CoreResult, DistanceCutoff, DistanceScratch, LcssParameters, MultivariateSeriesView,
    NumericVectorView,
};

/// Computes constrained LCSS distance in the interval `[0, 1]`.
///
/// Values match when their absolute difference is at most `epsilon`. The
/// returned value is one minus the constrained LCSS length divided by the
/// shorter input length. The cutoff is checked only against the completed
/// distance because the maximizing recurrence has no additive monotone bound.
///
/// # Errors
///
/// Returns an error for empty inputs, mismatched dtypes, negative epsilon, or
/// invalid internal access.
pub fn lcss(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    parameters: LcssParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_parameters(parameters)?;
    if first.is_empty() || second.is_empty() {
        return Err(CoreError::invalid_data("LCSS requires two nonempty series"));
    }
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "LCSS requires matching storage dtypes",
        ));
    }

    let result = lcss_core(
        first.len(),
        second.len(),
        parameters,
        scratch,
        |index| vector_value(first, index),
        |index| vector_value(second, index),
    )?;
    if cutoff.exceeded_by(result) {
        Ok(f64::INFINITY)
    } else {
        Ok(result)
    }
}

/// Computes independent multivariate LCSS by summing selected-channel
/// distances.
///
/// # Errors
///
/// Returns an error for incompatible inputs, invalid selected dimensions, or
/// invalid LCSS parameters.
pub fn lcss_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: LcssParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_parameters(parameters)?;
    validate_multivariate(first, second, selected)?;

    let mut total = 0.0;
    match selected {
        Some(selected) => {
            for &dimension in selected {
                total += lcss_channel(first, second, dimension, parameters, scratch)?;
                if cutoff.exceeded_by(total) {
                    return Ok(f64::INFINITY);
                }
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                total += lcss_channel(first, second, dimension, parameters, scratch)?;
                if cutoff.exceeded_by(total) {
                    return Ok(f64::INFINITY);
                }
            }
        }
    }
    Ok(total)
}

fn lcss_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    parameters: LcssParameters,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    lcss_core(
        first.time_count(),
        second.time_count(),
        parameters,
        scratch,
        |index| series_value(first, dimension, index),
        |index| series_value(second, dimension, index),
    )
}

fn lcss_core<First, Second>(
    first_length: usize,
    second_length: usize,
    parameters: LcssParameters,
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
    let epsilon = parameters.epsilon.get();
    let (mut previous, mut current) = scratch.integer_rolling_rows(column_count + 1);

    for row in 1..=row_count {
        current.fill(0);
        let start = 1.max(row.saturating_sub(window));
        let end = column_count.min(row.saturating_add(window));
        let row_value = if swapped {
            second_value(row - 1)?
        } else {
            first_value(row - 1)?
        };
        ensure_finite(row_value)?;

        // Values before the active band retain the best prefix score rather
        // than resetting the left recurrence to zero at the band boundary.
        if start > 1 {
            current[start - 1] = previous[start - 1];
        }
        for column in start..=end {
            let column_value = if swapped {
                first_value(column - 1)?
            } else {
                second_value(column - 1)?
            };
            ensure_finite(column_value)?;
            current[column] = if (row_value - column_value).abs() <= epsilon {
                previous[column - 1] + 1
            } else {
                previous[column].max(current[column - 1])
            };
        }
        std::mem::swap(&mut previous, &mut current);
    }

    #[allow(clippy::cast_precision_loss)]
    let similarity = previous[column_count] as f64 / column_count as f64;
    Ok(1.0 - similarity)
}

fn validate_parameters(parameters: LcssParameters) -> CoreResult<()> {
    if parameters.epsilon.get() < 0.0 {
        return Err(CoreError::invalid_configuration(
            "lcss_epsilon",
            "LCSS epsilon must be nonnegative",
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
            "independent LCSS requires matching storage dtypes",
        ));
    }
    if first.dimension_count() != second.dimension_count() {
        return Err(CoreError::shape_mismatch(
            "independent LCSS requires matching dimension counts",
        ));
    }
    if let Some(selected) = selected {
        if selected.is_empty() || selected.iter().any(|&d| d >= first.dimension_count()) {
            return Err(CoreError::invalid_data(
                "LCSS selected dimensions are empty or out of range",
            ));
        }
    }
    Ok(())
}

fn vector_value(series: NumericVectorView<'_>, index: usize) -> CoreResult<f64> {
    series
        .get_f64(index)
        .ok_or_else(|| CoreError::invalid_data("validated LCSS index was unavailable"))
}

fn series_value(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    time: usize,
) -> CoreResult<f64> {
    series
        .get_f64(dimension, time)
        .ok_or_else(|| CoreError::invalid_data("validated multivariate LCSS index was unavailable"))
}

#[inline]
fn ensure_finite(value: f64) -> CoreResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(CoreError::invalid_data(
            "LCSS requires finite numeric values",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{lcss, lcss_independent};
    use crate::{
        DistanceCutoff, DistanceScratch, FiniteF64, LcssParameters, MultivariateSeriesView,
        NumericVectorView,
    };

    fn parameters() -> LcssParameters {
        LcssParameters {
            window: None,
            epsilon: FiniteF64::new(0.0).expect("finite epsilon"),
        }
    }

    #[test]
    fn identical_series_have_zero_distance() {
        let values = [1.0_f64, 2.0, 3.0];
        assert_eq!(
            lcss(
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
    fn independent_distance_sums_channels() {
        let values =
            MultivariateSeriesView::contiguous_f32(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 2, 3)
                .expect("valid series");
        assert_eq!(
            lcss_independent(
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
