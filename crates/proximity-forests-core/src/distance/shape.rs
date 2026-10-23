//! Optimized histogram-of-gradients shape DTW distances.

use crate::{
    CoreError, CoreResult, DistanceCutoff, DistanceScratch, DtwParameters, MultivariateSeriesView,
    NumericVectorView, ShapeDtwParameters, dtw,
};

/// Computes one-dimensional histogram-of-gradients DTW.
///
/// Each input is differentiated exactly once conceptually. Histograms are
/// constructed directly from the source views in two passes, without allocating
/// intermediate derivative vectors.
///
/// # Errors
///
/// Returns an error for empty inputs, mismatched dtypes, or non-finite values.
pub fn shape_hog_1d_dtw(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    parameters: ShapeDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_vectors(first, second)?;
    let first_histogram = vector_histogram(first)?;
    let second_histogram = vector_histogram(second)?;
    histogram_dtw(
        &first_histogram,
        &second_histogram,
        parameters,
        cutoff,
        scratch,
    )
}

/// Computes independent multivariate one-dimensional ShapeHoG DTW.
///
/// Every selected channel is histogrammed and aligned independently. Channel
/// histograms are generated directly from the source view and released after
/// that channel is evaluated.
///
/// # Errors
///
/// Returns an error for incompatible inputs, invalid selected dimensions, or
/// non-finite values.
pub fn shape_hog_1d_dtw_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: ShapeDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_multivariate(first, second, selected)?;
    let mut total = 0.0;

    match selected {
        Some(selected) => {
            for &dimension in selected {
                let cost = shape_channel(
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
                let cost = shape_channel(
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

/// Computes dependent multivariate ShapeHoG DTW.
///
/// Selected channel derivatives share one global minimum and maximum and are
/// binned jointly in channel order. The implementation uses two direct passes
/// over the source view and never materializes the flattened gradient sequence.
///
/// # Errors
///
/// Returns an error for incompatible inputs, invalid selected dimensions, or
/// non-finite values.
pub fn shape_hog_dtw(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    parameters: ShapeDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    validate_multivariate(first, second, selected)?;
    let first_histogram = multivariate_histogram(first, selected)?;
    let second_histogram = multivariate_histogram(second, selected)?;
    histogram_dtw(
        &first_histogram,
        &second_histogram,
        parameters,
        cutoff,
        scratch,
    )
}

fn histogram_dtw(
    first: &[f64],
    second: &[f64],
    parameters: ShapeDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    dtw(
        NumericVectorView::from_f64(first),
        NumericVectorView::from_f64(second),
        DtwParameters {
            window: parameters.window,
        },
        cutoff,
        scratch,
    )
}

fn shape_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    parameters: ShapeDtwParameters,
    cutoff: DistanceCutoff,
    scratch: &mut DistanceScratch,
) -> CoreResult<f64> {
    let first_histogram = channel_histogram(first, dimension)?;
    let second_histogram = channel_histogram(second, dimension)?;
    histogram_dtw(
        &first_histogram,
        &second_histogram,
        parameters,
        cutoff,
        scratch,
    )
}

fn vector_histogram(series: NumericVectorView<'_>) -> CoreResult<Vec<f64>> {
    let bins = default_bin_count(series.len());
    let (minimum, maximum) = vector_gradient_range(series)?;
    let mut histogram = vec![0.0; bins];
    fill_vector_histogram(series, minimum, maximum, &mut histogram)?;
    Ok(histogram)
}

fn channel_histogram(series: MultivariateSeriesView<'_>, dimension: usize) -> CoreResult<Vec<f64>> {
    let bins = default_bin_count(series.time_count());
    let (minimum, maximum) = channel_gradient_range(series, dimension)?;
    let mut histogram = vec![0.0; bins];
    fill_channel_histogram(series, dimension, minimum, maximum, &mut histogram)?;
    Ok(histogram)
}

fn multivariate_histogram(
    series: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
) -> CoreResult<Vec<f64>> {
    let bins = default_bin_count(series.time_count());
    let (minimum, maximum, gradient_count) = multivariate_gradient_range(series, selected)?;
    let mut histogram = vec![0.0; bins];

    if minimum == maximum {
        #[allow(clippy::cast_precision_loss)]
        {
            histogram[0] = gradient_count as f64;
        }
        return Ok(histogram);
    }

    match selected {
        Some(selected) => {
            for &dimension in selected {
                fill_channel_histogram(series, dimension, minimum, maximum, &mut histogram)?;
            }
        }
        None => {
            for dimension in 0..series.dimension_count() {
                fill_channel_histogram(series, dimension, minimum, maximum, &mut histogram)?;
            }
        }
    }
    Ok(histogram)
}

fn vector_gradient_range(series: NumericVectorView<'_>) -> CoreResult<(f64, f64)> {
    let first = derivative_vector(series, 0)?;
    let mut minimum = first;
    let mut maximum = first;
    for time in 1..series.len() {
        let gradient = derivative_vector(series, time)?;
        minimum = minimum.min(gradient);
        maximum = maximum.max(gradient);
    }
    Ok((minimum, maximum))
}

fn channel_gradient_range(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
) -> CoreResult<(f64, f64)> {
    let first = derivative_channel(series, dimension, 0)?;
    let mut minimum = first;
    let mut maximum = first;
    for time in 1..series.time_count() {
        let gradient = derivative_channel(series, dimension, time)?;
        minimum = minimum.min(gradient);
        maximum = maximum.max(gradient);
    }
    Ok((minimum, maximum))
}

fn multivariate_gradient_range(
    series: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
) -> CoreResult<(f64, f64, usize)> {
    let first_dimension = selected.map_or(0, |dimensions| dimensions[0]);
    let first_gradient = derivative_channel(series, first_dimension, 0)?;
    let mut minimum = first_gradient;
    let mut maximum = first_gradient;
    let mut count = 0usize;

    match selected {
        Some(selected) => {
            for &dimension in selected {
                for time in 0..series.time_count() {
                    let gradient = derivative_channel(series, dimension, time)?;
                    minimum = minimum.min(gradient);
                    maximum = maximum.max(gradient);
                    count += 1;
                }
            }
        }
        None => {
            for dimension in 0..series.dimension_count() {
                for time in 0..series.time_count() {
                    let gradient = derivative_channel(series, dimension, time)?;
                    minimum = minimum.min(gradient);
                    maximum = maximum.max(gradient);
                    count += 1;
                }
            }
        }
    }
    Ok((minimum, maximum, count))
}

fn fill_vector_histogram(
    series: NumericVectorView<'_>,
    minimum: f64,
    maximum: f64,
    histogram: &mut [f64],
) -> CoreResult<()> {
    if minimum == maximum {
        #[allow(clippy::cast_precision_loss)]
        {
            histogram[0] += series.len() as f64;
        }
        return Ok(());
    }
    for time in 0..series.len() {
        let gradient = derivative_vector(series, time)?;
        histogram[bin_index(gradient, minimum, maximum, histogram.len())] += 1.0;
    }
    Ok(())
}

fn fill_channel_histogram(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    minimum: f64,
    maximum: f64,
    histogram: &mut [f64],
) -> CoreResult<()> {
    if minimum == maximum {
        #[allow(clippy::cast_precision_loss)]
        {
            histogram[0] += series.time_count() as f64;
        }
        return Ok(());
    }
    for time in 0..series.time_count() {
        let gradient = derivative_channel(series, dimension, time)?;
        histogram[bin_index(gradient, minimum, maximum, histogram.len())] += 1.0;
    }
    Ok(())
}

#[inline]
fn derivative_vector(series: NumericVectorView<'_>, time: usize) -> CoreResult<f64> {
    derivative_at(series.len(), time, |index| vector_value(series, index))
}

#[inline]
fn derivative_channel(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    time: usize,
) -> CoreResult<f64> {
    derivative_at(series.time_count(), time, |index| {
        series_value(series, dimension, index)
    })
}

fn derivative_at<Value>(length: usize, time: usize, mut value: Value) -> CoreResult<f64>
where
    Value: FnMut(usize) -> CoreResult<f64>,
{
    if length == 1 {
        value(0)?;
        return Ok(0.0);
    }
    if time == 0 {
        return Ok(value(1)? - value(0)?);
    }
    if time + 1 == length {
        return Ok(value(time)? - value(time - 1)?);
    }
    let previous = value(time - 1)?;
    let current = value(time)?;
    let next = value(time + 1)?;
    Ok(((current - previous) + (next - previous) / 2.0) / 2.0)
}

#[inline]
fn bin_index(value: f64, minimum: f64, maximum: f64, bins: usize) -> usize {
    #[allow(clippy::cast_precision_loss)]
    let scale = bins as f64 / (maximum - minimum);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let index = ((value - minimum) * scale) as usize;
    index.min(bins - 1)
}

fn vector_value(series: NumericVectorView<'_>, index: usize) -> CoreResult<f64> {
    let value = series.get_f64(index).ok_or_else(|| {
        CoreError::invalid_data("validated ShapeHoG vector index was unavailable")
    })?;
    ensure_finite(value)
}

fn series_value(
    series: MultivariateSeriesView<'_>,
    dimension: usize,
    time: usize,
) -> CoreResult<f64> {
    let value = series.get_f64(dimension, time).ok_or_else(|| {
        CoreError::invalid_data("validated ShapeHoG series index was unavailable")
    })?;
    ensure_finite(value)
}

#[inline]
fn ensure_finite(value: f64) -> CoreResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CoreError::invalid_data(
            "ShapeHoG DTW requires finite numeric values",
        ))
    }
}

fn default_bin_count(length: usize) -> usize {
    let mut root = 0usize;
    while root
        .saturating_add(1)
        .saturating_mul(root.saturating_add(1))
        <= length
    {
        root += 1;
    }
    root.max(1)
}

fn validate_vectors(first: NumericVectorView<'_>, second: NumericVectorView<'_>) -> CoreResult<()> {
    if first.is_empty() || second.is_empty() {
        return Err(CoreError::invalid_data(
            "one-dimensional ShapeHoG DTW requires nonempty series",
        ));
    }
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(
            "one-dimensional ShapeHoG DTW requires matching storage dtypes",
        ));
    }
    Ok(())
}

fn validate_multivariate(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
) -> CoreResult<()> {
    if first.time_count() == 0 || second.time_count() == 0 {
        return Err(CoreError::invalid_data(
            "multivariate ShapeHoG DTW requires nonempty time axes",
        ));
    }
    if first.scalar_kind() != second.scalar_kind()
        || first.dimension_count() != second.dimension_count()
    {
        return Err(CoreError::shape_mismatch(
            "multivariate ShapeHoG DTW requires matching dimensions and dtypes",
        ));
    }
    if let Some(selected) = selected {
        if selected.is_empty() {
            return Err(CoreError::invalid_data(
                "ShapeHoG requires at least one selected dimension",
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
    use super::{shape_hog_1d_dtw, shape_hog_1d_dtw_independent, shape_hog_dtw, vector_histogram};
    use crate::{
        DistanceCutoff, DistanceScratch, MultivariateSeriesView, NumericVectorView,
        ShapeDtwParameters,
    };

    #[test]
    fn constant_gradients_fill_the_first_bin() {
        assert_eq!(
            vector_histogram(NumericVectorView::from_f64(&[1.0, 3.0, 5.0, 7.0])),
            Ok(vec![4.0, 0.0])
        );
    }

    #[test]
    fn identical_univariate_series_have_zero_cost() {
        let values = [1.0_f64, 2.0, 4.0, 7.0];
        assert_eq!(
            shape_hog_1d_dtw(
                NumericVectorView::from_f64(&values),
                NumericVectorView::from_f64(&values),
                ShapeDtwParameters::default(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }

    #[test]
    fn identical_multivariate_series_have_zero_cost() {
        let values =
            MultivariateSeriesView::contiguous_f32(&[1.0, 2.0, 4.0, 10.0, 12.0, 15.0], 2, 3)
                .expect("valid series");
        assert_eq!(
            shape_hog_1d_dtw_independent(
                values,
                values,
                None,
                ShapeDtwParameters::default(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
        assert_eq!(
            shape_hog_dtw(
                values,
                values,
                None,
                ShapeDtwParameters::default(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }

    #[test]
    fn singleton_series_are_supported_without_special_buffers() {
        assert_eq!(
            shape_hog_1d_dtw(
                NumericVectorView::from_f64(&[2.0]),
                NumericVectorView::from_f64(&[9.0]),
                ShapeDtwParameters::default(),
                DistanceCutoff::None,
                &mut DistanceScratch::new(),
            ),
            Ok(0.0)
        );
    }
}
