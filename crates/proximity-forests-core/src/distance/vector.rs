//! Pointwise vector and independent multivariate distances.

use crate::{CoreError, CoreResult, DistanceCutoff, MultivariateSeriesView, NumericVectorView};

/// Computes Manhattan distance over all vector positions.
///
/// # Errors
///
/// Returns an error for unequal lengths or mismatched storage dtypes.
pub fn manhattan(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    validate_vectors(first, second, "Manhattan")?;
    match (first, second) {
        (NumericVectorView::F32(first), NumericVectorView::F32(second)) => {
            Ok(manhattan_f32(first, second, cutoff))
        }
        (NumericVectorView::F64(first), NumericVectorView::F64(second)) => {
            Ok(manhattan_f64(first, second, cutoff))
        }
        _ => Err(CoreError::shape_mismatch(
            "Manhattan distance requires matching storage dtypes",
        )),
    }
}

/// Computes Manhattan distance over selected vector positions.
///
/// # Errors
///
/// Returns an error for unequal lengths, mismatched storage dtypes, or an
/// empty or out-of-range selection.
pub fn manhattan_selected(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    selected: &[usize],
    cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    validate_vectors(first, second, "Manhattan")?;
    validate_selected(first.len(), selected, "Manhattan")?;
    match (first, second) {
        (NumericVectorView::F32(first), NumericVectorView::F32(second)) => {
            Ok(manhattan_selected_f32(first, second, selected, cutoff))
        }
        (NumericVectorView::F64(first), NumericVectorView::F64(second)) => {
            Ok(manhattan_selected_f64(first, second, selected, cutoff))
        }
        _ => Err(CoreError::shape_mismatch(
            "Manhattan distance requires matching storage dtypes",
        )),
    }
}

/// Computes independent multivariate Manhattan distance by summing complete
/// selected-channel costs.
///
/// # Errors
///
/// Returns an error for incompatible shapes or dtypes, invalid selected
/// dimensions, or unequal channel lengths.
pub fn manhattan_independent(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    validate_multivariate(first, second, selected, "Manhattan")?;
    let mut total = 0.0;

    match selected {
        Some(selected) => {
            for &dimension in selected {
                total +=
                    manhattan_channel(first, second, dimension, cutoff.remaining_after(total))?;
                if cutoff.exceeded_by(total) {
                    return Ok(total);
                }
            }
        }
        None => {
            for dimension in 0..first.dimension_count() {
                total +=
                    manhattan_channel(first, second, dimension, cutoff.remaining_after(total))?;
                if cutoff.exceeded_by(total) {
                    return Ok(total);
                }
            }
        }
    }

    Ok(total)
}

/// Computes cosine distance over all vector positions.
///
/// The cutoff is accepted for evaluator compatibility but cannot safely support
/// partial early abandoning because partial dot products and norms do not form
/// a monotone lower bound on final cosine distance.
///
/// # Errors
///
/// Returns an error for unequal lengths or mismatched storage dtypes.
pub fn cosine(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    _cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    validate_vectors(first, second, "cosine")?;
    let accumulated = match (first, second) {
        (NumericVectorView::F32(first), NumericVectorView::F32(second)) => {
            cosine_f32(first, second)
        }
        (NumericVectorView::F64(first), NumericVectorView::F64(second)) => {
            cosine_f64(first, second)
        }
        _ => {
            return Err(CoreError::shape_mismatch(
                "cosine distance requires matching storage dtypes",
            ));
        }
    };
    Ok(finish_cosine(accumulated))
}

/// Computes cosine distance over selected vector positions.
///
/// # Errors
///
/// Returns an error for unequal lengths, mismatched storage dtypes, or an
/// empty or out-of-range selection.
pub fn cosine_selected(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    selected: &[usize],
    _cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    validate_vectors(first, second, "cosine")?;
    validate_selected(first.len(), selected, "cosine")?;
    let accumulated = match (first, second) {
        (NumericVectorView::F32(first), NumericVectorView::F32(second)) => {
            cosine_selected_f32(first, second, selected)
        }
        (NumericVectorView::F64(first), NumericVectorView::F64(second)) => {
            cosine_selected_f64(first, second, selected)
        }
        _ => {
            return Err(CoreError::shape_mismatch(
                "cosine distance requires matching storage dtypes",
            ));
        }
    };
    Ok(finish_cosine(accumulated))
}

#[inline]
fn manhattan_f32(first: &[f32], second: &[f32], cutoff: DistanceCutoff) -> f64 {
    match cutoff {
        DistanceCutoff::None => manhattan_complete_f32(first, second),
        DistanceCutoff::Finite(limit) => {
            let mut total = 0.0;
            for index in 0..first.len() {
                total += (f64::from(first[index]) - f64::from(second[index])).abs();
                if total > limit {
                    break;
                }
            }
            total
        }
    }
}

#[inline]
fn manhattan_f64(first: &[f64], second: &[f64], cutoff: DistanceCutoff) -> f64 {
    match cutoff {
        DistanceCutoff::None => manhattan_complete_f64(first, second),
        DistanceCutoff::Finite(limit) => {
            let mut total = 0.0;
            for index in 0..first.len() {
                total += (first[index] - second[index]).abs();
                if total > limit {
                    break;
                }
            }
            total
        }
    }
}

#[inline]
fn manhattan_complete_f32(first: &[f32], second: &[f32]) -> f64 {
    let mut total = 0.0;
    let mut index = 0usize;
    let block_end = first.len() - first.len() % 4;

    while index < block_end {
        total += (f64::from(first[index]) - f64::from(second[index])).abs();
        total += (f64::from(first[index + 1]) - f64::from(second[index + 1])).abs();
        total += (f64::from(first[index + 2]) - f64::from(second[index + 2])).abs();
        total += (f64::from(first[index + 3]) - f64::from(second[index + 3])).abs();
        index += 4;
    }
    while index < first.len() {
        total += (f64::from(first[index]) - f64::from(second[index])).abs();
        index += 1;
    }
    total
}

#[inline]
fn manhattan_complete_f64(first: &[f64], second: &[f64]) -> f64 {
    let mut total = 0.0;
    let mut index = 0usize;
    let block_end = first.len() - first.len() % 4;

    while index < block_end {
        total += (first[index] - second[index]).abs();
        total += (first[index + 1] - second[index + 1]).abs();
        total += (first[index + 2] - second[index + 2]).abs();
        total += (first[index + 3] - second[index + 3]).abs();
        index += 4;
    }
    while index < first.len() {
        total += (first[index] - second[index]).abs();
        index += 1;
    }
    total
}

fn manhattan_selected_f32(
    first: &[f32],
    second: &[f32],
    selected: &[usize],
    cutoff: DistanceCutoff,
) -> f64 {
    let mut total = 0.0;
    for &index in selected {
        total += (f64::from(first[index]) - f64::from(second[index])).abs();
        if cutoff.exceeded_by(total) {
            break;
        }
    }
    total
}

fn manhattan_selected_f64(
    first: &[f64],
    second: &[f64],
    selected: &[usize],
    cutoff: DistanceCutoff,
) -> f64 {
    let mut total = 0.0;
    for &index in selected {
        total += (first[index] - second[index]).abs();
        if cutoff.exceeded_by(total) {
            break;
        }
    }
    total
}

fn manhattan_channel(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    dimension: usize,
    cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    match (
        first.contiguous_dimension(dimension),
        second.contiguous_dimension(dimension),
    ) {
        (Some(left), Some(right)) => manhattan(left, right, cutoff),
        _ => {
            let mut total = 0.0;
            for time in 0..first.time_count() {
                let left = first.get_f64(dimension, time).ok_or_else(|| {
                    CoreError::invalid_data("validated first Manhattan index was unavailable")
                })?;
                let right = second.get_f64(dimension, time).ok_or_else(|| {
                    CoreError::invalid_data("validated second Manhattan index was unavailable")
                })?;
                total += (left - right).abs();
                if cutoff.exceeded_by(total) {
                    break;
                }
            }
            Ok(total)
        }
    }
}

#[derive(Clone, Copy)]
struct CosineAccumulation {
    dot: f64,
    first_norm: f64,
    second_norm: f64,
}

#[inline]
fn cosine_f32(first: &[f32], second: &[f32]) -> CosineAccumulation {
    let mut result = CosineAccumulation {
        dot: 0.0,
        first_norm: 0.0,
        second_norm: 0.0,
    };
    let mut index = 0usize;
    let block_end = first.len() - first.len() % 4;

    while index < block_end {
        accumulate_cosine(
            &mut result,
            f64::from(first[index]),
            f64::from(second[index]),
        );
        accumulate_cosine(
            &mut result,
            f64::from(first[index + 1]),
            f64::from(second[index + 1]),
        );
        accumulate_cosine(
            &mut result,
            f64::from(first[index + 2]),
            f64::from(second[index + 2]),
        );
        accumulate_cosine(
            &mut result,
            f64::from(first[index + 3]),
            f64::from(second[index + 3]),
        );
        index += 4;
    }
    while index < first.len() {
        accumulate_cosine(
            &mut result,
            f64::from(first[index]),
            f64::from(second[index]),
        );
        index += 1;
    }
    result
}

#[inline]
fn cosine_f64(first: &[f64], second: &[f64]) -> CosineAccumulation {
    let mut result = CosineAccumulation {
        dot: 0.0,
        first_norm: 0.0,
        second_norm: 0.0,
    };
    let mut index = 0usize;
    let block_end = first.len() - first.len() % 4;

    while index < block_end {
        accumulate_cosine(&mut result, first[index], second[index]);
        accumulate_cosine(&mut result, first[index + 1], second[index + 1]);
        accumulate_cosine(&mut result, first[index + 2], second[index + 2]);
        accumulate_cosine(&mut result, first[index + 3], second[index + 3]);
        index += 4;
    }
    while index < first.len() {
        accumulate_cosine(&mut result, first[index], second[index]);
        index += 1;
    }
    result
}

fn cosine_selected_f32(first: &[f32], second: &[f32], selected: &[usize]) -> CosineAccumulation {
    let mut result = CosineAccumulation {
        dot: 0.0,
        first_norm: 0.0,
        second_norm: 0.0,
    };
    for &index in selected {
        accumulate_cosine(
            &mut result,
            f64::from(first[index]),
            f64::from(second[index]),
        );
    }
    result
}

fn cosine_selected_f64(first: &[f64], second: &[f64], selected: &[usize]) -> CosineAccumulation {
    let mut result = CosineAccumulation {
        dot: 0.0,
        first_norm: 0.0,
        second_norm: 0.0,
    };
    for &index in selected {
        accumulate_cosine(&mut result, first[index], second[index]);
    }
    result
}

#[inline(always)]
fn accumulate_cosine(result: &mut CosineAccumulation, left: f64, right: f64) {
    result.dot = left.mul_add(right, result.dot);
    result.first_norm = left.mul_add(left, result.first_norm);
    result.second_norm = right.mul_add(right, result.second_norm);
}

fn finish_cosine(accumulated: CosineAccumulation) -> f64 {
    if accumulated.first_norm == 0.0 {
        return if accumulated.second_norm == 0.0 {
            0.0
        } else {
            1.0
        };
    }
    if accumulated.second_norm == 0.0 {
        return 1.0;
    }
    let similarity = (accumulated.dot / (accumulated.first_norm * accumulated.second_norm).sqrt())
        .clamp(-1.0, 1.0);
    1.0 - similarity
}

fn validate_vectors(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    name: &str,
) -> CoreResult<()> {
    if first.len() != second.len() {
        return Err(CoreError::shape_mismatch(format!(
            "{name} distance requires equal lengths; received {} and {}",
            first.len(),
            second.len()
        )));
    }
    Ok(())
}

fn validate_selected(length: usize, selected: &[usize], name: &str) -> CoreResult<()> {
    if selected.is_empty() {
        return Err(CoreError::invalid_data(format!(
            "{name} distance requires at least one selected position"
        )));
    }
    if selected.iter().any(|&index| index >= length) {
        return Err(CoreError::invalid_data(format!(
            "a selected {name} position is outside vector length {length}"
        )));
    }
    Ok(())
}

fn validate_multivariate(
    first: MultivariateSeriesView<'_>,
    second: MultivariateSeriesView<'_>,
    selected: Option<&[usize]>,
    name: &str,
) -> CoreResult<()> {
    if first.scalar_kind() != second.scalar_kind() {
        return Err(CoreError::shape_mismatch(format!(
            "independent {name} distance requires matching storage dtypes"
        )));
    }
    if first.dimension_count() != second.dimension_count()
        || first.time_count() != second.time_count()
    {
        return Err(CoreError::shape_mismatch(format!(
            "independent {name} distance requires matching rectangular shapes"
        )));
    }
    if let Some(selected) = selected {
        validate_selected(first.dimension_count(), selected, name)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{cosine, manhattan, manhattan_independent};
    use crate::{DistanceCutoff, MultivariateSeriesView, NumericVectorView};

    #[test]
    fn manhattan_supports_f32_and_cutoffs() {
        assert_eq!(
            manhattan(
                NumericVectorView::from_f32(&[1.0, 2.0, 3.0]),
                NumericVectorView::from_f32(&[2.0, 4.0, 6.0]),
                DistanceCutoff::None,
            ),
            Ok(6.0)
        );
        assert_eq!(
            manhattan(
                NumericVectorView::from_f64(&[1.0, 2.0, 3.0]),
                NumericVectorView::from_f64(&[2.0, 4.0, 6.0]),
                DistanceCutoff::Finite(2.0),
            ),
            Ok(3.0)
        );
    }

    #[test]
    fn independent_manhattan_sums_complete_channels() {
        let first = MultivariateSeriesView::contiguous_f64(&[1.0, 2.0, 10.0, 20.0], 2, 2)
            .expect("valid first series");
        let second = MultivariateSeriesView::contiguous_f64(&[2.0, 4.0, 11.0, 18.0], 2, 2)
            .expect("valid second series");
        assert_eq!(
            manhattan_independent(first, second, None, DistanceCutoff::None),
            Ok(6.0)
        );
    }

    #[test]
    fn cosine_handles_zero_vectors_explicitly() {
        assert_eq!(
            cosine(
                NumericVectorView::from_f64(&[0.0, 0.0]),
                NumericVectorView::from_f64(&[0.0, 0.0]),
                DistanceCutoff::None,
            ),
            Ok(0.0)
        );
        assert_eq!(
            cosine(
                NumericVectorView::from_f64(&[0.0, 0.0]),
                NumericVectorView::from_f64(&[1.0, 0.0]),
                DistanceCutoff::None,
            ),
            Ok(1.0)
        );
    }

    #[test]
    fn cosine_complete_paths_match_known_values() {
        assert_eq!(
            cosine(
                NumericVectorView::from_f32(&[1.0, 0.0]),
                NumericVectorView::from_f32(&[0.0, 1.0]),
                DistanceCutoff::None,
            ),
            Ok(1.0)
        );
        assert_eq!(
            cosine(
                NumericVectorView::from_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]),
                NumericVectorView::from_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]),
                DistanceCutoff::None,
            ),
            Ok(0.0)
        );
    }
}
