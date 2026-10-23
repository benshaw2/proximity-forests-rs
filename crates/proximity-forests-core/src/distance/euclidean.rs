//! Optimized squared Euclidean distance for numeric vectors.

use crate::{CoreError, CoreResult, DistanceCutoff, NumericVectorView};

/// Computes squared Euclidean distance between equal-length vectors.
///
/// Both `f32` and `f64` storage accumulate into `f64`. Complete-distance and
/// finite-cutoff evaluations use separate kernels so the complete path has no
/// cutoff branch in its inner loop.
///
/// # Errors
///
/// Returns an error when vector lengths or storage dtypes differ.
pub fn squared_euclidean(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    validate_vectors(first, second, "squared Euclidean")?;

    match (first, second) {
        (NumericVectorView::F32(first), NumericVectorView::F32(second)) => {
            Ok(squared_f32(first, second, cutoff))
        }
        (NumericVectorView::F64(first), NumericVectorView::F64(second)) => {
            Ok(squared_f64(first, second, cutoff))
        }
        _ => Err(CoreError::shape_mismatch(
            "squared Euclidean distance requires matching storage dtypes",
        )),
    }
}

/// Computes squared Euclidean distance over selected vector positions.
///
/// Selected positions are borrowed and no sliced observation is allocated.
///
/// # Errors
///
/// Returns an error for mismatched dtypes, mismatched vector lengths, an empty
/// selection, or an out-of-bounds selected position.
pub fn squared_euclidean_selected(
    first: NumericVectorView<'_>,
    second: NumericVectorView<'_>,
    selected: &[usize],
    cutoff: DistanceCutoff,
) -> CoreResult<f64> {
    validate_vectors(first, second, "selected squared Euclidean")?;
    validate_selected(first.len(), selected)?;

    match (first, second) {
        (NumericVectorView::F32(first), NumericVectorView::F32(second)) => {
            Ok(squared_selected_f32(first, second, selected, cutoff))
        }
        (NumericVectorView::F64(first), NumericVectorView::F64(second)) => {
            Ok(squared_selected_f64(first, second, selected, cutoff))
        }
        _ => Err(CoreError::shape_mismatch(
            "selected squared Euclidean distance requires matching storage dtypes",
        )),
    }
}

#[inline]
fn squared_f32(first: &[f32], second: &[f32], cutoff: DistanceCutoff) -> f64 {
    match cutoff {
        DistanceCutoff::None => squared_complete_f32(first, second),
        DistanceCutoff::Finite(limit) => squared_cutoff_f32(first, second, limit),
    }
}

#[inline]
fn squared_f64(first: &[f64], second: &[f64], cutoff: DistanceCutoff) -> f64 {
    match cutoff {
        DistanceCutoff::None => squared_complete_f64(first, second),
        DistanceCutoff::Finite(limit) => squared_cutoff_f64(first, second, limit),
    }
}

/// Complete contiguous `f32` kernel with four independent accumulators.
///
/// Multiple accumulators shorten the floating-point dependency chain and make
/// this loop a stronger target for LLVM unrolling and vectorization.
#[inline]
fn squared_complete_f32(first: &[f32], second: &[f32]) -> f64 {
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut index = 0usize;
    let block_end = first.len() - first.len() % 4;

    while index < block_end {
        let difference0 = f64::from(first[index]) - f64::from(second[index]);
        let difference1 = f64::from(first[index + 1]) - f64::from(second[index + 1]);
        let difference2 = f64::from(first[index + 2]) - f64::from(second[index + 2]);
        let difference3 = f64::from(first[index + 3]) - f64::from(second[index + 3]);

        sum0 = difference0.mul_add(difference0, sum0);
        sum1 = difference1.mul_add(difference1, sum1);
        sum2 = difference2.mul_add(difference2, sum2);
        sum3 = difference3.mul_add(difference3, sum3);
        index += 4;
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);
    while index < first.len() {
        let difference = f64::from(first[index]) - f64::from(second[index]);
        total = difference.mul_add(difference, total);
        index += 1;
    }
    total
}

/// Complete contiguous `f64` kernel with four independent accumulators.
#[inline]
fn squared_complete_f64(first: &[f64], second: &[f64]) -> f64 {
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut index = 0usize;
    let block_end = first.len() - first.len() % 4;

    while index < block_end {
        let difference0 = first[index] - second[index];
        let difference1 = first[index + 1] - second[index + 1];
        let difference2 = first[index + 2] - second[index + 2];
        let difference3 = first[index + 3] - second[index + 3];

        sum0 = difference0.mul_add(difference0, sum0);
        sum1 = difference1.mul_add(difference1, sum1);
        sum2 = difference2.mul_add(difference2, sum2);
        sum3 = difference3.mul_add(difference3, sum3);
        index += 4;
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);
    while index < first.len() {
        let difference = first[index] - second[index];
        total = difference.mul_add(difference, total);
        index += 1;
    }
    total
}

#[inline]
fn squared_cutoff_f32(first: &[f32], second: &[f32], limit: f64) -> f64 {
    let mut total = 0.0;
    for index in 0..first.len() {
        let difference = f64::from(first[index]) - f64::from(second[index]);
        total = difference.mul_add(difference, total);
        if total > limit {
            return f64::INFINITY;
        }
    }
    total
}

#[inline]
fn squared_cutoff_f64(first: &[f64], second: &[f64], limit: f64) -> f64 {
    let mut total = 0.0;
    for index in 0..first.len() {
        let difference = first[index] - second[index];
        total = difference.mul_add(difference, total);
        if total > limit {
            return f64::INFINITY;
        }
    }
    total
}

#[inline]
fn squared_selected_f32(
    first: &[f32],
    second: &[f32],
    selected: &[usize],
    cutoff: DistanceCutoff,
) -> f64 {
    match cutoff {
        DistanceCutoff::None => {
            let mut total = 0.0;
            for &index in selected {
                let difference = f64::from(first[index]) - f64::from(second[index]);
                total = difference.mul_add(difference, total);
            }
            total
        }
        DistanceCutoff::Finite(limit) => {
            let mut total = 0.0;
            for &index in selected {
                let difference = f64::from(first[index]) - f64::from(second[index]);
                total = difference.mul_add(difference, total);
                if total > limit {
                    return f64::INFINITY;
                }
            }
            total
        }
    }
}

#[inline]
fn squared_selected_f64(
    first: &[f64],
    second: &[f64],
    selected: &[usize],
    cutoff: DistanceCutoff,
) -> f64 {
    match cutoff {
        DistanceCutoff::None => {
            let mut total = 0.0;
            for &index in selected {
                let difference = first[index] - second[index];
                total = difference.mul_add(difference, total);
            }
            total
        }
        DistanceCutoff::Finite(limit) => {
            let mut total = 0.0;
            for &index in selected {
                let difference = first[index] - second[index];
                total = difference.mul_add(difference, total);
                if total > limit {
                    return f64::INFINITY;
                }
            }
            total
        }
    }
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

fn validate_selected(length: usize, selected: &[usize]) -> CoreResult<()> {
    if selected.is_empty() {
        return Err(CoreError::invalid_data(
            "selected squared Euclidean distance requires at least one position",
        ));
    }
    if let Some(&invalid) = selected.iter().find(|&&index| index >= length) {
        return Err(CoreError::invalid_data(format!(
            "selected position {invalid} is outside vector length {length}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{squared_euclidean, squared_euclidean_selected};
    use crate::{DistanceCutoff, NumericVectorView};

    #[test]
    fn computes_f32_and_f64_without_converting_storage() {
        assert_eq!(
            squared_euclidean(
                NumericVectorView::from_f32(&[1.0, 2.0, 3.0]),
                NumericVectorView::from_f32(&[2.0, 4.0, 6.0]),
                DistanceCutoff::None,
            ),
            Ok(14.0)
        );
        assert_eq!(
            squared_euclidean(
                NumericVectorView::from_f64(&[1.0, 2.0, 3.0]),
                NumericVectorView::from_f64(&[2.0, 4.0, 6.0]),
                DistanceCutoff::None,
            ),
            Ok(14.0)
        );
    }

    #[test]
    fn complete_kernels_handle_non_multiple_of_four_lengths() {
        assert_eq!(
            squared_euclidean(
                NumericVectorView::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]),
                NumericVectorView::from_f32(&[2.0, 3.0, 4.0, 5.0, 6.0]),
                DistanceCutoff::None,
            ),
            Ok(5.0)
        );
        assert_eq!(
            squared_euclidean(
                NumericVectorView::from_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]),
                NumericVectorView::from_f64(&[2.0, 3.0, 4.0, 5.0, 6.0]),
                DistanceCutoff::None,
            ),
            Ok(5.0)
        );
    }

    #[test]
    fn cutoff_preserves_ties_and_abandons_only_above() {
        let first = NumericVectorView::from_f64(&[0.0, 0.0]);
        let second = NumericVectorView::from_f64(&[0.0, 2.0]);
        assert_eq!(
            squared_euclidean(
                first,
                second,
                DistanceCutoff::finite(4.0).expect("valid cutoff"),
            ),
            Ok(4.0)
        );
        assert_eq!(
            squared_euclidean(
                first,
                second,
                DistanceCutoff::finite(3.0).expect("valid cutoff"),
            ),
            Ok(f64::INFINITY)
        );
    }

    #[test]
    fn selected_positions_are_allocation_free_and_validated() {
        assert_eq!(
            squared_euclidean_selected(
                NumericVectorView::from_f32(&[1.0, 2.0, 3.0]),
                NumericVectorView::from_f32(&[2.0, 9.0, 5.0]),
                &[0, 2],
                DistanceCutoff::None,
            ),
            Ok(5.0)
        );
        assert!(
            squared_euclidean_selected(
                NumericVectorView::from_f64(&[1.0]),
                NumericVectorView::from_f64(&[1.0]),
                &[1],
                DistanceCutoff::None,
            )
            .is_err()
        );
        assert!(
            squared_euclidean_selected(
                NumericVectorView::from_f64(&[1.0]),
                NumericVectorView::from_f64(&[1.0]),
                &[],
                DistanceCutoff::None,
            )
            .is_err()
        );
    }
}
