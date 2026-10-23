//! Numeric single- and multi-target regression storage.

use crate::{CoreError, CoreResult, ScalarKind};

/// Owned row-major regression targets preserving `f32` or `f64` storage.
#[derive(Clone, Debug, PartialEq)]
pub enum RegressionTargets {
    /// Single-precision target storage.
    F32 {
        /// Row-major target values.
        values: Vec<f32>,
        /// Number of samples.
        sample_count: usize,
        /// Number of outputs per sample.
        output_count: usize,
    },
    /// Double-precision target storage.
    F64 {
        /// Row-major target values.
        values: Vec<f64>,
        /// Number of samples.
        sample_count: usize,
        /// Number of outputs per sample.
        output_count: usize,
    },
}

impl RegressionTargets {
    /// Creates row-major `f32` regression targets.
    pub fn from_f32(
        values: Vec<f32>,
        sample_count: usize,
        output_count: usize,
    ) -> CoreResult<Self> {
        validate_shape_and_values(&values, sample_count, output_count)?;
        Ok(Self::F32 {
            values,
            sample_count,
            output_count,
        })
    }

    /// Creates row-major `f64` regression targets.
    pub fn from_f64(
        values: Vec<f64>,
        sample_count: usize,
        output_count: usize,
    ) -> CoreResult<Self> {
        validate_shape_and_values(&values, sample_count, output_count)?;
        Ok(Self::F64 {
            values,
            sample_count,
            output_count,
        })
    }

    /// Returns the target storage dtype.
    #[must_use]
    pub const fn scalar_kind(&self) -> ScalarKind {
        match self {
            Self::F32 { .. } => ScalarKind::F32,
            Self::F64 { .. } => ScalarKind::F64,
        }
    }

    /// Returns the number of samples.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        match self {
            Self::F32 { sample_count, .. } | Self::F64 { sample_count, .. } => *sample_count,
        }
    }

    /// Returns the number of outputs per sample.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        match self {
            Self::F32 { output_count, .. } | Self::F64 { output_count, .. } => *output_count,
        }
    }

    /// Returns one target promoted to `f64` for scoring or prediction.
    #[must_use]
    pub fn get_f64(&self, sample: usize, output: usize) -> Option<f64> {
        if sample >= self.sample_count() || output >= self.output_count() {
            return None;
        }
        let offset = sample * self.output_count() + output;
        match self {
            Self::F32 { values, .. } => Some(f64::from(values[offset])),
            Self::F64 { values, .. } => Some(values[offset]),
        }
    }
}

fn validate_shape_and_values<T>(
    values: &[T],
    sample_count: usize,
    output_count: usize,
) -> CoreResult<()>
where
    T: Copy + Into<f64>,
{
    if output_count == 0 {
        return Err(CoreError::invalid_data(
            "regression targets require at least one output",
        ));
    }
    let expected = sample_count
        .checked_mul(output_count)
        .ok_or_else(|| CoreError::invalid_data("regression target shape overflowed usize"))?;
    if values.len() != expected {
        return Err(CoreError::shape_mismatch(format!(
            "regression target storage has {} values; expected {expected} for {sample_count} samples and {output_count} outputs",
            values.len()
        )));
    }
    if values
        .iter()
        .copied()
        .any(|value| !value.into().is_finite())
    {
        return Err(CoreError::invalid_data("regression targets must be finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::RegressionTargets;
    use crate::ScalarKind;

    #[test]
    fn preserves_f32_storage_and_promotes_on_access() {
        let targets = RegressionTargets::from_f32(vec![1.25, 2.5], 1, 2).expect("valid targets");
        assert_eq!(targets.scalar_kind(), ScalarKind::F32);
        assert_eq!(targets.get_f64(0, 1), Some(2.5));
    }

    #[test]
    fn supports_f64_multi_target_storage() {
        let targets =
            RegressionTargets::from_f64(vec![1.0, 2.0, 3.0, 4.0], 2, 2).expect("valid targets");
        assert_eq!(targets.scalar_kind(), ScalarKind::F64);
        assert_eq!(targets.get_f64(1, 0), Some(3.0));
    }

    #[test]
    fn rejects_non_finite_and_mismatched_targets() {
        assert!(RegressionTargets::from_f64(vec![f64::NAN], 1, 1).is_err());
        assert!(RegressionTargets::from_f32(vec![1.0], 2, 1).is_err());
    }
}
