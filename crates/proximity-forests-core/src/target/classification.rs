//! Compact integer-encoded classification targets.

use crate::{CoreError, CoreResult};

/// Compact class identifier used inside the Rust core.
pub type ClassId = u32;

/// Row-major single- or multi-output classification targets.
///
/// Python-facing code is responsible for encoding arbitrary user labels into
/// compact class identifiers and retaining inverse mappings for decoded
/// predictions and `classes_`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClassificationTargets {
    values: Vec<ClassId>,
    sample_count: usize,
    output_count: usize,
}

impl ClassificationTargets {
    /// Creates row-major classification targets.
    pub fn new(values: Vec<ClassId>, sample_count: usize, output_count: usize) -> CoreResult<Self> {
        validate_shape(values.len(), sample_count, output_count)?;
        Ok(Self {
            values,
            sample_count,
            output_count,
        })
    }

    /// Creates one-output classification targets.
    pub fn single_output(values: Vec<ClassId>) -> CoreResult<Self> {
        let sample_count = values.len();
        Self::new(values, sample_count, 1)
    }

    /// Returns the number of labeled samples.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the number of classification outputs per sample.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }

    /// Returns the class identifier at one sample and output.
    #[must_use]
    pub fn get(&self, sample: usize, output: usize) -> Option<ClassId> {
        self.offset(sample, output)
            .map(|offset| self.values[offset])
    }

    /// Returns all outputs for one sample.
    #[must_use]
    pub fn row(&self, sample: usize) -> Option<&[ClassId]> {
        if sample >= self.sample_count {
            return None;
        }
        let start = sample * self.output_count;
        Some(&self.values[start..start + self.output_count])
    }

    /// Returns the complete row-major storage.
    #[must_use]
    pub fn as_slice(&self) -> &[ClassId] {
        &self.values
    }

    fn offset(&self, sample: usize, output: usize) -> Option<usize> {
        if sample < self.sample_count && output < self.output_count {
            Some(sample * self.output_count + output)
        } else {
            None
        }
    }
}

fn validate_shape(value_count: usize, sample_count: usize, output_count: usize) -> CoreResult<()> {
    if output_count == 0 {
        return Err(CoreError::invalid_data(
            "classification targets require at least one output",
        ));
    }
    let expected = sample_count
        .checked_mul(output_count)
        .ok_or_else(|| CoreError::invalid_data("classification target shape overflowed usize"))?;
    if value_count != expected {
        return Err(CoreError::shape_mismatch(format!(
            "classification target storage has {value_count} values; expected {expected} for {sample_count} samples and {output_count} outputs"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ClassificationTargets;

    #[test]
    fn supports_multi_output_rows() {
        let targets = ClassificationTargets::new(vec![0, 1, 2, 3], 2, 2).expect("valid targets");
        assert_eq!(targets.row(0), Some(&[0, 1][..]));
        assert_eq!(targets.get(1, 1), Some(3));
        assert_eq!(targets.get(2, 0), None);
    }

    #[test]
    fn rejects_inconsistent_shapes() {
        assert!(ClassificationTargets::new(vec![0, 1, 2], 2, 2).is_err());
        assert!(ClassificationTargets::new(Vec::new(), 0, 0).is_err());
    }
}
