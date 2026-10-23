//! Borrowed one-dimensional numeric observations.

use crate::ScalarKind;

/// Borrowed `f32` or `f64` vector without dtype conversion.
///
/// The same physical representation can be interpreted by a caller as either
/// a tabular row or a univariate time series. The semantic distinction belongs
/// to the dataset schema rather than the primitive view.
#[derive(Clone, Copy, Debug)]
pub enum NumericVectorView<'a> {
    /// Borrowed single-precision values.
    F32(&'a [f32]),
    /// Borrowed double-precision values.
    F64(&'a [f64]),
}

impl<'a> NumericVectorView<'a> {
    /// Creates a single-precision view.
    #[must_use]
    pub const fn from_f32(values: &'a [f32]) -> Self {
        Self::F32(values)
    }

    /// Creates a double-precision view.
    #[must_use]
    pub const fn from_f64(values: &'a [f64]) -> Self {
        Self::F64(values)
    }

    /// Returns the storage dtype.
    #[must_use]
    pub const fn scalar_kind(self) -> ScalarKind {
        match self {
            Self::F32(_) => ScalarKind::F32,
            Self::F64(_) => ScalarKind::F64,
        }
    }

    /// Returns the number of values.
    #[must_use]
    pub const fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    /// Reports whether the view is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Returns one value promoted to `f64`.
    #[must_use]
    pub fn get_f64(self, index: usize) -> Option<f64> {
        match self {
            Self::F32(values) => values.get(index).copied().map(f64::from),
            Self::F64(values) => values.get(index).copied(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NumericVectorView;
    use crate::ScalarKind;

    #[test]
    fn preserves_storage_dtype() {
        let values = [1.0_f32, 2.0];
        let view = NumericVectorView::from_f32(&values);
        assert_eq!(view.scalar_kind(), ScalarKind::F32);
        assert_eq!(view.get_f64(1), Some(2.0));
    }
}
