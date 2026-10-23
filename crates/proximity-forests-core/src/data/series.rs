//! Dimension-major multivariate time-series views.

use crate::{CoreError, CoreResult, NumericVectorView, ScalarKind};

/// Borrowed rectangular multivariate series with logical shape
/// `dimension x time`.
///
/// Each observation has one common time length across its dimensions. Two
/// observations may have different time lengths, but distance evaluation must
/// validate compatible dimension counts. Element strides permit dimension-major
/// contiguous arrays as well as compatible strided NumPy views without copying.
#[derive(Clone, Copy, Debug)]
pub enum MultivariateSeriesView<'a> {
    /// Single-precision storage.
    F32(StridedSeries<'a, f32>),
    /// Double-precision storage.
    F64(StridedSeries<'a, f64>),
}

impl<'a> MultivariateSeriesView<'a> {
    /// Creates a contiguous dimension-major `f32` view.
    ///
    /// # Errors
    ///
    /// Returns an error for empty axes, shape overflow, or mismatched storage.
    pub fn contiguous_f32(
        values: &'a [f32],
        dimension_count: usize,
        time_count: usize,
    ) -> CoreResult<Self> {
        Ok(Self::F32(StridedSeries::contiguous(
            values,
            dimension_count,
            time_count,
        )?))
    }

    /// Creates a contiguous dimension-major `f64` view.
    ///
    /// # Errors
    ///
    /// Returns an error for empty axes, shape overflow, or mismatched storage.
    pub fn contiguous_f64(
        values: &'a [f64],
        dimension_count: usize,
        time_count: usize,
    ) -> CoreResult<Self> {
        Ok(Self::F64(StridedSeries::contiguous(
            values,
            dimension_count,
            time_count,
        )?))
    }

    /// Creates a strided dimension-major `f32` view.
    ///
    /// Strides are measured in elements, not bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for empty axes, zero strides, arithmetic overflow, or
    /// a backing slice too short for the requested shape and strides.
    pub fn strided_f32(
        values: &'a [f32],
        dimension_count: usize,
        time_count: usize,
        dimension_stride: usize,
        time_stride: usize,
    ) -> CoreResult<Self> {
        Ok(Self::F32(StridedSeries::new(
            values,
            dimension_count,
            time_count,
            dimension_stride,
            time_stride,
        )?))
    }

    /// Creates a strided dimension-major `f64` view.
    ///
    /// Strides are measured in elements, not bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for empty axes, zero strides, arithmetic overflow, or
    /// a backing slice too short for the requested shape and strides.
    pub fn strided_f64(
        values: &'a [f64],
        dimension_count: usize,
        time_count: usize,
        dimension_stride: usize,
        time_stride: usize,
    ) -> CoreResult<Self> {
        Ok(Self::F64(StridedSeries::new(
            values,
            dimension_count,
            time_count,
            dimension_stride,
            time_stride,
        )?))
    }

    /// Returns the storage dtype.
    #[must_use]
    pub const fn scalar_kind(self) -> ScalarKind {
        match self {
            Self::F32(_) => ScalarKind::F32,
            Self::F64(_) => ScalarKind::F64,
        }
    }

    /// Returns the number of channels/dimensions.
    #[must_use]
    pub const fn dimension_count(self) -> usize {
        match self {
            Self::F32(view) => view.dimension_count,
            Self::F64(view) => view.dimension_count,
        }
    }

    /// Returns the common time length within this observation.
    #[must_use]
    pub const fn time_count(self) -> usize {
        match self {
            Self::F32(view) => view.time_count,
            Self::F64(view) => view.time_count,
        }
    }

    /// Returns one value promoted to `f64`.
    #[must_use]
    pub fn get_f64(self, dimension: usize, time: usize) -> Option<f64> {
        match self {
            Self::F32(view) => view.get(dimension, time).map(f64::from),
            Self::F64(view) => view.get(dimension, time),
        }
    }

    /// Returns one complete dimension as a contiguous vector view when its
    /// time stride is one.
    #[must_use]
    pub fn contiguous_dimension(self, dimension: usize) -> Option<NumericVectorView<'a>> {
        match self {
            Self::F32(view) => view
                .contiguous_dimension(dimension)
                .map(NumericVectorView::F32),
            Self::F64(view) => view
                .contiguous_dimension(dimension)
                .map(NumericVectorView::F64),
        }
    }
}

/// Typed strided matrix view used internally by the dtype-erased series enum.
#[derive(Clone, Copy, Debug)]
pub struct StridedSeries<'a, T> {
    values: &'a [T],
    dimension_count: usize,
    time_count: usize,
    dimension_stride: usize,
    time_stride: usize,
}

impl<'a, T> StridedSeries<'a, T>
where
    T: Copy,
{
    fn contiguous(values: &'a [T], dimension_count: usize, time_count: usize) -> CoreResult<Self> {
        let expected = dimension_count
            .checked_mul(time_count)
            .ok_or_else(|| CoreError::invalid_data("multivariate series shape overflowed usize"))?;
        if values.len() != expected {
            return Err(CoreError::shape_mismatch(format!(
                "multivariate storage has {} values; expected {expected}",
                values.len()
            )));
        }
        Self::new(values, dimension_count, time_count, time_count, 1)
    }

    fn new(
        values: &'a [T],
        dimension_count: usize,
        time_count: usize,
        dimension_stride: usize,
        time_stride: usize,
    ) -> CoreResult<Self> {
        if dimension_count == 0 || time_count == 0 {
            return Err(CoreError::invalid_data(
                "multivariate series requires nonempty dimension and time axes",
            ));
        }
        if dimension_stride == 0 || time_stride == 0 {
            return Err(CoreError::invalid_data(
                "multivariate series strides must be positive",
            ));
        }
        let last_dimension = (dimension_count - 1)
            .checked_mul(dimension_stride)
            .ok_or_else(|| CoreError::invalid_data("dimension stride overflow"))?;
        let last_time = (time_count - 1)
            .checked_mul(time_stride)
            .ok_or_else(|| CoreError::invalid_data("time stride overflow"))?;
        let last_offset = last_dimension
            .checked_add(last_time)
            .ok_or_else(|| CoreError::invalid_data("series offset overflow"))?;
        if last_offset >= values.len() {
            return Err(CoreError::shape_mismatch(
                "backing storage is too short for the requested series view",
            ));
        }
        Ok(Self {
            values,
            dimension_count,
            time_count,
            dimension_stride,
            time_stride,
        })
    }

    fn get(self, dimension: usize, time: usize) -> Option<T> {
        if dimension >= self.dimension_count || time >= self.time_count {
            return None;
        }
        let offset = dimension * self.dimension_stride + time * self.time_stride;
        self.values.get(offset).copied()
    }

    fn contiguous_dimension(self, dimension: usize) -> Option<&'a [T]> {
        if dimension >= self.dimension_count || self.time_stride != 1 {
            return None;
        }
        let start = dimension * self.dimension_stride;
        self.values.get(start..start + self.time_count)
    }
}

#[cfg(test)]
mod tests {
    use super::MultivariateSeriesView;

    #[test]
    fn contiguous_dimension_major_access_works() {
        let values = [1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let view = MultivariateSeriesView::contiguous_f32(&values, 2, 3).expect("valid series");
        assert_eq!(view.get_f64(1, 2), Some(30.0));
        assert_eq!(view.contiguous_dimension(0).map(|row| row.len()), Some(3));
    }

    #[test]
    fn strided_time_major_backing_is_supported() {
        let values = [1.0_f64, 10.0, 2.0, 20.0, 3.0, 30.0];
        let view =
            MultivariateSeriesView::strided_f64(&values, 2, 3, 1, 2).expect("valid strided series");
        assert_eq!(view.get_f64(1, 2), Some(30.0));
        assert!(view.contiguous_dimension(0).is_none());
    }

    #[test]
    fn observations_may_have_different_time_lengths() {
        let short = MultivariateSeriesView::contiguous_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2)
            .expect("valid short series");
        let long = MultivariateSeriesView::contiguous_f64(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 2, 3)
            .expect("valid long series");
        assert_eq!(short.dimension_count(), long.dimension_count());
        assert_ne!(short.time_count(), long.time_count());
    }

    #[test]
    fn rejects_invalid_shapes() {
        assert!(MultivariateSeriesView::contiguous_f32(&[1.0, 2.0], 2, 2).is_err());
        assert!(MultivariateSeriesView::contiguous_f64(&[], 0, 0).is_err());
    }
}
