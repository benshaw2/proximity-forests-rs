//! Optional time-coordinate views for ordered series.

use crate::{CoreError, CoreResult, NumericVectorView};

/// Borrowed time coordinates associated with one series.
#[derive(Clone, Copy, Debug)]
pub struct TimeAxisView<'a> {
    values: NumericVectorView<'a>,
}

impl<'a> TimeAxisView<'a> {
    /// Creates a validated time axis.
    ///
    /// # Errors
    ///
    /// Returns an error when the axis is empty, contains a nonfinite value, or
    /// is not strictly increasing.
    pub fn new(values: NumericVectorView<'a>) -> CoreResult<Self> {
        if values.is_empty() {
            return Err(CoreError::invalid_data("time axis cannot be empty"));
        }
        let mut previous = None;
        for index in 0..values.len() {
            let value = values
                .get_f64(index)
                .ok_or_else(|| CoreError::invalid_data("invalid time-axis index"))?;
            if !value.is_finite() {
                return Err(CoreError::invalid_data("time coordinates must be finite"));
            }
            if previous.is_some_and(|prior| value <= prior) {
                return Err(CoreError::invalid_data(
                    "time coordinates must be strictly increasing",
                ));
            }
            previous = Some(value);
        }
        Ok(Self { values })
    }

    /// Returns the number of time coordinates.
    #[must_use]
    pub const fn len(self) -> usize {
        self.values.len()
    }

    /// Returns one coordinate promoted to `f64`.
    #[must_use]
    pub fn get_f64(self, index: usize) -> Option<f64> {
        self.values.get_f64(index)
    }
}

#[cfg(test)]
mod tests {
    use super::TimeAxisView;
    use crate::NumericVectorView;

    #[test]
    fn accepts_strictly_increasing_coordinates() {
        let values = [0.0, 0.5, 2.0];
        let axis =
            TimeAxisView::new(NumericVectorView::from_f64(&values)).expect("valid time axis");
        assert_eq!(axis.len(), 3);
    }

    #[test]
    fn rejects_duplicates_and_reversals() {
        assert!(TimeAxisView::new(NumericVectorView::from_f64(&[0.0, 0.0])).is_err());
        assert!(TimeAxisView::new(NumericVectorView::from_f32(&[1.0, 0.0])).is_err());
    }
}
