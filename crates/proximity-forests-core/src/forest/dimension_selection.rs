//! Dimension-selection configuration for node-level split search.

use crate::{CoreError, CoreResult};

/// Determines how many features or channels are considered at a tree node.
///
/// A node selects its dimensions once and shares that selection across all
/// split candidates evaluated at the node. For tabular data, dimensions are
/// features. For multivariate time series, dimensions are channels rather
/// than time points.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum DimensionSelection {
    /// Use every available dimension.
    All,

    /// Use `ceil(sqrt(d))` dimensions.
    Sqrt,

    /// Use `floor(log2(d)) + 1` dimensions.
    Log2,

    /// Use a fixed number of dimensions, clamped to the available count.
    FixedCount(usize),

    /// Use `ceil(proportion * d)` dimensions, clamped to `[1, d]`.
    Proportion(f64),
}

impl DimensionSelection {
    /// Validates configuration independently of a particular dataset.
    pub fn validate(self) -> CoreResult<()> {
        match self {
            Self::FixedCount(0) => Err(CoreError::invalid_configuration(
                "dimension_selection",
                "fixed dimension count must be greater than zero",
            )),
            Self::Proportion(value) if !value.is_finite() => Err(CoreError::invalid_configuration(
                "dimension_selection",
                "dimension proportion must be finite",
            )),
            Self::Proportion(value) if !(0.0 < value && value <= 1.0) => {
                Err(CoreError::invalid_configuration(
                    "dimension_selection",
                    "dimension proportion must be in the interval (0, 1]",
                ))
            }
            _ => Ok(()),
        }
    }

    /// Returns the number of dimensions to select from `available`.
    ///
    /// This method validates both the configuration and the available
    /// dimension count. It determines only the count; deterministic sampling
    /// of the actual distinct indices belongs to node split construction.
    pub fn selected_count(self, available: usize) -> CoreResult<usize> {
        self.validate()?;

        if available == 0 {
            return Err(CoreError::invalid_data(
                "dimension selection requires at least one available dimension",
            ));
        }

        let count = match self {
            Self::All => available,
            Self::Sqrt => ceil_sqrt(available),
            Self::Log2 => floor_log2(available) + 1,
            Self::FixedCount(count) => count.min(available),
            Self::Proportion(proportion) => {
                ((proportion * available as f64).ceil() as usize).clamp(1, available)
            }
        };

        Ok(count)
    }
}

fn ceil_sqrt(value: usize) -> usize {
    let floor = (value as f64).sqrt() as usize;
    if floor.saturating_mul(floor) == value {
        floor
    } else {
        floor + 1
    }
}

fn floor_log2(value: usize) -> usize {
    usize::BITS as usize - 1 - value.leading_zeros() as usize
}

#[cfg(test)]
mod tests {
    use super::DimensionSelection;

    #[test]
    fn all_uses_every_dimension() {
        assert_eq!(DimensionSelection::All.selected_count(7), Ok(7));
    }

    #[test]
    fn sqrt_rounds_up() {
        assert_eq!(DimensionSelection::Sqrt.selected_count(8), Ok(3));
        assert_eq!(DimensionSelection::Sqrt.selected_count(9), Ok(3));
    }

    #[test]
    fn log2_matches_java_rule() {
        assert_eq!(DimensionSelection::Log2.selected_count(1), Ok(1));
        assert_eq!(DimensionSelection::Log2.selected_count(8), Ok(4));
        assert_eq!(DimensionSelection::Log2.selected_count(9), Ok(4));
    }

    #[test]
    fn fixed_count_is_clamped() {
        assert_eq!(DimensionSelection::FixedCount(9).selected_count(4), Ok(4));
        assert!(DimensionSelection::FixedCount(0).selected_count(4).is_err());
    }

    #[test]
    fn proportion_rounds_up_and_is_clamped() {
        assert_eq!(
            DimensionSelection::Proportion(0.25).selected_count(10),
            Ok(3)
        );
        assert_eq!(
            DimensionSelection::Proportion(1.0).selected_count(10),
            Ok(10)
        );
        assert!(
            DimensionSelection::Proportion(0.0)
                .selected_count(10)
                .is_err()
        );
        assert!(
            DimensionSelection::Proportion(1.1)
                .selected_count(10)
                .is_err()
        );
        assert!(
            DimensionSelection::Proportion(f64::NAN)
                .selected_count(10)
                .is_err()
        );
    }

    #[test]
    fn zero_available_dimensions_is_rejected() {
        assert!(DimensionSelection::All.selected_count(0).is_err());
    }
}
