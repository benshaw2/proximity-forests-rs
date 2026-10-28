//! Allocation-free relative work estimates for coarse scheduling decisions.
//!
//! Estimates are deliberately computed at node, batch, or tile boundaries. They
//! are not elapsed-time predictions and are never called from numerical inner
//! loops. Saturating integer arithmetic makes them safe for untrusted shapes.

use super::DistanceConfig;

/// Saturating relative unit used to compare scheduling grain.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WorkUnits(u64);

impl WorkUnits {
    /// No estimated work.
    pub const ZERO: Self = Self(0);

    /// Creates work units from a relative integer value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the relative integer value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Saturating addition.
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Saturating multiplication by a structural count.
    #[must_use]
    pub fn saturating_mul(self, count: usize) -> Self {
        Self(self.0.saturating_mul(count_to_u64(count)))
    }
}

/// Shape metadata for one distance evaluation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DistanceWorkShape {
    /// Number of values along the first time or feature axis.
    pub first_length: usize,
    /// Number of values along the second time or feature axis.
    pub second_length: usize,
    /// Number of selected channels; vector distances use one.
    pub selected_dimensions: usize,
}

impl DistanceWorkShape {
    /// Creates a nonempty distance-work shape.
    #[must_use]
    pub const fn new(
        first_length: usize,
        second_length: usize,
        selected_dimensions: usize,
    ) -> Self {
        Self {
            first_length,
            second_length,
            selected_dimensions,
        }
    }
}

impl DistanceConfig {
    /// Estimates relative work for one evaluation from already-known metadata.
    ///
    /// This performs one enum dispatch and a small fixed number of saturating
    /// integer operations. Callers should compute it once before exposing a
    /// candidate batch or inference tile, never for each routed observation.
    #[must_use]
    pub fn estimate_work(&self, shape: DistanceWorkShape) -> WorkUnits {
        use DistanceConfig as C;
        let dimensions = count_to_u64(shape.selected_dimensions.max(1));
        let linear = linear_extent(shape).saturating_mul(dimensions);
        let pairwise = |window| {
            band_cells(shape.first_length, shape.second_length, window).saturating_mul(dimensions)
        };

        let value = match self {
            C::SquaredEuclidean
            | C::Manhattan
            | C::Cosine
            | C::NaNSquaredEuclidean { .. }
            | C::SquaredEuclideanIndependent
            | C::ManhattanIndependent
            | C::NaNSquaredEuclideanIndependent { .. } => linear,

            // CID adds linear complexity transforms to a linear pointwise pass.
            C::Cid | C::CidIndependent => linear.saturating_mul(3),

            // Direct SBD implementations evaluate a lag range over both inputs.
            C::Sbd | C::SbdIndependent => full_cells(shape).saturating_mul(dimensions),

            C::Dtw(parameters)
            | C::Ddtw(parameters)
            | C::DtwIndependent(parameters)
            | C::DdtwIndependent(parameters)
            | C::DtwDependent(parameters)
            | C::DdtwDependent(parameters) => pairwise(parameters.window),

            C::Wdtw(parameters)
            | C::Wddtw(parameters)
            | C::WdtwIndependent(parameters)
            | C::WddtwIndependent(parameters)
            | C::WdtwDependent(parameters)
            | C::WddtwDependent(parameters) => pairwise(parameters.window),

            C::Erp(parameters) | C::ErpIndependent(parameters) => pairwise(parameters.window),
            C::Lcss(parameters) | C::LcssIndependent(parameters) => pairwise(parameters.window),

            C::Msm(_) | C::MsmIndependent(_) | C::Twe(_) | C::TweIndependent(_) => {
                full_cells(shape).saturating_mul(dimensions)
            }

            // ShapeHoG performs linear descriptor construction and then aligns
            // compact histograms. The conservative bound retains both terms.
            C::ShapeHog1dDtw(parameters)
            | C::ShapeHog1dDtwIndependent(parameters)
            | C::ShapeHogDtw(parameters) => linear.saturating_add(pairwise(parameters.window)),

            C::DtwArow(parameters)
            | C::DtwArowIndependent(parameters)
            | C::DtwArowDependent(parameters) => pairwise(parameters.window),
        };
        WorkUnits::new(value)
    }
}

#[inline]
fn linear_extent(shape: DistanceWorkShape) -> u64 {
    count_to_u64(shape.first_length.max(shape.second_length))
}

#[inline]
fn full_cells(shape: DistanceWorkShape) -> u64 {
    count_to_u64(shape.first_length).saturating_mul(count_to_u64(shape.second_length))
}

/// Conservative cell bound for a Sakoe-Chiba-style absolute window.
fn band_cells(first: usize, second: usize, window: Option<usize>) -> u64 {
    let Some(window) = window else {
        return count_to_u64(first).saturating_mul(count_to_u64(second));
    };
    if first == 0 || second == 0 {
        return 0;
    }
    let band_width = window.saturating_mul(2).saturating_add(1).min(second);
    count_to_u64(first).saturating_mul(count_to_u64(band_width))
}

#[inline]
fn count_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{DistanceWorkShape, WorkUnits};
    use crate::{DistanceConfig, DtwParameters};

    #[test]
    fn linear_work_grows_with_length_and_dimensions() {
        let config = DistanceConfig::SquaredEuclideanIndependent;
        let small = config.estimate_work(DistanceWorkShape::new(16, 16, 2));
        let long = config.estimate_work(DistanceWorkShape::new(64, 64, 2));
        let wide = config.estimate_work(DistanceWorkShape::new(16, 16, 8));
        assert!(long > small);
        assert!(wide > small);
    }

    #[test]
    fn constrained_dynamic_programming_is_below_full_matrix_work() {
        let shape = DistanceWorkShape::new(128, 128, 1);
        let full = DistanceConfig::Dtw(DtwParameters { window: None }).estimate_work(shape);
        let narrow = DistanceConfig::Dtw(DtwParameters { window: Some(4) }).estimate_work(shape);
        assert!(narrow < full);
        assert_eq!(full.get(), 128 * 128);
    }

    #[test]
    fn work_arithmetic_saturates() {
        assert_eq!(
            WorkUnits::new(u64::MAX)
                .saturating_add(WorkUnits::new(1))
                .get(),
            u64::MAX
        );
        assert_eq!(WorkUnits::new(u64::MAX).saturating_mul(2).get(), u64::MAX);
    }
}
