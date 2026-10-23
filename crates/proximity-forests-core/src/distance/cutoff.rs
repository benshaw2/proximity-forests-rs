//! Early-abandoning cutoff semantics.

use crate::{CoreError, CoreResult};

/// Optional best-so-far threshold for exact distance evaluation.
///
/// A distance may abandon only after proving that its exact result is strictly
/// greater than the finite cutoff. Equality must remain exact so downstream
/// branch ties can be preserved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceCutoff {
    /// Compute the complete distance without early abandoning.
    None,
    /// Permit abandoning once the distance is proven strictly greater than the
    /// supplied finite, nonnegative threshold.
    Finite(f64),
}

impl DistanceCutoff {
    /// Creates a validated finite cutoff.
    ///
    /// # Errors
    ///
    /// Returns an error when `value` is negative, NaN, or infinite.
    pub fn finite(value: f64) -> CoreResult<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(CoreError::invalid_configuration(
                "distance_cutoff",
                "a finite cutoff must be finite and nonnegative",
            ));
        }
        Ok(Self::Finite(value))
    }

    /// Reports whether a lower bound is strictly greater than the cutoff.
    #[must_use]
    pub fn exceeded_by(self, accumulated: f64) -> bool {
        matches!(self, Self::Finite(limit) if accumulated > limit)
    }

    /// Returns the remaining nonnegative budget after `consumed` distance.
    #[must_use]
    pub fn remaining_after(self, consumed: f64) -> Self {
        match self {
            Self::None => Self::None,
            Self::Finite(limit) => Self::Finite((limit - consumed).max(0.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DistanceCutoff;

    #[test]
    fn equality_does_not_trigger_abandoning() {
        let cutoff = DistanceCutoff::finite(4.0).expect("valid cutoff");
        assert!(!cutoff.exceeded_by(4.0));
        assert!(cutoff.exceeded_by(f64::from_bits(4.0_f64.to_bits() + 1)));
    }

    #[test]
    fn invalid_cutoffs_are_rejected() {
        assert!(DistanceCutoff::finite(-1.0).is_err());
        assert!(DistanceCutoff::finite(f64::NAN).is_err());
        assert!(DistanceCutoff::finite(f64::INFINITY).is_err());
    }
}
