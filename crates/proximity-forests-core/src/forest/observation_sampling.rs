//! Per-tree observation-sampling policy.

use std::num::NonZeroUsize;

use crate::{CoreError, CoreResult};

/// Number of observations requested for one tree.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SampleSize {
    /// Resolve to the complete training-set size.
    All,
    /// Resolve to a fixed positive number of observations.
    Fixed(NonZeroUsize),
    /// Resolve to `ceil(proportion * n)`, with a proportion in `(0, 1]`.
    Proportion(f64),
}

impl SampleSize {
    /// Validates configuration independently of a dataset.
    pub fn validate(self) -> CoreResult<()> {
        match self {
            Self::Proportion(value) if !value.is_finite() => Err(CoreError::invalid_configuration(
                "sample_size",
                "proportion must be finite",
            )),
            Self::Proportion(value) if !(0.0 < value && value <= 1.0) => {
                Err(CoreError::invalid_configuration(
                    "sample_size",
                    "proportion must be in the interval (0, 1]",
                ))
            }
            _ => Ok(()),
        }
    }

    /// Resolves this request for a nonempty training set.
    ///
    /// Subsampling without replacement clamps the result to `population`;
    /// bootstrap sampling may request more draws than the population size.
    pub fn resolve(self, population: usize, without_replacement: bool) -> CoreResult<usize> {
        self.validate()?;
        if population == 0 {
            return Err(CoreError::invalid_data(
                "observation sampling requires at least one observation",
            ));
        }
        let requested = match self {
            Self::All => population,
            Self::Fixed(value) => value.get(),
            Self::Proportion(value) => ((value * population as f64).ceil() as usize).max(1),
        };
        Ok(if without_replacement {
            requested.min(population)
        } else {
            requested
        })
    }
}

/// How each tree obtains observations from the training set.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ObservationSampling {
    /// Use every training observation exactly once.
    Full,
    /// Draw observations with replacement.
    Bootstrap(SampleSize),
    /// Draw distinct observations without replacement.
    Subsample(SampleSize),
}

impl ObservationSampling {
    /// Validates the policy independently of a dataset.
    pub fn validate(self) -> CoreResult<()> {
        match self {
            Self::Full => Ok(()),
            Self::Bootstrap(size) | Self::Subsample(size) => size.validate(),
        }
    }

    /// Resolves the number of observations or draws for one tree.
    pub fn resolved_size(self, population: usize) -> CoreResult<usize> {
        match self {
            Self::Full => SampleSize::All.resolve(population, true),
            Self::Bootstrap(size) => size.resolve(population, false),
            Self::Subsample(size) => size.resolve(population, true),
        }
    }
}

impl Default for ObservationSampling {
    fn default() -> Self {
        Self::Bootstrap(SampleSize::All)
    }
}

#[cfg(test)]
mod tests {
    use super::{ObservationSampling, SampleSize};
    use std::num::NonZeroUsize;

    #[test]
    fn default_preserves_full_size_bootstrap() {
        assert_eq!(
            ObservationSampling::default().resolved_size(17).unwrap(),
            17
        );
    }

    #[test]
    fn fixed_subsample_clamps_to_population() {
        let policy =
            ObservationSampling::Subsample(SampleSize::Fixed(NonZeroUsize::new(256).unwrap()));
        assert_eq!(policy.resolved_size(90).unwrap(), 90);
        assert_eq!(policy.resolved_size(1_000).unwrap(), 256);
    }

    #[test]
    fn bootstrap_can_draw_more_than_population() {
        let policy =
            ObservationSampling::Bootstrap(SampleSize::Fixed(NonZeroUsize::new(256).unwrap()));
        assert_eq!(policy.resolved_size(90).unwrap(), 256);
    }

    #[test]
    fn proportion_uses_ceiling() {
        let policy = ObservationSampling::Subsample(SampleSize::Proportion(0.25));
        assert_eq!(policy.resolved_size(10).unwrap(), 3);
    }

    #[test]
    fn invalid_proportions_are_rejected() {
        for value in [0.0, -0.5, 1.1, f64::NAN, f64::INFINITY] {
            assert!(SampleSize::Proportion(value).validate().is_err());
        }
    }

    #[test]
    fn empty_population_is_rejected() {
        assert!(ObservationSampling::Full.resolved_size(0).is_err());
    }
}
