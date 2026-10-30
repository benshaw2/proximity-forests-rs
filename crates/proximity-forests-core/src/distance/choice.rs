//! Candidate-time distance choices and sampling contracts.
//!
//! A [`DistanceChoice::Fixed`] entry preserves one exact configuration. A
//! [`DistanceChoice::Sampled`] entry is resolved exactly once for each split
//! candidate using that candidate's deterministic distance random stream. The
//! winning split stores only the resulting [`DistanceConfig`].

use crate::{
    CoreError, CoreResult, DimensionSelectionSupport, DistanceConfig, DistanceWorkShape, ModelRng,
    WorkUnits,
};

/// Dataset statistics needed while resolving candidate-local parameters.
///
/// These values are computed at the fitting boundary and reused by every
/// candidate. They are never recomputed inside a distance kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceSamplingContext {
    /// Maximum relevant series length.
    pub maximum_length: usize,
    /// Population standard deviation used by data-scaled families.
    pub population_standard_deviation: f64,
}

impl DistanceSamplingContext {
    /// Creates a validated sampling context.
    pub fn new(maximum_length: usize, population_standard_deviation: f64) -> CoreResult<Self> {
        if maximum_length == 0 {
            return Err(CoreError::invalid_configuration(
                "maximum_length",
                "distance sampling requires a positive maximum series length",
            ));
        }
        if !population_standard_deviation.is_finite() || population_standard_deviation < 0.0 {
            return Err(CoreError::invalid_configuration(
                "population_standard_deviation",
                "distance sampling requires a finite nonnegative population standard deviation",
            ));
        }
        Ok(Self {
            maximum_length,
            population_standard_deviation,
        })
    }
}

/// One entry in a candidate-generation distance catalog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DistanceChoice {
    /// Reuses one fully specified immutable configuration.
    Fixed(DistanceConfig),
    /// Resolves a family once for each split candidate.
    Sampled(SampledDistanceFamily),
}

impl DistanceChoice {
    /// Returns dimension-selection semantics without resolving candidate randomness.
    #[must_use]
    pub const fn dimension_selection_support(&self) -> DimensionSelectionSupport {
        match self {
            Self::Fixed(config) => config.capabilities().dimension_selection,
            Self::Sampled(family) => family.dimension_selection_support(),
        }
    }

    /// Estimates representative work without resolving candidate randomness.
    ///
    /// Fixed choices use their concrete configuration. Every currently sampled
    /// family is dynamic-programming based, so sampled choices use a conservative
    /// full-matrix estimate. This method is intended for coarse scheduling only.
    #[must_use]
    pub fn estimate_work(&self, shape: DistanceWorkShape) -> WorkUnits {
        match self {
            Self::Fixed(config) => config.estimate_work(shape),
            Self::Sampled(_) => {
                let first = u64::try_from(shape.first_length).unwrap_or(u64::MAX);
                let second = u64::try_from(shape.second_length).unwrap_or(u64::MAX);
                let dimensions =
                    u64::try_from(shape.selected_dimensions.max(1)).unwrap_or(u64::MAX);
                WorkUnits::new(first.saturating_mul(second).saturating_mul(dimensions))
            }
        }
    }
}

impl From<DistanceConfig> for DistanceChoice {
    fn from(value: DistanceConfig) -> Self {
        Self::Fixed(value)
    }
}

/// Native families whose parameters are resolved per split candidate.
///
/// `Dtw` and `Ddtw` resolve to their Java-compatible unrestricted-window
/// forms. The `Cv` variants sample a restricted window. Independent (`_i`)
/// and dependent (`_d`) forms are represented separately so no implicit
/// multivariate fallback occurs.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SampledDistanceFamily {
    /// Unrestricted univariate dynamic time warping.
    Dtw,
    /// Univariate dynamic time warping with a sampled constrained window.
    DtwCv,
    /// Unrestricted univariate derivative dynamic time warping.
    Ddtw,
    /// Univariate derivative DTW with a sampled constrained window.
    DdtwCv,
    /// Univariate weighted dynamic time warping with sampled weighting.
    Wdtw,
    /// Univariate weighted derivative DTW with sampled weighting.
    Wddtw,
    /// Univariate ERP with candidate-local sampled parameters.
    Erp,
    /// Univariate LCSS with candidate-local sampled parameters.
    Lcss,
    /// Univariate MSM with candidate-local sampled cost.
    Msm,
    /// Univariate TWE with candidate-local sampled parameters.
    Twe,

    /// Independent multivariate DTW with an unrestricted window.
    DtwIndependent,
    /// Independent multivariate DTW with a sampled constrained window.
    DtwCvIndependent,
    /// Independent multivariate derivative DTW with an unrestricted window.
    DdtwIndependent,
    /// Independent multivariate derivative DTW with a sampled window.
    DdtwCvIndependent,
    /// Independent multivariate weighted DTW with sampled weighting.
    WdtwIndependent,
    /// Independent multivariate weighted derivative DTW with sampled weighting.
    WddtwIndependent,
    /// Independent multivariate ERP with sampled parameters.
    ErpIndependent,
    /// Independent multivariate LCSS with sampled parameters.
    LcssIndependent,
    /// Independent multivariate MSM with sampled cost.
    MsmIndependent,
    /// Independent multivariate TWE with sampled parameters.
    TweIndependent,

    /// Dependent multivariate DTW with an unrestricted window.
    DtwDependent,
    /// Dependent multivariate DTW with a sampled constrained window.
    DtwCvDependent,
    /// Dependent multivariate derivative DTW with an unrestricted window.
    DdtwDependent,
    /// Dependent multivariate derivative DTW with a sampled window.
    DdtwCvDependent,
    /// Dependent multivariate weighted DTW with sampled weighting.
    WdtwDependent,
    /// Dependent multivariate weighted derivative DTW with sampled weighting.
    WddtwDependent,
}

impl SampledDistanceFamily {
    /// Returns the dimension-selection semantics shared by every configuration
    /// that this sampled family can resolve to.
    #[must_use]
    pub const fn dimension_selection_support(self) -> DimensionSelectionSupport {
        use DimensionSelectionSupport as S;
        use SampledDistanceFamily as F;
        match self {
            F::Dtw
            | F::DtwCv
            | F::Ddtw
            | F::DdtwCv
            | F::Wdtw
            | F::Wddtw
            | F::Erp
            | F::Lcss
            | F::Msm
            | F::Twe => S::None,

            F::DtwIndependent
            | F::DtwCvIndependent
            | F::DdtwIndependent
            | F::DdtwCvIndependent
            | F::WdtwIndependent
            | F::WddtwIndependent
            | F::ErpIndependent
            | F::LcssIndependent
            | F::MsmIndependent
            | F::TweIndependent
            | F::DtwDependent
            | F::DtwCvDependent
            | F::DdtwDependent
            | F::DdtwCvDependent
            | F::WdtwDependent
            | F::WddtwDependent => S::MultivariateChannels,
        }
    }
}

/// Policy that turns a sampled family into a concrete configuration.
pub trait DistanceSamplingPolicy: Sync {
    /// Resolves one family using candidate-local randomness.
    fn sample(
        &self,
        family: SampledDistanceFamily,
        context: DistanceSamplingContext,
        rng: &mut ModelRng,
    ) -> CoreResult<DistanceConfig>;
}

/// Resolves one catalog entry into the configuration stored by a candidate.
pub fn resolve_distance_choice(
    choice: &DistanceChoice,
    context: DistanceSamplingContext,
    rng: &mut ModelRng,
    policy: &dyn DistanceSamplingPolicy,
) -> CoreResult<DistanceConfig> {
    match choice {
        DistanceChoice::Fixed(config) => Ok(config.clone()),
        DistanceChoice::Sampled(family) => policy.sample(*family, context, rng),
    }
}

#[cfg(test)]
mod work_tests {
    use super::{DistanceChoice, SampledDistanceFamily};
    use crate::{DistanceConfig, DistanceWorkShape};

    #[test]
    fn fixed_choice_uses_configured_distance_work() {
        let choice = DistanceChoice::Fixed(DistanceConfig::SquaredEuclidean);
        assert_eq!(
            choice
                .estimate_work(DistanceWorkShape::new(32, 32, 1))
                .get(),
            32
        );
    }

    #[test]
    fn sampled_choice_uses_conservative_dynamic_programming_work() {
        let choice = DistanceChoice::Sampled(SampledDistanceFamily::DtwCv);
        assert_eq!(
            choice
                .estimate_work(DistanceWorkShape::new(32, 24, 3))
                .get(),
            2304
        );
    }

    #[test]
    fn fixed_choice_delegates_dimension_selection_semantics() {
        use crate::DimensionSelectionSupport;
        assert_eq!(
            DistanceChoice::Fixed(DistanceConfig::SquaredEuclidean).dimension_selection_support(),
            DimensionSelectionSupport::VectorFeatures
        );
        assert_eq!(
            DistanceChoice::Fixed(DistanceConfig::Dtw(Default::default()))
                .dimension_selection_support(),
            DimensionSelectionSupport::None
        );
    }

    #[test]
    fn sampled_families_advertise_semantics_before_resolution() {
        use crate::DimensionSelectionSupport;
        assert_eq!(
            DistanceChoice::Sampled(SampledDistanceFamily::DtwCv).dimension_selection_support(),
            DimensionSelectionSupport::None
        );
        assert_eq!(
            DistanceChoice::Sampled(SampledDistanceFamily::DtwCvDependent)
                .dimension_selection_support(),
            DimensionSelectionSupport::MultivariateChannels
        );
    }
}
