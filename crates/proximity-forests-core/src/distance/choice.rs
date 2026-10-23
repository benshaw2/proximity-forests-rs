//! Candidate-time distance choices and sampling contracts.
//!
//! A [`DistanceChoice::Fixed`] entry preserves one exact configuration. A
//! [`DistanceChoice::Sampled`] entry is resolved exactly once for each split
//! candidate using that candidate's deterministic distance random stream. The
//! winning split stores only the resulting [`DistanceConfig`].

use crate::{CoreError, CoreResult, DistanceConfig, ModelRng};

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
