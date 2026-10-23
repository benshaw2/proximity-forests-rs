//! Java-compatible candidate-local distance-parameter sampling.

use crate::{
    CoreResult, DistanceConfig, DistanceSamplingContext, DistanceSamplingPolicy, DtwParameters,
    ErpParameters, FiniteF64, LcssParameters, ModelRng, MsmParameters, SampledDistanceFamily,
    TweParameters, WeightedDtwParameters,
};

/// Built-in policy matching the parameter-selection rules used by PFGAP Java.
#[derive(Clone, Copy, Debug, Default)]
pub struct JavaCompatibleDistanceSampling;

impl DistanceSamplingPolicy for JavaCompatibleDistanceSampling {
    fn sample(
        &self,
        family: SampledDistanceFamily,
        context: DistanceSamplingContext,
        rng: &mut ModelRng,
    ) -> CoreResult<DistanceConfig> {
        use SampledDistanceFamily as Family;

        Ok(match family {
            Family::Dtw => DistanceConfig::Dtw(full_window()),
            Family::DtwCv => DistanceConfig::Dtw(sampled_dtw(context, rng)?),
            Family::Ddtw => DistanceConfig::Ddtw(full_window()),
            Family::DdtwCv => DistanceConfig::Ddtw(sampled_dtw(context, rng)?),
            Family::Wdtw => DistanceConfig::Wdtw(sampled_weighted(rng)?),
            Family::Wddtw => DistanceConfig::Wddtw(sampled_weighted(rng)?),
            Family::Erp => DistanceConfig::Erp(sampled_erp(context, rng)?),
            Family::Lcss => DistanceConfig::Lcss(sampled_lcss(context, rng)?),
            Family::Msm => DistanceConfig::Msm(sampled_msm(rng)?),
            Family::Twe => DistanceConfig::Twe(sampled_twe(rng)?),

            Family::DtwIndependent => DistanceConfig::DtwIndependent(full_window()),
            Family::DtwCvIndependent => DistanceConfig::DtwIndependent(sampled_dtw(context, rng)?),
            Family::DdtwIndependent => DistanceConfig::DdtwIndependent(full_window()),
            Family::DdtwCvIndependent => {
                DistanceConfig::DdtwIndependent(sampled_dtw(context, rng)?)
            }
            Family::WdtwIndependent => DistanceConfig::WdtwIndependent(sampled_weighted(rng)?),
            Family::WddtwIndependent => DistanceConfig::WddtwIndependent(sampled_weighted(rng)?),
            Family::ErpIndependent => DistanceConfig::ErpIndependent(sampled_erp(context, rng)?),
            Family::LcssIndependent => DistanceConfig::LcssIndependent(sampled_lcss(context, rng)?),
            Family::MsmIndependent => DistanceConfig::MsmIndependent(sampled_msm(rng)?),
            Family::TweIndependent => DistanceConfig::TweIndependent(sampled_twe(rng)?),

            Family::DtwDependent => DistanceConfig::DtwDependent(full_window()),
            Family::DtwCvDependent => DistanceConfig::DtwDependent(sampled_dtw(context, rng)?),
            Family::DdtwDependent => DistanceConfig::DdtwDependent(full_window()),
            Family::DdtwCvDependent => DistanceConfig::DdtwDependent(sampled_dtw(context, rng)?),
            Family::WdtwDependent => DistanceConfig::WdtwDependent(sampled_weighted(rng)?),
            Family::WddtwDependent => DistanceConfig::WddtwDependent(sampled_weighted(rng)?),
        })
    }
}

#[inline]
const fn full_window() -> DtwParameters {
    DtwParameters { window: None }
}

fn sampled_dtw(context: DistanceSamplingContext, rng: &mut ModelRng) -> CoreResult<DtwParameters> {
    Ok(DtwParameters {
        window: Some(sample_window(context.maximum_length, rng)?),
    })
}

fn sampled_weighted(rng: &mut ModelRng) -> CoreResult<WeightedDtwParameters> {
    Ok(WeightedDtwParameters {
        window: None,
        stiffness: FiniteF64::new(rng.unit_f64())?,
    })
}

fn sampled_erp(context: DistanceSamplingContext, rng: &mut ModelRng) -> CoreResult<ErpParameters> {
    Ok(ErpParameters {
        window: Some(sample_window(context.maximum_length, rng)?),
        gap: FiniteF64::new(sample_scaled(context.population_standard_deviation, rng))?,
    })
}

fn sampled_lcss(
    context: DistanceSamplingContext,
    rng: &mut ModelRng,
) -> CoreResult<LcssParameters> {
    Ok(LcssParameters {
        window: Some(sample_window(context.maximum_length, rng)?),
        epsilon: FiniteF64::new(sample_scaled(context.population_standard_deviation, rng))?,
    })
}

fn sampled_msm(rng: &mut ModelRng) -> CoreResult<MsmParameters> {
    Ok(MsmParameters {
        cost: FiniteF64::new(sample_from(MSM_COSTS, rng)?)?,
    })
}

fn sampled_twe(rng: &mut ModelRng) -> CoreResult<TweParameters> {
    Ok(TweParameters {
        stiffness: FiniteF64::new(sample_from(TWE_NU, rng)?)?,
        penalty: FiniteF64::new(sample_from(TWE_LAMBDA, rng)?)?,
    })
}

fn sample_window(maximum_length: usize, rng: &mut ModelRng) -> CoreResult<usize> {
    let upper_exclusive = maximum_length.div_ceil(4);
    if upper_exclusive <= 1 {
        Ok(0)
    } else {
        rng.index(upper_exclusive)
    }
}

fn sample_scaled(scale: f64, rng: &mut ModelRng) -> f64 {
    if scale == 0.0 {
        0.0
    } else {
        let lower = scale * 0.2;
        lower + rng.unit_f64() * (scale - lower)
    }
}

fn sample_from(values: &[f64], rng: &mut ModelRng) -> CoreResult<f64> {
    Ok(values[rng.index(values.len())?])
}

const MSM_COSTS: &[f64] = &[
    0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.0375, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
    0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.55, 0.6, 0.65,
    0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5,
    3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0,
    38.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
];
const TWE_NU: &[f64] = &[1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0];
const TWE_LAMBDA: &[f64] = &[
    0.0,
    0.011_111_111_111_111_112,
    0.022_222_222_222_222_223,
    0.033_333_333_333_333_33,
    0.044_444_444_444_444_446,
    0.055_555_555_555_555_555,
    0.066_666_666_666_666_67,
    0.077_777_777_777_777_78,
    0.088_888_888_888_888_89,
    0.1,
];
