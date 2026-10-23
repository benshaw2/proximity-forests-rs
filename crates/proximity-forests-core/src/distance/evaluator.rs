//! Prepared dispatch for built-in distances.

use crate::{
    CoreError, CoreResult, DistanceConfig, DistanceCutoff, DistanceScratch, MultivariateSeriesView,
    NumericVectorView, cid, cid_independent, cosine, cosine_selected, ddtw, ddtw_dependent,
    ddtw_independent, dtw, dtw_arow, dtw_arow_dependent, dtw_arow_independent, dtw_dependent,
    dtw_independent, erp, erp_independent, lcss, lcss_independent, manhattan,
    manhattan_independent, manhattan_selected, msm, msm_independent, nan_squared_euclidean,
    nan_squared_euclidean_independent, nan_squared_euclidean_selected, sbd, sbd_independent,
    shape_hog_1d_dtw, shape_hog_1d_dtw_independent, shape_hog_dtw, squared_euclidean,
    squared_euclidean_independent, squared_euclidean_selected, twe, twe_independent, wddtw,
    wddtw_dependent, wddtw_independent, wdtw, wdtw_dependent, wdtw_independent,
};

/// Prepared native distance evaluator with dispatch outside numerical kernels.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DistanceEvaluator {
    config: DistanceConfig,
}

impl DistanceEvaluator {
    /// Creates an evaluator from typed configuration.
    #[must_use]
    #[inline]
    pub const fn new(config: DistanceConfig) -> Self {
        Self { config }
    }

    /// Returns the evaluator configuration.
    #[must_use]
    #[inline]
    pub const fn config(&self) -> &DistanceConfig {
        &self.config
    }

    /// Evaluates compatible vector observations.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible inputs or an unsupported layout.
    #[inline]
    pub fn evaluate_vector(
        &self,
        first: NumericVectorView<'_>,
        second: NumericVectorView<'_>,
        selected: Option<&[usize]>,
        cutoff: DistanceCutoff,
        scratch: &mut DistanceScratch,
    ) -> CoreResult<f64> {
        match &self.config {
            DistanceConfig::SquaredEuclidean => match selected {
                Some(selected) => squared_euclidean_selected(first, second, selected, cutoff),
                None => squared_euclidean(first, second, cutoff),
            },
            DistanceConfig::Manhattan => match selected {
                Some(selected) => manhattan_selected(first, second, selected, cutoff),
                None => manhattan(first, second, cutoff),
            },
            DistanceConfig::Cosine => match selected {
                Some(selected) => cosine_selected(first, second, selected, cutoff),
                None => cosine(first, second, cutoff),
            },
            DistanceConfig::Cid => cid(first, second, cutoff),
            DistanceConfig::Sbd => sbd(first, second, cutoff),
            DistanceConfig::NaNSquaredEuclidean { no_overlap } => match selected {
                Some(selected) => {
                    nan_squared_euclidean_selected(first, second, selected, cutoff, *no_overlap)
                }
                None => nan_squared_euclidean(first, second, cutoff, *no_overlap),
            },
            DistanceConfig::Dtw(parameters) => dtw(first, second, *parameters, cutoff, scratch),
            DistanceConfig::Ddtw(parameters) => ddtw(first, second, *parameters, cutoff, scratch),
            DistanceConfig::Wdtw(parameters) => wdtw(first, second, *parameters, cutoff, scratch),
            DistanceConfig::Wddtw(parameters) => wddtw(first, second, *parameters, cutoff, scratch),
            DistanceConfig::Erp(parameters) => erp(first, second, *parameters, cutoff, scratch),
            DistanceConfig::Lcss(parameters) => lcss(first, second, *parameters, cutoff, scratch),
            DistanceConfig::Msm(parameters) => msm(first, second, *parameters, cutoff, scratch),
            DistanceConfig::Twe(parameters) => twe(first, second, *parameters, cutoff, scratch),
            DistanceConfig::ShapeHog1dDtw(parameters) => {
                shape_hog_1d_dtw(first, second, *parameters, cutoff, scratch)
            }
            DistanceConfig::DtwArow(parameters) => {
                dtw_arow(first, second, *parameters, cutoff, scratch)
            }
            config => Err(wrong_layout(config, "vector")),
        }
    }

    /// Evaluates compatible multivariate series observations.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible inputs or an unsupported layout.
    #[inline]
    pub fn evaluate_multivariate(
        &self,
        first: MultivariateSeriesView<'_>,
        second: MultivariateSeriesView<'_>,
        selected: Option<&[usize]>,
        cutoff: DistanceCutoff,
        scratch: &mut DistanceScratch,
    ) -> CoreResult<f64> {
        match &self.config {
            DistanceConfig::SquaredEuclideanIndependent => {
                squared_euclidean_independent(first, second, selected, cutoff)
            }
            DistanceConfig::ManhattanIndependent => {
                manhattan_independent(first, second, selected, cutoff)
            }
            DistanceConfig::CidIndependent => cid_independent(first, second, selected, cutoff),
            DistanceConfig::SbdIndependent => sbd_independent(first, second, selected, cutoff),
            DistanceConfig::NaNSquaredEuclideanIndependent { no_overlap } => {
                nan_squared_euclidean_independent(first, second, selected, cutoff, *no_overlap)
            }
            DistanceConfig::DtwIndependent(parameters) => {
                dtw_independent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::DtwDependent(parameters) => {
                dtw_dependent(first, second, selected, parameters.window, cutoff, scratch)
            }
            DistanceConfig::DdtwIndependent(parameters) => {
                ddtw_independent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::DdtwDependent(parameters) => {
                ddtw_dependent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::WdtwIndependent(parameters) => {
                wdtw_independent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::WdtwDependent(parameters) => {
                wdtw_dependent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::WddtwIndependent(parameters) => {
                wddtw_independent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::WddtwDependent(parameters) => {
                wddtw_dependent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::ErpIndependent(parameters) => {
                erp_independent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::LcssIndependent(parameters) => {
                lcss_independent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::MsmIndependent(parameters) => {
                msm_independent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::TweIndependent(parameters) => {
                twe_independent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::ShapeHog1dDtwIndependent(parameters) => {
                shape_hog_1d_dtw_independent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::ShapeHogDtw(parameters) => {
                shape_hog_dtw(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::DtwArowIndependent(parameters) => {
                dtw_arow_independent(first, second, selected, *parameters, cutoff, scratch)
            }
            DistanceConfig::DtwArowDependent(parameters) => {
                dtw_arow_dependent(first, second, selected, *parameters, cutoff, scratch)
            }
            config => Err(wrong_layout(config, "multivariate series")),
        }
    }
}

#[cold]
fn wrong_layout(config: &DistanceConfig, layout: &str) -> CoreError {
    CoreError::unsupported(format!(
        "the {:?} distance does not support {layout} observations",
        config.kind()
    ))
}
