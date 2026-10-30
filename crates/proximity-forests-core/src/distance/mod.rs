//! Native distance contracts and built-in implementation catalog.

mod arow;
mod capabilities;
mod choice;
mod closest;
mod complexity;
mod config;
mod correlation;
mod cutoff;
mod derivative;
mod dtw;
mod erp;
mod euclidean;
mod evaluator;
mod kind;
mod lcss;
mod missing;
mod missing_multivariate;
mod msm;
mod multivariate_euclidean;
mod sampling;
mod scratch;
mod shape;
mod twe;
mod vector;
mod weighted_dtw;
mod work;

pub use arow::{dtw_arow, dtw_arow_dependent, dtw_arow_independent};
pub use capabilities::{DimensionSelectionSupport, DistanceCapabilities};
pub use choice::{
    DistanceChoice, DistanceSamplingContext, DistanceSamplingPolicy, SampledDistanceFamily,
    resolve_distance_choice,
};
pub use closest::{ClosestExemplar, closest_exemplar};
pub use complexity::{cid, cid_independent};
pub use config::{
    DistanceConfig, DtwArowParameters, DtwParameters, ErpParameters, FiniteF64, LcssParameters,
    MsmParameters, ShapeDtwParameters, TweParameters, WeightedDtwParameters,
};
pub use correlation::{sbd, sbd_independent};
pub use cutoff::DistanceCutoff;
pub use derivative::{ddtw, ddtw_dependent, ddtw_independent, derivative_transform};
pub use dtw::{dtw, dtw_dependent, dtw_independent};
pub use erp::{erp, erp_independent};
pub use euclidean::{squared_euclidean, squared_euclidean_selected};
pub use evaluator::DistanceEvaluator;
pub use kind::DistanceKind;
pub use lcss::{lcss, lcss_independent};
pub use missing::{NoOverlapPolicy, nan_squared_euclidean, nan_squared_euclidean_selected};
pub use missing_multivariate::nan_squared_euclidean_independent;
pub use msm::{msm, msm_independent};
pub use multivariate_euclidean::squared_euclidean_independent;
pub use sampling::JavaCompatibleDistanceSampling;
pub use scratch::DistanceScratch;
pub use shape::{shape_hog_1d_dtw, shape_hog_1d_dtw_independent, shape_hog_dtw};
pub use twe::{twe, twe_independent};
pub use vector::{cosine, cosine_selected, manhattan, manhattan_independent, manhattan_selected};
pub use weighted_dtw::{
    wddtw, wddtw_dependent, wddtw_independent, wdtw, wdtw_dependent, wdtw_independent,
};
pub use work::{DistanceWorkShape, WorkUnits};
