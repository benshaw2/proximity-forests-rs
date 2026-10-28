//! High-performance modeling core for Proximity Forests.
//!
//! This crate contains Python-independent model logic. Python bindings,
//! array conversion, packaging, and estimator ergonomics belong in the
//! separate binding crate.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations, missing_docs, unreachable_pub)]

pub mod data;
pub mod distance;
pub mod error;
pub mod forest;
pub mod purity;
pub mod random;
pub mod scalar;
pub mod target;

pub use data::{DataLayout, MultivariateSeriesView, NumericVectorView, TimeAxisView, VectorRole};
pub use distance::{
    ClosestExemplar, DistanceCapabilities, DistanceChoice, DistanceConfig, DistanceCutoff,
    DistanceEvaluator, DistanceKind, DistanceSamplingContext, DistanceSamplingPolicy,
    DistanceScratch, DistanceWorkShape, DtwArowParameters, DtwParameters, ErpParameters, FiniteF64,
    JavaCompatibleDistanceSampling, LcssParameters, MsmParameters, NoOverlapPolicy,
    SampledDistanceFamily, ShapeDtwParameters, TweParameters, WeightedDtwParameters, WorkUnits,
    cid, cid_independent, closest_exemplar, cosine, cosine_selected, ddtw, ddtw_dependent,
    ddtw_independent, derivative_transform, dtw, dtw_arow, dtw_arow_dependent,
    dtw_arow_independent, dtw_dependent, dtw_independent, erp, erp_independent, lcss,
    lcss_independent, manhattan, manhattan_independent, manhattan_selected, msm, msm_independent,
    nan_squared_euclidean, nan_squared_euclidean_independent, nan_squared_euclidean_selected,
    resolve_distance_choice, sbd, sbd_independent, shape_hog_1d_dtw, shape_hog_1d_dtw_independent,
    shape_hog_dtw, squared_euclidean, squared_euclidean_independent, squared_euclidean_selected,
    twe, twe_independent, wddtw, wddtw_dependent, wddtw_independent, wdtw, wdtw_dependent,
    wdtw_independent,
};
pub use error::{CoreError, CoreResult};
pub use forest::{
    BootstrapSample, ClassificationConfig, DimensionSelection, ForestConfig, IsolationConfig,
    NodeMembership, RegressionConfig, SampleId, TiePolicy, sample_dimensions,
    sample_weighted_without_replacement,
};
pub use purity::{IsolationCriterion, SupervisedCriterion};
pub use random::{ModelRng, RandomPurpose, SeedMixer};
pub use scalar::ScalarKind;
pub use target::{ClassId, ClassificationTargets, RegressionTargets};
