//! Typed configuration for the native distance catalog.

use super::{DistanceCapabilities, DistanceKind, NoOverlapPolicy};

/// Finite floating-point parameter stored by IEEE-754 bits.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FiniteF64(u64);

impl FiniteF64 {
    /// Creates a finite parameter.
    ///
    /// # Errors
    ///
    /// Returns an error for NaN or infinity.
    pub fn new(value: f64) -> crate::CoreResult<Self> {
        if !value.is_finite() {
            return Err(crate::CoreError::invalid_configuration(
                "distance_parameter",
                "distance parameters must be finite",
            ));
        }
        Ok(Self(value.to_bits()))
    }

    /// Returns the stored value.
    #[must_use]
    pub const fn get(self) -> f64 {
        f64::from_bits(self.0)
    }
}

/// Optional absolute warping-window constraint.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct DtwParameters {
    /// Maximum displacement, or no explicit constraint.
    pub window: Option<usize>,
}

/// Weighted-DTW parameters.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WeightedDtwParameters {
    /// Maximum displacement, or no explicit constraint.
    pub window: Option<usize>,
    /// Logistic weighting stiffness.
    pub stiffness: FiniteF64,
}

/// ERP parameters.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ErpParameters {
    /// Maximum displacement, or no explicit constraint.
    pub window: Option<usize>,
    /// Gap reference value.
    pub gap: FiniteF64,
}

/// LCSS parameters.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LcssParameters {
    /// Maximum displacement, or no explicit constraint.
    pub window: Option<usize>,
    /// Matching tolerance.
    pub epsilon: FiniteF64,
}

/// MSM parameters.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MsmParameters {
    /// Move/split/merge operation cost.
    pub cost: FiniteF64,
}

/// TWE parameters.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TweParameters {
    /// Temporal stiffness, traditionally `nu`.
    pub stiffness: FiniteF64,
    /// Edit penalty, traditionally `lambda`.
    pub penalty: FiniteF64,
}

/// ShapeHoG-DTW parameters.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ShapeDtwParameters {
    /// Optional DTW window over histogram vectors.
    pub window: Option<usize>,
}

/// DTW-AROW parameters.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct DtwArowParameters {
    /// Optional maximum displacement. A finite window is never widened.
    pub window: Option<usize>,
}

/// Fully specified native distance configuration.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum DistanceConfig {
    /// Squared Euclidean distance.
    SquaredEuclidean,
    /// Manhattan distance.
    Manhattan,
    /// Cosine distance.
    Cosine,
    /// Complexity-invariant distance.
    Cid,
    /// Shape-based distance.
    Sbd,
    /// Dynamic time warping.
    Dtw(DtwParameters),
    /// Derivative DTW.
    Ddtw(DtwParameters),
    /// Weighted DTW.
    Wdtw(WeightedDtwParameters),
    /// Weighted derivative DTW.
    Wddtw(WeightedDtwParameters),
    /// Edit distance with real penalty.
    Erp(ErpParameters),
    /// Longest common subsequence distance.
    Lcss(LcssParameters),
    /// Move-split-merge distance.
    Msm(MsmParameters),
    /// Time warp edit distance.
    Twe(TweParameters),
    /// One-dimensional ShapeHoG-DTW.
    ShapeHog1dDtw(ShapeDtwParameters),
    /// Independent squared Euclidean distance.
    SquaredEuclideanIndependent,
    /// Independent Manhattan distance.
    ManhattanIndependent,
    /// Independent CID.
    CidIndependent,
    /// Independent SBD.
    SbdIndependent,
    /// Independent DTW.
    DtwIndependent(DtwParameters),
    /// Independent derivative DTW.
    DdtwIndependent(DtwParameters),
    /// Independent weighted DTW.
    WdtwIndependent(WeightedDtwParameters),
    /// Independent weighted derivative DTW.
    WddtwIndependent(WeightedDtwParameters),
    /// Independent ERP.
    ErpIndependent(ErpParameters),
    /// Independent LCSS.
    LcssIndependent(LcssParameters),
    /// Independent MSM.
    MsmIndependent(MsmParameters),
    /// Independent TWE.
    TweIndependent(TweParameters),
    /// Independent one-dimensional ShapeHoG-DTW.
    ShapeHog1dDtwIndependent(ShapeDtwParameters),
    /// Dependent DTW.
    DtwDependent(DtwParameters),
    /// Dependent derivative DTW.
    DdtwDependent(DtwParameters),
    /// Dependent weighted DTW.
    WdtwDependent(WeightedDtwParameters),
    /// Dependent weighted derivative DTW.
    WddtwDependent(WeightedDtwParameters),
    /// Dependent multivariate ShapeHoG-DTW.
    ShapeHogDtw(ShapeDtwParameters),
    /// NaN-aware squared Euclidean distance.
    NaNSquaredEuclidean {
        /// Behavior when no positions overlap.
        no_overlap: NoOverlapPolicy,
    },
    /// Independent NaN-aware squared Euclidean distance.
    NaNSquaredEuclideanIndependent {
        /// Behavior when a channel has no overlap.
        no_overlap: NoOverlapPolicy,
    },
    /// Univariate DTW-AROW.
    DtwArow(DtwArowParameters),
    /// Independent DTW-AROW.
    DtwArowIndependent(DtwArowParameters),
    /// Dependent DTW-AROW.
    DtwArowDependent(DtwArowParameters),
}

impl DistanceConfig {
    /// Returns the canonical distance kind.
    #[must_use]
    pub const fn kind(&self) -> DistanceKind {
        use DistanceConfig as C;
        match self {
            C::SquaredEuclidean => DistanceKind::SquaredEuclidean,
            C::Manhattan => DistanceKind::Manhattan,
            C::Cosine => DistanceKind::Cosine,
            C::Cid => DistanceKind::Cid,
            C::Sbd => DistanceKind::Sbd,
            C::Dtw(_) => DistanceKind::Dtw,
            C::Ddtw(_) => DistanceKind::Ddtw,
            C::Wdtw(_) => DistanceKind::Wdtw,
            C::Wddtw(_) => DistanceKind::Wddtw,
            C::Erp(_) => DistanceKind::Erp,
            C::Lcss(_) => DistanceKind::Lcss,
            C::Msm(_) => DistanceKind::Msm,
            C::Twe(_) => DistanceKind::Twe,
            C::ShapeHog1dDtw(_) => DistanceKind::ShapeHog1dDtw,
            C::SquaredEuclideanIndependent => DistanceKind::SquaredEuclideanIndependent,
            C::ManhattanIndependent => DistanceKind::ManhattanIndependent,
            C::CidIndependent => DistanceKind::CidIndependent,
            C::SbdIndependent => DistanceKind::SbdIndependent,
            C::DtwIndependent(_) => DistanceKind::DtwIndependent,
            C::DdtwIndependent(_) => DistanceKind::DdtwIndependent,
            C::WdtwIndependent(_) => DistanceKind::WdtwIndependent,
            C::WddtwIndependent(_) => DistanceKind::WddtwIndependent,
            C::ErpIndependent(_) => DistanceKind::ErpIndependent,
            C::LcssIndependent(_) => DistanceKind::LcssIndependent,
            C::MsmIndependent(_) => DistanceKind::MsmIndependent,
            C::TweIndependent(_) => DistanceKind::TweIndependent,
            C::ShapeHog1dDtwIndependent(_) => DistanceKind::ShapeHog1dDtwIndependent,
            C::DtwDependent(_) => DistanceKind::DtwDependent,
            C::DdtwDependent(_) => DistanceKind::DdtwDependent,
            C::WdtwDependent(_) => DistanceKind::WdtwDependent,
            C::WddtwDependent(_) => DistanceKind::WddtwDependent,
            C::ShapeHogDtw(_) => DistanceKind::ShapeHogDtw,
            C::NaNSquaredEuclidean { .. } => DistanceKind::NaNSquaredEuclidean,
            C::NaNSquaredEuclideanIndependent { .. } => {
                DistanceKind::NaNSquaredEuclideanIndependent
            }
            C::DtwArow(_) => DistanceKind::DtwArow,
            C::DtwArowIndependent(_) => DistanceKind::DtwArowIndependent,
            C::DtwArowDependent(_) => DistanceKind::DtwArowDependent,
        }
    }

    /// Returns static estimator-validation capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> DistanceCapabilities {
        capabilities_for(self.kind())
    }

    /// Reports whether the numerical kernel is implemented.
    #[must_use]
    pub const fn is_implemented(&self) -> bool {
        true
    }
}

const fn capabilities_for(kind: DistanceKind) -> DistanceCapabilities {
    use DistanceKind as K;
    let missing = matches!(
        kind,
        K::NaNSquaredEuclidean
            | K::NaNSquaredEuclideanIndependent
            | K::DtwArow
            | K::DtwArowIndependent
            | K::DtwArowDependent
    );
    let unequal = matches!(
        kind,
        K::Dtw
            | K::Ddtw
            | K::Wdtw
            | K::Wddtw
            | K::Erp
            | K::Lcss
            | K::Msm
            | K::Twe
            | K::ShapeHog1dDtw
            | K::DtwIndependent
            | K::DdtwIndependent
            | K::WdtwIndependent
            | K::WddtwIndependent
            | K::ErpIndependent
            | K::LcssIndependent
            | K::MsmIndependent
            | K::TweIndependent
            | K::ShapeHog1dDtwIndependent
            | K::DtwDependent
            | K::DdtwDependent
            | K::WdtwDependent
            | K::WddtwDependent
            | K::ShapeHogDtw
            | K::DtwArow
            | K::DtwArowIndependent
            | K::DtwArowDependent
    );
    let early = !matches!(
        kind,
        K::Cosine | K::Sbd | K::SbdIndependent | K::Lcss | K::LcssIndependent
    );
    DistanceCapabilities {
        selected_dimensions: true,
        unequal_lengths: unequal,
        early_abandoning: early,
        missing_values: missing,
        time_coordinates: false,
        batch_evaluation: false,
    }
}
