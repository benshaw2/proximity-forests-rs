//! Stable identities for native built-in distance families.

/// Canonical native distance identity.
///
/// Variants cover the planned tabular, univariate, independent multivariate,
/// dependent multivariate, shape-based, and missing-compatible catalog. Graph,
/// language-interop, and meta-file distances are intentionally excluded.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum DistanceKind {
    // Vector/tabular and univariate distances.
    /// Squared Euclidean distance for complete vectors.
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
    Dtw,
    /// Derivative dynamic time warping.
    Ddtw,
    /// Weighted dynamic time warping.
    Wdtw,
    /// Weighted derivative dynamic time warping.
    Wddtw,
    /// Edit distance with real penalty.
    Erp,
    /// Longest common subsequence distance.
    Lcss,
    /// Move-split-merge distance.
    Msm,
    /// Time warp edit distance.
    Twe,
    /// One-dimensional histogram-of-gradients shape DTW.
    ShapeHog1dDtw,

    // Independent multivariate distances.
    /// Sum of per-channel squared Euclidean distances.
    SquaredEuclideanIndependent,
    /// Sum of per-channel Manhattan distances.
    ManhattanIndependent,
    /// Independent multivariate complexity-invariant distance.
    CidIndependent,
    /// Independent multivariate shape-based distance.
    SbdIndependent,
    /// Independent multivariate DTW.
    DtwIndependent,
    /// Independent multivariate derivative DTW.
    DdtwIndependent,
    /// Independent multivariate weighted DTW.
    WdtwIndependent,
    /// Independent multivariate weighted derivative DTW.
    WddtwIndependent,
    /// Independent multivariate ERP.
    ErpIndependent,
    /// Independent multivariate LCSS.
    LcssIndependent,
    /// Independent multivariate MSM.
    MsmIndependent,
    /// Independent multivariate TWE.
    TweIndependent,
    /// Independent multivariate one-dimensional shape DTW.
    ShapeHog1dDtwIndependent,

    // Dependent multivariate distances.
    /// Dependent multivariate DTW.
    DtwDependent,
    /// Dependent multivariate derivative DTW.
    DdtwDependent,
    /// Dependent multivariate weighted DTW.
    WdtwDependent,
    /// Dependent multivariate weighted derivative DTW.
    WddtwDependent,
    /// Dependent multivariate histogram-of-gradients shape DTW.
    ShapeHogDtw,

    // Missing-compatible distances.
    /// NaN-aware squared Euclidean distance for incomplete vectors.
    NaNSquaredEuclidean,
    /// Independent multivariate NaN-aware squared Euclidean distance.
    NaNSquaredEuclideanIndependent,
    /// Missing-compatible univariate DTW-AROW.
    DtwArow,
    /// Missing-compatible independent multivariate DTW-AROW.
    DtwArowIndependent,
    /// Missing-compatible dependent multivariate DTW-AROW.
    DtwArowDependent,
}
