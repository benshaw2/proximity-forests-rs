//! Static capabilities advertised by a distance implementation.

/// Layout semantics for selecting a subset of input dimensions.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum DimensionSelectionSupport {
    /// Dimension selection is not meaningful for this distance.
    ///
    /// Vector positions are typically ordered time points rather than
    /// exchangeable tabular features.
    #[default]
    None,

    /// Selected vector positions are interpreted as tabular features.
    VectorFeatures,

    /// Selected multivariate dimensions are interpreted as channels.
    MultivariateChannels,
}

impl DimensionSelectionSupport {
    /// Reports whether vector-layout feature subsets are supported.
    #[must_use]
    pub const fn supports_vector_features(self) -> bool {
        matches!(self, Self::VectorFeatures)
    }

    /// Reports whether multivariate channel subsets are supported.
    #[must_use]
    pub const fn supports_multivariate_channels(self) -> bool {
        matches!(self, Self::MultivariateChannels)
    }
}

/// Capabilities used to validate a configured distance before training.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct DistanceCapabilities {
    /// Describes whether and how selected dimensions are meaningful.
    pub dimension_selection: DimensionSelectionSupport,

    /// Supports unequal series lengths.
    pub unequal_lengths: bool,

    /// Supports early abandoning from a best-so-far cutoff.
    pub early_abandoning: bool,

    /// Defines explicit missing-value semantics.
    pub missing_values: bool,

    /// Uses optional time-coordinate metadata.
    pub time_coordinates: bool,

    /// Supports batched evaluation through a specialized path.
    pub batch_evaluation: bool,
}
