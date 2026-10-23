//! Static capabilities advertised by a distance implementation.

/// Capabilities used to validate a configured distance before training.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct DistanceCapabilities {
    /// Supports selected tabular features or multivariate channels.
    pub selected_dimensions: bool,

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
