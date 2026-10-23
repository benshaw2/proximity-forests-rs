//! Typed purposes used to isolate independent random streams.

/// Semantic purpose for a deterministic random stream.
///
/// Explicit discriminants are persisted as part of the reproducibility
/// contract and must not be renumbered after release.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u64)]
pub enum RandomPurpose {
    /// Bootstrap sampling for one tree.
    Bootstrap = 1,
    /// Selection of dimensions at one node.
    Dimensions = 2,
    /// Selection of a candidate distance configuration.
    CandidateDistance = 3,
    /// Selection of candidate exemplars.
    CandidateExemplars = 4,
    /// Assignment of one observation to a branch, including tie resolution.
    BranchAssignment = 5,
    /// Selection of a classification output for exemplar grouping.
    ClassificationOutput = 6,
    /// Prediction-time tie resolution.
    PredictionTie = 7,
    /// Majority-vote tie resolution.
    MajorityVoteTie = 8,
}

impl RandomPurpose {
    /// Returns the stable numeric discriminator for seed derivation.
    #[must_use]
    pub const fn discriminator(self) -> u64 {
        self as u64
    }
}
