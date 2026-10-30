//! Fitted forest containers owning all exemplar observations.

use std::path::{Path, PathBuf};

use crate::{ForestConfig, ScalarKind, TiePolicy};

use super::{OobMembership, OwnedObservationStore, ProximityTree, TreeSample};

/// Policy controlling complete training-data retention.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum DataRetention {
    /// Retain fitted exemplars and compact analytical metadata only.
    #[default]
    Exemplars,
    /// Embed an immutable copy of the complete fitting collection.
    Copy,
    /// Resolve the complete fitting collection through an external descriptor.
    Reference,
}

/// Persistent external training-data format.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExternalDataFormat {
    /// A NumPy `.npy` file that can be reopened read-only and memory mapped.
    Npy,
}

/// Serializable descriptor for an externally retained fitting collection.
///
/// The descriptor owns no live pointer or file handle. A runtime adapter must
/// reopen the path and validate the recorded metadata before exposing values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalTrainingData {
    path: PathBuf,
    format: ExternalDataFormat,
    scalar_kind: ScalarKind,
    shape: Box<[usize]>,
    payload_offset: u64,
    payload_length: u64,
    file_size: u64,
    fingerprint: Option<Box<str>>,
}

impl ExternalTrainingData {
    /// Creates an external training-data descriptor.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        path: PathBuf,
        format: ExternalDataFormat,
        scalar_kind: ScalarKind,
        shape: Box<[usize]>,
        payload_offset: u64,
        payload_length: u64,
        file_size: u64,
        fingerprint: Option<Box<str>>,
    ) -> Self {
        Self {
            path,
            format,
            scalar_kind,
            shape,
            payload_offset,
            payload_length,
            file_size,
            fingerprint,
        }
    }

    /// Returns the persistent container path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the external container format.
    #[must_use]
    pub const fn format(&self) -> ExternalDataFormat {
        self.format
    }

    /// Returns the numeric scalar kind.
    #[must_use]
    pub const fn scalar_kind(&self) -> ScalarKind {
        self.scalar_kind
    }

    /// Returns the logical observation shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the numeric payload offset from the start of the file.
    #[must_use]
    pub const fn payload_offset(&self) -> u64 {
        self.payload_offset
    }

    /// Returns the numeric payload byte length.
    #[must_use]
    pub const fn payload_length(&self) -> u64 {
        self.payload_length
    }

    /// Returns the file size recorded at fit time.
    #[must_use]
    pub const fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Returns the optional content fingerprint.
    #[must_use]
    pub fn fingerprint(&self) -> Option<&str> {
        self.fingerprint.as_deref()
    }
}

/// Complete training data retained by the fitted forest.
#[derive(Clone, Debug, PartialEq)]
pub enum RetainedTrainingData {
    /// No complete raw training collection is embedded.
    None,
    /// Complete immutable model-owned fitting collection.
    Owned(OwnedObservationStore),
    /// Persistent external descriptor used to reopen donor values lazily.
    Reference(ExternalTrainingData),
}

/// Bootstrap/OOB metadata retained per tree for analytical proximities.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingMetadata {
    sample_count: usize,
    bootstrap: Vec<TreeSample>,
}

impl RetainedTrainingData {
    /// Reports whether complete donor values are available through this model.
    #[must_use]
    pub const fn has_training_data(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Returns an embedded complete fitting collection when present.
    #[must_use]
    pub const fn owned(&self) -> Option<&OwnedObservationStore> {
        match self {
            Self::Owned(store) => Some(store),
            Self::None | Self::Reference(_) => None,
        }
    }

    /// Returns the external descriptor when reference retention is active.
    #[must_use]
    pub const fn external(&self) -> Option<&ExternalTrainingData> {
        match self {
            Self::Reference(descriptor) => Some(descriptor),
            Self::None | Self::Owned(_) => None,
        }
    }
}

impl TrainingMetadata {
    /// Creates compact per-tree fitting metadata.
    #[must_use]
    pub const fn new(sample_count: usize, bootstrap: Vec<TreeSample>) -> Self {
        Self {
            sample_count,
            bootstrap,
        }
    }

    /// Returns training sample count.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns bootstrap and OOB metadata in tree order.
    #[must_use]
    pub fn bootstrap(&self) -> &[TreeSample] {
        &self.bootstrap
    }

    /// Derives compact sample-to-tree OOB membership from persisted tree samples.
    ///
    /// The derived index is intentionally not serialized; callers that compute
    /// OOB results build it once per operation and reuse its contiguous rows.
    pub fn oob_membership(&self) -> crate::CoreResult<OobMembership> {
        OobMembership::from_tree_samples(self.sample_count, &self.bootstrap)
    }
}

/// Shared fitted forest state.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedForest {
    config: ForestConfig,
    trees: Vec<ProximityTree>,
    retained_training_data: RetainedTrainingData,
    training_metadata: TrainingMetadata,
}

impl FittedForest {
    /// Creates a complete fitted forest.
    #[must_use]
    pub fn new(
        config: ForestConfig,
        trees: Vec<ProximityTree>,
        retained_training_data: RetainedTrainingData,
        training_metadata: TrainingMetadata,
    ) -> Self {
        Self {
            config,
            trees,
            retained_training_data,
            training_metadata,
        }
    }

    /// Returns fitted trees in stable tree-index order.
    #[must_use]
    pub fn trees(&self) -> &[ProximityTree] {
        &self.trees
    }

    /// Returns model-owned exemplar observations.
    #[must_use]
    pub const fn retained_training_data(&self) -> &RetainedTrainingData {
        &self.retained_training_data
    }

    /// Returns compact bootstrap/OOB metadata.
    #[must_use]
    pub const fn training_metadata(&self) -> &TrainingMetadata {
        &self.training_metadata
    }

    /// Returns the configured retention policy.
    #[must_use]
    pub const fn data_retention(&self) -> DataRetention {
        self.config.data_retention
    }

    /// Reports whether complete donor values are embedded or externally resolvable.
    #[must_use]
    pub const fn has_training_data(&self) -> bool {
        self.retained_training_data.has_training_data()
    }

    /// Reports whether complete donor values are embedded in the model artifact.
    #[must_use]
    pub const fn has_embedded_training_data(&self) -> bool {
        matches!(self.retained_training_data, RetainedTrainingData::Owned(_))
    }

    /// Returns fitted configuration.
    #[must_use]
    pub const fn config(&self) -> &ForestConfig {
        &self.config
    }

    /// Returns prediction-time branch tie policy.
    #[must_use]
    pub const fn tie_policy(&self) -> TiePolicy {
        self.config.branch_tie_policy
    }
}

/// Fitted classification forest.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationForest {
    forest: FittedForest,
    output_count: usize,
}

impl ClassificationForest {
    /// Serializes the fitted classifier to the stable versioned binary format.
    pub fn to_bytes(&self) -> crate::CoreResult<Vec<u8>> {
        super::serialization::classification_to_bytes(self)
    }

    /// Restores a fitted classifier from the stable versioned binary format.
    pub fn from_bytes(bytes: &[u8]) -> crate::CoreResult<Self> {
        super::serialization::classification_from_bytes(bytes)
    }

    /// Creates a fitted classification forest.
    #[must_use]
    pub const fn new(forest: FittedForest, output_count: usize) -> Self {
        Self {
            forest,
            output_count,
        }
    }

    /// Returns shared fitted forest state.
    #[must_use]
    pub const fn forest(&self) -> &FittedForest {
        &self.forest
    }

    /// Returns classification output count.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }
}

/// Fitted regression forest.
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionForest {
    forest: FittedForest,
    output_count: usize,
}

impl RegressionForest {
    /// Serializes the fitted regressor to the stable versioned binary format.
    pub fn to_bytes(&self) -> crate::CoreResult<Vec<u8>> {
        super::serialization::regression_to_bytes(self)
    }

    /// Restores a fitted regressor from the stable versioned binary format.
    pub fn from_bytes(bytes: &[u8]) -> crate::CoreResult<Self> {
        super::serialization::regression_from_bytes(bytes)
    }

    /// Creates a fitted regression forest.
    #[must_use]
    pub const fn new(forest: FittedForest, output_count: usize) -> Self {
        Self {
            forest,
            output_count,
        }
    }

    /// Returns shared fitted forest state.
    #[must_use]
    pub const fn forest(&self) -> &FittedForest {
        &self.forest
    }

    /// Returns regression output count.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }
}

/// Fitted isolation forest.
#[derive(Clone, Debug, PartialEq)]
pub struct IsolationForest {
    forest: FittedForest,
    normalization_size: usize,
}

impl IsolationForest {
    /// Serializes the fitted isolation forest to the stable versioned binary format.
    pub fn to_bytes(&self) -> crate::CoreResult<Vec<u8>> {
        super::serialization::isolation_to_bytes(self)
    }

    /// Restores a fitted isolation forest from the stable versioned binary format.
    pub fn from_bytes(bytes: &[u8]) -> crate::CoreResult<Self> {
        super::serialization::isolation_from_bytes(bytes)
    }

    /// Creates a fitted isolation forest.
    #[must_use]
    pub const fn new(forest: FittedForest, normalization_size: usize) -> Self {
        Self {
            forest,
            normalization_size,
        }
    }

    /// Returns shared fitted forest state.
    #[must_use]
    pub const fn forest(&self) -> &FittedForest {
        &self.forest
    }

    /// Returns path-length normalization sample size.
    #[must_use]
    pub const fn normalization_size(&self) -> usize {
        self.normalization_size
    }
}
