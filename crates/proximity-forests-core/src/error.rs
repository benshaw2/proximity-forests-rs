//! Errors produced by the Proximity Forests core.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Result type used by fallible core operations.
pub type CoreResult<T> = Result<T, CoreError>;

/// Error returned when model configuration, data, or fitted state is invalid.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CoreError {
    /// A user-supplied configuration value is invalid.
    InvalidConfiguration {
        /// Name of the invalid setting.
        setting: &'static str,
        /// Human-readable explanation of the violated requirement.
        reason: String,
    },

    /// Input data does not satisfy the requested operation's contract.
    InvalidData {
        /// Human-readable explanation of the data problem.
        reason: String,
    },

    /// Two related inputs have incompatible shapes or lengths.
    ShapeMismatch {
        /// Human-readable explanation of the mismatch.
        reason: String,
    },

    /// An operation requires capabilities that the selected component lacks.
    UnsupportedOperation {
        /// Human-readable explanation of the missing capability.
        reason: String,
    },

    /// An operation requires a fitted model.
    NotFitted,
}

impl CoreError {
    /// Creates an invalid-configuration error.
    #[must_use]
    pub fn invalid_configuration(setting: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidConfiguration {
            setting,
            reason: reason.into(),
        }
    }

    /// Creates an invalid-data error.
    #[must_use]
    pub fn invalid_data(reason: impl Into<String>) -> Self {
        Self::InvalidData {
            reason: reason.into(),
        }
    }

    /// Creates a shape-mismatch error.
    #[must_use]
    pub fn shape_mismatch(reason: impl Into<String>) -> Self {
        Self::ShapeMismatch {
            reason: reason.into(),
        }
    }

    /// Creates an unsupported-operation error.
    #[must_use]
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self::UnsupportedOperation {
            reason: reason.into(),
        }
    }
}

impl Display for CoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration { setting, reason } => {
                write!(formatter, "invalid configuration `{setting}`: {reason}")
            }
            Self::InvalidData { reason } => write!(formatter, "invalid data: {reason}"),
            Self::ShapeMismatch { reason } => write!(formatter, "shape mismatch: {reason}"),
            Self::UnsupportedOperation { reason } => {
                write!(formatter, "unsupported operation: {reason}")
            }
            Self::NotFitted => formatter.write_str("the model is not fitted"),
        }
    }
}

impl Error for CoreError {}
