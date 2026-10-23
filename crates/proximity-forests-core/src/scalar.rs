//! Numeric storage kinds supported by the core model.

use std::fmt::{Display, Formatter};
use std::str::FromStr;

use crate::error::CoreError;

/// Numeric dtype used to store observations and numeric targets.
///
/// Storage precision is intentionally separate from accumulation precision.
/// Distances, impurity scores, predictions, and proximities may accumulate in
/// `f64` even when observations are stored as `f32`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ScalarKind {
    /// IEEE-754 single-precision storage.
    F32,
    /// IEEE-754 double-precision storage.
    F64,
}

impl ScalarKind {
    /// Returns the number of bytes occupied by one scalar value.
    #[must_use]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
            Self::F64 => size_of::<f64>(),
        }
    }

    /// Returns the canonical NumPy-compatible dtype name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }
}

impl Display for ScalarKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ScalarKind {
    type Err = CoreError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" => Ok(Self::F32),
            "f64" | "float64" | "double" => Ok(Self::F64),
            _ => Err(CoreError::invalid_configuration(
                "dtype",
                format!("unsupported numeric dtype `{value}`; expected float32 or float64"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ScalarKind;
    use std::str::FromStr;

    #[test]
    fn parses_supported_names_case_insensitively() {
        assert_eq!(ScalarKind::from_str("f32"), Ok(ScalarKind::F32));
        assert_eq!(ScalarKind::from_str("FLOAT32"), Ok(ScalarKind::F32));
        assert_eq!(ScalarKind::from_str("f64"), Ok(ScalarKind::F64));
        assert_eq!(ScalarKind::from_str("double"), Ok(ScalarKind::F64));
    }

    #[test]
    fn reports_storage_width() {
        assert_eq!(ScalarKind::F32.size_in_bytes(), 4);
        assert_eq!(ScalarKind::F64.size_in_bytes(), 8);
    }

    #[test]
    fn rejects_unknown_names() {
        assert!(ScalarKind::from_str("float16").is_err());
    }
}
