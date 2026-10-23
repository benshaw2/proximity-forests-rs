//! Conversion of Rust-core errors into stable Python exceptions.

use proximity_forests_core::CoreError;
use pyo3::PyErr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};

/// Converts a core error without exposing Rust implementation details.
pub fn core_error(error: CoreError) -> PyErr {
    let message = error.to_string();
    // Shape, data, and configuration failures are user-facing value errors.
    // Unexpected internal-state failures remain runtime errors. This textual
    // fallback can be replaced by matching public CoreError variants once the
    // error enum is frozen as an API contract.
    if message.contains("shape")
        || message.contains("requires")
        || message.contains("must")
        || message.contains("invalid")
        || message.contains("outside")
    {
        PyValueError::new_err(message)
    } else {
        PyRuntimeError::new_err(message)
    }
}
