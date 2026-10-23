//! NumPy input validation and owned row extraction.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

/// Owned homogeneous two-dimensional floating-point input.
///
/// Python arrays are copied once before releasing the GIL. The Rust core then
/// owns fitted observations, so no Python or NumPy lifetime crosses fitting.
#[derive(Debug)]
pub enum MatrixInput {
    F32 {
        values: Vec<f32>,
        rows: usize,
        columns: usize,
    },
    F64 {
        values: Vec<f64>,
        rows: usize,
        columns: usize,
    },
}

impl MatrixInput {
    pub fn from_f32(array: PyReadonlyArray2<'_, f32>) -> PyResult<Self> {
        let shape = array.shape();
        validate_shape(shape)?;
        Ok(Self::F32 {
            values: array.as_array().iter().copied().collect(),
            rows: shape[0],
            columns: shape[1],
        })
    }

    pub fn from_f64(array: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let shape = array.shape();
        validate_shape(shape)?;
        Ok(Self::F64 {
            values: array.as_array().iter().copied().collect(),
            rows: shape[0],
            columns: shape[1],
        })
    }

    pub const fn rows(&self) -> usize {
        match self {
            Self::F32 { rows, .. } | Self::F64 { rows, .. } => *rows,
        }
    }

    pub const fn columns(&self) -> usize {
        match self {
            Self::F32 { columns, .. } | Self::F64 { columns, .. } => *columns,
        }
    }

    pub fn contains_infinity(&self) -> bool {
        match self {
            Self::F32 { values, .. } => values.iter().any(|value| value.is_infinite()),
            Self::F64 { values, .. } => values.iter().any(|value| value.is_infinite()),
        }
    }
}

fn validate_shape(shape: &[usize]) -> PyResult<()> {
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(PyValueError::new_err(
            "X must be a nonempty two-dimensional NumPy array",
        ));
    }
    Ok(())
}
