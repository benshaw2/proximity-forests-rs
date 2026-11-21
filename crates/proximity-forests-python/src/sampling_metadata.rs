//! Dense sampling matrices derived lazily from fitted metadata.
use numpy::{IntoPyArray, PyArray2};
use proximity_forests_core::forest::FittedForest;
use pyo3::exceptions::{PyOverflowError, PyRuntimeError};
use pyo3::prelude::*;

pub(crate) fn oob_mask<'py>(
    py: Python<'py>,
    forest: &FittedForest,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let m = forest.training_metadata();
    let n = m.sample_count();
    let t = m.bootstrap().len();
    let mut v = vec![
        false;
        n.checked_mul(t)
            .ok_or_else(|| PyOverflowError::new_err("OOB mask shape overflow"))?
    ];
    for (j, s) in m.bootstrap().iter().enumerate() {
        for &i in s.out_of_bag_positions() {
            let x = v
                .get_mut(i * t + j)
                .ok_or_else(|| PyRuntimeError::new_err("OOB index outside training matrix"))?;
            *x = true;
        }
    }
    matrix(py, n, t, v, "OOB mask")
}
pub(crate) fn in_bag_counts<'py>(
    py: Python<'py>,
    forest: &FittedForest,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let m = forest.training_metadata();
    let n = m.sample_count();
    let t = m.bootstrap().len();
    let mut v = vec![
        0u32;
        n.checked_mul(t)
            .ok_or_else(|| PyOverflowError::new_err("in-bag shape overflow"))?
    ];
    for (j, s) in m.bootstrap().iter().enumerate() {
        if s.multiplicities().len() != n {
            return Err(PyRuntimeError::new_err("multiplicity width mismatch"));
        }
        for (i, &c) in s.multiplicities().iter().enumerate() {
            v[i * t + j] = u32::try_from(c)
                .map_err(|_| PyOverflowError::new_err("multiplicity exceeds uint32"))?;
        }
    }
    matrix(py, n, t, v, "in-bag counts")
}
fn matrix<'py, T: numpy::Element>(
    py: Python<'py>,
    n: usize,
    t: usize,
    v: Vec<T>,
    name: &str,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    let a = numpy::ndarray::Array2::from_shape_vec((n, t), v)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid {name}: {e}")))?;
    Ok(a.into_pyarray(py))
}
