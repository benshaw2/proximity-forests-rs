#![allow(dead_code)]
//! NumPy input validation and zero-copy typed matrix borrowing.

use std::path::PathBuf;

use numpy::{PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use proximity_forests_core::forest::{
    DataRetention, ExternalDataFormat, ExternalTrainingData, ForestConfig,
};
use proximity_forests_core::{
    DistanceSamplingContext, MultivariateSeriesView, NumericVectorView, ScalarKind,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PySequence};

/// Borrowed homogeneous two-dimensional floating-point input.
///
/// Compatible C-contiguous NumPy arrays are borrowed for the duration of one
/// native call. Fitting copies only state required by the selected retention
/// policy and committed tree exemplars.
#[derive(Debug)]
pub enum MatrixInput<'py> {
    /// Borrowed single-precision matrix.
    F32(PyReadonlyArray2<'py, f32>),
    /// Borrowed double-precision matrix.
    F64(PyReadonlyArray2<'py, f64>),
}

impl<'py> MatrixInput<'py> {
    /// Extracts a C-contiguous `float32` or `float64` NumPy matrix.
    pub fn extract(value: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(array) = value.extract::<PyReadonlyArray2<'py, f32>>() {
            validate_shape(array.shape())?;
            ensure_contiguous_f32(&array)?;
            return Ok(Self::F32(array));
        }
        if let Ok(array) = value.extract::<PyReadonlyArray2<'py, f64>>() {
            validate_shape(array.shape())?;
            ensure_contiguous_f64(&array)?;
            return Ok(Self::F64(array));
        }
        Err(PyValueError::new_err(
            "X must be a C-contiguous two-dimensional NumPy array with dtype float32 or float64",
        ))
    }

    /// Returns the number of observations.
    pub fn rows(&self) -> usize {
        match self {
            Self::F32(array) => array.shape()[0],
            Self::F64(array) => array.shape()[0],
        }
    }

    /// Returns the number of columns or time points.
    pub fn columns(&self) -> usize {
        match self {
            Self::F32(array) => array.shape()[1],
            Self::F64(array) => array.shape()[1],
        }
    }

    /// Returns the logical row/column shape.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.columns())
    }

    /// Returns the NumPy scalar storage kind without converting values.
    pub const fn scalar_kind(&self) -> ScalarKind {
        match self {
            Self::F32(_) => ScalarKind::F32,
            Self::F64(_) => ScalarKind::F64,
        }
    }

    /// Reports whether the matrix contains positive or negative infinity.
    pub fn contains_infinity(&self) -> PyResult<bool> {
        match self {
            Self::F32(array) => Ok(array.as_slice()?.iter().any(|value| value.is_infinite())),
            Self::F64(array) => Ok(array.as_slice()?.iter().any(|value| value.is_infinite())),
        }
    }

    /// Reports whether the matrix contains NaN.
    pub fn contains_nan(&self) -> PyResult<bool> {
        match self {
            Self::F32(array) => Ok(array.as_slice()?.iter().any(|value| value.is_nan())),
            Self::F64(array) => Ok(array.as_slice()?.iter().any(|value| value.is_nan())),
        }
    }

    /// Computes dataset-level statistics used by sampled distance families.
    pub fn sampling_context(&self) -> PyResult<DistanceSamplingContext> {
        let standard_deviation = match self {
            Self::F32(array) => population_standard_deviation(
                array.as_slice()?.iter().map(|&value| f64::from(value)),
            )?,
            Self::F64(array) => population_standard_deviation(array.as_slice()?.iter().copied())?,
        };
        DistanceSamplingContext::new(self.columns(), standard_deviation)
            .map_err(crate::error::core_error)
    }

    /// Calls `operation` with zero-copy vector views over every matrix row.
    pub fn with_vector_views<R>(
        &self,
        operation: impl FnOnce(&[NumericVectorView<'_>]) -> PyResult<R>,
    ) -> PyResult<R> {
        let columns = self.columns();
        match self {
            Self::F32(array) => {
                let values = array.as_slice()?;
                let views = values
                    .chunks_exact(columns)
                    .map(NumericVectorView::from_f32)
                    .collect::<Vec<_>>();
                operation(&views)
            }
            Self::F64(array) => {
                let values = array.as_slice()?;
                let views = values
                    .chunks_exact(columns)
                    .map(NumericVectorView::from_f64)
                    .collect::<Vec<_>>();
                operation(&views)
            }
        }
    }
}

/// Borrowed equal-length multivariate collection with logical shape
/// `(cases, channels, timepoints)`.
#[derive(Debug)]
pub enum SeriesArrayInput<'py> {
    /// Borrowed single-precision collection.
    F32(PyReadonlyArray3<'py, f32>),
    /// Borrowed double-precision collection.
    F64(PyReadonlyArray3<'py, f64>),
}

impl<'py> SeriesArrayInput<'py> {
    /// Extracts a C-contiguous three-dimensional NumPy collection.
    pub fn extract(value: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(array) = value.extract::<PyReadonlyArray3<'py, f32>>() {
            validate_series_shape(array.shape())?;
            array.as_slice().map_err(|_| {
                PyValueError::new_err(
                    "multivariate X must be C-contiguous; use numpy.ascontiguousarray(X)",
                )
            })?;
            return Ok(Self::F32(array));
        }
        if let Ok(array) = value.extract::<PyReadonlyArray3<'py, f64>>() {
            validate_series_shape(array.shape())?;
            array.as_slice().map_err(|_| {
                PyValueError::new_err(
                    "multivariate X must be C-contiguous; use numpy.ascontiguousarray(X)",
                )
            })?;
            return Ok(Self::F64(array));
        }
        Err(PyValueError::new_err(
            "multivariate X must be a C-contiguous three-dimensional NumPy array with dtype float32 or float64",
        ))
    }

    /// Returns the number of cases.
    pub fn cases(&self) -> usize {
        match self {
            Self::F32(array) => array.shape()[0],
            Self::F64(array) => array.shape()[0],
        }
    }

    /// Returns the shared channel count.
    pub fn channels(&self) -> usize {
        match self {
            Self::F32(array) => array.shape()[1],
            Self::F64(array) => array.shape()[1],
        }
    }

    /// Returns the shared number of time points.
    pub fn timepoints(&self) -> usize {
        match self {
            Self::F32(array) => array.shape()[2],
            Self::F64(array) => array.shape()[2],
        }
    }

    /// Returns the scalar storage kind.
    pub const fn scalar_kind(&self) -> ScalarKind {
        match self {
            Self::F32(_) => ScalarKind::F32,
            Self::F64(_) => ScalarKind::F64,
        }
    }

    /// Computes candidate-local sampling statistics over all values.
    pub fn sampling_context(&self) -> PyResult<DistanceSamplingContext> {
        let standard_deviation = match self {
            Self::F32(array) => population_standard_deviation(
                array.as_slice()?.iter().map(|&value| f64::from(value)),
            )?,
            Self::F64(array) => population_standard_deviation(array.as_slice()?.iter().copied())?,
        };
        DistanceSamplingContext::new(self.timepoints(), standard_deviation)
            .map_err(crate::error::core_error)
    }

    /// Calls `operation` with zero-copy per-case multivariate views.
    pub fn with_series_views<R>(
        &self,
        operation: impl FnOnce(&[MultivariateSeriesView<'_>]) -> PyResult<R>,
    ) -> PyResult<R> {
        let channels = self.channels();
        let timepoints = self.timepoints();
        let case_width = channels
            .checked_mul(timepoints)
            .ok_or_else(|| PyValueError::new_err("multivariate case shape overflowed usize"))?;
        match self {
            Self::F32(array) => {
                let views = array
                    .as_slice()?
                    .chunks_exact(case_width)
                    .map(|values| {
                        MultivariateSeriesView::contiguous_f32(values, channels, timepoints)
                            .map_err(crate::error::core_error)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                operation(&views)
            }
            Self::F64(array) => {
                let views = array
                    .as_slice()?
                    .chunks_exact(case_width)
                    .map(|values| {
                        MultivariateSeriesView::contiguous_f64(values, channels, timepoints)
                            .map_err(crate::error::core_error)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                operation(&views)
            }
        }
    }
}

/// Borrowed unequal-length collection represented by a Python sequence of
/// C-contiguous two-dimensional arrays shaped `(channels, timepoints_i)`.
///
/// All cases must use one floating-point dtype and one common channel count.
/// Individual time-axis lengths may differ.
#[derive(Debug)]
pub enum UnequalSeriesInput<'py> {
    /// Borrowed single-precision cases.
    F32(Vec<PyReadonlyArray2<'py, f32>>),
    /// Borrowed double-precision cases.
    F64(Vec<PyReadonlyArray2<'py, f64>>),
}

impl<'py> UnequalSeriesInput<'py> {
    /// Extracts a homogeneous Python sequence of rectangular series cases.
    pub fn extract(value: &Bound<'py, PyAny>) -> PyResult<Self> {
        let sequence = value.cast::<PySequence>().map_err(|_| {
            PyValueError::new_err(
                "unequal-length X must be a sequence of two-dimensional NumPy arrays",
            )
        })?;
        let case_count = sequence.len()?;
        if case_count < 2 {
            return Err(PyValueError::new_err(
                "unequal-length X must contain at least two cases",
            ));
        }
        let first = sequence.get_item(0)?;
        if first.extract::<PyReadonlyArray2<'py, f32>>().is_ok() {
            let mut arrays = Vec::with_capacity(case_count);
            for index in 0..case_count {
                let item = sequence.get_item(index)?;
                let array = item.extract::<PyReadonlyArray2<'py, f32>>().map_err(|_| {
                    PyValueError::new_err(
                        "all unequal-length cases must be float32 two-dimensional NumPy arrays",
                    )
                })?;
                validate_unequal_case(array.shape(), index)?;
                array.as_slice().map_err(|_| {
                    PyValueError::new_err(format!(
                        "unequal-length case {index} must be C-contiguous",
                    ))
                })?;
                arrays.push(array);
            }
            validate_common_channels_f32(&arrays)?;
            return Ok(Self::F32(arrays));
        }
        if first.extract::<PyReadonlyArray2<'py, f64>>().is_ok() {
            let mut arrays = Vec::with_capacity(case_count);
            for index in 0..case_count {
                let item = sequence.get_item(index)?;
                let array = item.extract::<PyReadonlyArray2<'py, f64>>().map_err(|_| {
                    PyValueError::new_err(
                        "all unequal-length cases must be float64 two-dimensional NumPy arrays",
                    )
                })?;
                validate_unequal_case(array.shape(), index)?;
                array.as_slice().map_err(|_| {
                    PyValueError::new_err(format!(
                        "unequal-length case {index} must be C-contiguous",
                    ))
                })?;
                arrays.push(array);
            }
            validate_common_channels_f64(&arrays)?;
            return Ok(Self::F64(arrays));
        }
        Err(PyValueError::new_err(
            "unequal-length cases must be float32 or float64 two-dimensional NumPy arrays",
        ))
    }

    /// Returns the number of cases.
    pub fn cases(&self) -> usize {
        match self {
            Self::F32(arrays) => arrays.len(),
            Self::F64(arrays) => arrays.len(),
        }
    }

    /// Returns the common number of channels.
    pub fn channels(&self) -> usize {
        match self {
            Self::F32(arrays) => arrays[0].shape()[0],
            Self::F64(arrays) => arrays[0].shape()[0],
        }
    }

    /// Returns the maximum case length, used as the sampling scale for
    /// candidate-local elastic-distance windows.
    pub fn maximum_timepoints(&self) -> usize {
        match self {
            Self::F32(arrays) => arrays.iter().map(|a| a.shape()[1]).max().unwrap_or(0),
            Self::F64(arrays) => arrays.iter().map(|a| a.shape()[1]).max().unwrap_or(0),
        }
    }

    /// Computes pooled dataset statistics without materializing a padded array.
    pub fn sampling_context(&self) -> PyResult<DistanceSamplingContext> {
        let standard_deviation = match self {
            Self::F32(arrays) => population_standard_deviation(
                arrays
                    .iter()
                    .map(PyReadonlyArray2::as_slice)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .map(|&value| f64::from(value)),
            )?,
            Self::F64(arrays) => population_standard_deviation(
                arrays
                    .iter()
                    .map(PyReadonlyArray2::as_slice)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .copied(),
            )?,
        };
        DistanceSamplingContext::new(self.maximum_timepoints(), standard_deviation)
            .map_err(crate::error::core_error)
    }

    /// Calls `operation` with one zero-copy multivariate view per case.
    pub fn with_series_views<R>(
        &self,
        operation: impl FnOnce(&[MultivariateSeriesView<'_>]) -> PyResult<R>,
    ) -> PyResult<R> {
        match self {
            Self::F32(arrays) => {
                let views = arrays
                    .iter()
                    .map(|array| {
                        MultivariateSeriesView::contiguous_f32(
                            array.as_slice()?,
                            array.shape()[0],
                            array.shape()[1],
                        )
                        .map_err(crate::error::core_error)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                operation(&views)
            }
            Self::F64(arrays) => {
                let views = arrays
                    .iter()
                    .map(|array| {
                        MultivariateSeriesView::contiguous_f64(
                            array.as_slice()?,
                            array.shape()[0],
                            array.shape()[1],
                        )
                        .map_err(crate::error::core_error)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                operation(&views)
            }
        }
    }
}

fn validate_unequal_case(shape: &[usize], index: usize) -> PyResult<()> {
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(PyValueError::new_err(format!(
            "unequal-length case {index} must have nonempty shape (channels, timepoints)",
        )));
    }
    Ok(())
}

fn validate_common_channels_f32(arrays: &[PyReadonlyArray2<'_, f32>]) -> PyResult<()> {
    validate_common_channels(arrays.iter().map(|array| array.shape()[0]))
}

fn validate_common_channels_f64(arrays: &[PyReadonlyArray2<'_, f64>]) -> PyResult<()> {
    validate_common_channels(arrays.iter().map(|array| array.shape()[0]))
}

fn validate_common_channels(mut channels: impl Iterator<Item = usize>) -> PyResult<()> {
    let expected = channels.next().unwrap_or(0);
    if channels.any(|value| value != expected) {
        return Err(PyValueError::new_err(
            "all unequal-length cases must have the same number of channels",
        ));
    }
    Ok(())
}

fn validate_series_shape(shape: &[usize]) -> PyResult<()> {
    if shape.len() != 3 || shape.iter().any(|&value| value == 0) {
        return Err(PyValueError::new_err(
            "multivariate X must have nonempty shape (n_cases, n_channels, n_timepoints)",
        ));
    }
    Ok(())
}

fn ensure_contiguous_f32(array: &PyReadonlyArray2<'_, f32>) -> PyResult<()> {
    array.as_slice().map(|_| ()).map_err(|_| {
        PyValueError::new_err("X must be C-contiguous; use numpy.ascontiguousarray(X)")
    })
}

fn ensure_contiguous_f64(array: &PyReadonlyArray2<'_, f64>) -> PyResult<()> {
    array.as_slice().map(|_| ()).map_err(|_| {
        PyValueError::new_err("X must be C-contiguous; use numpy.ascontiguousarray(X)")
    })
}

fn validate_shape(shape: &[usize]) -> PyResult<()> {
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(PyValueError::new_err(
            "X must be a nonempty two-dimensional NumPy array",
        ));
    }
    Ok(())
}

fn population_standard_deviation(values: impl IntoIterator<Item = f64>) -> PyResult<f64> {
    let mut count = 0_u64;
    let mut mean = 0.0;
    let mut squared_deviations = 0.0;
    for value in values {
        if !value.is_finite() {
            return Err(PyValueError::new_err(
                "X must contain only finite values for distance-parameter sampling",
            ));
        }
        count += 1;
        let delta = value - mean;
        mean += delta / count as f64;
        squared_deviations += delta * (value - mean);
    }
    Ok(if count == 0 {
        0.0
    } else {
        (squared_deviations / count as f64).sqrt()
    })
}

/// Parses the Python-facing fitted-data retention policy.
pub fn parse_data_retention(value: &str) -> PyResult<DataRetention> {
    match value {
        "exemplars" => Ok(DataRetention::Exemplars),
        "copy" => Ok(DataRetention::Copy),
        "reference" => Ok(DataRetention::Reference),
        _ => Err(PyValueError::new_err(
            "data_retention must be one of: 'exemplars', 'copy', or 'reference'",
        )),
    }
}

/// Metadata supplied by the Python boundary for a persistent `.npy` mapping.
#[derive(Clone, Debug)]
pub struct NpyReference {
    /// Canonical or user-supplied persistent file path.
    pub path: PathBuf,
    /// Byte offset of the numeric payload inside the `.npy` file.
    pub payload_offset: u64,
    /// Complete backing-file size in bytes.
    pub file_size: u64,
    /// Optional externally calculated integrity fingerprint.
    pub fingerprint: Option<Box<str>>,
}

/// Builds the core external-data descriptor for a validated matrix mapping.
pub fn external_training_data(
    input: &MatrixInput<'_>,
    reference: NpyReference,
) -> PyResult<ExternalTrainingData> {
    let shape = input.shape();
    let scalar_bytes = input.scalar_kind().size_in_bytes();
    let element_count = shape
        .0
        .checked_mul(shape.1)
        .ok_or_else(|| PyValueError::new_err("memory-mapped matrix shape overflowed usize"))?;
    let payload_length = element_count.checked_mul(scalar_bytes).ok_or_else(|| {
        PyValueError::new_err("memory-mapped matrix byte length overflowed usize")
    })?;
    let payload_length = u64::try_from(payload_length)
        .map_err(|_| PyValueError::new_err("memory-mapped matrix is too large to describe"))?;
    let payload_end = reference
        .payload_offset
        .checked_add(payload_length)
        .ok_or_else(|| PyValueError::new_err("memory-mapped payload range overflowed u64"))?;
    if payload_end > reference.file_size {
        return Err(PyValueError::new_err(
            "memory-mapped matrix payload extends beyond its backing file",
        ));
    }
    Ok(ExternalTrainingData::new(
        reference.path,
        ExternalDataFormat::Npy,
        input.scalar_kind(),
        vec![shape.0, shape.1].into_boxed_slice(),
        reference.payload_offset,
        payload_length,
        reference.file_size,
        reference.fingerprint,
    ))
}

/// Applies a Python-facing retention policy to a forest configuration.
///
/// Reference retention remains blocked until the estimator passes validated
/// persistent mapping metadata alongside the borrowed matrix.
pub fn configure_data_retention(config: &mut ForestConfig, value: &str) -> PyResult<()> {
    let retention = parse_data_retention(value)?;
    if retention == DataRetention::Reference {
        return Err(PyValueError::new_err(
            "data_retention='reference' requires a persistent NumPy .npy memory map; this estimator boundary does not expose mapping metadata yet",
        ));
    }
    config.data_retention = retention;
    config.external_training_data = None;
    Ok(())
}
