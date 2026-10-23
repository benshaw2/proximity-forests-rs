//! Native regression estimator state exposed to Python.

use std::num::NonZeroUsize;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use proximity_forests_core::{
    DistanceChoice, DistanceSamplingContext, NumericVectorView, RegressionTargets,
    SupervisedCriterion,
};

use proximity_forests_core::forest::{
    DimensionSelection, ForestConfig, RegressionForest, apply_vector_many,
    fit_vector_regression_forest, predict_vector_regression_many,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::{distance_spec::parse_vector_catalog, error::core_error};

/// Rust-backed regressor used by the public Python estimator wrapper.
#[pyclass(name = "_NativeProximityForestRegressor")]
pub struct NativeProximityForestRegressor {
    n_estimators: usize,
    n_candidates: usize,
    n_exemplars: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    bootstrap: bool,
    random_state: u64,
    n_jobs: usize,
    distance_catalog: Box<[DistanceChoice]>,
    model: Option<RegressionForest>,
    n_features_in: Option<usize>,
}

#[pymethods]
impl NativeProximityForestRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, n_candidates=5, n_exemplars=2,
        max_depth=None, min_samples_split=2, bootstrap=true,
        random_state=0, n_jobs=0, distances=vec![String::from("squared_euclidean")]))]
    fn new(
        n_estimators: usize,
        n_candidates: usize,
        n_exemplars: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        bootstrap: bool,
        random_state: u64,
        n_jobs: usize,
        distances: Vec<String>,
    ) -> PyResult<Self> {
        if n_estimators == 0 || n_candidates == 0 || n_exemplars == 0 || min_samples_split == 0 {
            return Err(PyValueError::new_err(
                "n_estimators, n_candidates, n_exemplars, and min_samples_split must be positive",
            ));
        }
        Ok(Self {
            n_estimators,
            n_candidates,
            n_exemplars,
            max_depth,
            min_samples_split,
            bootstrap,
            random_state,
            n_jobs,
            distance_catalog: parse_vector_catalog(distances)?,
            model: None,
            n_features_in: None,
        })
    }

    /// Fits one or more row-major floating-point regression targets.
    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let shape = x.shape();
        let target_shape = y.shape();
        if shape[0] == 0 || shape[1] == 0 {
            return Err(PyValueError::new_err("X must be a nonempty 2D array"));
        }
        if target_shape[0] != shape[0] {
            return Err(PyValueError::new_err(
                "X and y contain different numbers of samples",
            ));
        }
        if target_shape[1] == 0 {
            return Err(PyValueError::new_err(
                "y must contain at least one regression output",
            ));
        }
        if y.as_array().iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("y must contain only finite values"));
        }
        if self.n_exemplars > shape[0] {
            return Err(PyValueError::new_err(
                "n_exemplars cannot exceed the number of fitting samples",
            ));
        }

        let distance_sampling = vector_sampling_context(x.as_array())?;
        let rows = copy_rows(x.as_array());
        let views: Vec<_> = rows
            .iter()
            .map(|row| NumericVectorView::from_f64(row))
            .collect();
        let target_values = y.as_array().iter().copied().collect();
        let targets = RegressionTargets::from_f64(target_values, target_shape[0], target_shape[1])
            .map_err(core_error)?;
        let mut config = ForestConfig::new(
            NonZeroUsize::new(self.n_estimators).expect("validated positive"),
            NonZeroUsize::new(self.n_candidates).expect("validated positive"),
            self.random_state,
        );
        config.bootstrap = self.bootstrap;
        config.maximum_depth = self.max_depth;
        config.minimum_split_size =
            NonZeroUsize::new(self.min_samples_split).expect("validated positive");
        config.dimension_selection = DimensionSelection::All;

        let model = fit_vector_regression_forest(
            config,
            &views,
            &targets,
            &self.distance_catalog,
            distance_sampling,
            shape[1],
            self.n_exemplars,
            SupervisedCriterion::Variance,
        )
        .map_err(core_error)?;

        self.model = Some(model);
        self.n_features_in = Some(shape[1]);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let predictions = self.predict_batch(x)?;
        let values = predictions
            .into_iter()
            .map(|outputs| {
                outputs
                    .first()
                    .copied()
                    .ok_or_else(|| PyRuntimeError::new_err("regression output was unavailable"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(values.into_pyarray(py))
    }

    fn predict_matrix<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let predictions = self.predict_batch(x)?;
        let output_count = self.model.as_ref().ok_or_else(not_fitted)?.output_count();
        let values: Vec<f64> = predictions.into_iter().flatten().collect();
        let rows = values.len() / output_count;
        let array = numpy::ndarray::Array2::from_shape_vec((rows, output_count), values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        self.validate_features(x.shape())?;
        let rows = copy_rows(x.as_array());
        let views: Vec<_> = rows
            .iter()
            .map(|row| NumericVectorView::from_f64(row))
            .collect();
        let model = self.model.as_ref().ok_or_else(not_fitted)?;
        let result = apply_vector_many(model.forest(), &views, self.random_state, self.n_jobs)
            .map_err(core_error)?;
        let array = numpy::ndarray::Array2::from_shape_vec(
            (result.sample_count(), result.tree_count()),
            result.into_leaf_indices().into_vec(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.n_features_in.ok_or_else(not_fitted)
    }

    #[getter]
    fn n_outputs_(&self) -> PyResult<usize> {
        self.model
            .as_ref()
            .map(RegressionForest::output_count)
            .ok_or_else(not_fitted)
    }
}

impl NativeProximityForestRegressor {
    fn predict_batch(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Vec<f64>>> {
        self.validate_features(x.shape())?;
        let rows = copy_rows(x.as_array());
        let views: Vec<_> = rows
            .iter()
            .map(|row| NumericVectorView::from_f64(row))
            .collect();
        predict_vector_regression_many(
            self.model.as_ref().ok_or_else(not_fitted)?,
            &views,
            self.random_state,
            self.n_jobs,
        )
        .map_err(core_error)
    }

    fn validate_features(&self, shape: &[usize]) -> PyResult<()> {
        let expected = self.n_features_in.ok_or_else(not_fitted)?;
        if shape.len() != 2 || shape[1] != expected {
            return Err(PyValueError::new_err(format!(
                "X has {} features, but this estimator expects {expected}",
                shape.get(1).copied().unwrap_or(0)
            )));
        }
        Ok(())
    }
}

fn vector_sampling_context(
    array: numpy::ndarray::ArrayView2<'_, f64>,
) -> PyResult<DistanceSamplingContext> {
    let mut count = 0_u64;
    let mut mean = 0.0;
    let mut sum_squared_deviations = 0.0;

    for &value in array {
        if !value.is_finite() {
            return Err(PyValueError::new_err(
                "X must contain only finite values for distance-parameter sampling",
            ));
        }
        count += 1;
        let delta = value - mean;
        mean += delta / count as f64;
        sum_squared_deviations += delta * (value - mean);
    }

    let population_standard_deviation = if count == 0 {
        0.0
    } else {
        (sum_squared_deviations / count as f64).sqrt()
    };
    DistanceSamplingContext::new(array.ncols(), population_standard_deviation).map_err(core_error)
}

fn copy_rows(array: numpy::ndarray::ArrayView2<'_, f64>) -> Vec<Vec<f64>> {
    array.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn not_fitted() -> PyErr {
    PyRuntimeError::new_err("This ProximityForestRegressor instance is not fitted yet")
}
