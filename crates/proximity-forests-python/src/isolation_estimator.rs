//! Native proximity-isolation estimator state exposed to Python.

use std::num::NonZeroUsize;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use proximity_forests_core::{DistanceConfig, IsolationCriterion, NumericVectorView};

use proximity_forests_core::forest::{
    DimensionSelection, ForestConfig, IsolationForest, anomaly_scores_vector_many,
    apply_vector_many, fit_vector_isolation_forest,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::error::core_error;

/// Rust-backed isolation estimator used by the public Python wrapper.
#[pyclass(name = "_NativeProximityIsolationForest")]
pub struct NativeProximityIsolationForest {
    n_estimators: usize,
    n_candidates: usize,
    n_exemplars: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    bootstrap: bool,
    random_state: u64,
    n_jobs: usize,
    model: Option<IsolationForest>,
    n_features_in: Option<usize>,
}

#[pymethods]
impl NativeProximityIsolationForest {
    #[new]
    #[pyo3(signature = (n_estimators=100, n_candidates=5, n_exemplars=2,
        max_depth=None, min_samples_split=2, bootstrap=false,
        random_state=0, n_jobs=0))]
    fn new(
        n_estimators: usize,
        n_candidates: usize,
        n_exemplars: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        bootstrap: bool,
        random_state: u64,
        n_jobs: usize,
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
            model: None,
            n_features_in: None,
        })
    }

    /// Fits a vector-layout proximity isolation forest.
    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let shape = x.shape();
        if shape[0] < 2 || shape[1] == 0 {
            return Err(PyValueError::new_err(
                "X must contain at least two samples and one feature",
            ));
        }
        if self.n_exemplars > shape[0] {
            return Err(PyValueError::new_err(
                "n_exemplars cannot exceed the number of fitting samples",
            ));
        }
        let rows = copy_rows(x.as_array());
        let views: Vec<_> = rows
            .iter()
            .map(|row| NumericVectorView::from_f64(row))
            .collect();
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

        let catalog = [DistanceConfig::SquaredEuclidean];
        let model = fit_vector_isolation_forest(
            config,
            &views,
            &catalog,
            shape[1],
            self.n_exemplars,
            IsolationCriterion::MinChild,
        )
        .map_err(core_error)?;
        self.model = Some(model);
        self.n_features_in = Some(shape[1]);
        Ok(())
    }

    /// Returns the original-paper score: larger values are more anomalous.
    fn anomaly_score<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.validate_features(x.shape())?;
        let rows = copy_rows(x.as_array());
        let views: Vec<_> = rows
            .iter()
            .map(|row| NumericVectorView::from_f64(row))
            .collect();
        let values = anomaly_scores_vector_many(
            self.model.as_ref().ok_or_else(not_fitted)?,
            &views,
            self.random_state,
            self.n_jobs,
        )
        .map_err(core_error)?;
        Ok(values.into_pyarray(py))
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<usize>>> {
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
            result.leaf_indices().to_vec(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.n_features_in.ok_or_else(not_fitted)
    }
}

impl NativeProximityIsolationForest {
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

fn copy_rows(array: numpy::ndarray::ArrayView2<'_, f64>) -> Vec<Vec<f64>> {
    array.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn not_fitted() -> PyErr {
    PyRuntimeError::new_err("This ProximityIsolationForest instance is not fitted yet")
}
