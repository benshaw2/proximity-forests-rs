//! Native regression estimator state exposed to Python.

use std::num::NonZeroUsize;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use proximity_forests_core::{RegressionTargets, SupervisedCriterion};

use proximity_forests_core::forest::{
    DimensionSelection, ForestConfig, RegressionForest, apply_multivariate_many, apply_vector_many,
    fit_multivariate_regression_forest, fit_vector_regression_forest,
    predict_multivariate_regression_many, predict_vector_regression_many,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::{
    distance_spec::{parse_multivariate_catalog, parse_vector_catalog},
    error::core_error,
    input::{MatrixInput, SeriesArrayInput, UnequalSeriesInput, configure_data_retention},
};

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
    data_retention: String,
    distance_tokens: Box<[String]>,
    model: Option<RegressionForest>,
    n_features_in: Option<usize>,
    n_channels_in: Option<usize>,
    n_timepoints_in: Option<usize>,
}

#[pymethods]
impl NativeProximityForestRegressor {
    /// Returns the stable native fitted-model payload used by Python pickle.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self
            .model
            .as_ref()
            .ok_or_else(not_fitted)?
            .to_bytes()
            .map_err(core_error)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Loads a stable native payload into this estimator's configured shell.
    fn load_bytes(
        &mut self,
        payload: &[u8],
        n_features_in: usize,
        n_channels_in: Option<usize>,
        n_timepoints_in: Option<usize>,
    ) -> PyResult<()> {
        if n_features_in == 0
            || n_channels_in.is_none() && n_timepoints_in.is_some()
            || n_channels_in == Some(0)
            || n_timepoints_in == Some(0)
        {
            return Err(PyValueError::new_err(
                "serialized regressor metadata is invalid",
            ));
        }
        self.model = Some(RegressionForest::from_bytes(payload).map_err(core_error)?);
        self.n_features_in = Some(n_features_in);
        self.n_channels_in = n_channels_in;
        self.n_timepoints_in = n_timepoints_in;
        Ok(())
    }

    #[new]
    #[pyo3(signature = (n_estimators=100, n_candidates=5, n_exemplars=2,
        max_depth=None, min_samples_split=2, bootstrap=true,
        random_state=0, n_jobs=0, data_retention=String::from("exemplars"), distances=vec![String::from("squared_euclidean")]))]
    fn new(
        n_estimators: usize,
        n_candidates: usize,
        n_exemplars: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        bootstrap: bool,
        random_state: u64,
        n_jobs: usize,
        data_retention: String,
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
            data_retention,
            distance_tokens: distances.into_boxed_slice(),
            model: None,
            n_features_in: None,
            n_channels_in: None,
            n_timepoints_in: None,
        })
    }

    /// Fits one or more row-major floating-point regression targets.
    fn fit(&mut self, x: &Bound<'_, PyAny>, y: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let input = MatrixInput::extract(x)?;
        let target_shape = y.shape();
        if target_shape[0] != input.rows() {
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
        if self.n_exemplars > input.rows() {
            return Err(PyValueError::new_err(
                "n_exemplars cannot exceed the number of fitting samples",
            ));
        }
        let distance_sampling = input.sampling_context()?;
        let distance_catalog = parse_vector_catalog(self.distance_tokens.to_vec())?;
        let targets = RegressionTargets::from_f64(
            y.as_array().iter().copied().collect(),
            target_shape[0],
            target_shape[1],
        )
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
        configure_data_retention(&mut config, &self.data_retention)?;
        let feature_count = input.columns();
        let model = input.with_vector_views(|views| {
            fit_vector_regression_forest(
                config,
                views,
                &targets,
                &distance_catalog,
                distance_sampling,
                feature_count,
                self.n_exemplars,
                SupervisedCriterion::Variance,
            )
            .map_err(core_error)
        })?;
        self.model = Some(model);
        self.n_features_in = Some(feature_count);
        Ok(())
    }

    fn fit_multivariate(
        &mut self,
        x: &Bound<'_, PyAny>,
        y: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let input = SeriesArrayInput::extract(x)?;
        let shape = y.shape();
        if shape[0] != input.cases() || shape[1] == 0 {
            return Err(PyValueError::new_err("X and y shapes are incompatible"));
        }
        let targets =
            RegressionTargets::from_f64(y.as_array().iter().copied().collect(), shape[0], shape[1])
                .map_err(core_error)?;
        let catalog = parse_multivariate_catalog(self.distance_tokens.to_vec())?;
        let sampling = input.sampling_context()?;
        let mut config = ForestConfig::new(
            NonZeroUsize::new(self.n_estimators).unwrap(),
            NonZeroUsize::new(self.n_candidates).unwrap(),
            self.random_state,
        );
        config.bootstrap = self.bootstrap;
        config.maximum_depth = self.max_depth;
        config.minimum_split_size = NonZeroUsize::new(self.min_samples_split).unwrap();
        config.dimension_selection = DimensionSelection::All;
        configure_data_retention(&mut config, &self.data_retention)?;
        self.model = Some(input.with_series_views(|views| {
            fit_multivariate_regression_forest(
                config,
                views,
                &targets,
                &catalog,
                sampling,
                self.n_exemplars,
                SupervisedCriterion::Variance,
            )
            .map_err(core_error)
        })?);
        self.n_features_in = Some(input.channels());
        self.n_channels_in = Some(input.channels());
        self.n_timepoints_in = Some(input.timepoints());
        Ok(())
    }
    fn predict_multivariate_matrix<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let input = SeriesArrayInput::extract(x)?;
        self.validate_multivariate_shape(input.channels(), input.timepoints())?;
        let predictions = input.with_series_views(|views| {
            predict_multivariate_regression_many(
                self.model.as_ref().ok_or_else(not_fitted)?,
                views,
                self.random_state,
                self.n_jobs,
            )
            .map_err(core_error)
        })?;
        let rows = predictions.len();
        let outputs = self.model.as_ref().unwrap().output_count();
        Ok(numpy::ndarray::Array2::from_shape_vec(
            (rows, outputs),
            predictions.into_iter().flatten().collect(),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .into_pyarray(py))
    }
    fn apply_multivariate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let input = SeriesArrayInput::extract(x)?;
        self.validate_multivariate_shape(input.channels(), input.timepoints())?;
        let model = self.model.as_ref().ok_or_else(not_fitted)?;
        let result = input.with_series_views(|views| {
            apply_multivariate_many(model.forest(), views, self.random_state, self.n_jobs)
                .map_err(core_error)
        })?;
        Ok(numpy::ndarray::Array2::from_shape_vec(
            (result.sample_count(), result.tree_count()),
            result.into_leaf_indices().into_vec(),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .into_pyarray(py))
    }
    #[getter]
    fn n_channels_in_(&self) -> PyResult<usize> {
        self.n_channels_in.ok_or_else(not_fitted)
    }
    #[getter]
    fn n_timepoints_in_(&self) -> PyResult<usize> {
        self.n_timepoints_in.ok_or_else(not_fitted)
    }

    fn fit_unequal(&mut self, x: &Bound<'_, PyAny>, y: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let input = UnequalSeriesInput::extract(x)?;
        let shape = y.shape();
        if shape[0] != input.cases() || shape[1] == 0 {
            return Err(PyValueError::new_err("X and y shapes are incompatible"));
        }
        if self.n_exemplars > input.cases() {
            return Err(PyValueError::new_err(
                "n_exemplars cannot exceed the number of fitting cases",
            ));
        }
        let targets =
            RegressionTargets::from_f64(y.as_array().iter().copied().collect(), shape[0], shape[1])
                .map_err(core_error)?;
        let catalog = parse_multivariate_catalog(self.distance_tokens.to_vec())?;
        let sampling = input.sampling_context()?;
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
        configure_data_retention(&mut config, &self.data_retention)?;
        self.model = Some(input.with_series_views(|views| {
            fit_multivariate_regression_forest(
                config,
                views,
                &targets,
                &catalog,
                sampling,
                self.n_exemplars,
                SupervisedCriterion::Variance,
            )
            .map_err(core_error)
        })?);
        self.n_features_in = Some(input.channels());
        self.n_channels_in = Some(input.channels());
        self.n_timepoints_in = None;
        Ok(())
    }

    fn predict_unequal_matrix<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let input = UnequalSeriesInput::extract(x)?;
        self.validate_unequal_channels(input.channels())?;
        let predictions = input.with_series_views(|views| {
            predict_multivariate_regression_many(
                self.model.as_ref().ok_or_else(not_fitted)?,
                views,
                self.random_state,
                self.n_jobs,
            )
            .map_err(core_error)
        })?;
        let rows = predictions.len();
        let outputs = self.model.as_ref().ok_or_else(not_fitted)?.output_count();
        let array = numpy::ndarray::Array2::from_shape_vec(
            (rows, outputs),
            predictions.into_iter().flatten().collect(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    fn apply_unequal<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let input = UnequalSeriesInput::extract(x)?;
        self.validate_unequal_channels(input.channels())?;
        let model = self.model.as_ref().ok_or_else(not_fitted)?;
        let result = input.with_series_views(|views| {
            apply_multivariate_many(model.forest(), views, self.random_state, self.n_jobs)
                .map_err(core_error)
        })?;
        let array = numpy::ndarray::Array2::from_shape_vec(
            (result.sample_count(), result.tree_count()),
            result.into_leaf_indices().into_vec(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
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
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let predictions = self.predict_batch(x)?;
        let output_count = self.model.as_ref().ok_or_else(not_fitted)?.output_count();
        let rows = predictions.len();
        let values = predictions.into_iter().flatten().collect::<Vec<_>>();
        let array = numpy::ndarray::Array2::from_shape_vec((rows, output_count), values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let input = MatrixInput::extract(x)?;
        self.validate_features(input.columns())?;
        let model = self.model.as_ref().ok_or_else(not_fitted)?;
        let result = input.with_vector_views(|views| {
            apply_vector_many(model.forest(), views, self.random_state, self.n_jobs)
                .map_err(core_error)
        })?;
        let array = numpy::ndarray::Array2::from_shape_vec(
            (result.sample_count(), result.tree_count()),
            result.into_leaf_indices().into_vec(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    #[getter]
    fn data_retention_(&self) -> PyResult<&'static str> {
        let forest = self.model.as_ref().ok_or_else(not_fitted)?.forest();
        Ok(match forest.data_retention() {
            proximity_forests_core::forest::DataRetention::Exemplars => "exemplars",
            proximity_forests_core::forest::DataRetention::Copy => "copy",
            proximity_forests_core::forest::DataRetention::Reference => "reference",
        })
    }

    #[getter]
    fn has_training_data_(&self) -> PyResult<bool> {
        Ok(self
            .model
            .as_ref()
            .ok_or_else(not_fitted)?
            .forest()
            .has_training_data())
    }

    #[getter]
    fn has_embedded_training_data_(&self) -> PyResult<bool> {
        Ok(self
            .model
            .as_ref()
            .ok_or_else(not_fitted)?
            .forest()
            .has_embedded_training_data())
    }

    #[getter]
    fn supports_proximities_(&self) -> PyResult<bool> {
        Ok(true)
    }

    #[getter]
    fn supports_self_contained_imputation_(&self) -> PyResult<bool> {
        self.has_embedded_training_data_()
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
    fn predict_batch(&self, x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let input = MatrixInput::extract(x)?;
        self.validate_features(input.columns())?;
        input.with_vector_views(|views| {
            predict_vector_regression_many(
                self.model.as_ref().ok_or_else(not_fitted)?,
                views,
                self.random_state,
                self.n_jobs,
            )
            .map_err(core_error)
        })
    }

    fn validate_multivariate_shape(&self, channels: usize, timepoints: usize) -> PyResult<()> {
        if self.n_channels_in != Some(channels) || self.n_timepoints_in != Some(timepoints) {
            return Err(PyValueError::new_err("multivariate X shape mismatch"));
        }
        Ok(())
    }

    fn validate_unequal_channels(&self, channels: usize) -> PyResult<()> {
        if self.n_channels_in != Some(channels) || self.n_timepoints_in.is_some() {
            return Err(PyValueError::new_err(
                "unequal-length X does not match the fitted model layout or channel count",
            ));
        }
        Ok(())
    }

    fn validate_features(&self, feature_count: usize) -> PyResult<()> {
        let expected = self.n_features_in.ok_or_else(not_fitted)?;
        if feature_count != expected {
            return Err(PyValueError::new_err(format!(
                "X has {feature_count} features, but this estimator expects {expected}",
            )));
        }
        Ok(())
    }
}

fn not_fitted() -> PyErr {
    PyRuntimeError::new_err("This ProximityForestRegressor instance is not fitted yet")
}
