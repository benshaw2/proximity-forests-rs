//! Native classification estimator state exposed to Python.

use std::num::NonZeroUsize;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use proximity_forests_core::{ClassificationTargets, SupervisedCriterion};

use proximity_forests_core::forest::{
    ClassificationForest, DimensionSelection, ForestClassificationPrediction, ForestConfig,
    apply_multivariate_many, apply_vector_many, fit_multivariate_classification_forest,
    fit_vector_classification_forest, predict_multivariate_classification_many,
    predict_vector_classification_many,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::{
    distance_spec::{
        parse_multivariate_catalog, parse_vector_catalog, validate_missing_compatibility,
    },
    error::core_error,
    input::{MatrixInput, SeriesArrayInput, UnequalSeriesInput, configure_data_retention},
};

/// Rust-backed classifier used by the public Python estimator wrapper.
#[pyclass(name = "_NativeProximityForestClassifier")]
pub struct NativeProximityForestClassifier {
    n_estimators: usize,
    n_candidates: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    bootstrap: bool,
    random_state: u64,
    n_jobs: usize,
    data_retention: String,
    distance_tokens: Box<[String]>,
    model: Option<ClassificationForest>,
    n_features_in: Option<usize>,
    n_channels_in: Option<usize>,
    n_timepoints_in: Option<usize>,
    class_counts: Option<Box<[usize]>>,
}

#[pymethods]
impl NativeProximityForestClassifier {
    /// Updates the worker request used by subsequent fit and inference operations.
    fn set_n_jobs(&mut self, n_jobs: usize) {
        self.n_jobs = n_jobs;
    }
    /// Returns the worker request used by subsequent native operations.
    #[getter]
    fn n_jobs_(&self) -> usize {
        self.n_jobs
    }
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
        class_counts: Vec<usize>,
        n_channels_in: Option<usize>,
        n_timepoints_in: Option<usize>,
    ) -> PyResult<()> {
        if n_features_in == 0 || class_counts.is_empty() || class_counts.contains(&0) {
            return Err(PyValueError::new_err(
                "serialized classifier metadata is invalid",
            ));
        }
        if n_channels_in.is_none() && n_timepoints_in.is_some()
            || n_channels_in == Some(0)
            || n_timepoints_in == Some(0)
        {
            return Err(PyValueError::new_err(
                "serialized classifier layout metadata is invalid",
            ));
        }
        let model = ClassificationForest::from_bytes(payload).map_err(core_error)?;
        if model.output_count() != class_counts.len() {
            return Err(PyValueError::new_err(
                "serialized classifier output count disagrees with Python metadata",
            ));
        }
        self.model = Some(model);
        self.n_features_in = Some(n_features_in);
        self.n_channels_in = n_channels_in;
        self.n_timepoints_in = n_timepoints_in;
        self.class_counts = Some(class_counts.into_boxed_slice());
        Ok(())
    }

    #[new]
    #[pyo3(signature = (n_estimators=100, n_candidates=5, max_depth=None,
        min_samples_split=2, bootstrap=true, random_state=0, n_jobs=1, data_retention=String::from("exemplars"), distances=vec![String::from("squared_euclidean")]))]
    fn new(
        n_estimators: usize,
        n_candidates: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        bootstrap: bool,
        random_state: u64,
        n_jobs: usize,
        data_retention: String,
        distances: Vec<String>,
    ) -> PyResult<Self> {
        if n_estimators == 0 || n_candidates == 0 || min_samples_split == 0 {
            return Err(PyValueError::new_err(
                "n_estimators, n_candidates, and min_samples_split must be positive",
            ));
        }
        Ok(Self {
            n_estimators,
            n_candidates,
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
            class_counts: None,
        })
    }

    /// Fits row-major encoded single- or multi-output class identifiers.
    fn fit(
        &mut self,
        x: &Bound<'_, PyAny>,
        y: PyReadonlyArray2<'_, u32>,
        class_counts: Vec<usize>,
    ) -> PyResult<()> {
        let input = MatrixInput::extract(x)?;
        input.contains_nan()?;
        let target_shape = y.shape();
        if target_shape[0] != input.rows() {
            return Err(PyValueError::new_err(
                "X and y contain different numbers of samples",
            ));
        }
        if target_shape[1] == 0 {
            return Err(PyValueError::new_err(
                "y must contain at least one classification output",
            ));
        }
        if class_counts.len() != target_shape[1] {
            return Err(PyValueError::new_err(
                "class_counts must contain one entry per classification output",
            ));
        }
        if class_counts.iter().any(|&count| count < 2) {
            return Err(PyValueError::new_err(
                "each classification output requires at least two classes",
            ));
        }
        for (output, &count) in class_counts.iter().enumerate() {
            let count_u32 = u32::try_from(count)
                .map_err(|_| PyValueError::new_err("class count does not fit in u32"))?;
            if y.as_array()
                .column(output)
                .iter()
                .any(|&class| class >= count_u32)
            {
                return Err(PyValueError::new_err(
                    "encoded class identifier is outside its output class range",
                ));
            }
        }
        let distance_sampling = input.sampling_context()?;
        let distance_catalog = parse_vector_catalog(self.distance_tokens.to_vec())?;
        validate_missing_compatibility(&distance_catalog, input.contains_nan()?)?;
        let target_values = y.as_array().iter().copied().collect();
        let targets = ClassificationTargets::new(target_values, target_shape[0], target_shape[1])
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
            fit_vector_classification_forest(
                config,
                self.n_jobs,
                views,
                &targets,
                &distance_catalog,
                distance_sampling,
                feature_count,
                SupervisedCriterion::Gini,
            )
            .map_err(core_error)
        })?;
        self.model = Some(model);
        self.n_features_in = Some(feature_count);
        self.class_counts = Some(class_counts.into_boxed_slice());
        Ok(())
    }

    /// Fits an equal-length multivariate collection shaped (cases, channels, timepoints).
    fn fit_multivariate(
        &mut self,
        x: &Bound<'_, PyAny>,
        y: PyReadonlyArray2<'_, u32>,
        class_counts: Vec<usize>,
    ) -> PyResult<()> {
        let input = SeriesArrayInput::extract(x)?;
        let shape = y.shape();
        if shape[0] != input.cases() || shape[1] == 0 || class_counts.len() != shape[1] {
            return Err(PyValueError::new_err(
                "X, y, and class_counts have incompatible shapes",
            ));
        }
        if class_counts.iter().any(|&count| count < 2) {
            return Err(PyValueError::new_err(
                "each classification output requires at least two classes",
            ));
        }
        let targets =
            ClassificationTargets::new(y.as_array().iter().copied().collect(), shape[0], shape[1])
                .map_err(core_error)?;
        let catalog = parse_multivariate_catalog(self.distance_tokens.to_vec())?;
        validate_missing_compatibility(&catalog, input.contains_nan()?)?;
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
            fit_multivariate_classification_forest(
                config,
                self.n_jobs,
                views,
                &targets,
                &catalog,
                sampling,
                SupervisedCriterion::Gini,
            )
            .map_err(core_error)
        })?);
        self.n_features_in = Some(input.channels());
        self.n_channels_in = Some(input.channels());
        self.n_timepoints_in = Some(input.timepoints());
        self.class_counts = Some(class_counts.into_boxed_slice());
        Ok(())
    }

    fn predict_multivariate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let predictions = self.predict_multivariate_batch(x)?;
        if self.output_count()? != 1 {
            return Err(PyValueError::new_err(
                "use predict_multivariate_matrix for multi-output models",
            ));
        }
        Ok(predictions
            .iter()
            .map(|p| p.classes()[0])
            .collect::<Vec<_>>()
            .into_pyarray(py))
    }

    fn predict_multivariate_matrix<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let predictions = self.predict_multivariate_batch(x)?;
        let outputs = self.output_count()?;
        let values = predictions
            .iter()
            .flat_map(|p| p.classes().iter().copied())
            .collect();
        let array = numpy::ndarray::Array2::from_shape_vec((predictions.len(), outputs), values)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    fn predict_multivariate_proba_output<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        output: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let predictions = self.predict_multivariate_batch(x)?;
        let n_classes = *self
            .class_counts
            .as_deref()
            .ok_or_else(not_fitted)?
            .get(output)
            .ok_or_else(|| PyValueError::new_err("classification output index is out of range"))?;
        let mut values = vec![0.0; predictions.len() * n_classes];
        for (sample, prediction) in predictions.iter().enumerate() {
            for &(class, probability) in prediction
                .probabilities(output)
                .ok_or_else(|| PyRuntimeError::new_err("probabilities unavailable"))?
            {
                values[sample * n_classes + class as usize] = probability;
            }
        }
        let array = numpy::ndarray::Array2::from_shape_vec((predictions.len(), n_classes), values)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    fn apply_multivariate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let input = SeriesArrayInput::extract(x)?;
        input.contains_nan()?;
        self.validate_multivariate_shape(input.channels(), input.timepoints())?;
        let model = self.model.as_ref().ok_or_else(not_fitted)?;
        let result = input.with_series_views(|views| {
            apply_multivariate_many(model.forest(), views, self.random_state, self.n_jobs)
                .map_err(core_error)
        })?;
        let array = numpy::ndarray::Array2::from_shape_vec(
            (result.sample_count(), result.tree_count()),
            result.into_leaf_indices().into_vec(),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    #[getter]
    fn n_channels_in_(&self) -> PyResult<usize> {
        self.n_channels_in.ok_or_else(not_fitted)
    }
    #[getter]
    fn n_timepoints_in_(&self) -> PyResult<usize> {
        self.n_timepoints_in.ok_or_else(not_fitted)
    }

    /// Fits an unequal-length collection of `(channels, timepoints_i)` cases.
    fn fit_unequal(
        &mut self,
        x: &Bound<'_, PyAny>,
        y: PyReadonlyArray2<'_, u32>,
        class_counts: Vec<usize>,
    ) -> PyResult<()> {
        let input = UnequalSeriesInput::extract(x)?;
        if input.cases() < 2 {
            return Err(PyValueError::new_err(
                "fitting unequal-length X requires at least two cases",
            ));
        }
        let shape = y.shape();
        if shape[0] != input.cases() || shape[1] == 0 || class_counts.len() != shape[1] {
            return Err(PyValueError::new_err(
                "X, y, and class_counts have incompatible shapes",
            ));
        }
        if class_counts.iter().any(|&count| count < 2) {
            return Err(PyValueError::new_err(
                "each classification output requires at least two classes",
            ));
        }
        let targets =
            ClassificationTargets::new(y.as_array().iter().copied().collect(), shape[0], shape[1])
                .map_err(core_error)?;
        let catalog = parse_multivariate_catalog(self.distance_tokens.to_vec())?;
        validate_missing_compatibility(&catalog, input.contains_nan()?)?;
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
            fit_multivariate_classification_forest(
                config,
                self.n_jobs,
                views,
                &targets,
                &catalog,
                sampling,
                SupervisedCriterion::Gini,
            )
            .map_err(core_error)
        })?);
        self.n_features_in = Some(input.channels());
        self.n_channels_in = Some(input.channels());
        self.n_timepoints_in = None;
        self.class_counts = Some(class_counts.into_boxed_slice());
        Ok(())
    }

    fn predict_unequal<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        if self.output_count()? != 1 {
            return Err(PyValueError::new_err(
                "use predict_unequal_matrix for multi-output models",
            ));
        }
        let predictions = self.predict_unequal_batch(x)?;
        Ok(predictions
            .iter()
            .map(|prediction| prediction.classes()[0])
            .collect::<Vec<_>>()
            .into_pyarray(py))
    }

    fn predict_unequal_matrix<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let predictions = self.predict_unequal_batch(x)?;
        let outputs = self.output_count()?;
        let values = predictions
            .iter()
            .flat_map(|prediction| prediction.classes().iter().copied())
            .collect();
        let array = numpy::ndarray::Array2::from_shape_vec((predictions.len(), outputs), values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    fn predict_unequal_proba_output<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        output: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let predictions = self.predict_unequal_batch(x)?;
        let n_classes = *self
            .class_counts
            .as_deref()
            .ok_or_else(not_fitted)?
            .get(output)
            .ok_or_else(|| PyValueError::new_err("classification output index is out of range"))?;
        let mut values = vec![0.0; predictions.len() * n_classes];
        for (sample, prediction) in predictions.iter().enumerate() {
            for &(class, probability) in prediction
                .probabilities(output)
                .ok_or_else(|| PyRuntimeError::new_err("probabilities unavailable"))?
            {
                values[sample * n_classes + class as usize] = probability;
            }
        }
        let array = numpy::ndarray::Array2::from_shape_vec((predictions.len(), n_classes), values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    fn apply_unequal<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let input = UnequalSeriesInput::extract(x)?;
        input.contains_nan()?;
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

    /// Predicts encoded classes for a single fitted output.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        if self.output_count()? != 1 {
            return Err(PyValueError::new_err(
                "predict is only available for one-output models; use predict_matrix",
            ));
        }
        let predictions = self.predict_batch(x)?;
        let values = predictions
            .iter()
            .map(|prediction| prediction.classes()[0])
            .collect::<Vec<_>>();
        Ok(values.into_pyarray(py))
    }

    /// Predicts encoded classes for all fitted outputs.
    fn predict_matrix<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let predictions = self.predict_batch(x)?;
        let output_count = self.output_count()?;
        let mut values = Vec::with_capacity(predictions.len() * output_count);
        for prediction in &predictions {
            if prediction.classes().len() != output_count {
                return Err(PyRuntimeError::new_err(
                    "native classification prediction has the wrong output width",
                ));
            }
            values.extend_from_slice(prediction.classes());
        }
        let array =
            numpy::ndarray::Array2::from_shape_vec((predictions.len(), output_count), values)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    /// Returns probabilities for one fitted output.
    fn predict_proba_output<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        output: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let predictions = self.predict_batch(x)?;
        let class_counts = self.class_counts.as_deref().ok_or_else(not_fitted)?;
        let &n_classes = class_counts
            .get(output)
            .ok_or_else(|| PyValueError::new_err("classification output index is out of range"))?;
        let mut values = vec![0.0; predictions.len() * n_classes];
        for (sample, prediction) in predictions.iter().enumerate() {
            let probabilities = prediction.probabilities(output).ok_or_else(|| {
                PyRuntimeError::new_err("classification output probability was unavailable")
            })?;
            for &(class, probability) in probabilities {
                let class = usize::try_from(class)
                    .map_err(|_| PyRuntimeError::new_err("class identifier overflow"))?;
                if class >= n_classes {
                    return Err(PyRuntimeError::new_err(
                        "native class identifier is outside fitted classes",
                    ));
                }
                values[sample * n_classes + class] = probability;
            }
        }
        let array = numpy::ndarray::Array2::from_shape_vec((predictions.len(), n_classes), values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let input = MatrixInput::extract(x)?;
        input.contains_nan()?;
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
            .map(|model| model.output_count())
            .ok_or_else(not_fitted)
    }
}

impl NativeProximityForestClassifier {
    fn predict_batch(&self, x: &Bound<'_, PyAny>) -> PyResult<Vec<ForestClassificationPrediction>> {
        let input = MatrixInput::extract(x)?;
        input.contains_nan()?;
        self.validate_features(input.columns())?;
        input.with_vector_views(|views| {
            predict_vector_classification_many(
                self.model.as_ref().ok_or_else(not_fitted)?,
                views,
                self.random_state,
                self.n_jobs,
            )
            .map_err(core_error)
        })
    }

    fn predict_multivariate_batch(
        &self,
        x: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<ForestClassificationPrediction>> {
        let input = SeriesArrayInput::extract(x)?;
        input.contains_nan()?;
        self.validate_multivariate_shape(input.channels(), input.timepoints())?;
        input.with_series_views(|views| {
            predict_multivariate_classification_many(
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
            return Err(PyValueError::new_err(
                "multivariate X shape does not match fitted channels and timepoints",
            ));
        }
        Ok(())
    }

    fn predict_unequal_batch(
        &self,
        x: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<ForestClassificationPrediction>> {
        let input = UnequalSeriesInput::extract(x)?;
        input.contains_nan()?;
        self.validate_unequal_channels(input.channels())?;
        input.with_series_views(|views| {
            predict_multivariate_classification_many(
                self.model.as_ref().ok_or_else(not_fitted)?,
                views,
                self.random_state,
                self.n_jobs,
            )
            .map_err(core_error)
        })
    }

    fn validate_unequal_channels(&self, channels: usize) -> PyResult<()> {
        if self.n_channels_in != Some(channels) || self.n_timepoints_in.is_some() {
            return Err(PyValueError::new_err(
                "unequal-length X does not match the fitted model layout or channel count",
            ));
        }
        Ok(())
    }

    fn output_count(&self) -> PyResult<usize> {
        self.model
            .as_ref()
            .map(ClassificationForest::output_count)
            .ok_or_else(not_fitted)
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
    PyRuntimeError::new_err("This ProximityForestClassifier instance is not fitted yet")
}
