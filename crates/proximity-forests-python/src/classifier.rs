//! Native classification estimator state exposed to Python.

use std::num::NonZeroUsize;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use proximity_forests_core::{
    ClassificationTargets, DistanceChoice, DistanceSamplingContext, NumericVectorView,
    SupervisedCriterion,
};

use proximity_forests_core::forest::{
    ClassificationForest, DimensionSelection, ForestClassificationPrediction, ForestConfig,
    apply_vector_many, fit_vector_classification_forest, predict_vector_classification_many,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::{distance_spec::parse_vector_catalog, error::core_error};

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
    distance_catalog: Box<[DistanceChoice]>,
    model: Option<ClassificationForest>,
    n_features_in: Option<usize>,
    class_counts: Option<Box<[usize]>>,
}

#[pymethods]
impl NativeProximityForestClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, n_candidates=5, max_depth=None,
        min_samples_split=2, bootstrap=true, random_state=0, n_jobs=0, distances=vec![String::from("squared_euclidean")]))]
    fn new(
        n_estimators: usize,
        n_candidates: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        bootstrap: bool,
        random_state: u64,
        n_jobs: usize,
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
            distance_catalog: parse_vector_catalog(distances)?,
            model: None,
            n_features_in: None,
            class_counts: None,
        })
    }

    /// Fits row-major encoded single- or multi-output class identifiers.
    fn fit(
        &mut self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray2<'_, u32>,
        class_counts: Vec<usize>,
    ) -> PyResult<()> {
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

        let distance_sampling = vector_sampling_context(x.as_array())?;
        let rows = copy_rows(x.as_array());
        let views: Vec<_> = rows
            .iter()
            .map(|row| NumericVectorView::from_f64(row))
            .collect();
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

        let model = fit_vector_classification_forest(
            config,
            &views,
            &targets,
            &self.distance_catalog,
            distance_sampling,
            shape[1],
            SupervisedCriterion::Gini,
        )
        .map_err(core_error)?;
        self.model = Some(model);
        self.n_features_in = Some(shape[1]);
        self.class_counts = Some(class_counts.into_boxed_slice());
        Ok(())
    }

    /// Predicts encoded classes for a single fitted output.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        if self.output_count()? != 1 {
            return Err(PyValueError::new_err(
                "predict is only available for one-output models; use predict_matrix",
            ));
        }
        let predictions = self.predict_batch(x)?;
        let values: Vec<u32> = predictions
            .iter()
            .map(|prediction| prediction.classes()[0])
            .collect();
        Ok(values.into_pyarray(py))
    }

    /// Predicts encoded classes for all fitted outputs.
    fn predict_matrix<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
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
        x: PyReadonlyArray2<'py, f64>,
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
            .map(|model| model.output_count())
            .ok_or_else(not_fitted)
    }
}

impl NativeProximityForestClassifier {
    fn predict_batch(
        &self,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Vec<ForestClassificationPrediction>> {
        self.validate_features(x.shape())?;
        let rows = copy_rows(x.as_array());
        let views: Vec<_> = rows
            .iter()
            .map(|row| NumericVectorView::from_f64(row))
            .collect();
        predict_vector_classification_many(
            self.model.as_ref().ok_or_else(not_fitted)?,
            &views,
            self.random_state,
            self.n_jobs,
        )
        .map_err(core_error)
    }

    fn output_count(&self) -> PyResult<usize> {
        self.model
            .as_ref()
            .map(ClassificationForest::output_count)
            .ok_or_else(not_fitted)
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
    PyRuntimeError::new_err("This ProximityForestClassifier instance is not fitted yet")
}
