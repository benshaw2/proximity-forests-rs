//! Native classification estimator state exposed to Python.

use std::num::NonZeroUsize;

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
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
    n_classes: Option<usize>,
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
            n_classes: None,
        })
    }

    /// Fits encoded single-output class identifiers.
    fn fit(
        &mut self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, u32>,
        n_classes: usize,
    ) -> PyResult<()> {
        let shape = x.shape();
        if shape[0] == 0 || shape[1] == 0 {
            return Err(PyValueError::new_err("X must be a nonempty 2D array"));
        }
        if y.len() != shape[0] {
            return Err(PyValueError::new_err(
                "X and y contain different numbers of samples",
            ));
        }
        if n_classes < 2 {
            return Err(PyValueError::new_err(
                "classification requires at least two classes",
            ));
        }

        let distance_sampling = vector_sampling_context(x.as_array())?;
        let rows = copy_rows(x.as_array());
        let views: Vec<_> = rows
            .iter()
            .map(|row| NumericVectorView::from_f64(row))
            .collect();
        let targets =
            ClassificationTargets::single_output(y.as_slice()?.to_vec()).map_err(core_error)?;
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

        // The first Python slice intentionally starts with a conservative,
        // parameter-free distance catalog. Additional typed distances will be
        // exposed as the Python parameter translator is expanded.
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
        self.n_classes = Some(n_classes);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let predictions = self.predict_batch(x)?;
        let values: Vec<u32> = predictions
            .iter()
            .map(|prediction| prediction.classes()[0])
            .collect();
        Ok(values.into_pyarray(py))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let predictions = self.predict_batch(x)?;
        let n_classes = self.n_classes.ok_or_else(not_fitted)?;
        let mut values = vec![0.0; predictions.len() * n_classes];
        for (sample, prediction) in predictions.iter().enumerate() {
            let probabilities = prediction.probabilities(0).ok_or_else(|| {
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
