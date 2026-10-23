//! Python extension module for proximity-forests.

mod classifier;
mod error;
mod input;
mod isolation_estimator;
mod regressor;

use classifier::NativeProximityForestClassifier;
use isolation_estimator::NativeProximityIsolationForest;
use pyo3::prelude::*;
use regressor::NativeProximityForestRegressor;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<NativeProximityForestClassifier>()?;
    module.add_class::<NativeProximityForestRegressor>()?;
    module.add_class::<NativeProximityIsolationForest>()?;
    module.add_function(wrap_pyfunction!(runtime_info, module)?)?;
    Ok(())
}

#[pyfunction]
fn runtime_info() -> (&'static str, &'static str) {
    (env!("CARGO_PKG_VERSION"), std::env::consts::ARCH)
}
