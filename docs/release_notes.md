# Release notes

## 0.1.0b1

This beta release provides the first public Python package backed by the Rust
proximity-forest core.

### Estimators

- `ProximityForestClassifier`, including probability prediction and multi-output
  classification;
- `ProximityForestRegressor`, including multi-target regression;
- `ProximityIsolationForest`, including anomaly scores, decision scores, and
  inlier/outlier predictions.

### Data and distances

- two-dimensional vector input;
- equal-length multivariate time series;
- unequal-length time-series collections;
- configurable vector, independent multivariate, and dependent multivariate
  distance catalogs;
- missing-value-aware distance families;
- observation and dimension sampling.

### Fitting and fitted artifacts

- deterministic parallel fitting and inference through `n_jobs`;
- out-of-bag results for classification and regression;
- terminal-node assignments through `apply`;
- versioned native fitted-model serialization;
- exemplar-only, copied, and `.npy` reference-backed retention modes.

### Beta compatibility notice

The package is in beta. Public APIs and serialized-model compatibility may
change before the first stable release. Persist the package version alongside
long-lived model artifacts.
