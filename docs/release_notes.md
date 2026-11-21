# Release notes

## 0.1.0b2

Second public beta, adding exact per-tree sampling metadata for downstream
forest-proximity construction.

### Added

- `get_oob_mask()` on classifier, regressor, and isolation estimators, returning
  exact out-of-bag membership with shape `(n_samples_fit, n_estimators)`
- `get_in_bag_counts()` on all three estimator families, returning exact
  bootstrap or subsample multiplicities with the same sample-by-tree orientation
- lazy NumPy materialization from the fitted model's existing persisted sampling
  metadata
- coverage for bootstrap, subsample, full sampling, fitted-state errors,
  data-retention policies, and pickle round trips

### Integration

- enables external packages to construct uniform, KeRF, OOB, and GAP forest
  proximities without reconstructing samples from random seeds
- preserves bootstrap multiplicities greater than one, as required by GAP
  weighting

### Compatibility

- no fitted-model serialization format change
- no additional fitted-model memory unless a sampling matrix is requested
- existing `0.1.0b1` model payloads remain compatible with this beta release

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
