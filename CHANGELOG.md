# Changelog

## 0.1.0b1

First public beta with installable Python packaging and a Rust-native core.

### Included
- classification, regression, and proximity isolation estimators
- tabular plus equal- and unequal-length univariate/multivariate inputs
- configurable bootstrap, subsampling, and full-data observation sampling
- dimension subset selection with distance/layout compatibility checks
- bounded deterministic parallel fitting and inference through `n_jobs`
- out-of-bag predictions, vote counts, scores, and zero-vote diagnostics
- missing-compatible distances, multi-output targets, and versioned pickle payloads
- exemplar-only and single-copy embedded training-data retention modes

### Beta limitations
- serialized models should be loaded with a compatible package version
- public reference-backed retention, proximity matrices, and imputation remain future work
- the Proximity Forest distance preset still requires its planned alignment audit
- prebuilt wheel availability depends on the release platform matrix


## 0.1.0a2

Second private alpha milestone focused on worker handling, input compatibility,
and development packaging.

### Fixed

- honor `n_jobs` during classification, regression, and isolation fitting
- default estimators to sequential execution with `n_jobs=1`, while preserving
  `None` and `-1` as automatic worker selection
- allow single-case inference for prediction, probability, leaf assignment,
  regression, isolation scoring, and unequal-length inputs
- accept aeon-style univariate collections shaped `(n_cases, 1, n_timepoints)`
  and canonicalize them to the univariate distance path

### Changed

- require Python 3.10 or newer
- consolidate runtime, test, notebook, build, and benchmark dependencies in
  `pyproject.toml`
- provide `test` and `dev` installation extras

- replace independent native thread groups with one bounded operation-scoped
  execution context
- add cost-gated candidate evaluation and calibrated query-tree inference tiling
- allow fitted native estimators to update their worker request without changing
  serialized forest state
### Validation

- verify exact fit and inference equivalence across worker counts
- calibrate candidate and inference activation thresholds in release mode

- add regression coverage for single-case inference and one-channel 3D input
- add controlled fit and inference granularity benchmarks to guide the upcoming
  bounded shared-pool parallelization refactor

### Not yet included

- persistent estimator-level executors and finer cross-platform worker calibration
- native proximity-matrix APIs
- native imputation APIs
- public reference-retention workflow
- comparative performance claims

## 0.1.0a1

Initial private alpha milestone.

### Included

- classifier, regressor, and isolation estimators
- tabular, equal/unequal univariate, and equal/unequal multivariate input
- float32 and float64 support
- missing-compatible NaN-aware distances with strict catalog validation
- deterministic missing-data routing semantics
- versioned serialization and pickle round trips
- exemplar-only and copied-training-data retention

### Not yet included

- public reference-retention workflow
- native proximity-matrix APIs
- native imputation APIs
- comparative performance claims
