# Changelog

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

### Validation

- add regression coverage for single-case inference and one-channel 3D input
- add controlled fit and inference granularity benchmarks to guide the upcoming
  bounded shared-pool parallelization refactor

### Not yet included

- shared-pool hierarchical parallelism
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
