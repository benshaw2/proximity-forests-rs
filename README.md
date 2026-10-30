# proximity-forests

`proximity-forests` is a beta-stage Python package backed by a Rust core for
proximity-forest classification, regression, and isolation models.

> **Beta status:** Public APIs may still change before a stable release.
> Serialized models should be loaded with a compatible package version.

## Current scope

- classification, multi-output classification, and multilabel targets
- single- and multi-target regression
- proximity-based isolation forests
- tabular and equal-length univariate input as 2D NumPy arrays
- equal-length multivariate input as 3D NumPy arrays shaped
  `(n_cases, n_channels, n_timepoints)`
- unequal-length univariate input as a list of 1D arrays
- unequal-length multivariate input as a list of 2D arrays shaped
  `(n_channels, n_timepoints_i)`
- native `float32` and `float64` input paths
- missing-data fitting and prediction when every configured distance advertises
  missing-value support
- versioned fitted-model serialization through Python `pickle`
- exemplar-only and embedded-copy training-data retention modes

## Installation

Install the published package from PyPI:

```bash
python -m pip install proximity-forests
```

Pre-release versions may require:

```bash
python -m pip install --pre proximity-forests
```

For editable source development, a Rust toolchain and Maturin are required:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
maturin develop --release
python -m pytest
```

## Minimal example

```python
import numpy as np
from proximity_forests import ProximityForestClassifier

X = np.array(
    [[0.0, 1.0], [0.1, 1.1], [4.0, 5.0], [4.1, 5.1]],
    dtype=np.float32,
)
y = np.array([0, 0, 1, 1])

model = ProximityForestClassifier(
    n_estimators=100,
    random_state=42,
).fit(X, y)

predictions = model.predict(X)
```

## Parallel execution

All three estimators accept `n_jobs`. The public worker contract is:

- `n_jobs=1`, the default, requests sequential native execution.
- `n_jobs=None` and `n_jobs=-1` request automatic selection from the logical CPUs
  visible to the process.
- a positive integer requests that many native workers. Values larger than the
  available CPU count are accepted but usually provide no benefit.

The Rust core creates one bounded, operation-scoped worker pool. Fitting may use
parallelism across trees and, for sufficiently expensive nodes, across split
candidates. Prediction, probability estimation, isolation scoring, and `apply`
may tile work across both queries and fitted trees. Small inference matrices stay
sequential to avoid pool and scheduling overhead. These thresholds are internal
calibration details and are not part of the stable public API.

Scheduling does not define model semantics. Stable seeds, indexed collection, and
ascending tree-order reduction preserve fitted structures and outputs across
worker counts for the same inputs and configuration. Floating-point predictions
are reduced in original tree order. Errors from indexed native work are reported
in stable index order.

`n_jobs` is an estimator parameter, not fitted-model state. Pickle payloads contain
the fitted forest and Python estimator parameters, but no worker pool or other
runtime executor state. See [`docs/concurrency.md`](docs/concurrency.md) for the
implementation contract, calibration scope, and deferred work.

## Missing values

NaN is accepted only when every possible distance in the configured catalog
supports missing values. Positive and negative infinity are always rejected.
Missing-compatible distances may also be used with complete data.

```python
X_missing = X.copy()
X_missing[0, 0] = np.nan

model = ProximityForestClassifier(
    distances=["nan_squared_euclidean"],
    random_state=42,
).fit(X_missing, y)
```

For multivariate series, use `dtw_arow_i` or `dtw_arow_d`.

## Known beta limitations

- reference-backed training-data retention is not yet exposed publicly
- native proximity-matrix and imputation APIs remain planned work
- serialized-model compatibility is versioned but not yet guaranteed across all
  future beta releases
- prebuilt wheel availability depends on the release platform matrix

## Development checks

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
./rebuild_and_test.sh
```

Parallel performance validation is intentionally separate from the test suite.
See [`benchmarks/PARALLEL_VALIDATION.md`](benchmarks/PARALLEL_VALIDATION.md) for
release-mode smoke checks and extended calibration commands.
