# proximity-forests

`proximity-forests` is an alpha-stage Python package backed by a Rust core for
proximity-forest classification, regression, and isolation models.

> **Alpha status:** The estimator, proximity, and future imputation APIs may
> change before a stable release. Serialized models should be loaded with a
> compatible package version.

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

## Installation from a local wheel

Building a wheel requires a Rust toolchain and Maturin. Installing the resulting
wheel does not require Rust.

```bash
python -m pip install "maturin>=1.9,<2.0"
maturin build --release --out dist
python -m pip install dist/proximity_forests-0.1.0a1-*.whl
```

For editable development:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip maturin
maturin develop --release
python -m pytest python/tests
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

## Known alpha limitations

- reference-backed training-data retention is not yet exposed publicly
- native proximity-matrix and imputation APIs remain planned work
- serialized-model compatibility is versioned but not yet guaranteed across all
  future alpha releases
- prebuilt wheel availability depends on the release platform matrix

## Development checks

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
./rebuild_and_test.sh
```
