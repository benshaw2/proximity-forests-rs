# proximity-forests

`proximity-forests` is a beta Python package backed by a Rust core for
proximity-forest classification, regression, and isolation.

> **Beta status:** Public APIs and serialized-model compatibility may change
> before a stable release.

## Installation

```bash
python -m pip install proximity-forests
```

Pre-release versions may require:

```bash
python -m pip install --pre proximity-forests
```

## Quick start

```python
import numpy as np
from proximity_forests import ProximityForestClassifier

X = np.array(
    [[0.0, 0.1], [0.2, 0.0], [0.9, 1.0], [1.1, 0.8]],
    dtype=np.float64,
)
y = np.array([0, 0, 1, 1])

model = ProximityForestClassifier(
    n_estimators=100,
    random_state=7,
).fit(X, y)

predictions = model.predict(X)
```

## Estimators

- `ProximityForestClassifier`
- `ProximityForestRegressor`
- `ProximityIsolationForest`

The package supports vector input, equal-length multivariate time series, and
unequal-length time-series collections. It includes configurable distance
catalogs, observation and dimension sampling, parallel execution, OOB results,
and versioned fitted-model persistence.

## Documentation

- [Getting started](docs/getting_started.md)
- [Input data](docs/input_data.md)
- [Distances](docs/distances.md)
- [Sampling and dimensions](docs/sampling_and_dimensions.md)
- [Out-of-bag results](docs/oob_results.md)
- [Persistence and retention](docs/model_persistence.md)
- [Examples](docs/examples/index.md)
- [API reference](docs/api_reference/index.md)

Build the local HTML documentation with:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -E -W --keep-going -b html docs docs/_build/html
```

## Development

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
maturin develop --release
./rebuild_and_test.sh
```
