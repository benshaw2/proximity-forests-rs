# Getting started

The package exposes three estimators:

- {class}`proximity_forests.ProximityForestClassifier` for categorical targets;
- {class}`proximity_forests.ProximityForestRegressor` for continuous targets;
- {class}`proximity_forests.ProximityIsolationForest` for unsupervised isolation scores.

## Classification

```python
import numpy as np
from proximity_forests import ProximityForestClassifier

X = np.asarray(
    [[0.0, 0.1], [0.2, 0.1], [1.0, 0.9], [0.8, 1.0]],
    dtype=np.float64,
)
y = np.asarray([0, 0, 1, 1])

model = ProximityForestClassifier(
    n_estimators=100,
    random_state=0,
)
model.fit(X, y)
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

## Regression

```python
import numpy as np
from proximity_forests import ProximityForestRegressor

X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
y = np.asarray([0.1, 1.2, 1.9, 3.1], dtype=np.float64)

model = ProximityForestRegressor(
    n_estimators=100,
    random_state=0,
)
model.fit(X, y)
predictions = model.predict(X)
```

## Isolation

```python
import numpy as np
from proximity_forests import ProximityIsolationForest

X = np.asarray([[0.0], [0.1], [0.2], [5.0]], dtype=np.float64)

model = ProximityIsolationForest(
    n_estimators=100,
    random_state=0,
)
model.fit(X)
scores = model.score_samples(X)
labels = model.predict(X)
```

The estimator pages list the complete constructor parameters, fitted attributes,
and methods.
