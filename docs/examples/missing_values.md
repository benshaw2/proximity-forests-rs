# Missing values

NaN values are accepted only when every configured distance supports missing
values. Positive and negative infinity are always rejected.

## Vector input

```python
import numpy as np
from proximity_forests import ProximityForestClassifier

X = np.array(
    [[0.0, np.nan], [0.2, 0.1], [0.9, 1.0], [1.1, np.nan]],
    dtype=np.float64,
)
y = np.array([0, 0, 1, 1])

model = ProximityForestClassifier(
    n_estimators=25,
    distances=["nan_squared_euclidean"],
    random_state=7,
)
model.fit(X, y)
predictions = model.predict(X)
```

## Multivariate time series

Use `dtw_arow_i` for independent channel handling or `dtw_arow_d` for dependent
channel handling.

```python
X_series = np.array(
    [
        [[0.0, np.nan, 0.2], [1.0, 0.9, 0.8]],
        [[0.1, 0.2, 0.3], [np.nan, 0.8, 0.7]],
        [[1.0, 0.8, 0.6], [0.0, 0.1, np.nan]],
        [[0.9, 0.7, 0.5], [0.1, 0.2, 0.3]],
    ],
    dtype=np.float64,
)

model = ProximityForestClassifier(
    n_estimators=25,
    distances=["dtw_arow_i"],
    random_state=7,
)
model.fit(X_series, y)
```
