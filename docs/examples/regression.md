# Regression

`ProximityForestRegressor` supports one or several continuous targets.

```python
import numpy as np
from proximity_forests import ProximityForestRegressor

X_train = np.array(
    [[0.0, 0.1], [0.2, 0.0], [0.9, 1.0], [1.1, 0.8]],
    dtype=np.float64,
)
y_train = np.array([0.0, 0.3, 1.8, 2.1], dtype=np.float64)
X_test = np.array([[0.1, 0.1], [1.0, 0.9]], dtype=np.float64)

model = ProximityForestRegressor(
    n_estimators=100,
    n_exemplars=2,
    distances=["squared_euclidean"],
    random_state=7,
    n_jobs=-1,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
leaf_indices = model.apply(X_test)
```

## Multiple targets

```python
y_multi = np.column_stack([y_train, y_train**2])
model.fit(X_train, y_multi)
predictions = model.predict(X_test)
```

The resulting prediction matrix has one column per target.
