# Classification

`ProximityForestClassifier` accepts categorical labels and exposes class
probabilities.

```python
import numpy as np
from proximity_forests import ProximityForestClassifier

X_train = np.array(
    [[0.0, 0.1], [0.2, 0.0], [0.9, 1.0], [1.1, 0.8]],
    dtype=np.float64,
)
y_train = np.array(["low", "low", "high", "high"])
X_test = np.array([[0.1, 0.1], [1.0, 0.9]], dtype=np.float64)

model = ProximityForestClassifier(
    n_estimators=100,
    distances=["squared_euclidean"],
    random_state=7,
    n_jobs=-1,
)
model.fit(X_train, y_train)

labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
leaf_indices = model.apply(X_test)
```

`labels` uses the original label values. `probabilities` has one column for each
entry in `classes_`. `leaf_indices` has shape `(n_test, n_estimators)`.

## Multi-output targets

Supply a two-dimensional target with one column per output:

```python
y_multi = np.array(
    [["low", "left"], ["low", "right"],
     ["high", "left"], ["high", "right"]],
)

model.fit(X_train, y_multi)
labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

For multi-output classification, `probabilities` is a list containing one
probability matrix per output.
