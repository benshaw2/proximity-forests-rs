# Equal-length multivariate time series

Store equal-length series as a three-dimensional array shaped
`(n_cases, n_channels, n_timepoints)`.

```python
import numpy as np
from proximity_forests import ProximityForestClassifier

rng = np.random.default_rng(7)
X_train = rng.normal(size=(24, 3, 32)).astype(np.float32)
y_train = np.repeat([0, 1], 12)
X_test = rng.normal(size=(4, 3, 32)).astype(np.float32)

model = ProximityForestClassifier(
    n_estimators=50,
    distances=["dtw_i", "dtw_d"],
    max_features="sqrt",
    random_state=7,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Independent distances use an `_i` suffix. Dependent distances use `_d`.
`max_features` samples channels for multivariate input.
