# Unequal-length time series

Use a sequence containing one array per case. Each case is shaped
`(n_channels, n_timepoints_i)`, and every case uses the same channel count.

```python
import numpy as np
from proximity_forests import ProximityForestClassifier

X_train = [
    np.array([[0.0, 0.2, 0.4]], dtype=np.float64),
    np.array([[0.1, 0.2, 0.3, 0.5]], dtype=np.float64),
    np.array([[1.0, 0.8, 0.6]], dtype=np.float64),
    np.array([[1.1, 0.9, 0.7, 0.5, 0.3]], dtype=np.float64),
]
y_train = np.array([0, 0, 1, 1])

model = ProximityForestClassifier(
    n_estimators=25,
    distances=["dtw_i"],
    random_state=7,
)
model.fit(X_train, y_train)
predictions = model.predict(X_train)
```

Reference retention is not currently available for unequal-length collections.
Use `data_retention="exemplars"` or `"copy"`.
