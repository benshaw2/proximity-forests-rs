# Isolation

`ProximityIsolationForest` fits without a target. Lower `score_samples` values
represent more abnormal observations.

```python
import numpy as np
from proximity_forests import ProximityIsolationForest

X = np.array([[0.0], [0.1], [0.2], [0.15], [5.0]], dtype=np.float64)

model = ProximityIsolationForest(
    n_estimators=100,
    contamination=0.2,
    random_state=7,
    n_jobs=-1,
)
model.fit(X)

normality_scores = model.score_samples(X)
shifted_scores = model.decision_function(X)
labels = model.predict(X)
raw_anomaly_scores = model.anomaly_score(X)
```

`predict` returns `1` for inliers and `-1` for outliers. `anomaly_score` uses the
native anomaly-oriented direction, where larger values are more abnormal.
