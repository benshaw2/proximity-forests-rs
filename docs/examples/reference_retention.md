# Reference-backed fitting

Reference retention stores structural metadata for a persistent `.npy` mapping
instead of embedding the complete training collection.

```python
import pickle
import numpy as np
from proximity_forests import ProximityForestClassifier

X = np.array(
    [[0.0, 0.1], [0.2, 0.0], [0.9, 1.0], [1.1, 0.8]],
    dtype=np.float64,
)
y = np.array([0, 0, 1, 1])

np.save("training.npy", X)
X_mapped = np.load("training.npy", mmap_mode="r")

model = ProximityForestClassifier(
    n_estimators=25,
    data_retention="reference",
    random_state=7,
)
model.fit(X_mapped, y)

metadata = model.external_training_data_
validated = model.validate_training_data_reference()
restored = pickle.loads(pickle.dumps(model))
predictions = restored.predict(X[:2])
```

Prediction uses retained fitted exemplars and does not automatically reopen the
external file. Call `validate_training_data_reference` when explicit lifecycle
validation or access to the complete mapped collection is required.
