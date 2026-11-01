# Model persistence and training-data retention

Fitted estimators support Python serialization. The native fitted forest is
stored as a versioned binary payload inside the Python estimator state.

## Pickle round trip

```python
import pickle

payload = pickle.dumps(model)
restored = pickle.loads(payload)
predictions = restored.predict(X_test)
```

The serialized estimator retains constructor parameters, fitted metadata, tree
structure, leaf state, and the observations required by its retention policy.
Runtime worker pools and threads are not serialized.

## Retention modes

Set `data_retention` before fitting.

### `"exemplars"`

This is the default. The fitted artifact embeds only observations selected as
splitter exemplars. Ordinary prediction, scoring, and `apply` remain available.
The complete training collection is not retained.

```python
model = ProximityForestClassifier(data_retention="exemplars")
```

### `"copy"`

The fitted artifact embeds one complete copy of the training collection in
addition to the fitted exemplar pool.

```python
model = ProximityForestClassifier(data_retention="copy")
```

This mode produces a self-contained artifact when later operations require all
training values. It also increases retained memory and serialized size.

### `"reference"`

Reference mode stores a descriptor for a dense `.npy` file rather than embedding
the complete training collection. It supports dense 2D vector data and dense 3D
equal-length time-series data.

Create a persistent mapping with:

```python
import numpy as np

np.save("training.npy", X)
X_mapped = np.load("training.npy", mmap_mode="r")

model = ProximityForestClassifier(data_retention="reference")
model.fit(X_mapped, y)
```

The mapping must:

- be an `np.memmap` opened from a `.npy` file;
- use `float32` or `float64`;
- be C-contiguous;
- represent the complete payload rather than a slice;
- have no trailing payload data.

Reference retention does not currently support unequal-length collections.

## Inspecting a reference

After fitting:

```python
descriptor = model.external_training_data_
```

The descriptor includes the path, dtype, shape, payload offset, payload length,
file size, and optional fingerprint.

Validate and reopen it explicitly with:

```python
mapped = model.validate_training_data_reference()
```

Validation checks the file and `.npy` metadata. Ordinary inference uses embedded
exemplars and does not automatically open the referenced file.

## Retention capability attributes

A fitted estimator exposes:

- `has_training_data_`;
- `has_embedded_training_data_`;
- `supports_proximities_`;
- `supports_self_contained_imputation_`.

These attributes describe the fitted artifact rather than the constructor alone.
