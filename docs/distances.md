# Distances

Candidate splitters route observations by comparing them with exemplars. The
`distances` parameter controls which distance families may be selected while a
forest is fitted.

## Selecting distances

Use the default catalog:

```python
model = ProximityForestClassifier(distances="default")
```

Use the package's broader proximity-forest catalog:

```python
model = ProximityForestClassifier(distances="pf")
```

Restrict fitting to one family:

```python
model = ProximityForestClassifier(distances=["dtw"])
```

Provide several candidate families:

```python
model = ProximityForestClassifier(
    distances=["squared_euclidean", "dtw", "erp"],
)
```

A plain string selects a distance family. Parameters for that family may be
sampled independently for candidate splitters when the implementation defines a
sampling rule.

## Fixed distance configurations

Use {class}`proximity_forests.DistanceSpec` when an exact configuration is
required:

```python
from proximity_forests import DistanceSpec, ProximityForestClassifier

model = ProximityForestClassifier(
    distances=[DistanceSpec(name="dtw", window=12)],
)
```

A dictionary with a `name` entry is equivalent:

```python
model = ProximityForestClassifier(
    distances=[{"name": "dtw", "window": 12}],
)
```

Supported fixed fields are:

- `dependent`, for choosing independent or dependent multivariate behavior;
- `window`, for a nonnegative alignment window;
- `stiffness`;
- `gap`;
- `epsilon`;
- `cost`;
- `penalty`;
- `no_overlap`.

Only parameters used by the selected family affect its computation. Invalid
combinations are rejected during normalization or fitting.

## Vector and univariate catalogs

Two-dimensional input uses the vector catalog. The default is
`squared_euclidean`. The broader `"pf"` catalog contains:

```text
squared_euclidean
dtw
ddtw
wdtw
wddtw
erp
lcss
msm
twe
```

A three-dimensional collection with one channel is canonicalized to this same
path.

## Multivariate catalogs

Multivariate names use a suffix:

- `_i` computes a channel-independent distance;
- `_d` computes a channel-dependent distance.

The multivariate default contains:

```text
squared_euclidean_i
dtw_i
dtw_d
```

The broader multivariate `"pf"` catalog contains:

```text
squared_euclidean_i  manhattan_i  cid_i    sbd_i
dtw_i                ddtw_i       wdtw_i   wddtw_i
erp_i                lcss_i       msm_i    twe_i
dtw_d                ddtw_d       wdtw_d   wddtw_d
```

When `DistanceSpec` uses a base name without a suffix, `dependent=False` selects
`_i` and `dependent=True` selects `_d`.

## Input compatibility

Distance availability depends on the fitted input layout. A multivariate-only
suffix cannot be used with vector input. The package validates the selected
catalog against the available feature or channel count and against dimension
subsampling.

Use the same input layout at fitting and inference. The fitted estimator records
that layout in `input_layout_`.
