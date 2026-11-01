# Observation and dimension sampling

The estimators can sample training observations per tree and dimensions per
candidate splitter. These controls affect model construction independently.

## Observation sampling

Set `sampling` to one of:

- `"bootstrap"`: sample observations with replacement;
- `"subsample"`: sample distinct observations without replacement;
- `"full"`: use every observation exactly once in original order.

For classification and regression, the default is full-size bootstrap sampling:

```python
model = ProximityForestClassifier(
    sampling="bootstrap",
    sample_size=None,
)
```

For isolation, the defaults are:

```python
model = ProximityIsolationForest(
    sampling="subsample",
    sample_size=256,
)
```

`sample_size` accepts:

- `None`, meaning the training-set size;
- a positive integer, meaning an absolute count;
- a float in `(0, 1]`, meaning a proportion of the training set.

With `sampling="full"`, every observation is used once and `sample_size` is
ignored.

```python
model = ProximityForestRegressor(
    sampling="subsample",
    sample_size=0.5,
)
```

## Dimension sampling

`max_features` controls vector-feature or time-series-channel sampling for each
candidate splitter.

Accepted values are:

- `None`, meaning all available dimensions;
- `"sqrt"`, meaning the square root of the available dimension count;
- `"log2"`, meaning the base-2 logarithm of the available dimension count;
- a positive integer, meaning an absolute count;
- a float in `(0, 1]`, meaning a proportion.

```python
model = ProximityForestClassifier(max_features="sqrt")
```

For two-dimensional input, dimensions are columns. For multivariate time-series
input, dimensions are channels. The selected distance must support the resolved
subset behavior.

## Tree size controls

`max_depth` caps the depth of every tree. `None` allows growth until another
stopping rule applies.

`min_samples_split` is the minimum number of sampled observations required to
attempt a split.

Regression and isolation also expose `n_exemplars`, the number of branch
exemplars generated for each candidate splitter.

## Reproducibility

`random_state` controls observation sampling, dimension sampling, candidate
configuration, exemplar selection, and random tie resolution. Use an explicit
integer when reproducible fitted structure is required.
