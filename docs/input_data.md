# Input data

The estimators accept numeric vector data and time-series collections.

## Vector or tabular input

Use a two-dimensional NumPy array shaped:

```text
(n_cases, n_features)
```

## Equal-length time series

Use a three-dimensional NumPy array shaped:

```text
(n_cases, n_channels, n_timepoints)
```

A one-channel time-series collection therefore has shape
`(n_cases, 1, n_timepoints)`.

## Unequal-length time series

Use a sequence containing one array per case. Each case has shape:

```text
(n_channels, n_timepoints_for_this_case)
```

All cases must use a consistent channel count.

## Numeric storage

`float32` and `float64` inputs are supported. Contiguous arrays avoid input
normalization copies. Individual distance families may impose additional
requirements concerning missing values, equal lengths, or input layout.
