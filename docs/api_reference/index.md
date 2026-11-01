# API reference

The estimator pages are generated from the public Python classes and document
constructor parameters, fitted attributes, methods, return values, and examples.

```{toctree}
:maxdepth: 2

classification
regression
isolation
distance_configuration
```

## Shared conventions

All three estimators expose `get_params` and `set_params`, return `self` from
`fit`, and provide `apply` for retrieving terminal-node assignments. Input data
must use the same layout at fitting and inference.
