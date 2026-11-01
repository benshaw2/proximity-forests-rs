# Concurrency contract

## Public worker semantics

`n_jobs=1` is sequential. `None` and `-1` are translated by the Python wrappers
to native request `0`, which resolves once per operation to the logical CPU count
available to the process. Positive values are explicit worker requests.

The worker request controls both fitting and inference. It is runtime scheduling
configuration, not mathematical model state.

## Bounded execution

Each native operation creates one `ExecutionContext`. A parallel context owns one
fixed-size Rayon pool; nested tree and candidate phases install work into that same
pool. Sequential contexts allocate no pool. This prevents independent nested pools
and provides one fixed worker budget per operation.

## Fitting hierarchy

Independent trees may run concurrently. Within a tree, candidate splits may run
concurrently only when the calibrated distance-aware work estimate reaches the
internal threshold. Candidate results are collected by stable candidate index, so
first-tie behavior and lowest-index error reporting are preserved.

## Inference hierarchy

Batch inference is represented as a query-by-tree matrix. Above the internal
activation boundary, the matrix is divided into stable rectangular tiles. Tiles
may run concurrently, including the single-query, many-tree case. Below the
boundary, the worker request is reduced to one before context construction, so
the operation remains genuinely pool-free.

Classification probabilities, regression outputs, and isolation path lengths are
restored to query-major and tree-major order before aggregation. Floating-point
addition therefore follows ascending original tree order regardless of worker
scheduling. `apply` and depth outputs use the same stable matrix coordinates.

## Determinism

For fixed estimator parameters, data, and random seed, changing `n_jobs` must not
change the fitted tree structure, predictions, probabilities, anomaly scores, or
leaf assignments. Indexed errors are selected in ascending logical index order.
The release-mode calibration benchmarks additionally use exact or zero-tolerance
comparisons across worker counts.

## Serialization

Fitted forest payloads serialize model structure, retained observations, and
training metadata. They do not serialize an `ExecutionContext`, Rayon pool, worker
thread, or other runtime-only scheduling state. Python pickle separately retains
the estimator parameter `n_jobs` and reconstructs a configured native shell before
loading the stable fitted-model payload.

## Threshold status

Candidate and inference thresholds are internal implementation details derived
from calibration on development hardware. They may change without an API change.
Users should choose `n_jobs`; the library chooses which eligible scheduling level
can profitably consume that budget.

