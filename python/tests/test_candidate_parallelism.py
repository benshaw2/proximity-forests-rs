import numpy as np

from proximity_forests import (
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)


def candidate_parallel_data():
    rng = np.random.default_rng(20261027)
    X = np.ascontiguousarray(rng.normal(size=(320, 24)), dtype=np.float64)
    y_class = ((X[:, 0] + 0.35 * X[:, 1]) > 0.0).astype(np.int64)
    y_regression = np.column_stack(
        (X[:, 0] - X[:, 1], X[:, 2] + 0.5 * X[:, 3])
    )
    return X, y_class, y_regression


def assert_same_fitted_structure(first, second, X):
    np.testing.assert_array_equal(first.apply(X), second.apply(X))


def test_classifier_candidate_parallelism_matches_sequential_fit():
    X, y, _ = candidate_parallel_data()
    common = dict(
        n_estimators=1,
        n_candidates=16,
        max_depth=7,
        bootstrap=False,
        random_state=1729,
    )
    sequential = ProximityForestClassifier(n_jobs=1, **common).fit(X, y)
    parallel = ProximityForestClassifier(n_jobs=4, **common).fit(X, y)
    assert_same_fitted_structure(sequential, parallel, X)
    np.testing.assert_array_equal(sequential.predict(X), parallel.predict(X))
    np.testing.assert_allclose(
        sequential.predict_proba(X), parallel.predict_proba(X), rtol=0, atol=0
    )


def test_regressor_candidate_parallelism_matches_sequential_fit():
    X, _, y = candidate_parallel_data()
    common = dict(
        n_estimators=1,
        n_candidates=8,
        n_exemplars=2,
        max_depth=7,
        bootstrap=False,
        random_state=1733,
    )
    sequential = ProximityForestRegressor(n_jobs=1, **common).fit(X, y)
    parallel = ProximityForestRegressor(n_jobs=4, **common).fit(X, y)
    assert_same_fitted_structure(sequential, parallel, X)
    np.testing.assert_allclose(
        sequential.predict(X), parallel.predict(X), rtol=0, atol=0
    )


def test_isolation_candidate_parallelism_matches_sequential_fit():
    X, _, _ = candidate_parallel_data()
    common = dict(
        n_estimators=1,
        n_candidates=8,
        n_exemplars=2,
        max_depth=7,
        bootstrap=False,
        random_state=1741,
    )
    sequential = ProximityIsolationForest(n_jobs=1, **common).fit(X)
    parallel = ProximityIsolationForest(n_jobs=4, **common).fit(X)
    assert_same_fitted_structure(sequential, parallel, X)
    np.testing.assert_allclose(
        sequential.anomaly_score(X), parallel.anomaly_score(X), rtol=0, atol=0
    )
