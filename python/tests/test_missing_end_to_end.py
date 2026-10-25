import pickle

import numpy as np
import pytest

from proximity_forests import (
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)


def targets(n):
    yc = np.arange(n) % 2
    yr = np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float) ** 2])
    return yc, yr


def exercise(X, classifier_distance, regressor_distance=None, isolation_distance=None):
    n = len(X)
    yc, yr = targets(n)
    regressor_distance = regressor_distance or classifier_distance
    isolation_distance = isolation_distance or classifier_distance

    clf = ProximityForestClassifier(
        n_estimators=5,
        n_candidates=4,
        bootstrap=False,
        random_state=101,
        distances=[classifier_distance],
    ).fit(X, yc)
    pred = clf.predict(X)
    proba = clf.predict_proba(X)
    leaves = clf.apply(X)
    assert pred.shape == (n,)
    assert proba.shape == (n, 2)
    assert leaves.shape == (n, 5)
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(restored.predict(X), pred)
    np.testing.assert_allclose(restored.predict_proba(X), proba)
    np.testing.assert_array_equal(restored.apply(X), leaves)

    reg = ProximityForestRegressor(
        n_estimators=5,
        n_candidates=4,
        n_exemplars=2,
        bootstrap=False,
        random_state=102,
        distances=[regressor_distance],
    ).fit(X, yr)
    reg_pred = reg.predict(X)
    reg_leaves = reg.apply(X)
    assert reg_pred.shape == (n, 2)
    assert reg_leaves.shape == (n, 5)
    restored = pickle.loads(pickle.dumps(reg))
    np.testing.assert_allclose(restored.predict(X), reg_pred)
    np.testing.assert_array_equal(restored.apply(X), reg_leaves)

    iso = ProximityIsolationForest(
        n_estimators=5,
        n_candidates=4,
        n_exemplars=2,
        bootstrap=False,
        random_state=103,
        distances=[isolation_distance],
    ).fit(X)
    scores = iso.anomaly_score(X)
    iso_leaves = iso.apply(X)
    assert scores.shape == (n,)
    assert iso_leaves.shape == (n, 5)
    restored = pickle.loads(pickle.dumps(iso))
    np.testing.assert_allclose(restored.anomaly_score(X), scores)
    np.testing.assert_array_equal(restored.apply(X), iso_leaves)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vector_missing_end_to_end(dtype):
    rng = np.random.default_rng(1)
    X = np.ascontiguousarray(rng.normal(size=(16, 12)), dtype=dtype)
    X[0, 1] = np.nan
    X[3, 4:7] = np.nan
    exercise(X, "nan_squared_euclidean")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_equal_multivariate_missing_end_to_end(dtype):
    rng = np.random.default_rng(2)
    X = np.ascontiguousarray(rng.normal(size=(16, 3, 12)), dtype=dtype)
    X[0, 0, 1] = np.nan
    X[4, 1, 3:6] = np.nan
    exercise(X, "dtw_arow_i", "dtw_arow_d", "dtw_arow_i")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unequal_multivariate_missing_end_to_end(dtype):
    rng = np.random.default_rng(3)
    X = [
        np.ascontiguousarray(rng.normal(size=(2, length)), dtype=dtype)
        for length in (8, 11, 9, 14, 10, 12, 7, 13, 15, 9, 12, 10, 16, 8, 14, 11)
    ]
    X[0][0, 1] = np.nan
    X[5][1, 2:5] = np.nan
    exercise(X, "dtw_arow_i", "dtw_arow_d", "dtw_arow_i")


def test_all_missing_query_routes_deterministically():
    X = np.array(
        [[0.0, 1.0], [0.1, 1.1], [4.0, 5.0], [4.1, 5.1]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1])
    model = ProximityForestClassifier(
        n_estimators=5,
        n_candidates=4,
        bootstrap=False,
        random_state=23,
        distances=["nan_squared_euclidean"],
    ).fit(X, y)
    query = np.full((2, 2), np.nan)
    first = model.predict(query)
    second = model.predict(query)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(model.predict_proba(query), model.predict_proba(query))
    np.testing.assert_array_equal(model.apply(query), model.apply(query))


def test_query_infinity_rejected_even_for_leaf_only_model():
    X = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
    y = np.array([0, 0, 1, 1])
    model = ProximityForestClassifier(
        n_estimators=1,
        bootstrap=False,
        distances=["nan_squared_euclidean"],
    ).fit(X, y)
    with pytest.raises(ValueError, match="infinity"):
        model.predict(np.array([[np.inf], [0.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="infinity"):
        model.apply(np.array([[np.inf], [0.0]], dtype=np.float64))
