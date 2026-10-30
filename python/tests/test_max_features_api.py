import pickle

import numpy as np
import pytest

from proximity_forests import (
    ProximityForestClassifier, ProximityForestRegressor, ProximityIsolationForest,
)


def vector_data():
    rng = np.random.default_rng(410)
    X = rng.normal(size=(24, 8))
    return X, (X[:, 0] > 0).astype(int), X[:, 0] - X[:, 2]


def multivariate_data():
    rng = np.random.default_rng(411)
    X = rng.normal(size=(24, 4, 9))
    return X, (X[:, 0].mean(axis=1) > 0).astype(int), X[:, 1].mean(axis=1)


@pytest.mark.parametrize("value", [None, "sqrt", "log2", 3, 0.5])
def test_classifier_max_features_parameter_protocol_and_fit(value):
    X, y, _ = vector_data()
    model = ProximityForestClassifier(
        n_estimators=3, n_candidates=2, max_features=value,
        distances=["squared_euclidean"], sampling="full", random_state=7,
    ).fit(X, y)
    assert model.get_params()["max_features"] == value
    assert model.apply(X).shape == (X.shape[0], 3)


@pytest.mark.parametrize("estimator", [
    ProximityForestRegressor(n_estimators=3, max_features="sqrt", distances=["squared_euclidean"], sampling="full", random_state=8),
    ProximityIsolationForest(n_estimators=3, max_features=0.5, distances=["squared_euclidean"], sampling="full", random_state=9),
])
def test_vector_max_features_fits_other_estimators(estimator):
    X, _, y = vector_data()
    if isinstance(estimator, ProximityForestRegressor):
        estimator.fit(X, y)
    else:
        estimator.fit(X)
    assert estimator.apply(X).shape == (X.shape[0], 3)


@pytest.mark.parametrize("estimator", [
    ProximityForestClassifier(n_estimators=3, max_features=0.5, distances=["dtw_i", "dtw_d"], sampling="full", random_state=10),
    ProximityForestRegressor(n_estimators=3, max_features=2, distances=["dtw_i", "dtw_d"], sampling="full", random_state=11),
    ProximityIsolationForest(n_estimators=3, max_features="sqrt", distances=["dtw_i", "dtw_d"], sampling="full", random_state=12),
])
def test_multivariate_channel_selection_fits(estimator):
    X, yc, yr = multivariate_data()
    if isinstance(estimator, ProximityForestClassifier):
        estimator.fit(X, yc)
    elif isinstance(estimator, ProximityForestRegressor):
        estimator.fit(X, yr)
    else:
        estimator.fit(X)
    assert estimator.apply(X).shape == (X.shape[0], 3)


def test_vector_temporal_distance_rejects_max_features():
    X, y, _ = vector_data()
    with pytest.raises(ValueError, match="does not support selecting vector-layout features"):
        ProximityForestClassifier(
            n_estimators=3, max_features="sqrt", distances=["dtw"],
            sampling="full", random_state=13,
        ).fit(X, y)


@pytest.mark.parametrize("value, error", [
    (True, TypeError), (0, ValueError), (-1, ValueError),
    (0.0, ValueError), (1.1, ValueError), ("bad", ValueError),
])
def test_invalid_max_features_fails_before_native_fit(value, error):
    X, y, _ = vector_data()
    with pytest.raises(error, match="max_features"):
        ProximityForestClassifier(max_features=value).fit(X, y)


def test_unfitted_pickle_and_set_params_preserve_max_features():
    model = ProximityForestClassifier(max_features="sqrt")
    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params()["max_features"] == "sqrt"
    assert restored.set_params(max_features=2) is restored
    assert restored.max_features == 2
