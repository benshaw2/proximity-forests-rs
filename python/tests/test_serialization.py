import pickle

import numpy as np
import pytest

from proximity_forests import (
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)


def data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 6))
    return X, (X[:, 0] > 0).astype(int), np.column_stack([X[:, 0], X[:, 1] - X[:, 2]])


@pytest.mark.parametrize("retention", ["exemplars", "copy"])
def test_classifier_pickle_round_trip(retention):
    X, y, _ = data()
    model = ProximityForestClassifier(
        n_estimators=7, sampling="full", random_state=5, data_retention=retention
    ).fit(X, y)
    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X), rtol=0, atol=0)
    np.testing.assert_array_equal(restored.apply(X), model.apply(X))
    assert restored.data_retention_ == retention


@pytest.mark.parametrize("retention", ["exemplars", "copy"])
def test_regressor_pickle_round_trip(retention):
    X, _, y = data()
    model = ProximityForestRegressor(
        n_estimators=7, sampling="full", random_state=5, data_retention=retention
    ).fit(X, y)
    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=0, atol=0)
    np.testing.assert_array_equal(restored.apply(X), model.apply(X))
    assert restored.data_retention_ == retention


@pytest.mark.parametrize("retention", ["exemplars", "copy"])
def test_isolation_pickle_round_trip(retention):
    X, _, _ = data()
    model = ProximityIsolationForest(
        n_estimators=7, random_state=5, data_retention=retention
    ).fit(X)
    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_allclose(restored.anomaly_score(X), model.anomaly_score(X), rtol=0, atol=0)
    np.testing.assert_array_equal(restored.apply(X), model.apply(X))
    assert restored.data_retention_ == retention


def test_unfitted_estimator_pickle_round_trip():
    model = ProximityForestClassifier(n_estimators=3)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params() == model.get_params()
    with pytest.raises(RuntimeError, match="not fitted"):
        restored.predict(np.zeros((2, 2)))
