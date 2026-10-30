import numpy as np
import pytest

from proximity_forests import (
    ProximityForestClassifier, ProximityForestRegressor, ProximityIsolationForest,
)


def data():
    X = np.arange(96, dtype=float).reshape(24, 4)
    y = np.asarray([0, 1] * 12)
    return X, y


@pytest.mark.parametrize("estimator", [
    ProximityForestClassifier(n_estimators=3, sampling="subsample", sample_size=7, random_state=3),
    ProximityForestRegressor(n_estimators=3, sampling="subsample", sample_size=0.5, random_state=3),
    ProximityIsolationForest(n_estimators=3, sampling="subsample", sample_size=7, random_state=3),
])
def test_sampling_modes_fit(estimator):
    X, y = data()
    if isinstance(estimator, ProximityForestClassifier): estimator.fit(X, y)
    elif isinstance(estimator, ProximityForestRegressor): estimator.fit(X, y.astype(float))
    else: estimator.fit(X)


def test_invalid_sampling_contracts_fail_before_native_fit():
    X, y = data()
    with pytest.raises(ValueError): ProximityForestClassifier(sampling="bad").fit(X, y)
    with pytest.raises(TypeError): ProximityForestClassifier(sample_size=True).fit(X, y)
    with pytest.raises(ValueError): ProximityForestClassifier(sample_size=0).fit(X, y)
    with pytest.raises(ValueError): ProximityForestClassifier(sample_size=1.1).fit(X, y)


def test_get_params_exposes_new_api_only():
    params = ProximityForestClassifier().get_params()
    assert params["sampling"] == "bootstrap"
    assert params["sample_size"] is None
    assert "bootstrap" not in params


def test_full_sampling_ignores_sample_size():
    X, y = data()
    model = ProximityForestClassifier(
        n_estimators=3, sampling="full", sample_size=3, random_state=8
    ).fit(X, y)
    assert model.predict(X).shape == (X.shape[0],)
