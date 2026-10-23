import numpy as np
import pytest

from proximity_forests import ProximityForestRegressor


def toy_data():
    X = np.array(
        [[0.0, 0.0], [0.1, 0.2], [1.0, 1.0], [1.2, 0.9], [2.0, 2.1]],
        dtype=np.float64,
    )
    y = np.array([0.0, 0.15, 1.0, 1.1, 2.05], dtype=np.float64)
    return X, y


def test_regressor_fit_predict_and_apply():
    X, y = toy_data()
    model = ProximityForestRegressor(
        n_estimators=5,
        n_candidates=3,
        n_exemplars=2,
        max_depth=5,
        bootstrap=False,
        random_state=11,
        n_jobs=1,
    ).fit(X, y)

    predicted = model.predict(X)
    leaves = model.apply(X)

    assert predicted.shape == (X.shape[0],)
    assert np.isfinite(predicted).all()
    assert leaves.shape == (X.shape[0], 5)
    assert model.n_features_in_ == 2
    assert model.n_outputs_ == 1


def test_regressor_is_deterministic_across_tree_thread_counts():
    X, y = toy_data()
    first = ProximityForestRegressor(
        n_estimators=7,
        n_exemplars=2,
        bootstrap=False,
        random_state=23,
        n_jobs=1,
    ).fit(X, y)
    second = ProximityForestRegressor(
        n_estimators=7,
        n_exemplars=2,
        bootstrap=False,
        random_state=23,
        n_jobs=2,
    ).fit(X, y)

    assert np.array_equal(first.apply(X), second.apply(X))
    assert np.allclose(first.predict(X), second.predict(X))


def test_regressor_rejects_multioutput_until_binding_is_enabled():
    X, y = toy_data()
    with pytest.raises(ValueError, match="one-dimensional y"):
        ProximityForestRegressor().fit(X, np.column_stack([y, y]))


def test_regressor_parameter_protocol():
    model = ProximityForestRegressor(n_estimators=3, n_exemplars=2)
    assert model.get_params()["n_exemplars"] == 2
    assert model.set_params(max_depth=6) is model
    assert model.max_depth == 6
