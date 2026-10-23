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


def test_regressor_supports_multi_target_predictions():
    X, y = toy_data()
    targets = np.column_stack((y, 2.0 * y + 1.0))
    model = ProximityForestRegressor(
        n_estimators=5,
        n_candidates=3,
        n_exemplars=2,
        max_depth=5,
        bootstrap=False,
        random_state=13,
        n_jobs=1,
    ).fit(X, targets)

    predicted = model.predict(X)

    assert predicted.shape == targets.shape
    assert np.isfinite(predicted).all()
    assert model.n_outputs_ == 2


def test_single_column_2d_target_preserves_2d_prediction_shape():
    X, y = toy_data()
    target_matrix = y[:, None]
    model = ProximityForestRegressor(
        n_estimators=3,
        n_exemplars=2,
        bootstrap=False,
        random_state=17,
        n_jobs=1,
    ).fit(X, target_matrix)

    assert model.predict(X).shape == (X.shape[0], 1)
    assert model.n_outputs_ == 1


def test_regressor_is_deterministic_across_tree_thread_counts():
    X, y = toy_data()
    targets = np.column_stack((y, y**2))
    first = ProximityForestRegressor(
        n_estimators=7,
        n_exemplars=2,
        bootstrap=False,
        random_state=23,
        n_jobs=1,
    ).fit(X, targets)
    second = ProximityForestRegressor(
        n_estimators=7,
        n_exemplars=2,
        bootstrap=False,
        random_state=23,
        n_jobs=2,
    ).fit(X, targets)

    assert np.array_equal(first.apply(X), second.apply(X))
    assert np.allclose(first.predict(X), second.predict(X))


def test_regressor_rejects_invalid_target_rank():
    X, y = toy_data()
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        ProximityForestRegressor().fit(X, y.reshape(1, 1, -1))


def test_regressor_rejects_nonfinite_multi_target_values():
    X, y = toy_data()
    targets = np.column_stack((y, y))
    targets[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ProximityForestRegressor().fit(X, targets)


def test_regressor_parameter_protocol():
    model = ProximityForestRegressor(n_estimators=3, n_exemplars=2)
    assert model.get_params()["n_exemplars"] == 2
    assert model.set_params(max_depth=6) is model
    assert model.max_depth == 6
