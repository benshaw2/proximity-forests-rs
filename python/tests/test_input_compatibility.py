import numpy as np
import pytest

from proximity_forests import (
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)


def _data(dtype):
    X2 = np.asarray(
        [[0.0, 0.2, 0.1, 0.3], [0.1, 0.3, 0.2, 0.4],
         [4.0, 4.2, 4.1, 4.3], [4.1, 4.3, 4.2, 4.4]],
        dtype=dtype,
    )
    return X2, X2[:, None, :]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_one_channel_3d_is_aeon_style_univariate(dtype):
    X2, X3 = _data(dtype)
    y = np.asarray([0, 0, 1, 1])
    model = ProximityForestClassifier(
        n_estimators=5, sampling="full", random_state=7, distances="pf"
    ).fit(X3, y)
    assert model.input_layout_ == "vector"
    assert model.n_features_in_ == X3.shape[2]
    np.testing.assert_array_equal(model.predict(X2), model.predict(X3))
    assert model.predict(X3[:1]).shape == (1,)
    assert model.predict_proba(X3[:1]).shape == (1, 2)
    assert model.apply(X3[:1]).shape == (1, 5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_case_inference_for_all_estimators(dtype):
    X2, X3 = _data(dtype)
    yc = np.asarray([0, 0, 1, 1])
    yr = np.asarray([0.0, 0.2, 4.0, 4.2])

    clf = ProximityForestClassifier(
        n_estimators=3, sampling="full", random_state=3
    ).fit(X3, yc)
    reg = ProximityForestRegressor(
        n_estimators=3, n_exemplars=2, sampling="full", random_state=3
    ).fit(X3, yr)
    iso = ProximityIsolationForest(
        n_estimators=3, n_exemplars=2, sampling="full", random_state=3
    ).fit(X3)

    assert clf.predict(X2[:1]).shape == (1,)
    assert reg.predict(X3[:1]).shape == (1,)
    assert np.asarray(iso.anomaly_score(X3[:1])).shape == (1,)


def test_fit_still_requires_two_cases():
    X = np.asarray([[[0.0, 1.0, 2.0]]], dtype=np.float64)
    with pytest.raises(ValueError, match="at least 2 cases"):
        ProximityForestRegressor(sampling="full").fit(X, np.asarray([0.0]))
    with pytest.raises(ValueError, match="at least 2 cases"):
        ProximityIsolationForest(sampling="full").fit(X)


def test_single_unequal_case_is_valid_for_inference_only():
    X = [
        np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64),
        np.asarray([[0.1, 1.1, 2.1, 3.1]], dtype=np.float64),
        np.asarray([[4.0, 5.0, 6.0]], dtype=np.float64),
        np.asarray([[4.1, 5.1, 6.1, 7.1]], dtype=np.float64),
    ]
    y = np.asarray([0, 0, 1, 1])
    model = ProximityForestClassifier(
        n_estimators=3, sampling="full", random_state=5, distances=["dtw_i"]
    ).fit(X, y)
    assert model.predict([X[0]]).shape == (1,)
    assert model.apply([X[0]]).shape == (1, 3)
