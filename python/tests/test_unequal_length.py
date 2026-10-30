import pickle
import numpy as np
import pytest
from proximity_forests import ProximityForestClassifier, ProximityForestRegressor, ProximityIsolationForest


def make_cases(dtype, channels=2):
    rng = np.random.default_rng(13)
    return [np.ascontiguousarray(rng.normal(size=(channels, length)), dtype=dtype) for length in (8, 11, 9, 14, 10, 12, 7, 13)]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unequal_multivariate_end_to_end(dtype):
    X = make_cases(dtype)
    yc = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    yr = np.arange(len(X), dtype=float)
    clf = ProximityForestClassifier(n_estimators=5, sampling="full", random_state=4, distances=["dtw_i"]).fit(X, yc)
    assert clf.input_layout_ == "unequal_multivariate"
    assert clf.n_channels_in_ == 2 and clf.n_timepoints_in_ is None
    assert clf.predict(X).shape == (8,) and clf.predict_proba(X).shape == (8, 2) and clf.apply(X).shape == (8, 5)
    reg = ProximityForestRegressor(n_estimators=5, n_exemplars=2, sampling="full", random_state=4, distances=["dtw_d"]).fit(X, yr)
    assert reg.predict(X).shape == (8,) and reg.apply(X).shape == (8, 5)
    iso = ProximityIsolationForest(n_estimators=5, n_exemplars=2, sampling="full", random_state=4, distances=["dtw_i"]).fit(X)
    assert iso.anomaly_score(X).shape == (8,) and iso.apply(X).shape == (8, 5)
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(restored.predict(X), clf.predict(X))


def test_unequal_univariate_convenience_form():
    X = [case[0] for case in make_cases(np.float64, channels=1)]
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    model = ProximityForestClassifier(n_estimators=3, sampling="full", distances=["dtw_i"]).fit(X, y)
    assert model.n_channels_in_ == 1
    assert model.predict(X).shape == (8,)


def test_unequal_rejects_mixed_ranks_and_channels():
    with pytest.raises(ValueError, match="either all 1D or all 2D"):
        ProximityForestClassifier().fit([np.ones(4), np.ones((1, 5))], [0, 1])
    with pytest.raises(ValueError, match="same channel count"):
        ProximityForestClassifier().fit([np.ones((1, 4)), np.ones((2, 5))], [0, 1])
