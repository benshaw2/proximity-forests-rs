import numpy as np
import pytest

from proximity_forests import ProximityIsolationForest


def toy_data():
    rng = np.random.default_rng(8)
    inliers = rng.normal(0.0, 0.25, size=(30, 2))
    outliers = np.array([[6.0, 6.0], [-6.0, -5.5]], dtype=np.float64)
    return np.vstack([inliers, outliers])


def test_isolation_api_shapes_and_score_identity():
    X = toy_data()
    model = ProximityIsolationForest(
        n_estimators=11,
        n_candidates=4,
        n_exemplars=2,
        bootstrap=False,
        random_state=17,
        n_jobs=1,
    ).fit(X)

    raw = model.score_samples(X)
    decision = model.decision_function(X)
    labels = model.predict(X)
    leaves = model.apply(X)

    assert raw.shape == (X.shape[0],)
    assert np.allclose(decision, raw - model.offset_)
    assert set(np.unique(labels)).issubset({-1, 1})
    assert leaves.shape == (X.shape[0], 11)
    assert model.n_features_in_ == 2
    assert model.max_samples_ == X.shape[0]


def test_numeric_contamination_sets_requested_boundary_fraction():
    X = toy_data()
    model = ProximityIsolationForest(
        n_estimators=13,
        contamination=0.125,
        bootstrap=False,
        random_state=21,
    ).fit(X)
    observed = np.mean(model.predict(X) == -1)
    assert abs(observed - 0.125) <= 1.0 / X.shape[0]


def test_auto_contamination_uses_standard_offset():
    X = toy_data()
    model = ProximityIsolationForest(contamination="auto", random_state=1).fit(X)
    assert model.offset_ == -0.5


def test_invalid_contamination_is_rejected():
    X = toy_data()
    with pytest.raises(ValueError, match="contamination"):
        ProximityIsolationForest(contamination=0.75).fit(X)

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_isolation_supports_native_float_dtypes(dtype):
    X = np.ascontiguousarray(toy_data(), dtype=dtype)
    model = ProximityIsolationForest(
        n_estimators=5, bootstrap=False, random_state=47
    ).fit(X)
    assert model.anomaly_score(X).shape == (X.shape[0],)
    assert model.apply(X).dtype == np.uint32

