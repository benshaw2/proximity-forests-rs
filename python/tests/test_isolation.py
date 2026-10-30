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
        sampling="full",
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
        sampling="full",
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
        n_estimators=5, sampling="full", random_state=47
    ).fit(X)
    assert model.anomaly_score(X).shape == (X.shape[0],)
    assert model.apply(X).dtype == np.uint32

def test_default_isolation_sampling_is_fixed_256_subsample():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(300, 3))
    model = ProximityIsolationForest(n_estimators=3, random_state=4).fit(X)
    assert model.sampling == "subsample"
    assert model.sample_size == 256
    assert model.max_samples_ == 256


def test_default_isolation_sample_size_clamps_to_small_dataset():
    X = toy_data()
    model = ProximityIsolationForest(n_estimators=3, random_state=5).fit(X)
    assert model.max_samples_ == X.shape[0]


def test_proportional_isolation_sample_size_uses_ceiling():
    rng = np.random.default_rng(32)
    X = rng.normal(size=(21, 3))
    model = ProximityIsolationForest(
        n_estimators=3, sampling="subsample", sample_size=0.25, random_state=6
    ).fit(X)
    assert model.max_samples_ == 6


def test_explicit_full_isolation_retains_full_normalization_size():
    X = toy_data()
    model = ProximityIsolationForest(
        n_estimators=3, sampling="full", sample_size=None, random_state=7
    ).fit(X)
    assert model.max_samples_ == X.shape[0]

