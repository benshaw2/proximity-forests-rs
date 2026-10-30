import pickle

import numpy as np
import pytest

from proximity_forests import (
    DistanceSpec,
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)


def vector_data(dtype=np.float64):
    rng = np.random.default_rng(71)
    X = np.ascontiguousarray(rng.normal(size=(24, 10)), dtype=dtype)
    X[0, 1] = np.nan
    X[5, 3:6] = np.nan
    yc = np.arange(X.shape[0]) % 2
    yr = np.column_stack([np.arange(X.shape[0]), np.arange(X.shape[0]) ** 2]).astype(float)
    return X, yc, yr


@pytest.mark.parametrize("retention", ["exemplars", "copy"])
@pytest.mark.parametrize("policy", ["infinite", "zero"])
def test_nan_euclidean_classifier_round_trip(retention, policy):
    X, y, _ = vector_data()
    model = ProximityForestClassifier(
        n_estimators=7,
        n_candidates=4,
        sampling="full",
        random_state=73,
        data_retention=retention,
        distances=[DistanceSpec("nan_squared_euclidean", no_overlap=policy)],
    ).fit(X, y)
    before = (model.predict(X), model.predict_proba(X), model.apply(X))
    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_array_equal(restored.predict(X), before[0])
    np.testing.assert_allclose(restored.predict_proba(X), before[1], rtol=0, atol=0)
    np.testing.assert_array_equal(restored.apply(X), before[2])
    assert restored.data_retention_ == retention
    assert restored.has_embedded_training_data_ is (retention == "copy")


@pytest.mark.parametrize("retention", ["exemplars", "copy"])
def test_nan_euclidean_regression_and_isolation_round_trip(retention):
    X, _, y = vector_data(np.float32)
    reg = ProximityForestRegressor(
        n_estimators=7,
        n_candidates=4,
        n_exemplars=2,
        sampling="full",
        random_state=79,
        data_retention=retention,
        distances=["nan_squared_euclidean"],
    ).fit(X, y)
    restored_reg = pickle.loads(pickle.dumps(reg))
    np.testing.assert_allclose(restored_reg.predict(X), reg.predict(X), rtol=0, atol=0)
    np.testing.assert_array_equal(restored_reg.apply(X), reg.apply(X))

    iso = ProximityIsolationForest(
        n_estimators=7,
        n_candidates=4,
        n_exemplars=2,
        sampling="full",
        random_state=83,
        data_retention=retention,
        distances=["nan_squared_euclidean"],
    ).fit(X)
    restored_iso = pickle.loads(pickle.dumps(iso))
    np.testing.assert_allclose(
        restored_iso.anomaly_score(X), iso.anomaly_score(X), rtol=0, atol=0
    )
    np.testing.assert_array_equal(restored_iso.apply(X), iso.apply(X))


@pytest.mark.parametrize("distance", ["dtw_arow_i", "dtw_arow_d"])
def test_dtw_arow_multivariate_round_trip(distance):
    rng = np.random.default_rng(89)
    X = np.ascontiguousarray(rng.normal(size=(20, 3, 14)), dtype=np.float64)
    X[0, 0, 2] = np.nan
    X[7, 2, 5:8] = np.nan
    y = np.arange(X.shape[0]) % 2
    model = ProximityForestClassifier(
        n_estimators=5,
        n_candidates=4,
        sampling="full",
        random_state=97,
        distances=[DistanceSpec(distance, window=4)],
        data_retention="copy",
    ).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X), rtol=0, atol=0)
    np.testing.assert_array_equal(restored.apply(X), model.apply(X))
    assert restored.input_layout_ == "multivariate"
    assert restored.n_channels_in_ == 3
    assert restored.n_timepoints_in_ == 14


def test_unequal_dtw_arow_round_trip_with_missing_values():
    rng = np.random.default_rng(101)
    X = [
        np.ascontiguousarray(rng.normal(size=(2, length)), dtype=np.float32)
        for length in (8, 11, 9, 14, 10, 12, 7, 13, 15, 9, 12, 10)
    ]
    X[0][0, 1] = np.nan
    X[4][1, 3:5] = np.nan
    y = np.arange(len(X)) % 2
    model = ProximityForestClassifier(
        n_estimators=5,
        n_candidates=4,
        sampling="full",
        random_state=103,
        distances=["dtw_arow_i"],
        data_retention="copy",
    ).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_array_equal(restored.apply(X), model.apply(X))
    assert restored.input_layout_ == "unequal_multivariate"
    assert restored.n_timepoints_in_ is None


def test_complete_data_missing_distance_round_trip():
    X, y, _ = vector_data()
    X = np.nan_to_num(X)
    model = ProximityForestClassifier(
        n_estimators=5,
        sampling="full",
        random_state=107,
        distances=["squared_euclidean", "nan_squared_euclidean"],
    ).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
