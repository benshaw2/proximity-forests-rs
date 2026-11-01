import pickle

import numpy as np

from proximity_forests import (
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)


def test_documented_classification_workflow():
    X = np.array(
        [[0.0, 0.1], [0.2, 0.0], [0.9, 1.0], [1.1, 0.8]],
        dtype=np.float64,
    )
    y = np.array(["low", "low", "high", "high"])
    model = ProximityForestClassifier(
        n_estimators=5,
        distances=["squared_euclidean"],
        random_state=7,
    ).fit(X, y)
    assert model.predict(X[:2]).shape == (2,)
    assert model.predict_proba(X[:2]).shape == (2, 2)
    assert model.apply(X[:2]).shape == (2, 5)


def test_documented_multioutput_classification_workflow():
    X = np.array([[0.0], [0.2], [0.9], [1.1]], dtype=np.float64)
    y = np.array(
        [["low", "left"], ["low", "right"],
         ["high", "left"], ["high", "right"]],
    )
    model = ProximityForestClassifier(
        n_estimators=5,
        distances=["squared_euclidean"],
        random_state=7,
    ).fit(X, y)
    assert model.predict(X).shape == (4, 2)
    probabilities = model.predict_proba(X)
    assert len(probabilities) == 2
    assert all(values.shape == (4, 2) for values in probabilities)


def test_documented_regression_workflow():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.1, 1.0, 2.1, 3.0], dtype=np.float64)
    model = ProximityForestRegressor(
        n_estimators=5,
        n_exemplars=2,
        distances=["squared_euclidean"],
        random_state=7,
    ).fit(X, y)
    assert model.predict(X[:2]).shape == (2,)
    assert model.apply(X[:2]).shape == (2, 5)


def test_documented_multitarget_regression_workflow():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.column_stack([X[:, 0], X[:, 0] ** 2])
    model = ProximityForestRegressor(
        n_estimators=5,
        distances=["squared_euclidean"],
        random_state=7,
    ).fit(X, y)
    assert model.predict(X[:2]).shape == (2, 2)


def test_documented_isolation_workflow():
    X = np.array([[0.0], [0.1], [0.2], [0.15], [5.0]], dtype=np.float64)
    model = ProximityIsolationForest(
        n_estimators=5,
        sample_size=5,
        contamination=0.2,
        random_state=7,
    ).fit(X)
    assert model.score_samples(X).shape == (5,)
    assert model.decision_function(X).shape == (5,)
    assert model.predict(X).shape == (5,)
    assert model.anomaly_score(X).shape == (5,)


def test_documented_equal_length_multivariate_workflow():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 3, 8)).astype(np.float32)
    y = np.repeat([0, 1], 6)
    model = ProximityForestClassifier(
        n_estimators=3,
        distances=["dtw_i", "dtw_d"],
        max_features="sqrt",
        random_state=7,
    ).fit(X, y)
    assert model.predict(X[:2]).shape == (2,)


def test_documented_unequal_length_workflow():
    X = [
        np.array([[0.0, 0.2, 0.4]], dtype=np.float64),
        np.array([[0.1, 0.2, 0.3, 0.5]], dtype=np.float64),
        np.array([[1.0, 0.8, 0.6]], dtype=np.float64),
        np.array([[1.1, 0.9, 0.7, 0.5, 0.3]], dtype=np.float64),
    ]
    y = np.array([0, 0, 1, 1])
    model = ProximityForestClassifier(
        n_estimators=3,
        distances=["dtw_i"],
        random_state=7,
    ).fit(X, y)
    assert model.predict(X).shape == (4,)


def test_documented_reference_retention_workflow(tmp_path):
    path = tmp_path / "training.npy"
    original = np.array(
        [[0.0, 0.1], [0.2, 0.0], [0.9, 1.0], [1.1, 0.8]],
        dtype=np.float64,
    )
    np.save(path, original)
    X = np.load(path, mmap_mode="r")
    y = np.array([0, 0, 1, 1])
    model = ProximityForestClassifier(
        n_estimators=3,
        data_retention="reference",
        random_state=7,
    ).fit(X, y)
    assert model.external_training_data_["shape"] == X.shape
    np.testing.assert_array_equal(model.validate_training_data_reference(), X)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.predict(original[:2]).shape == (2,)
