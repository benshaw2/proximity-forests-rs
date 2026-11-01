import pickle
import numpy as np
import pytest
from proximity_forests import ProximityForestClassifier, ProximityForestRegressor, ProximityIsolationForest


def mapped(tmp_path, dtype=np.float64):
    rng = np.random.default_rng(711)
    path = tmp_path / "training.npy"
    np.save(path, np.asarray(rng.normal(size=(48, 7)), dtype=dtype))
    return path, np.load(path, mmap_mode="r")


def models(X):
    y = (np.asarray(X[:, 0]) > 0).astype(int)
    r = np.asarray(X[:, 0] - X[:, 1])
    common = dict(n_estimators=5, sampling="full", random_state=19, data_retention="reference")
    return [
        ProximityForestClassifier(**common).fit(X, y),
        ProximityForestRegressor(**common).fit(X, r),
        ProximityIsolationForest(**common).fit(X),
    ]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vector_npy_reference_round_trip(tmp_path, dtype):
    _, X = mapped(tmp_path, dtype)
    for model in models(X):
        before = model.apply(X)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.apply(X), before)
        assert restored.data_retention_ == "reference"
        assert restored.has_training_data_ is True
        assert restored.has_embedded_training_data_ is False


def test_reference_rejects_ordinary_loaded_array(tmp_path):
    path, _ = mapped(tmp_path)
    ordinary = np.load(path)
    y = (ordinary[:, 0] > 0).astype(int)
    with pytest.raises(ValueError, match="memmap"):
        ProximityForestClassifier(data_retention="reference").fit(ordinary, y)


def test_reference_rejects_sliced_memmap(tmp_path):
    _, X = mapped(tmp_path)
    y = (np.asarray(X[:-1, 0]) > 0).astype(int)
    with pytest.raises(ValueError, match="complete"):
        ProximityForestClassifier(data_retention="reference").fit(X[:-1], y)



@pytest.mark.parametrize("estimator", [
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
])
def test_reference_descriptor_is_inspectable_and_validatable(tmp_path, estimator):
    path, X = mapped(tmp_path)
    common = dict(n_estimators=3, sampling="full", random_state=29, data_retention="reference")
    if estimator is ProximityForestClassifier:
        model = estimator(**common).fit(X, (np.asarray(X[:, 0]) > 0).astype(int))
    elif estimator is ProximityForestRegressor:
        model = estimator(**common).fit(X, np.asarray(X[:, 0]))
    else:
        model = estimator(**common).fit(X)
    descriptor = model.external_training_data_
    assert descriptor["path"] == str(path.resolve())
    assert descriptor["shape"] == X.shape
    assert descriptor["dtype"] == X.dtype.name
    assert descriptor["fingerprint"] is None
    reopened = model.validate_training_data_reference()
    assert isinstance(reopened, np.memmap)
    np.testing.assert_array_equal(reopened, X)


def test_missing_reference_does_not_break_inference_but_fails_validation(tmp_path):
    path, X = mapped(tmp_path)
    y = (np.asarray(X[:, 0]) > 0).astype(int)
    model = ProximityForestClassifier(
        n_estimators=3, sampling="full", random_state=31, data_retention="reference"
    ).fit(X, y)
    query = np.asarray(X[:4]).copy()
    expected = model.predict(query)
    del X
    path.unlink()
    np.testing.assert_array_equal(model.predict(query), expected)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        model.validate_training_data_reference()


def test_changed_reference_size_is_rejected(tmp_path):
    path, X = mapped(tmp_path)
    model = ProximityIsolationForest(
        n_estimators=3, sampling="full", random_state=37, data_retention="reference"
    ).fit(X)
    del X
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="file size changed"):
        model.validate_training_data_reference()


def test_replaced_same_size_reference_header_is_rejected(tmp_path):
    path, X = mapped(tmp_path, np.float64)
    model = ProximityIsolationForest(
        n_estimators=3, sampling="full", random_state=41, data_retention="reference"
    ).fit(X)
    original_size = path.stat().st_size
    del X
    # Same payload bytes, but a different logical shape with equal element count.
    np.save(path, np.zeros((24, 14), dtype=np.float64))
    assert path.stat().st_size == original_size
    with pytest.raises(ValueError, match="shape changed"):
        model.validate_training_data_reference()


def test_optional_fingerprint_requirement_is_explicit(tmp_path):
    _, X = mapped(tmp_path)
    model = ProximityIsolationForest(
        n_estimators=3, sampling="full", random_state=43, data_retention="reference"
    ).fit(X)
    with pytest.raises(ValueError, match="no content fingerprint"):
        model.validate_training_data_reference(require_fingerprint=True)


def test_non_reference_model_has_no_external_descriptor():
    X = np.arange(24, dtype=np.float64).reshape(8, 3)
    model = ProximityIsolationForest(n_estimators=3, sampling="full").fit(X)
    assert model.external_training_data_ is None
    with pytest.raises(ValueError, match="does not retain external"):
        model.validate_training_data_reference()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels", [1, 3])
def test_dense_3d_npy_reference_round_trip(tmp_path, dtype, channels):
    rng = np.random.default_rng(811 + channels)
    path = tmp_path / f"series_{channels}.npy"
    np.save(path, np.asarray(rng.normal(size=(32, channels, 9)), dtype=dtype))
    X = np.load(path, mmap_mode="r")
    y = (np.asarray(X[:, 0, 0]) > 0).astype(int)
    r = np.asarray(X[:, 0, 0] - X[:, 0, 1])
    distance = "squared_euclidean" if channels == 1 else "dtw_i"
    common = dict(
        n_estimators=3,
        sampling="full",
        random_state=47,
        data_retention="reference",
        distances=[distance],
    )
    models = [
        ProximityForestClassifier(**common).fit(X, y),
        ProximityForestRegressor(**common).fit(X, r),
        ProximityIsolationForest(**common).fit(X),
    ]
    for model in models:
        before = model.apply(X)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.apply(X), before)
        assert restored.external_training_data_["shape"] == X.shape
        reopened = restored.validate_training_data_reference()
        assert reopened.shape == X.shape
        np.testing.assert_array_equal(reopened, X)


def test_reference_still_rejects_unequal_length_collection():
    X = [np.zeros((2, 5)), np.zeros((2, 7))]
    with pytest.raises(ValueError, match="memmap"):
        ProximityIsolationForest(data_retention="reference").fit(X)
