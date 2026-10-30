import numpy as np
import pytest

from proximity_forests import ProximityForestClassifier


def toy_data():
    X = np.array(
        [[0.0, 0.0], [0.1, 0.2], [4.9, 5.0], [5.1, 4.8]],
        dtype=np.float64,
    )
    y = np.array(["left", "left", "right", "right"], dtype=object)
    return X, y


def test_classifier_fit_predict_proba_and_apply():
    X, y = toy_data()
    model = ProximityForestClassifier(
        n_estimators=5,
        n_candidates=3,
        max_depth=4,
        sampling="full",
        random_state=7,
        n_jobs=1,
    ).fit(X, y)

    predicted = model.predict(X)
    probabilities = model.predict_proba(X)
    leaves = model.apply(X)

    assert predicted.shape == (4,)
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert leaves.shape == (4, 5)
    assert model.n_features_in_ == 2
    assert model.n_outputs_ == 1
    assert model.n_classes_ == 2
    assert isinstance(model.classes_, np.ndarray)


def test_classifier_supports_multiclass_multioutput():
    X = np.array(
        [[0.0, 0.0], [0.1, 0.2], [2.0, 2.1], [2.2, 1.9], [4.0, 4.1], [4.2, 3.9]],
        dtype=np.float64,
    )
    y = np.column_stack((
        np.array(["low", "low", "mid", "mid", "high", "high"], dtype=object),
        np.array(["a", "b", "a", "b", "a", "b"], dtype=object),
    ))
    model = ProximityForestClassifier(
        n_estimators=7,
        n_candidates=4,
        sampling="full",
        random_state=17,
        n_jobs=1,
    ).fit(X, y)

    predicted = model.predict(X)
    probabilities = model.predict_proba(X)

    assert predicted.shape == y.shape
    assert model.n_outputs_ == 2
    assert model.target_type_ == "multiclass-multioutput"
    assert isinstance(model.classes_, list)
    assert model.n_classes_ == [3, 2]
    assert len(probabilities) == 2
    assert probabilities[0].shape == (X.shape[0], 3)
    assert probabilities[1].shape == (X.shape[0], 2)
    assert np.allclose(probabilities[0].sum(axis=1), 1.0)
    assert np.allclose(probabilities[1].sum(axis=1), 1.0)


def test_classifier_supports_multilabel_indicator_targets():
    X = np.array(
        [[0.0, 0.0], [0.2, 0.1], [1.0, 1.2], [1.2, 1.0], [2.0, 2.1], [2.2, 1.9]],
        dtype=np.float64,
    )
    y = np.array(
        [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0]],
        dtype=np.uint8,
    )
    model = ProximityForestClassifier(
        n_estimators=7,
        n_candidates=4,
        sampling="full",
        random_state=29,
        n_jobs=1,
    ).fit(X, y)

    predicted = model.predict(X)
    probabilities = model.predict_proba(X)

    assert predicted.shape == y.shape
    assert set(np.unique(predicted)).issubset({0, 1})
    assert model.target_type_ == "multilabel-indicator"
    assert model.n_outputs_ == 3
    assert model.n_classes_ == [2, 2, 2]
    assert len(probabilities) == 3
    assert all(matrix.shape == (X.shape[0], 2) for matrix in probabilities)


def test_classifier_is_deterministic_across_tree_thread_counts():
    X, y = toy_data()
    targets = np.column_stack((y, np.array([0, 1, 0, 1])))
    first = ProximityForestClassifier(
        n_estimators=7,
        sampling="full",
        random_state=19,
        n_jobs=1,
    ).fit(X, targets)
    second = ProximityForestClassifier(
        n_estimators=7,
        sampling="full",
        random_state=19,
        n_jobs=2,
    ).fit(X, targets)

    assert np.array_equal(first.apply(X), second.apply(X))
    assert np.array_equal(first.predict(X), second.predict(X))
    for first_output, second_output in zip(
        first.predict_proba(X), second.predict_proba(X)
    ):
        assert np.allclose(first_output, second_output)


def test_classifier_rejects_constant_output_column():
    X, y = toy_data()
    targets = np.column_stack((y, np.zeros(X.shape[0], dtype=np.uint8)))
    with pytest.raises(ValueError, match="output 1 requires at least two classes"):
        ProximityForestClassifier().fit(X, targets)


def test_estimator_parameter_protocol_without_sklearn():
    model = ProximityForestClassifier(n_estimators=3)
    assert model.get_params()["n_estimators"] == 3
    assert model.set_params(max_depth=8) is model
    assert model.max_depth == 8
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(unknown_parameter=True)

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_classifier_supports_native_float_dtypes(dtype):
    X, y = toy_data()
    X = np.ascontiguousarray(X, dtype=dtype)
    model = ProximityForestClassifier(
        n_estimators=5, sampling="full", random_state=41
    ).fit(X, y)
    assert model.predict(X).shape == (X.shape[0],)
    assert model.predict_proba(X).shape == (X.shape[0], 2)
    assert model.apply(X).dtype == np.uint32

