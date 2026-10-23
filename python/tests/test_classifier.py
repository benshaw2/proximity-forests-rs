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
        bootstrap=False,
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


def test_classifier_is_deterministic_across_tree_thread_counts():
    X, y = toy_data()
    first = ProximityForestClassifier(
        n_estimators=7,
        bootstrap=False,
        random_state=19,
        n_jobs=1,
    ).fit(X, y)
    second = ProximityForestClassifier(
        n_estimators=7,
        bootstrap=False,
        random_state=19,
        n_jobs=2,
    ).fit(X, y)

    assert np.array_equal(first.apply(X), second.apply(X))
    assert np.allclose(first.predict_proba(X), second.predict_proba(X))


def test_estimator_parameter_protocol_without_sklearn():
    model = ProximityForestClassifier(n_estimators=3)
    assert model.get_params()["n_estimators"] == 3
    assert model.set_params(max_depth=8) is model
    assert model.max_depth == 8
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(unknown_parameter=True)
