import pickle

import numpy as np
import pytest

from proximity_forests import (
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)


def vector_data():
    rng = np.random.default_rng(510)
    X = np.ascontiguousarray(rng.normal(size=(32, 9)), dtype=np.float64)
    yc = (X[:, 0] + X[:, 3] > 0).astype(int)
    yr = X[:, 0] - 0.5 * X[:, 4]
    return X, yc, yr


def equal_multivariate_data():
    rng = np.random.default_rng(511)
    X = np.ascontiguousarray(rng.normal(size=(28, 4, 10)), dtype=np.float64)
    yc = (X[:, 0].mean(axis=1) > 0).astype(int)
    yr = X[:, 2].mean(axis=1)
    return X, yc, yr


def unequal_multivariate_data():
    rng = np.random.default_rng(512)
    lengths = [8, 11, 9, 13, 10, 12, 7, 14] * 3
    X = [np.ascontiguousarray(rng.normal(size=(4, length))) for length in lengths]
    yc = np.asarray([int(case[0].mean() > 0) for case in X])
    yr = np.asarray([case[2].mean() for case in X])
    return X, yc, yr


def payload(model):
    return model.__getstate__()["_native_payload"]


def outputs(model, X):
    leaves = model.apply(X)
    if isinstance(model, ProximityForestClassifier):
        return leaves, model.predict(X), model.predict_proba(X)
    if isinstance(model, ProximityForestRegressor):
        return leaves, model.predict(X)
    return leaves, model.anomaly_score(X), model.score_samples(X), model.predict(X)


def assert_exact_outputs(first, second):
    assert len(first) == len(second)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def round_trip(model, X):
    before_payload = payload(model)
    before_outputs = outputs(model, X)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.max_features == model.max_features
    assert payload(restored) == before_payload
    assert_exact_outputs(outputs(restored, X), before_outputs)
    restored.n_jobs = 4
    assert_exact_outputs(outputs(restored, X), before_outputs)
    assert payload(restored) == before_payload
    return restored


@pytest.mark.parametrize("max_features", [None, "sqrt", "log2", 3, 0.5])
def test_vector_policy_forms_round_trip_exactly(max_features):
    X, yc, _ = vector_data()
    model = ProximityForestClassifier(
        n_estimators=5, n_candidates=3, max_features=max_features,
        distances=["squared_euclidean", "manhattan"], sampling="full",
        random_state=51, n_jobs=1,
    ).fit(X, yc)
    round_trip(model, X)


@pytest.mark.parametrize("layout", ["equal", "unequal"])
@pytest.mark.parametrize("kind", ["classifier", "regressor", "isolation"])
def test_multivariate_channel_selection_round_trip(layout, kind):
    X, yc, yr = equal_multivariate_data() if layout == "equal" else unequal_multivariate_data()
    common = dict(
        n_estimators=5, n_candidates=3, max_features=0.5,
        distances=["dtw_i", "dtw_d"], sampling="full", random_state=52, n_jobs=1,
    )
    if kind == "classifier":
        model = ProximityForestClassifier(**common).fit(X, yc)
    elif kind == "regressor":
        model = ProximityForestRegressor(n_exemplars=2, **common).fit(X, yr)
    else:
        model = ProximityIsolationForest(n_exemplars=2, **common).fit(X)
    round_trip(model, X)


def test_default_none_matches_explicit_all_dimension_behavior():
    X, yc, _ = vector_data()
    common = dict(
        n_estimators=7, n_candidates=3, distances=["squared_euclidean"],
        sampling="full", random_state=53, n_jobs=1,
    )
    implicit = ProximityForestClassifier(**common).fit(X, yc)
    explicit = ProximityForestClassifier(max_features=None, **common).fit(X, yc)
    assert payload(implicit) == payload(explicit)
    assert_exact_outputs(outputs(implicit, X), outputs(explicit, X))


def test_selected_dimensions_affect_fitted_payload_but_remain_deterministic():
    X, yc, _ = vector_data()
    common = dict(
        n_estimators=7, n_candidates=4, distances=["squared_euclidean"],
        sampling="full", random_state=54, n_jobs=1,
    )
    first = ProximityForestClassifier(max_features=2, **common).fit(X, yc)
    second = ProximityForestClassifier(max_features=2, **common).fit(X, yc)
    all_features = ProximityForestClassifier(max_features=None, **common).fit(X, yc)
    assert payload(first) == payload(second)
    assert_exact_outputs(outputs(first, X), outputs(second, X))
    assert payload(first) != payload(all_features)


def test_parallel_fit_is_deterministic_with_feature_selection():
    X, yc, _ = vector_data()
    common = dict(
        n_estimators=9, n_candidates=5, max_features="sqrt",
        distances=["squared_euclidean", "manhattan"], sampling="full", random_state=55,
    )
    sequential = ProximityForestClassifier(n_jobs=1, **common).fit(X, yc)
    parallel = ProximityForestClassifier(n_jobs=4, **common).fit(X, yc)
    assert payload(sequential) == payload(parallel)
    assert_exact_outputs(outputs(sequential, X), outputs(parallel, X))
