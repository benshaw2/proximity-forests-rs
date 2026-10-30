import pickle

import numpy as np
import pytest

from proximity_forests import (
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)


def data():
    rng = np.random.default_rng(612)
    X = np.ascontiguousarray(rng.normal(size=(96, 12)), dtype=np.float64)
    yc = (X[:, 0] + 0.25 * X[:, 1] > 0.0).astype(np.int64)
    yr = np.column_stack((X[:, 0] - X[:, 1], X[:, 2] + X[:, 3]))
    return X, yc, yr


def fitted_cases():
    X, yc, yr = data()
    return [
        (
            ProximityForestClassifier(
                n_estimators=67,
                n_candidates=3,
                sampling="full",
                random_state=612,
                n_jobs=1,
            ).fit(X, yc),
            X,
            lambda model, query: (
                np.asarray(model.predict(query)),
                np.asarray(model.predict_proba(query)),
                np.asarray(model.apply(query)),
            ),
        ),
        (
            ProximityForestRegressor(
                n_estimators=67,
                n_candidates=3,
                n_exemplars=2,
                sampling="full",
                random_state=612,
                n_jobs=1,
            ).fit(X, yr),
            X,
            lambda model, query: (
                np.asarray(model.predict(query)),
                np.asarray(model.apply(query)),
            ),
        ),
        (
            ProximityIsolationForest(
                n_estimators=67,
                n_candidates=3,
                n_exemplars=2,
                sampling="full",
                random_state=612,
                n_jobs=1,
            ).fit(X),
            X,
            lambda model, query: (
                np.asarray(model.anomaly_score(query)),
                np.asarray(model.score_samples(query)),
                np.asarray(model.apply(query)),
            ),
        ),
    ]


def assert_outputs_equal(expected, actual):
    assert len(actual) == len(expected)
    for left, right in zip(expected, actual):
        if left.dtype.kind in "iuOb" and right.dtype.kind in "iuOb":
            np.testing.assert_array_equal(right, left)
        else:
            np.testing.assert_allclose(right, left, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("case", range(3))
def test_mutating_n_jobs_changes_native_budget_without_changing_payload(case):
    model, X, outputs = fitted_cases()[case]
    baseline_payload = bytes(model._native.to_bytes())
    baseline_outputs = outputs(model, X)

    assert model.set_params(n_jobs=4) is model
    assert model.n_jobs == 4
    assert model._native.n_jobs_ == 4
    assert bytes(model._native.to_bytes()) == baseline_payload
    assert_outputs_equal(baseline_outputs, outputs(model, X))

    model.set_params(n_jobs=None)
    assert model.n_jobs is None
    assert model._native.n_jobs_ == 0
    assert bytes(model._native.to_bytes()) == baseline_payload
    assert_outputs_equal(baseline_outputs, outputs(model, X))

    model.set_params(n_jobs=1)
    assert model._native.n_jobs_ == 1
    assert bytes(model._native.to_bytes()) == baseline_payload


@pytest.mark.parametrize("case", range(3))
@pytest.mark.parametrize("n_jobs, native_jobs", [(1, 1), (4, 4), (None, 0), (-1, 0)])
def test_pickle_preserves_worker_request_and_fitted_payload(case, n_jobs, native_jobs):
    model, X, outputs = fitted_cases()[case]
    model.set_params(n_jobs=n_jobs)
    payload_before = bytes(model._native.to_bytes())
    expected = outputs(model, X)

    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))

    assert restored.n_jobs == n_jobs
    assert restored._native.n_jobs_ == native_jobs
    assert bytes(restored._native.to_bytes()) == payload_before
    assert_outputs_equal(expected, outputs(restored, X))


@pytest.mark.parametrize("case", range(3))
def test_worker_request_remains_mutable_after_pickle(case):
    model, X, outputs = fitted_cases()[case]
    expected = outputs(model, X)
    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))

    restored.set_params(n_jobs=4)
    assert restored._native.n_jobs_ == 4
    assert_outputs_equal(expected, outputs(restored, X))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ProximityForestClassifier(n_estimators=3, n_jobs=None),
        lambda: ProximityForestRegressor(n_estimators=3, n_jobs=-1),
        lambda: ProximityIsolationForest(n_estimators=3, n_jobs=4),
    ],
)
def test_unfitted_pickle_preserves_public_worker_parameter(factory):
    model = factory()
    restored = pickle.loads(pickle.dumps(model))
    assert restored.n_jobs == model.n_jobs
    assert restored.get_params()["n_jobs"] == model.n_jobs
    assert restored._native is None
