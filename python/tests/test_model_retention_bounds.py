import pickle
import numpy as np
import pytest

from proximity_forests import (
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)


def _data(dtype=np.float64):
    rng = np.random.default_rng(20261031)
    X = np.ascontiguousarray(rng.normal(size=(96, 24)), dtype=dtype)
    yc = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(np.int64)
    yr = X[:, 0] - X[:, 1]
    return X, yc, yr


def _payload(model):
    return bytes(model._native.to_bytes())


def _factory(kind, tree_count, retention):
    common = dict(
        n_estimators=tree_count,
        n_candidates=5,
        max_depth=9,
        sampling="full",
        data_retention=retention,
        random_state=731,
        n_jobs=1,
    )
    X, yc, yr = _data()
    if kind == "classifier":
        return X, ProximityForestClassifier(**common).fit(X, yc)
    if kind == "regressor":
        return X, ProximityForestRegressor(n_exemplars=3, **common).fit(X, yr)
    return X, ProximityIsolationForest(n_exemplars=3, **common).fit(X)


@pytest.mark.parametrize("kind", ["classifier", "regressor", "isolation"])
def test_exemplar_round_trip_remains_exact(kind):
    X, model = _factory(kind, 11, "exemplars")
    leaves = model.apply(X)
    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_array_equal(restored.apply(X), leaves)
    assert restored.data_retention_ == "exemplars"
    assert restored.has_embedded_training_data_ is False


@pytest.mark.parametrize("kind", ["classifier", "regressor", "isolation"])
def test_copy_payload_never_adds_more_than_one_dataset(kind):
    X, exemplar_model = _factory(kind, 17, "exemplars")
    _, copy_model = _factory(kind, 17, "copy")
    dataset_bytes = X.nbytes
    added_bytes = len(_payload(copy_model)) - len(_payload(exemplar_model))
    # Copy must add at most one dataset, never one dataset per tree. Zero is a
    # valid and desirable result when the forest-wide exemplar pool already
    # spans every training row: A7 then uses that one full pool for both routing
    # and retained-data semantics. The allowance covers framing and offsets.
    assert 0 <= added_bytes <= dataset_bytes + 8 * (X.shape[0] + 64)


@pytest.mark.parametrize("kind", ["classifier", "regressor", "isolation"])
def test_exemplar_payload_growth_is_tree_structure_not_repeated_dataset(kind):
    _, one = _factory(kind, 1, "exemplars")
    _, many = _factory(kind, 17, "exemplars")
    X, _, _ = _data()
    growth = len(_payload(many)) - len(_payload(one))
    # Additional trees contribute nodes and branches, but may not contribute a
    # repeated full exemplar dataset per tree.
    assert growth < 16 * X.nbytes


@pytest.mark.parametrize("kind", ["classifier", "regressor", "isolation"])
def test_copy_payload_contains_at_most_one_raw_dataset_plus_model_structure(kind):
    X, one = _factory(kind, 1, "copy")
    _, exemplar_one = _factory(kind, 1, "exemplars")
    non_data_structure = len(_payload(exemplar_one))
    allowance = 8 * (X.shape[0] + 64)
    assert len(_payload(one)) <= non_data_structure + X.nbytes + allowance
