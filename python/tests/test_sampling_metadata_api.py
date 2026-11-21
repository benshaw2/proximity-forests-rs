import pickle

import numpy as np
import pytest

from proximity_forests import (
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)


def data():
    rng = np.random.default_rng(7241)
    X = np.ascontiguousarray(rng.normal(size=(48, 6)))
    yc = (X[:, 0] > 0).astype(np.int64)
    yr = X[:, 0] - X[:, 1]
    return X, yc, yr


def fitted(kind, sampling, sample_size=None, retention="exemplars"):
    X, yc, yr = data()
    kwargs = dict(
        n_estimators=11, sampling=sampling, sample_size=sample_size,
        random_state=91, n_jobs=1, data_retention=retention,
    )
    if kind == "classifier":
        return ProximityForestClassifier(**kwargs).fit(X, yc)
    if kind == "regressor":
        return ProximityForestRegressor(**kwargs).fit(X, yr)
    return ProximityIsolationForest(**kwargs).fit(X)


@pytest.mark.parametrize("kind", ["classifier", "regressor", "isolation"])
@pytest.mark.parametrize(
    ("sampling", "sample_size", "column_sum"),
    [("bootstrap", None, 48), ("subsample", 24, 24), ("full", None, 48)],
)
def test_sampling_matrix_contract(kind, sampling, sample_size, column_sum):
    model = fitted(kind, sampling, sample_size)
    oob = model.get_oob_mask()
    counts = model.get_in_bag_counts()
    assert oob.shape == counts.shape == (48, 11)
    assert oob.dtype == np.bool_
    assert counts.dtype == np.uint32
    np.testing.assert_array_equal(oob, counts == 0)
    np.testing.assert_array_equal(counts.sum(axis=0), column_sum)
    if sampling in {"subsample", "full"}:
        assert np.all(counts <= 1)
    if sampling == "full":
        assert not oob.any()
        assert np.all(counts == 1)


def test_gap_retains_bootstrap_multiplicity():
    model = fitted("classifier", "bootstrap")
    counts = model.get_in_bag_counts()
    assert np.any(counts > 1), "fixture must distinguish counts from membership"
    np.testing.assert_array_equal(model.get_oob_mask(), counts == 0)


@pytest.mark.parametrize("retention", ["exemplars", "copy"])
def test_metadata_survives_pickle_without_requiring_training_copy(retention):
    model = fitted("regressor", "bootstrap", retention=retention)
    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_array_equal(restored.get_oob_mask(), model.get_oob_mask())
    np.testing.assert_array_equal(
        restored.get_in_bag_counts(), model.get_in_bag_counts()
    )


@pytest.mark.parametrize(
    "model",
    [ProximityForestClassifier(n_estimators=3),
     ProximityForestRegressor(n_estimators=3),
     ProximityIsolationForest(n_estimators=3)],
)
def test_sampling_metadata_requires_fit(model):
    with pytest.raises(RuntimeError, match="not fitted"):
        model.get_oob_mask()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.get_in_bag_counts()
