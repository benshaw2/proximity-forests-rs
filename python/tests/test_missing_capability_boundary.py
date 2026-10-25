import numpy as np
import pytest

from proximity_forests import DistanceSpec, ProximityForestClassifier


def vector_data(dtype=np.float64):
    X = np.ascontiguousarray(
        [[0.0, 1.0, 2.0], [0.1, 1.1, 2.1], [4.0, 5.0, 6.0], [4.1, 5.1, 6.1]],
        dtype=dtype,
    )
    return X, np.array([0, 0, 1, 1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_missing_distance_runs_on_complete_data(dtype):
    X, y = vector_data(dtype)
    model = ProximityForestClassifier(
        n_estimators=3,
        bootstrap=False,
        random_state=7,
        distances=["nan_squared_euclidean"],
    ).fit(X, y)
    assert model.predict(X).shape == (4,)


def test_complete_data_allows_mixed_regular_and_missing_catalog():
    X, y = vector_data()
    model = ProximityForestClassifier(
        n_estimators=3,
        bootstrap=False,
        random_state=7,
        distances=["squared_euclidean", "nan_squared_euclidean"],
    ).fit(X, y)
    assert model.apply(X).shape == (4, 3)


def test_nan_data_requires_every_distance_to_support_missing_values():
    X, y = vector_data()
    X[0, 1] = np.nan
    with pytest.raises(ValueError, match="every configured distance"):
        ProximityForestClassifier(
            distances=["nan_squared_euclidean", "squared_euclidean"],
            bootstrap=False,
        ).fit(X, y)


def test_nan_aware_sampling_and_fixed_missing_distance_fit():
    X, y = vector_data()
    X[0, 1] = np.nan
    model = ProximityForestClassifier(
        n_estimators=3,
        bootstrap=False,
        random_state=9,
        distances=[DistanceSpec("nan_squared_euclidean", no_overlap="infinite")],
    ).fit(X, y)
    assert model.predict(X).shape == (4,)
    assert model.predict_proba(X).shape == (4, 2)
    assert model.apply(X).shape == (4, 3)


def test_infinity_and_all_missing_collection_are_rejected():
    X, y = vector_data()
    X[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        ProximityForestClassifier(distances=["nan_squared_euclidean"]).fit(X, y)
    with pytest.raises(ValueError, match="observed non-NaN"):
        ProximityForestClassifier(distances=["nan_squared_euclidean"]).fit(
            np.full_like(X, np.nan), y
        )
