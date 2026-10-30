import numpy as np
import pytest

from proximity_forests import DistanceSpec, ProximityForestClassifier
from proximity_forests._distances import normalize_distances, normalize_vector_distances


def toy_data():
    X = np.array(
        [[0.0, 0.0], [0.2, 0.1], [3.8, 4.0], [4.1, 3.9]],
        dtype=np.float64,
    )
    y = np.array(["left", "left", "right", "right"], dtype=object)
    return X, y


def test_fixed_parameter_distance_specs_fit():
    X, y = toy_data()
    model = ProximityForestClassifier(
        n_estimators=3,
        n_candidates=2,
        distances=[
            "squared_euclidean",
            DistanceSpec("dtw", window=1),
            {"name": "erp", "window": 1, "gap": 0.0},
        ],
        sampling="full",
        random_state=7,
    ).fit(X, y)
    assert model.predict(X).shape == (X.shape[0],)
    assert model.apply(X).dtype == np.uint32


def test_plain_family_and_distance_spec_remain_distinct():
    assert normalize_vector_distances(["dtw"]) == ("dtw",)
    assert normalize_vector_distances([DistanceSpec("dtw")]) == (
        "dtw?fixed=true",
    )
    assert normalize_vector_distances([DistanceSpec("dtw", window=20)]) == (
        "dtw?fixed=true&window=20",
    )


def test_cv_multivariate_names_are_available_to_multivariate_normalizer():
    assert normalize_distances(
        ["dtw_cv_i", "dtw_cv_d", "ddtw_cv_i", "ddtw_cv_d"],
        layout="multivariate",
    ) == ("dtw_cv_i", "dtw_cv_d", "ddtw_cv_i", "ddtw_cv_d")


def test_fixed_cv_spec_is_rejected():
    with pytest.raises(ValueError, match="fixed"):
        DistanceSpec("dtw_cv").native_token(layout="vector")


def test_unknown_distance_parameter_is_rejected():
    X, y = toy_data()
    with pytest.raises(ValueError, match="not valid"):
        ProximityForestClassifier(
            distances=["dtw?gap=1.0"],
            sampling="full",
        ).fit(X, y)


def test_multivariate_suffix_is_rejected_by_current_vector_binding():
    X, y = toy_data()
    with pytest.raises(ValueError, match="multivariate input"):
        ProximityForestClassifier(
            distances=["dtw_i"],
            sampling="full",
        ).fit(X, y)


def test_distance_spec_rejects_conflicting_dependence():
    with pytest.raises(ValueError, match="conflicts"):
        DistanceSpec("dtw_i", dependent=True).native_token(layout="multivariate")
