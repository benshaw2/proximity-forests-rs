import numpy as np
import pytest

from proximity_forests import (
    DistanceSpec,
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)


SAMPLED_VECTOR_FAMILIES = (
    "dtw",
    "dtw_cv",
    "ddtw",
    "ddtw_cv",
    "wdtw",
    "wddtw",
    "erp",
    "lcss",
    "msm",
    "twe",
)


def vector_data():
    rng = np.random.default_rng(12345)
    left = rng.normal(0.0, 0.15, size=(8, 12))
    right = rng.normal(2.0, 0.15, size=(8, 12))
    X = np.ascontiguousarray(np.vstack((left, right)), dtype=np.float64)
    y_classification = np.array(["left"] * 8 + ["right"] * 8, dtype=object)
    y_regression = np.linspace(-1.0, 1.0, X.shape[0], dtype=np.float64)
    return X, y_classification, y_regression


@pytest.mark.parametrize("family", SAMPLED_VECTOR_FAMILIES)
def test_each_sampled_family_fits_all_three_estimators(family):
    X, y_classification, y_regression = vector_data()

    classifier = ProximityForestClassifier(
        n_estimators=3,
        n_candidates=2,
        distances=[family],
        sampling="full",
        random_state=17,
        n_jobs=1,
    ).fit(X, y_classification)
    assert classifier.apply(X).shape == (X.shape[0], 3)

    regressor = ProximityForestRegressor(
        n_estimators=3,
        n_candidates=2,
        n_exemplars=2,
        distances=[family],
        sampling="full",
        random_state=17,
        n_jobs=1,
    ).fit(X, y_regression)
    assert regressor.apply(X).shape == (X.shape[0], 3)

    isolation = ProximityIsolationForest(
        n_estimators=3,
        n_candidates=2,
        n_exemplars=2,
        distances=[family],
        sampling="full",
        random_state=17,
        n_jobs=1,
    ).fit(X)
    assert isolation.apply(X).shape == (X.shape[0], 3)


def test_fixed_specs_fit_without_becoming_sampled_families():
    X, y, _ = vector_data()
    model = ProximityForestClassifier(
        n_estimators=4,
        n_candidates=3,
        distances=[
            DistanceSpec("dtw", window=2),
            DistanceSpec("wdtw", stiffness=0.2),
            DistanceSpec("erp", window=2, gap=0.0),
            DistanceSpec("lcss", window=2, epsilon=0.25),
            DistanceSpec("msm", cost=1.0),
            DistanceSpec("twe", stiffness=0.001, penalty=0.1),
        ],
        sampling="full",
        random_state=29,
        n_jobs=1,
    ).fit(X, y)
    assert model.predict(X).shape == (X.shape[0],)


def test_sampled_catalog_is_reproducible():
    X, y, _ = vector_data()
    kwargs = dict(
        n_estimators=7,
        n_candidates=4,
        distances=list(SAMPLED_VECTOR_FAMILIES),
        sampling="bootstrap",
        random_state=101,
        n_jobs=1,
    )
    first = ProximityForestClassifier(**kwargs).fit(X, y)
    second = ProximityForestClassifier(**kwargs).fit(X, y)

    np.testing.assert_array_equal(first.apply(X), second.apply(X))
    np.testing.assert_array_equal(first.predict(X), second.predict(X))


def test_sampled_catalog_is_scheduling_independent():
    X, y, _ = vector_data()
    common = dict(
        n_estimators=9,
        n_candidates=4,
        distances=list(SAMPLED_VECTOR_FAMILIES),
        sampling="bootstrap",
        random_state=202,
    )
    serial = ProximityForestClassifier(n_jobs=1, **common).fit(X, y)
    parallel = ProximityForestClassifier(n_jobs=2, **common).fit(X, y)

    np.testing.assert_array_equal(serial.apply(X), parallel.apply(X))
    np.testing.assert_array_equal(serial.predict(X), parallel.predict(X))


def test_nan_input_rejects_catalog_without_missing_support():
    X, y, _ = vector_data()
    X = X.copy()
    X[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="every configured distance must support missing values",
    ):
        ProximityForestClassifier(
            n_estimators=3,
            sampling="full",
            random_state=7,
        ).fit(X, y)
