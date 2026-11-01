import numpy as np
import pytest

from proximity_forests import ProximityForestRegressor, ProximityIsolationForest


@pytest.mark.parametrize("kind", ["regressor", "isolation"])
def test_configured_exemplar_count_is_capped_by_each_node_distinct_count(kind):
    rng = np.random.default_rng(4401)
    X = np.ascontiguousarray(rng.normal(size=(48, 8)), dtype=np.float64)
    common = dict(
        n_estimators=3,
        n_candidates=4,
        n_exemplars=3,
        max_depth=10,
        min_samples_split=2,
        sampling="full",
        random_state=44,
        n_jobs=1,
    )
    if kind == "regressor":
        y = X[:, 0] - X[:, 1]
        model = ProximityForestRegressor(**common).fit(X, y)
    else:
        model = ProximityIsolationForest(**common).fit(X)
    assert model.apply(X).shape == (X.shape[0], common["n_estimators"])
