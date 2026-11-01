import inspect

import pytest

from proximity_forests import (
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)


@pytest.mark.parametrize(
    "estimator",
    [ProximityForestClassifier, ProximityForestRegressor, ProximityIsolationForest],
)
def test_estimator_docstring_has_public_api_sections(estimator):
    doc = inspect.getdoc(estimator)
    assert doc
    for section in ("Parameters", "Attributes", "Examples"):
        assert section in doc
    for parameter in inspect.signature(estimator).parameters:
        assert parameter in doc


@pytest.mark.parametrize(
    "estimator",
    [ProximityForestClassifier, ProximityForestRegressor, ProximityIsolationForest],
)
def test_common_public_methods_are_documented(estimator):
    for name in ("fit", "predict", "apply", "get_params", "set_params"):
        assert inspect.getdoc(getattr(estimator, name))


def test_classifier_probability_method_is_documented():
    assert inspect.getdoc(ProximityForestClassifier.predict_proba)


def test_isolation_scoring_methods_are_documented():
    for name in ("anomaly_score", "score_samples", "decision_function"):
        assert inspect.getdoc(getattr(ProximityIsolationForest, name))
