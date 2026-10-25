"""Proximity forest estimators with a scikit-learn-shaped API."""

from ._classifier import ProximityForestClassifier
from importlib.metadata import version as _distribution_version

from ._core import runtime_info
from ._distances import DistanceSpec
from ._isolation import ProximityIsolationForest
from ._regressor import ProximityForestRegressor

__version__ = _distribution_version("proximity-forests")
del _distribution_version

__all__ = [
    "DistanceSpec",
    "ProximityForestClassifier",
    "ProximityForestRegressor",
    "ProximityIsolationForest",
    "__version__",
    "runtime_info",
]
