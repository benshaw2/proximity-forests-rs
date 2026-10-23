"""Scikit-learn-shaped classifier without a runtime sklearn dependency."""

from __future__ import annotations

import numpy as np

from ._core import _NativeProximityForestClassifier
from ._distances import normalize_vector_distances


class ProximityForestClassifier:
    """Proximity forest classifier with a familiar estimator API.

    The class intentionally implements ``get_params`` and ``set_params``
    directly. It therefore works with estimator-style tooling and ForestGeom
    adapters without requiring scikit-learn at runtime.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        n_candidates: int = 5,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        distances="default",
        bootstrap: bool = True,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_candidates = n_candidates
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.distances = distances
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("The initial classifier supports one-dimensional y")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y contain different numbers of samples")

        self.classes_, encoded = np.unique(y, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("classification requires at least two classes")
        encoded = np.asarray(encoded, dtype=np.uint32)
        self._native = _NativeProximityForestClassifier(
            n_estimators=int(self.n_estimators),
            n_candidates=int(self.n_candidates),
            max_depth=self.max_depth,
            min_samples_split=int(self.min_samples_split),
            bootstrap=bool(self.bootstrap),
            random_state=0 if self.random_state is None else int(self.random_state),
            n_jobs=0 if self.n_jobs in (None, -1) else int(self.n_jobs),
            distances=list(normalize_vector_distances(self.distances)),
        )
        self._native.fit(X, encoded, int(self.classes_.size))
        self.n_features_in_ = self._native.n_features_in_
        self.n_outputs_ = self._native.n_outputs_
        self.n_classes_ = int(self.classes_.size)
        return self

    def predict(self, X):
        native = self._check_is_fitted()
        encoded = np.asarray(native.predict(self._validate_X(X)), dtype=np.int64)
        return self.classes_[encoded]

    def predict_proba(self, X):
        native = self._check_is_fitted()
        return np.asarray(native.predict_proba(self._validate_X(X)))

    def apply(self, X):
        """Return terminal node indices with shape ``(n_samples, n_trees)``."""
        native = self._check_is_fitted()
        return np.asarray(native.apply(self._validate_X(X)))

    def get_params(self, deep: bool = True):
        del deep
        return {
            "n_estimators": self.n_estimators,
            "n_candidates": self.n_candidates,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "distances": self.distances,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def set_params(self, **params):
        valid = self.get_params(deep=False)
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"Invalid parameter {name!r} for this estimator")
            setattr(self, name, value)
        return self

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty two-dimensional array")
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float64)
        # First binding slice normalizes to f64. Native f32 dispatch follows.
        return np.ascontiguousarray(X, dtype=np.float64)

    def _check_is_fitted(self):
        native = getattr(self, "_native", None)
        if native is None:
            raise RuntimeError(
                "This ProximityForestClassifier instance is not fitted yet"
            )
        return native
