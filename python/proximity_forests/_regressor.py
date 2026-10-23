"""Scikit-learn-shaped proximity forest regressor."""
from __future__ import annotations

import numpy as np

from ._core import _NativeProximityForestRegressor
from ._distances import normalize_vector_distances


class ProximityForestRegressor:
    """Proximity forest regressor for single- and multi-target regression.

    A one-dimensional target produces one-dimensional predictions. A
    two-dimensional target with shape ``(n_samples, n_outputs)`` produces a
    prediction matrix with the same output width.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        n_candidates: int = 5,
        n_exemplars: int = 2,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        distances="default",
        bootstrap: bool = True,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_candidates = n_candidates
        self.n_exemplars = n_exemplars
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.distances = distances
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim not in (1, 2):
            raise ValueError("y must be one- or two-dimensional")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y contain different numbers of samples")
        if y.ndim == 2 and y.shape[1] == 0:
            raise ValueError("y must contain at least one regression output")
        if not np.isfinite(y).all():
            raise ValueError("y must contain only finite values")

        self._target_was_1d = y.ndim == 1
        y_matrix = y.reshape(-1, 1) if self._target_was_1d else y

        self._native = _NativeProximityForestRegressor(
            n_estimators=int(self.n_estimators),
            n_candidates=int(self.n_candidates),
            n_exemplars=int(self.n_exemplars),
            max_depth=self.max_depth,
            min_samples_split=int(self.min_samples_split),
            bootstrap=bool(self.bootstrap),
            random_state=0 if self.random_state is None else int(self.random_state),
            n_jobs=0 if self.n_jobs in (None, -1) else int(self.n_jobs),
            distances=list(normalize_vector_distances(self.distances)),
        )
        self._native.fit(X, np.ascontiguousarray(y_matrix, dtype=np.float64))
        self.n_features_in_ = self._native.n_features_in_
        self.n_outputs_ = self._native.n_outputs_
        return self

    def predict(self, X):
        native = self._check_is_fitted()
        X = self._validate_X(X)
        if self._target_was_1d:
            return np.asarray(native.predict(X))
        return np.asarray(native.predict_matrix(X))

    def apply(self, X):
        """Return terminal node indices with shape ``(n_samples, n_trees)``."""
        native = self._check_is_fitted()
        return np.asarray(native.apply(self._validate_X(X)))

    def get_params(self, deep: bool = True):
        del deep
        return {
            "n_estimators": self.n_estimators,
            "n_candidates": self.n_candidates,
            "n_exemplars": self.n_exemplars,
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
        return np.ascontiguousarray(X, dtype=np.float64)

    def _check_is_fitted(self):
        native = getattr(self, "_native", None)
        if native is None:
            raise RuntimeError(
                "This ProximityForestRegressor instance is not fitted yet"
            )
        return native
