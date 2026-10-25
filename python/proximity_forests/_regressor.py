"""Scikit-learn-shaped proximity forest regressor."""
from __future__ import annotations

import numpy as np

from ._core import _NativeProximityForestRegressor
from ._distances import normalize_distances


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
        data_retention: str = "exemplars",
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
        self.data_retention = data_retention

    def fit(self, X, y):
        X, layout = self._validate_X(X)
        n_cases = len(X) if layout == "unequal_multivariate" else X.shape[0]
        y = np.asarray(y, dtype=np.float64)
        self._target_was_1d = y.ndim == 1
        if y.ndim not in (1, 2) or y.shape[0] != n_cases:
            raise ValueError("y must be one- or two-dimensional with one row per case")
        targets = np.ascontiguousarray(y.reshape(-1, 1) if y.ndim == 1 else y)
        distance_layout = "multivariate" if layout == "unequal_multivariate" else layout
        distances = list(normalize_distances(self.distances, layout=distance_layout))
        self._native = _NativeProximityForestRegressor(
            n_estimators=int(self.n_estimators), n_candidates=int(self.n_candidates), n_exemplars=int(self.n_exemplars),
            max_depth=self.max_depth, min_samples_split=int(self.min_samples_split), bootstrap=bool(self.bootstrap),
            random_state=0 if self.random_state is None else int(self.random_state),
            n_jobs=0 if self.n_jobs in (None, -1) else int(self.n_jobs), data_retention=str(self.data_retention), distances=distances,
        )
        if layout == "unequal_multivariate":
            self._native.fit_unequal(X, targets)
            self.n_channels_in_, self.n_timepoints_in_ = X[0].shape[0], None
        elif layout == "multivariate":
            self._native.fit_multivariate(X, targets)
            self.n_channels_in_, self.n_timepoints_in_ = X.shape[1:]
        else:
            self._native.fit(X, targets)
        self.input_layout_ = layout
        self.n_features_in_ = X[0].shape[0] if layout == "unequal_multivariate" else X.shape[1]
        self.n_outputs_ = targets.shape[1]
        self.data_retention_ = str(self.data_retention)
        return self

    def predict(self, X):
        native = self._check_is_fitted()
        X, layout = self._validate_X(X)
        if layout != self.input_layout_:
            raise ValueError("X layout does not match the fitted model")
        if layout == "unequal_multivariate":
            values = native.predict_unequal_matrix(X)
        elif layout == "multivariate":
            values = native.predict_multivariate_matrix(X)
        else:
            values = native.predict_matrix(X)
        values = np.asarray(values)
        return values[:, 0] if self._target_was_1d else values

    def apply(self, X):
        native = self._check_is_fitted()
        X, layout = self._validate_X(X)
        if layout != self.input_layout_:
            raise ValueError("X layout does not match the fitted model")
        if layout == "unequal_multivariate":
            return native.apply_unequal(X)
        return native.apply_multivariate(X) if layout == "multivariate" else native.apply(X)

    def __getstate__(self):
        state = self.__dict__.copy()
        native = state.pop("_native", None)
        state["_native_payload"] = None if native is None else bytes(native.to_bytes())
        return state

    def __setstate__(self, state):
        payload = state.pop("_native_payload", None)
        self.__dict__.update(state)
        if payload is None:
            self._native = None
            return
        self._native = _NativeProximityForestRegressor(
            n_estimators=int(self.n_estimators),
            n_candidates=int(self.n_candidates),
            n_exemplars=int(self.n_exemplars),
            max_depth=self.max_depth,
            min_samples_split=int(self.min_samples_split),
            bootstrap=bool(self.bootstrap),
            random_state=0 if self.random_state is None else int(self.random_state),
            n_jobs=0 if self.n_jobs in (None, -1) else int(self.n_jobs),
            data_retention=str(self.data_retention),
            distances=list(normalize_distances(self.distances, layout=("multivariate" if getattr(self, "input_layout_", "vector") == "unequal_multivariate" else getattr(self, "input_layout_", "vector")))),
        )
        self._native.load_bytes(
            payload,
            int(self.n_features_in_),
            getattr(self, "n_channels_in_", None),
            getattr(self, "n_timepoints_in_", None),
        )


    @property
    def has_training_data_(self):
        """Whether fitted training observations remain reachable by the model."""
        self._check_is_fitted()
        return getattr(self, "data_retention_", self.data_retention) != "exemplars"

    @property
    def has_embedded_training_data_(self):
        """Whether the fitted artifact embeds a complete training-data copy."""
        self._check_is_fitted()
        return getattr(self, "data_retention_", self.data_retention) == "copy"

    @property
    def supports_proximities_(self):
        """Whether tree leaf assignments can be used to derive proximities."""
        self._check_is_fitted()
        return True

    @property
    def supports_self_contained_imputation_(self):
        """Whether the artifact embeds all training values needed for imputation."""
        self._check_is_fitted()
        return getattr(self, "data_retention_", self.data_retention) == "copy"

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
            "data_retention": self.data_retention,
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
        if isinstance(X, (list, tuple)):
            if len(X) < 2:
                raise ValueError("unequal-length X must contain at least two cases")
            cases = [np.asarray(case) for case in X]
            ranks = {case.ndim for case in cases}
            if not ranks.issubset({1, 2}) or len(ranks) != 1:
                raise ValueError(
                    "unequal-length X must contain either all 1D or all 2D arrays"
                )
            if 1 in ranks:
                cases = [case.reshape(1, -1) for case in cases]
            if any(case.shape[0] == 0 or case.shape[1] == 0 for case in cases):
                raise ValueError("unequal-length cases must have nonempty axes")
            channels = cases[0].shape[0]
            if any(case.shape[0] != channels for case in cases):
                raise ValueError(
                    "all unequal-length cases must have the same channel count"
                )
            dtype = (
                np.float32
                if all(case.dtype == np.float32 for case in cases)
                else np.float64
            )
            normalized = [
                np.ascontiguousarray(case, dtype=dtype) for case in cases
            ]
            return normalized, "unequal_multivariate"

        X = np.asarray(X)
        if X.ndim == 2:
            if X.shape[0] < 2 or X.shape[1] == 0:
                raise ValueError(
                    "2D X must contain at least two cases and one feature or timepoint"
                )
            layout = "vector"
        elif X.ndim == 3:
            if X.shape[0] < 2 or X.shape[1] == 0 or X.shape[2] == 0:
                raise ValueError(
                    "3D X must have nonempty shape (cases, channels, timepoints)"
                )
            layout = "multivariate"
        else:
            raise ValueError(
                "X must be a 2D/3D NumPy array or a list of 1D/2D arrays"
            )
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float64)
        return np.ascontiguousarray(X), layout

    def _check_is_fitted(self):
        native = getattr(self, "_native", None)
        if native is None:
            raise RuntimeError(
                "This ProximityForestRegressor instance is not fitted yet"
            )
        return native
