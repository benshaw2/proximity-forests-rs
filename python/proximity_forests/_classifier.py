"""Scikit-learn-shaped classifier without a runtime sklearn dependency."""
from __future__ import annotations

import numpy as np

from ._core import _NativeProximityForestClassifier
from ._distances import normalize_distances


class ProximityForestClassifier:
    """Proximity forest classifier for single- and multi-output targets.

    One-dimensional targets preserve the existing single-output API. A
    two-dimensional target is encoded independently by column and produces a
    prediction matrix. Binary indicator matrices are identified as multilabel
    targets while sharing the same native multi-output representation.
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
        data_retention: str = "exemplars",
    ) -> None:
        self.n_estimators = n_estimators
        self.n_candidates = n_candidates
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
        y = np.asarray(y)
        if y.ndim not in (1, 2) or y.shape[0] != n_cases:
            raise ValueError("y must be one- or two-dimensional with one row per case")
        self._target_was_1d = y.ndim == 1
        target_matrix = y.reshape(-1, 1) if self._target_was_1d else y
        classes, encoded = [], []
        for output in range(target_matrix.shape[1]):
            values, inverse = np.unique(target_matrix[:, output], return_inverse=True)
            if len(values) < 2:
                raise ValueError(f"classification output {output} requires at least two classes")
            classes.append(values)
            encoded.append(inverse.astype(np.uint32, copy=False))
        encoded_y = np.ascontiguousarray(np.column_stack(encoded), dtype=np.uint32)
        distance_layout = "multivariate" if layout == "unequal_multivariate" else layout
        distances = list(normalize_distances(self.distances, layout=distance_layout))
        self._native = _NativeProximityForestClassifier(
            n_estimators=int(self.n_estimators), n_candidates=int(self.n_candidates),
            max_depth=self.max_depth, min_samples_split=int(self.min_samples_split),
            bootstrap=bool(self.bootstrap), random_state=0 if self.random_state is None else int(self.random_state),
            n_jobs=0 if self.n_jobs in (None, -1) else int(self.n_jobs),
            data_retention=str(self.data_retention), distances=distances,
        )
        if layout == "unequal_multivariate":
            self._native.fit_unequal(X, encoded_y, [len(v) for v in classes])
            self.n_channels_in_, self.n_timepoints_in_ = X[0].shape[0], None
        elif layout == "multivariate":
            self._native.fit_multivariate(X, encoded_y, [len(v) for v in classes])
            self.n_channels_in_, self.n_timepoints_in_ = X.shape[1:]
        else:
            self._native.fit(X, encoded_y, [len(v) for v in classes])
        self.input_layout_ = layout
        self.classes_ = classes[0] if self._target_was_1d else classes
        self.n_classes_ = len(classes[0]) if self._target_was_1d else [len(v) for v in classes]
        self.n_outputs_ = len(classes)
        if self._target_was_1d:
            self.target_type_ = "binary" if len(classes[0]) == 2 else "multiclass"
        elif np.issubdtype(target_matrix.dtype, np.integer) and np.all((target_matrix == 0) | (target_matrix == 1)):
            self.target_type_ = "multilabel-indicator"
        else:
            self.target_type_ = "multiclass-multioutput"
        self.n_features_in_ = X[0].shape[0] if layout == "unequal_multivariate" else X.shape[1]
        self.data_retention_ = str(self.data_retention)
        return self

    def predict(self, X):
        native = self._check_is_fitted()
        X, layout = self._validate_X(X)
        if layout != self.input_layout_:
            raise ValueError("X layout does not match the fitted model")
        if layout == "unequal_multivariate":
            encoded = native.predict_unequal(X) if self._target_was_1d else native.predict_unequal_matrix(X)
        elif layout == "multivariate":
            encoded = native.predict_multivariate(X) if self._target_was_1d else native.predict_multivariate_matrix(X)
        else:
            encoded = native.predict(X) if self._target_was_1d else native.predict_matrix(X)
        if self._target_was_1d:
            return self.classes_[np.asarray(encoded, dtype=np.intp)]
        encoded = np.asarray(encoded, dtype=np.intp)
        return np.column_stack([self.classes_[i][encoded[:, i]] for i in range(self.n_outputs_)])

    def predict_proba(self, X):
        native = self._check_is_fitted()
        X, layout = self._validate_X(X)
        if layout != self.input_layout_:
            raise ValueError("X layout does not match the fitted model")
        if layout == "unequal_multivariate":
            call = native.predict_unequal_proba_output
        elif layout == "multivariate":
            call = native.predict_multivariate_proba_output
        else:
            call = native.predict_proba_output
        probabilities = [call(X, output) for output in range(self.n_outputs_)]
        return probabilities[0] if self._target_was_1d else probabilities

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
        self._native = _NativeProximityForestClassifier(
            n_estimators=int(self.n_estimators),
            n_candidates=int(self.n_candidates),
            max_depth=self.max_depth,
            min_samples_split=int(self.min_samples_split),
            bootstrap=bool(self.bootstrap),
            random_state=0 if self.random_state is None else int(self.random_state),
            n_jobs=0 if self.n_jobs in (None, -1) else int(self.n_jobs),
            data_retention=str(self.data_retention),
            distances=list(normalize_distances(self.distances, layout=("multivariate" if getattr(self, "input_layout_", "vector") == "unequal_multivariate" else getattr(self, "input_layout_", "vector")))),
        )
        class_counts = (
            [int(self.n_classes_)]
            if self._target_was_1d
            else [int(value) for value in self.n_classes_]
        )
        self._native.load_bytes(
            payload,
            int(self.n_features_in_),
            class_counts,
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
                "This ProximityForestClassifier instance is not fitted yet"
            )
        return native
