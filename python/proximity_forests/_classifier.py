"""Scikit-learn-shaped classifier without a runtime sklearn dependency."""
from __future__ import annotations

import numpy as np

from ._core import _NativeProximityForestClassifier
from ._distances import normalize_vector_distances


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
        if y.ndim not in (1, 2):
            raise ValueError("y must be one- or two-dimensional")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y contain different numbers of samples")
        if y.ndim == 2 and y.shape[1] == 0:
            raise ValueError("y must contain at least one classification output")

        self._target_was_1d = y.ndim == 1
        target_matrix = y.reshape(-1, 1) if self._target_was_1d else y
        classes = []
        encoded_columns = []
        for output in range(target_matrix.shape[1]):
            output_classes, encoded = np.unique(
                target_matrix[:, output], return_inverse=True
            )
            if output_classes.size < 2:
                raise ValueError(
                    f"classification output {output} requires at least two classes"
                )
            classes.append(output_classes)
            encoded_columns.append(np.asarray(encoded, dtype=np.uint32))

        encoded_matrix = np.ascontiguousarray(
            np.column_stack(encoded_columns), dtype=np.uint32
        )
        class_counts = [int(output_classes.size) for output_classes in classes]

        if self._target_was_1d:
            self.target_type_ = "binary" if class_counts[0] == 2 else "multiclass"
            self.classes_ = classes[0]
            self.n_classes_ = class_counts[0]
        else:
            is_binary_indicator = all(count == 2 for count in class_counts)
            if is_binary_indicator:
                try:
                    is_binary_indicator = bool(
                        np.isin(target_matrix, (0, 1, False, True)).all()
                    )
                except TypeError:
                    is_binary_indicator = False
            self.target_type_ = (
                "multilabel-indicator"
                if is_binary_indicator
                else "multiclass-multioutput"
            )
            self.classes_ = classes
            self.n_classes_ = class_counts

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
        self._native.fit(X, encoded_matrix, class_counts)
        self.n_features_in_ = self._native.n_features_in_
        self.n_outputs_ = self._native.n_outputs_
        return self

    def predict(self, X):
        native = self._check_is_fitted()
        X = self._validate_X(X)
        if self._target_was_1d:
            encoded = np.asarray(native.predict(X), dtype=np.int64)
            return self.classes_[encoded]

        encoded = np.asarray(native.predict_matrix(X), dtype=np.int64)
        decoded_columns = [
            output_classes[encoded[:, output]]
            for output, output_classes in enumerate(self.classes_)
        ]
        return np.column_stack(decoded_columns)

    def predict_proba(self, X):
        native = self._check_is_fitted()
        X = self._validate_X(X)
        if self._target_was_1d:
            return np.asarray(native.predict_proba_output(X, 0))
        return [
            np.asarray(native.predict_proba_output(X, output))
            for output in range(self.n_outputs_)
        ]

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
        return np.ascontiguousarray(X, dtype=np.float64)

    def _check_is_fitted(self):
        native = getattr(self, "_native", None)
        if native is None:
            raise RuntimeError(
                "This ProximityForestClassifier instance is not fitted yet"
            )
        return native
