"""Scikit-learn-shaped proximity isolation forest."""

from __future__ import annotations

import numbers
import numpy as np

from ._core import _NativeProximityIsolationForest


class ProximityIsolationForest:
    """Isolation Forest scoring with proximity-based node splits.

    ``score_samples`` follows scikit-learn's orientation: lower values are more
    abnormal. ``decision_function`` subtracts ``offset_`` so negative values are
    classified as outliers, and ``predict`` returns ``-1`` for outliers and
    ``1`` for inliers.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        n_candidates: int = 5,
        n_exemplars: int = 2,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        contamination: str | float = "auto",
        bootstrap: bool = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_candidates = n_candidates
        self.n_exemplars = n_exemplars
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.contamination = contamination
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        del y
        X = self._validate_X(X)
        self._validate_contamination()
        self._native = _NativeProximityIsolationForest(
            n_estimators=int(self.n_estimators),
            n_candidates=int(self.n_candidates),
            n_exemplars=int(self.n_exemplars),
            max_depth=self.max_depth,
            min_samples_split=int(self.min_samples_split),
            bootstrap=bool(self.bootstrap),
            random_state=0 if self.random_state is None else int(self.random_state),
            n_jobs=0 if self.n_jobs in (None, -1) else int(self.n_jobs),
        )
        self._native.fit(X)
        self.n_features_in_ = self._native.n_features_in_
        self.max_samples_ = X.shape[0]

        training_scores = self.score_samples(X)
        if self.contamination == "auto":
            self.offset_ = -0.5
        else:
            # Match sklearn's boundary convention: approximately the requested
            # contamination fraction has decision_function < 0.
            self.offset_ = float(
                np.percentile(training_scores, 100.0 * float(self.contamination))
            )
        return self

    def anomaly_score(self, X):
        """Return scores in (0, 1], with larger values more anomalous."""
        native = self._check_is_fitted()
        return np.asarray(
            native.anomaly_score(
                self._validate_X(X)
            )
        )

    def score_samples(self, X):
        """Return normality-oriented raw scores; lower means more abnormal."""
        return -self.anomaly_score(X)

    def decision_function(self, X):
        """Return shifted scores with zero as the fitted outlier boundary."""
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        """Return ``1`` for inliers and ``-1`` for outliers."""
        return np.where(self.decision_function(X) < 0.0, -1, 1).astype(np.int64)

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
            "contamination": self.contamination,
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

    def _validate_contamination(self):
        if self.contamination == "auto":
            return
        if not isinstance(self.contamination, numbers.Real):
            raise ValueError("contamination must be 'auto' or a float in (0, 0.5]")
        if not 0.0 < float(self.contamination) <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0:
            raise ValueError("X must contain at least two samples and one feature")
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float64)
        return np.ascontiguousarray(X, dtype=np.float64)

    def _check_is_fitted(self):
        native = getattr(self, "_native", None)
        if native is None:
            raise RuntimeError(
                "This ProximityIsolationForest instance is not fitted yet"
            )
        return native
