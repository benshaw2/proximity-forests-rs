"""Scikit-learn-shaped proximity isolation forest."""

from __future__ import annotations

import numbers
import numpy as np

from ._core import _NativeProximityIsolationForest
from ._distances import normalize_distances
from ._sampling import normalize_sampling
from ._max_features import normalize_max_features
from ._reference import prepare_reference_input, validate_reference_descriptor


class ProximityIsolationForest:
    """Proximity-based isolation forest for anomaly detection.

    Trees use distance-based exemplar splits and isolation leaf statistics.
    ``score_samples`` returns normality-oriented scores, where lower values are
    more abnormal. ``decision_function`` shifts scores by ``offset_`` and
    ``predict`` returns ``-1`` for outliers and ``1`` for inliers.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    n_candidates : int, default=5
        Number of candidate splitters evaluated at each splittable node.
    n_exemplars : int, default=2
        Number of branch exemplars generated for each isolation candidate.
    max_depth : int or None, default=None
        Maximum tree depth. ``None`` allows growth until another stopping rule
        applies.
    min_samples_split : int, default=2
        Minimum number of sampled observations required to split a node.
    max_features : {None, "sqrt", "log2"}, int, or float, default=None
        Number of features or channels sampled for a candidate splitter.
        ``None`` uses all available dimensions. An integer selects an absolute
        count. A float in ``(0, 1]`` selects a proportion.
    distances : str, distance specification, or sequence, default="default"
        Distance catalog used to generate candidate splitters. The accepted
        distances depend on whether ``X`` is vector, equal-length time-series,
        or unequal-length time-series data.
    contamination : {"auto"} or float, default="auto"
        Expected outlier proportion used to establish the prediction threshold.
        A float must be in ``(0, 0.5]``. ``"auto"`` uses an offset of ``-0.5``.
    sampling : {"bootstrap", "subsample", "full"}, default="subsample"
        Per-tree observation sampling policy. ``"bootstrap"`` samples with
        replacement, ``"subsample"`` samples distinct observations, and
        ``"full"`` uses every observation in its original order.
    sample_size : int, float, or None, default=256
        Per-tree sample size. An integer is an absolute count and a float in
        ``(0, 1]`` is a proportion. ``None`` uses the training-set size.
    random_state : int or None, default=None
        Seed controlling observation sampling, candidate generation, parameter
        sampling, exemplar selection, and random tie resolution. ``None`` is
        currently equivalent to seed ``0``.
    n_jobs : int or None, default=1
        Worker budget. ``1`` requests sequential execution. ``None`` and ``-1``
        use the available logical CPUs.
    data_retention : {"exemplars", "copy", "reference"}, default="exemplars"
        Training-data retention policy. ``"exemplars"`` embeds only observations
        selected by fitted splitters. ``"copy"`` embeds the complete training
        collection once. ``"reference"`` stores metadata for a complete,
        C-contiguous ``.npy`` memory map while retaining fitted exemplars for
        ordinary inference.

    Attributes
    ----------
    offset_ : float
        Threshold subtracted by ``decision_function``. Negative shifted scores
        are predicted as outliers.
    max_samples_ : int
        Resolved number of observations sampled for each tree.
    input_layout_ : {"vector", "multivariate", "unequal_multivariate"}
        Input layout established during fitting.
    n_features_in_ : int
        Number of vector features, or number of channels for time-series input.
    n_channels_in_ : int
        Number of channels for multivariate time-series input.
    n_timepoints_in_ : int or None
        Time points per case for equal-length multivariate input, or ``None``
        for unequal-length input.
    data_retention_ : str
        Retention policy used by the fitted model.

    Examples
    --------
    >>> import numpy as np
    >>> from proximity_forests import ProximityIsolationForest
    >>> X = np.array([[0.0], [0.1], [0.2], [5.0]])
    >>> model = ProximityIsolationForest(n_estimators=10, random_state=7)
    >>> model.fit(X).score_samples(X).shape
    (4,)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        n_candidates: int = 5,
        n_exemplars: int = 2,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features=None,
        contamination: str | float = "auto",
        distances="default",
        sampling: str = "subsample",
        sample_size=256,
        random_state: int | None = None,
        n_jobs: int | None = 1,
        data_retention: str = "exemplars",
    ) -> None:
        self.n_estimators = n_estimators
        self.n_candidates = n_candidates
        self.n_exemplars = n_exemplars
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.contamination = contamination
        self.distances = distances
        self.sampling = sampling
        self.sample_size = sample_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.data_retention = data_retention

    def _resolved_sample_size(self, n_cases: int) -> int:
        if self.sampling == "full":
            return n_cases
        if self.sample_size is None:
            return n_cases
        if isinstance(self.sample_size, (int, np.integer)) and not isinstance(self.sample_size, bool):
            requested = int(self.sample_size)
        else:
            requested = max(1, int(np.ceil(float(self.sample_size) * n_cases)))
        return min(requested, n_cases) if self.sampling == "subsample" else requested

    def fit(self, X, y=None):
        """Fit the isolation forest.

        Parameters
        ----------
        X : array-like or sequence of arrays
            Training observations in a supported input layout.
        y : ignored
            Accepted for estimator-interface compatibility.

        Returns
        -------
        self : ProximityIsolationForest
            Fitted estimator.
        """
        del y
        self._validate_contamination()
        X, reference_metadata = prepare_reference_input(X, self.data_retention)
        X, layout = self._validate_X(X, fitting=True)
        n_cases = len(X) if layout == "unequal_multivariate" else X.shape[0]
        distance_layout = "multivariate" if layout == "unequal_multivariate" else layout
        distances = list(normalize_distances(self.distances, layout=distance_layout))
        sampling, sample_size = normalize_sampling(self.sampling, self.sample_size, default_sampling="subsample")
        self._native = _NativeProximityIsolationForest(
            n_estimators=int(self.n_estimators), n_candidates=int(self.n_candidates), n_exemplars=int(self.n_exemplars),
            max_depth=self.max_depth, min_samples_split=int(self.min_samples_split), max_features=normalize_max_features(self.max_features), sampling=sampling, sample_size=sample_size,
            random_state=0 if self.random_state is None else int(self.random_state),
            n_jobs=0 if self.n_jobs in (None, -1) else int(self.n_jobs), data_retention=str(self.data_retention), distances=distances,
        )
        if layout == "unequal_multivariate":
            self._native.fit_unequal(X)
            self.n_channels_in_, self.n_timepoints_in_ = X[0].shape[0], None
            scores = np.asarray(self._native.anomaly_score_unequal(X))
        elif layout == "multivariate":
            self._native.fit_multivariate(X, *reference_metadata)
            self.n_channels_in_, self.n_timepoints_in_ = X.shape[1:]
            scores = np.asarray(self._native.anomaly_score_multivariate(X))
        else:
            self._native.fit(X, *reference_metadata)
            scores = np.asarray(self._native.anomaly_score(X))
        self.input_layout_ = layout
        self.n_features_in_ = X[0].shape[0] if layout == "unequal_multivariate" else X.shape[1]
        self.max_samples_ = self._resolved_sample_size(n_cases)
        self.data_retention_ = str(self.data_retention)
        self.offset_ = -0.5 if self.contamination == "auto" else -float(np.quantile(scores, 1.0 - float(self.contamination)))
        return self

    def anomaly_score(self, X):
        """Return native anomaly-oriented scores; larger is more abnormal.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Raw proximity-isolation anomaly scores.
        """
        native = self._check_is_fitted()
        X, layout = self._validate_X(X)
        if layout != self.input_layout_:
            raise ValueError("X layout does not match the fitted model")
        if layout == "unequal_multivariate":
            return native.anomaly_score_unequal(X)
        return native.anomaly_score_multivariate(X) if layout == "multivariate" else native.anomaly_score(X)

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
        """Return the terminal node reached in every tree.

        Returns
        -------
        leaves : ndarray of shape (n_samples, n_estimators)
            Stable terminal-node indices for each observation and tree.
        """
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
        sampling, sample_size = normalize_sampling(self.sampling, self.sample_size, default_sampling="subsample")
        self._native = _NativeProximityIsolationForest(
            n_estimators=int(self.n_estimators),
            n_candidates=int(self.n_candidates),
            n_exemplars=int(self.n_exemplars),
            max_depth=self.max_depth,
            min_samples_split=int(self.min_samples_split),
            max_features=normalize_max_features(self.max_features), sampling=sampling, sample_size=sample_size,
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
    def external_training_data_(self):
        """Persisted external descriptor, or ``None`` when not in reference mode."""
        native = self._check_is_fitted()
        metadata = native.external_training_data_
        if metadata is None:
            return None
        path, dtype, shape, offset, length, file_size, fingerprint = metadata
        return {
            "path": path,
            "dtype": dtype,
            "shape": tuple(shape),
            "payload_offset": int(offset),
            "payload_length": int(length),
            "file_size": int(file_size),
            "fingerprint": fingerprint,
        }

    def validate_training_data_reference(self, *, require_fingerprint=False):
        """Validate and reopen the referenced training collection read-only.

        Validation is explicit. Normal prediction and evaluation continue to
        use embedded exemplars and do not access the external file.
        """
        native = self._check_is_fitted()
        return validate_reference_descriptor(
            native.external_training_data_,
            require_fingerprint=bool(require_fingerprint),
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
        """Return constructor parameters.

        Parameters
        ----------
        deep : bool, default=True
            Accepted for estimator-interface compatibility.

        Returns
        -------
        params : dict
            Parameter names mapped to their current values.
        """
        del deep
        return {
            "n_estimators": self.n_estimators,
            "n_candidates": self.n_candidates,
            "n_exemplars": self.n_exemplars,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
            "contamination": self.contamination,
            "distances": self.distances,
            "sampling": self.sampling,
            "sample_size": self.sample_size,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "data_retention": self.data_retention,
        }

    def set_params(self, **params):
        """Set constructor parameters.

        Returns
        -------
        self
            Estimator with updated parameters.
        """
        valid = self.get_params(deep=False)
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"Invalid parameter {name!r} for this estimator")
            setattr(self, name, value)
        if "n_jobs" in params:
            native = getattr(self, "_native", None)
            if native is not None:
                native.set_n_jobs(
                    0 if self.n_jobs in (None, -1) else int(self.n_jobs)
                )
        return self

    def _validate_contamination(self):
        if self.contamination == "auto":
            return
        if not isinstance(self.contamination, numbers.Real):
            raise ValueError("contamination must be 'auto' or a float in (0, 0.5]")
        if not 0.0 < float(self.contamination) <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")

    @staticmethod
    def _validate_X(X, *, fitting: bool = False):
        if isinstance(X, (list, tuple)):
            minimum_cases = 2 if fitting else 1
            if len(X) < minimum_cases:
                operation = "fitting" if fitting else "inference"
                raise ValueError(
                    f"unequal-length X must contain at least {minimum_cases} "
                    f"case{'s' if minimum_cases != 1 else ''} for {operation}"
                )
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
        minimum_cases = 2 if fitting else 1
        if X.ndim == 2:
            if X.shape[0] < minimum_cases or X.shape[1] == 0:
                raise ValueError(
                    f"2D X must contain at least {minimum_cases} case"
                    f"{'s' if minimum_cases != 1 else ''} and one feature or timepoint"
                )
            layout = "vector"
        elif X.ndim == 3:
            if (
                X.shape[0] < minimum_cases
                or X.shape[1] == 0
                or X.shape[2] == 0
            ):
                raise ValueError(
                    f"3D X must contain at least {minimum_cases} case"
                    f"{'s' if minimum_cases != 1 else ''} and nonempty channel/time axes"
                )
            if X.shape[1] == 1:
                # aeon-style univariate collection: (cases, 1, timepoints).
                # Canonicalize to the vector/univariate path so 2D and 3D
                # one-channel inputs are interchangeable at inference and
                # use the univariate default/PF distance catalogs.
                X = X[:, 0, :]
                layout = "vector"
            else:
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
                "This ProximityIsolationForest instance is not fitted yet"
            )
        return native
