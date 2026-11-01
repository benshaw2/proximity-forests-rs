"""Scikit-learn-shaped classifier without a runtime sklearn dependency."""
from __future__ import annotations

import numpy as np
import warnings

from ._core import _NativeProximityForestClassifier
from ._distances import normalize_distances
from ._sampling import normalize_sampling
from ._max_features import normalize_max_features
from ._reference import prepare_reference_input, validate_reference_descriptor


class ProximityForestClassifier:
    """Proximity-forest classifier for vector and time-series data.

    Each tree recursively evaluates candidate exemplar-based splitters and
    routes an observation to the branch with the nearest exemplar. Forest
    probabilities are formed from the fitted classification leaves. One- and
    two-dimensional targets are supported.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    n_candidates : int, default=5
        Number of candidate splitters evaluated at each splittable node.
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
    sampling : {"bootstrap", "subsample", "full"}, default="bootstrap"
        Per-tree observation sampling policy. ``"bootstrap"`` samples with
        replacement, ``"subsample"`` samples distinct observations, and
        ``"full"`` uses every observation in its original order.
    sample_size : int, float, or None, default=None
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
    oob_score : bool, default=False
        Whether to compute out-of-bag predictions and a classification accuracy
        score after fitting. Requires a sampling policy that can leave
        observations out of a tree sample.

    Attributes
    ----------
    classes_ : ndarray or list of ndarray
        Class labels for each output. A one-dimensional target produces one
        array; a multi-output target produces one array per output.
    n_classes_ : int or list of int
        Number of classes for each output.
    n_outputs_ : int
        Number of target outputs.
    target_type_ : str
        Resolved target type: ``"binary"``, ``"multiclass"``,
        ``"multilabel-indicator"``, or ``"multiclass-multioutput"``.
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
    oob_decision_function_ : ndarray or list of ndarray
        Out-of-bag class probabilities when ``oob_score=True``. Rows without a
        vote contain ``NaN``.
    oob_prediction_ : ndarray
        Decoded out-of-bag predictions when ``oob_score=True``.
    oob_vote_counts_ : ndarray of shape (n_samples,)
        Number of out-of-bag tree votes per training observation.
    oob_score_ : float
        Accuracy over training observations that received at least one OOB vote.

    Notes
    -----
    Two-dimensional ``X`` is interpreted as vector or univariate-series input.
    Equal-length multivariate series use shape
    ``(n_cases, n_channels, n_timepoints)``. Unequal-length series use one
    ``(n_channels, n_timepoints_i)`` array per case.

    Examples
    --------
    >>> import numpy as np
    >>> from proximity_forests import ProximityForestClassifier
    >>> X = np.array([[0.0], [0.2], [0.8], [1.0]])
    >>> y = np.array(["low", "low", "high", "high"])
    >>> model = ProximityForestClassifier(n_estimators=10, random_state=7)
    >>> model.fit(X, y).predict([[0.1], [0.9]]).shape
    (2,)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        n_candidates: int = 5,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features=None,
        distances="default",
        sampling: str = "bootstrap",
        sample_size=None,
        random_state: int | None = None,
        n_jobs: int | None = 1,
        data_retention: str = "exemplars",
        oob_score: bool = False,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_candidates = n_candidates
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.distances = distances
        self.sampling = sampling
        self.sample_size = sample_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.data_retention = data_retention
        self.oob_score = oob_score

    def fit(self, X, y):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like or sequence of arrays
            Training observations in a supported input layout.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Categorical target labels.

        Returns
        -------
        self : ProximityForestClassifier
            Fitted estimator.
        """
        X, reference_metadata = prepare_reference_input(X, self.data_retention)
        X, layout = self._validate_X(X, fitting=True)
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
        sampling, sample_size = normalize_sampling(self.sampling, self.sample_size, default_sampling="bootstrap")
        for name in ("oob_decision_function_", "oob_prediction_", "oob_vote_counts_", "oob_score_"):
            self.__dict__.pop(name, None)
        self._native = _NativeProximityForestClassifier(
            n_estimators=int(self.n_estimators), n_candidates=int(self.n_candidates),
            max_depth=self.max_depth, min_samples_split=int(self.min_samples_split),
            max_features=normalize_max_features(self.max_features), sampling=sampling, sample_size=sample_size, random_state=0 if self.random_state is None else int(self.random_state),
            n_jobs=0 if self.n_jobs in (None, -1) else int(self.n_jobs),
            data_retention=str(self.data_retention), distances=distances,
        )
        if layout == "unequal_multivariate":
            self._native.fit_unequal(X, encoded_y, [len(v) for v in classes])
            self.n_channels_in_, self.n_timepoints_in_ = X[0].shape[0], None
        elif layout == "multivariate":
            self._native.fit_multivariate(
                X, encoded_y, [len(v) for v in classes], *reference_metadata
            )
            self.n_channels_in_, self.n_timepoints_in_ = X.shape[1:]
        else:
            self._native.fit(
                X, encoded_y, [len(v) for v in classes], *reference_metadata
            )
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
        if self.oob_score:
            method = self._native.oob_unequal if layout == "unequal_multivariate" else (
                self._native.oob_multivariate if layout == "multivariate" else self._native.oob
            )
            probabilities, encoded_predictions, votes, score = method(
                X, encoded_y, [len(v) for v in classes]
            )
            self.oob_vote_counts_ = np.asarray(votes, dtype=np.int64)
            matrices = [np.asarray(values, dtype=float) for values in probabilities]
            self.oob_decision_function_ = matrices[0] if self._target_was_1d else matrices
            encoded = np.asarray(encoded_predictions, dtype=object).reshape(len(X), len(classes))
            decoded = np.empty(encoded.shape, dtype=object)
            decoded[:] = np.nan
            for output, labels in enumerate(classes):
                valid = np.array([value is not None for value in encoded[:, output]])
                if np.any(valid):
                    decoded[valid, output] = labels[np.asarray(encoded[valid, output], dtype=int)]
            self.oob_prediction_ = decoded[:, 0] if self._target_was_1d else decoded
            self.oob_score_ = float(score)
            zero_votes = int(np.count_nonzero(self.oob_vote_counts_ == 0))
            if zero_votes:
                warnings.warn(
                    f"{zero_votes} of {len(self.oob_vote_counts_)} training samples received no "
                    "OOB votes; their OOB results are NaN. Increase n_estimators for more "
                    "complete OOB coverage.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return self

    def predict(self, X):
        """Predict class labels for ``X``.

        Returns
        -------
        y_pred : ndarray
            Predicted labels, with one column per output for multi-output fits.
        """
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
        """Predict class probabilities for ``X``.

        Returns
        -------
        probabilities : ndarray or list of ndarray
            One matrix for a single output, or one matrix per output.
        """
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
        self.oob_score = bool(getattr(self, "oob_score", False))
        if payload is None:
            self._native = None
            return
        sampling, sample_size = normalize_sampling(self.sampling, self.sample_size, default_sampling="bootstrap")
        self._native = _NativeProximityForestClassifier(
            n_estimators=int(self.n_estimators),
            n_candidates=int(self.n_candidates),
            max_depth=self.max_depth,
            min_samples_split=int(self.min_samples_split),
            max_features=normalize_max_features(self.max_features), sampling=sampling, sample_size=sample_size,
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
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
            "distances": self.distances,
            "sampling": self.sampling,
            "sample_size": self.sample_size,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "data_retention": self.data_retention,
            "oob_score": self.oob_score,
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
                "This ProximityForestClassifier instance is not fitted yet"
            )
        return native
