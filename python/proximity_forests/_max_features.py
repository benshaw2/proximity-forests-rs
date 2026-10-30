"""Validation and native encoding for node-level feature/channel selection."""
from __future__ import annotations
import numbers


def normalize_max_features(max_features) -> str:
    """Encode a scikit-learn-shaped max_features value for the native layer."""
    if max_features is None:
        return "all"
    if isinstance(max_features, bool):
        raise TypeError("max_features must be None, 'sqrt', 'log2', a positive integer, or a float in (0, 1]")
    if isinstance(max_features, str):
        value = max_features.strip().lower()
        if value not in {"sqrt", "log2"}:
            raise ValueError("string max_features must be 'sqrt' or 'log2'")
        return value
    if isinstance(max_features, numbers.Integral):
        value = int(max_features)
        if value < 1:
            raise ValueError("integer max_features must be positive")
        return f"fixed:{value}"
    if isinstance(max_features, numbers.Real):
        value = float(max_features)
        if not 0.0 < value <= 1.0:
            raise ValueError("floating max_features must be in the interval (0, 1]")
        return f"proportion:{value!r}"
    raise TypeError("max_features must be None, 'sqrt', 'log2', a positive integer, or a float in (0, 1]")
