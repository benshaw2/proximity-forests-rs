"""Validation and native encoding for per-tree observation sampling."""

from __future__ import annotations

import numbers

_VALID_SAMPLING = {"full", "bootstrap", "subsample"}


def normalize_sampling(sampling, sample_size, *, default_sampling: str):
    """Return a validated ``(sampling, native_sample_size)`` pair.

    ``native_sample_size`` is ``"all"``, ``"fixed:<n>"``, or
    ``"proportion:<p>"``. Booleans are rejected even though ``bool`` is an
    integer subclass.
    """
    mode = default_sampling if sampling is None else str(sampling).strip().lower()
    if mode not in _VALID_SAMPLING:
        raise ValueError("sampling must be 'full', 'bootstrap', or 'subsample'")
    if mode == "full":
        # Full sampling is authoritative: every observation is used exactly
        # once. Ignore sample_size so isolation's default of 256 does not make
        # an explicit sampling="full" request invalid.
        return mode, "all"
    if sample_size is None:
        return mode, "all"
    if isinstance(sample_size, bool):
        raise TypeError("sample_size must be None, a positive integer, or a float in (0, 1]")
    if isinstance(sample_size, numbers.Integral):
        value = int(sample_size)
        if value < 1:
            raise ValueError("integer sample_size must be positive")
        return mode, f"fixed:{value}"
    if isinstance(sample_size, numbers.Real):
        value = float(sample_size)
        if not 0.0 < value <= 1.0:
            raise ValueError("floating sample_size must be in the interval (0, 1]")
        return mode, f"proportion:{value!r}"
    raise TypeError("sample_size must be None, a positive integer, or a float in (0, 1]")
