"""Typed distance specifications and catalog normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlencode

_VECTOR_DEFAULT = ("squared_euclidean",)
_VECTOR_PF = (
    "squared_euclidean", "dtw", "ddtw", "wdtw", "wddtw",
    "erp", "lcss", "msm", "twe",
)
_MULTIVARIATE_DEFAULT = ("squared_euclidean_i", "dtw_i", "dtw_d")
_MULTIVARIATE_PF = (
    "squared_euclidean_i", "manhattan_i", "cid_i", "sbd_i",
    "dtw_i", "ddtw_i", "wdtw_i", "wddtw_i", "erp_i", "lcss_i",
    "msm_i", "twe_i", "dtw_d", "ddtw_d", "wdtw_d", "wddtw_d",
)


@dataclass(frozen=True)
class DistanceSpec:
    """One exact, fixed distance configuration.

    Plain strings select candidate-local families. ``DistanceSpec`` and mapping
    inputs always request a fixed configuration, even if no optional parameter
    is supplied. The internal ``fixed=true`` marker preserves that distinction
    across the Python/Rust boundary.
    """

    name: str
    dependent: bool | None = None
    window: int | None = None
    stiffness: float | None = None
    gap: float | None = None
    epsilon: float | None = None
    cost: float | None = None
    penalty: float | None = None

    def native_token(self, *, layout: str) -> str:
        name = _normalize_name(self.name)
        if layout == "multivariate":
            name = _resolve_multivariate_name(name, self.dependent)
        elif self.dependent is not None:
            raise ValueError("dependent is only valid for multivariate input")

        if name.endswith("_cv") or "_cv_" in name:
            raise ValueError(
                "DistanceSpec is fixed; use a plain *_cv string for sampled windows"
            )
        if self.window is not None and self.window < 0:
            raise ValueError("distance window must be nonnegative")

        parameters = {
            key: value
            for key, value in (
                ("fixed", "true"),
                ("window", self.window),
                ("stiffness", self.stiffness),
                ("gap", self.gap),
                ("epsilon", self.epsilon),
                ("cost", self.cost),
                ("penalty", self.penalty),
            )
            if value is not None
        }
        return f"{name}?{urlencode(parameters)}"


def normalize_distances(distances, *, layout: str) -> tuple[str, ...]:
    """Return validated native tokens for one input layout."""
    if layout not in {"vector", "multivariate"}:
        raise ValueError("layout must be 'vector' or 'multivariate'")
    if distances is None or distances == "default":
        return _VECTOR_DEFAULT if layout == "vector" else _MULTIVARIATE_DEFAULT
    if distances == "pf":
        return _VECTOR_PF if layout == "vector" else _MULTIVARIATE_PF

    values = (
        (distances,)
        if isinstance(distances, (str, Mapping, DistanceSpec))
        else tuple(distances)
    )
    if not values:
        raise ValueError("distances must contain at least one distance")
    return tuple(_native_token(value, layout=layout) for value in values)


def normalize_vector_distances(distances) -> tuple[str, ...]:
    """Normalize the catalog for current two-dimensional estimators."""
    return normalize_distances(distances, layout="vector")


def _native_token(value, *, layout: str) -> str:
    if isinstance(value, DistanceSpec):
        return value.native_token(layout=layout)
    if isinstance(value, Mapping):
        if "name" not in value:
            raise ValueError("distance dictionaries require a 'name' entry")
        return DistanceSpec(**dict(value)).native_token(layout=layout)
    if isinstance(value, str):
        name = _normalize_name(value)
        if layout == "vector" and name.endswith(("_i", "_d")):
            raise ValueError(f"{value!r} requires multivariate input")
        return name
    raise TypeError("each distance must be a string, mapping, or DistanceSpec")


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise TypeError("distance name must be a nonempty string")
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_multivariate_name(name: str, dependent: bool | None) -> str:
    if name.endswith(("_i", "_d")):
        if dependent is not None:
            expected = "_d" if dependent else "_i"
            if not name.endswith(expected):
                raise ValueError("distance suffix conflicts with dependent")
        return name
    if dependent is None:
        dependent = False
    return f"{name}_{'d' if dependent else 'i'}"
