"""Persistent NumPy reference validation for fitted-data retention."""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np


def prepare_reference_input(X, retention):
    """Return ``(X, native_metadata)`` before estimator normalization.

    Metadata is ``(path, payload_offset, file_size, shape, fingerprint)``. Full
    fingerprinting is deliberately absent from the default fit path.
    """
    if str(retention) != "reference":
        return X, (None, None, None, None, None)
    if not isinstance(X, np.memmap):
        raise ValueError(
            "data_retention='reference' requires a NumPy .npy memmap opened "
            "with numpy.load(path, mmap_mode='r' or 'r+'); an ordinary ndarray "
            "does not retain a trustworthy backing-file descriptor"
        )
    if X.ndim not in (2, 3):
        raise ValueError("reference retention supports dense 2D or 3D .npy inputs")
    if X.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("reference retention requires a float32 or float64 .npy payload")
    if not X.flags.c_contiguous:
        raise ValueError("reference retention requires a C-contiguous .npy memmap")
    filename = getattr(X, "filename", None)
    if filename is None:
        raise ValueError("reference memmap does not expose a persistent backing path")
    path = Path(filename).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".npy":
        raise ValueError("reference retention currently requires a .npy backing file")
    offset = int(getattr(X, "offset", 0))
    file_size = int(path.stat().st_size)
    expected_end = offset + int(X.nbytes)
    if expected_end != file_size:
        raise ValueError(
            "reference retention requires the complete C-order .npy payload, "
            "not a sliced view or a file with trailing data"
        )
    # Verify that the file header describes exactly the supplied mapping. This
    # reads metadata only; it neither hashes nor materializes the numeric payload.
    probe = np.load(path, mmap_mode="r", allow_pickle=False)
    if probe.shape != X.shape or probe.dtype != X.dtype or int(probe.offset) != offset:
        raise ValueError("reference memmap metadata disagrees with its .npy header")
    return X, (os.fspath(path), offset, file_size, tuple(int(v) for v in X.shape), None)


_DTYPE_NAMES = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}


def validate_reference_descriptor(metadata, *, require_fingerprint=False):
    """Validate a persisted .npy descriptor and return a read-only memmap.

    This operation is explicit and potentially touches filesystem metadata and
    the .npy header. It is never called by predict, apply, or score methods.
    """
    if metadata is None:
        raise ValueError("this fitted model does not retain external training data")
    (
        stored_path,
        scalar_kind,
        stored_shape,
        payload_offset,
        payload_length,
        stored_file_size,
        fingerprint,
    ) = metadata
    path = Path(stored_path).expanduser()
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"referenced training-data file does not exist: {stored_path}"
        ) from error
    actual_file_size = int(path.stat().st_size)
    if actual_file_size != int(stored_file_size):
        raise ValueError(
            "referenced training-data file size changed: "
            f"expected {stored_file_size}, found {actual_file_size}"
        )
    if require_fingerprint and fingerprint is None:
        raise ValueError("this reference descriptor has no content fingerprint")
    if fingerprint is not None:
        raise NotImplementedError(
            "fingerprint verification is not implemented for this fingerprint scheme"
        )
    try:
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise ValueError(f"referenced training-data file is not a valid .npy array: {error}") from error
    expected_dtype = _DTYPE_NAMES.get(str(scalar_kind))
    if expected_dtype is None:
        raise ValueError(f"unsupported referenced scalar kind: {scalar_kind}")
    if mapped.dtype != expected_dtype:
        raise ValueError(
            f"referenced training-data dtype changed: expected {expected_dtype}, found {mapped.dtype}"
        )
    expected_shape = tuple(int(value) for value in stored_shape)
    if mapped.shape != expected_shape:
        raise ValueError(
            f"referenced training-data shape changed: expected {expected_shape}, found {mapped.shape}"
        )
    if not mapped.flags.c_contiguous:
        raise ValueError("referenced training-data array is not C-contiguous")
    if int(mapped.offset) != int(payload_offset):
        raise ValueError(
            "referenced training-data payload offset changed: "
            f"expected {payload_offset}, found {mapped.offset}"
        )
    if int(mapped.nbytes) != int(payload_length):
        raise ValueError(
            "referenced training-data payload length changed: "
            f"expected {payload_length}, found {mapped.nbytes}"
        )
    if int(mapped.offset) + int(mapped.nbytes) != actual_file_size:
        raise ValueError("referenced .npy payload does not occupy the expected complete file")
    return mapped
