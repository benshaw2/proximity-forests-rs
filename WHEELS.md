# Wheel builds

This project contains a compiled Rust extension. Wheels are therefore specific
to a Python version, operating system, and CPU architecture. Installing a
compatible wheel does not require a Rust toolchain.

## Initial beta matrix

- Linux x86_64 and aarch64
- Windows x86_64
- macOS x86_64 and arm64
- CPython 3.9 through 3.13

The CI workflow uploads build artifacts for private distribution and does not
publish them to PyPI. Additional targets should be added when they are tested
and needed by users.

## Installation

```bash
python -m pip install /path/to/proximity_forests-*.whl
```

Test an installed wheel outside the source checkout to avoid accidentally
importing the local package directory.
