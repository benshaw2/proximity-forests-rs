# Installation

## From PyPI

```bash
python -m pip install proximity-forests
```

## Development installation

From a repository checkout, create and activate a virtual environment, then run:

```bash
python -m pip install -e ".[dev]"
```

The package contains a native extension and requires a supported Python version
and platform wheel, or a Rust toolchain when building from source.

## Documentation dependencies

```bash
python -m pip install -e ".[docs]"
```

Build the documentation with:

```bash
python -m sphinx -E -W --keep-going -b html docs docs/_build/html
```
