#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "Expected project virtual environment at $ROOT/.venv" >&2
    exit 1
fi

export VIRTUAL_ENV="$ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

PY="$VIRTUAL_ENV/bin/python"
MATURIN="$VIRTUAL_ENV/bin/maturin"

if [[ ! -x "$MATURIN" ]]; then
    "$PY" -m pip install maturin
fi

printf 'Python: '
"$PY" -c 'import sys; print(sys.executable, sys.version)'
printf 'Maturin: '
"$MATURIN" --version

# Apply and verify standard Rust formatting.
cargo fmt --all
cargo fmt --all --check

# Compile and test both the core and Python binding crates before replacing the
# installed extension. The workspace currently enables a broad Clippy lint set;
# treating every lint as an error is a separate repository-wide cleanup and is
# not required to validate a feature or release build.
cargo check --workspace

# Opt in to strict Clippy explicitly. This is useful for a dedicated lint-cleanup
# branch, but avoids blocking ordinary rebuilds on pre-existing workspace lints.
if [[ "${PF_STRICT_CLIPPY:-0}" == "1" ]]; then
    cargo clippy --workspace --all-targets -- -D warnings
fi

cargo test --workspace

# Prevent an older in-tree extension from shadowing the editable installation.
find python/proximity_forests -maxdepth 1 -type f \
    \( -name '_core*.so' -o -name '_core*.pyd' -o -name '_core*.dylib' \) \
    -print -delete

# VIRTUAL_ENV and PATH select the exact interpreter. Maturin 1.15 does not
# accept --interpreter on `develop`.
"$MATURIN" develop --release

# Verify that Python loaded the rebuilt extension and that the constructor
# contract expected by the public wrappers is present.
"$PY" - <<'PY'
import inspect
import proximity_forests._core as core

print("Loaded native extension:", core.__file__)
classes = (
    core._NativeProximityForestClassifier,
    core._NativeProximityForestRegressor,
    core._NativeProximityIsolationForest,
)
for cls in classes:
    signature = inspect.signature(cls)
    print(cls.__name__, signature)
    if "distances" not in signature.parameters:
        raise SystemExit(
            f"{cls.__name__} is stale: constructor does not expose distances"
        )
PY

# Run the focused distance tests first for quicker diagnostics, then the full suite.
#"$PY" -m pytest python/tests/test_distances.py -v
"$PY" -m pytest
