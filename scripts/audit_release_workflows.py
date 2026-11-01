#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
docs = (root / ".github/workflows/docs.yml").read_text(encoding="utf-8")
wheels = (root / ".github/workflows/build-wheels.yml").read_text(encoding="utf-8")

for token in (
    "actions/deploy-pages@v4",
    "actions/upload-pages-artifact@v3",
    "python -m sphinx",
    "-W",
    "scripts/audit_documentation.py",
):
    if token not in docs:
        raise SystemExit(f"documentation workflow missing {token}")


for token in (
    "PyO3/maturin-action@v1",
    'command: build',
    'rust-toolchain: "1.85.0"',
    "-r docs/requirements.txt",
    "--find-links dist",
    "sccache: true",
    "Verify documentation import target",
    "working-directory: ${{ runner.temp }}",
):
    if token not in docs:
        raise SystemExit(f"documentation native-build setup missing {token}")
if 'pip install -e ".[docs]"' in docs:
    raise SystemExit("documentation workflow must not use pip editable native build")
if "command: develop" in docs:
    raise SystemExit("documentation workflow must build and install a wheel, not use maturin develop")
if "cache: sccache" in docs:
    raise SystemExit("maturin-action v1 expects sccache: true, not cache: sccache")

for token in (
    "fail-fast: false",
    "rust-toolchain: \"1.85.0\"",
    "--locked",
    "verify-distributions:",
    "needs:",
    "release-distributions",
    "expected_wheels = 5 * 4",
    "python -m twine check",
    "SHA256SUMS",
):
    if token not in wheels:
        raise SystemExit(f"distribution workflow missing {token}")

for forbidden in ("pypa/gh-action-pypi-publish", "gh release create", "softprops/action-gh-release"):
    if forbidden in wheels:
        raise SystemExit("A1 must verify distributions but must not publish yet")

print("Release workflow audit passed")
