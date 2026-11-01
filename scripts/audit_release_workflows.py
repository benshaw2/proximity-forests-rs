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
