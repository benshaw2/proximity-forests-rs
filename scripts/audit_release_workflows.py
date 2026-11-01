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




for token in (
    "github-release:",
    "publish-pypi:",
    "needs:\n      - verify-distributions",
    "needs:\n      - github-release",
    "gh release create",
    "--verify-tag",
    "--prerelease",
    "name: pypi",
    "id-token: write",
    "pypa/gh-action-pypi-publish@release/v1",
    "rm -f dist/SHA256SUMS",
):
    if token not in wheels:
        raise SystemExit(f"tagged publication workflow missing {token}")

for job_name in ("github-release", "publish-pypi"):
    job = wheels.split(f"  {job_name}:", 1)[1]
    if job_name == "github-release":
        job = job.split("  publish-pypi:", 1)[0]
    if "github.event_name == 'push'" not in job or "github.ref_type == 'tag'" not in job:
        raise SystemExit(f"{job_name} must be restricted to pushed tags")

pypi_job = wheels.split("  publish-pypi:", 1)[1]
if "contents: write" in pypi_job:
    raise SystemExit("PyPI job must not receive repository write permission")
if "password:" in pypi_job or "username:" in pypi_job:
    raise SystemExit("PyPI publication must use Trusted Publishing without stored credentials")

print("Release workflow audit passed")
