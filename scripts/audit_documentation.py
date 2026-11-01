#!/usr/bin/env python3
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
docs = root / "docs"
conf = (docs / "conf.py").read_text(encoding="utf-8")
excluded = set()
match = re.search(r"exclude_patterns\s*=\s*\[(.*?)\]", conf, re.S)
if match:
    excluded = set(re.findall(r'"([^"]+)"', match.group(1)))

sources = [path for path in docs.rglob("*") if path.suffix in {".md", ".rst"}]
for path in sources:
    relative = path.relative_to(docs).as_posix()
    if relative in excluded or any(part in {"_build", "_templates", "generated"} for part in path.parts):
        continue
    text = path.read_text(encoding="utf-8")
    if "reference-backed training-data retention is not yet exposed" in text:
        raise SystemExit(f"stale reference-retention claim in {relative}")
    if "scikit-learn" in text.lower() or "nearest neighbor" in text.lower() or re.search(r"\bknn\b", text, re.I):
        raise SystemExit(f"unrelated estimator discussion in {relative}")

index = (docs / "index.md").read_text(encoding="utf-8")
for required in (
    "getting_started", "input_data", "distances", "sampling_and_dimensions",
    "oob_results", "model_persistence", "concurrency", "examples/index",
    "api_reference/index", "release_notes",
):
    if required not in index:
        raise SystemExit(f"main documentation navigation omits {required}")

pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
requirements = [line.strip() for line in (docs / "requirements.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
for requirement in requirements:
    package = re.split(r"[<>=!~]", requirement, maxsplit=1)[0].strip().lower()
    if package not in pyproject.lower():
        raise SystemExit(f"docs dependency missing from pyproject: {package}")

readme_lines = (root / "README.md").read_text(encoding="utf-8").splitlines()
if len(readme_lines) > 100:
    raise SystemExit("README exceeds the concise landing-page limit")
print(f"Documentation audit passed: {len(sources)} source pages")
