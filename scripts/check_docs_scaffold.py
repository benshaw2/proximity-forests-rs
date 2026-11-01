#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
required = [
    "docs/conf.py",
    "docs/index.md",
    "docs/getting_started.md",
    "docs/input_data.md",
    "docs/api_reference/classification.rst",
    "docs/api_reference/regression.rst",
    "docs/api_reference/isolation.rst",
    "docs/requirements.txt",
]
missing = [relative for relative in required if not (root / relative).is_file()]
if missing:
    raise SystemExit("Missing documentation files: " + ", ".join(missing))
for name in (
    "ProximityForestClassifier",
    "ProximityForestRegressor",
    "ProximityIsolationForest",
):
    if not any(name in (root / relative).read_text(encoding="utf-8") for relative in required):
        raise SystemExit(f"Estimator absent from documentation scaffold: {name}")
print("Documentation scaffold is complete")

conf_text = (root / "docs/conf.py").read_text(encoding="utf-8")
if '"_templates"' not in conf_text:
    raise SystemExit("docs/_templates must be excluded from source documents")
if "nitpicky = True" in conf_text:
    raise SystemExit("nitpicky mode is incompatible with public value-domain type prose")
template_text = (root / "docs/_templates/autosummary/class.rst").read_text(encoding="utf-8")
if ".. autosummary::" in template_text:
    raise SystemExit("class template must not emit unresolved member autosummary links")

shared_guides = [
    "docs/distances.md",
    "docs/sampling_and_dimensions.md",
    "docs/oob_results.md",
    "docs/model_persistence.md",
]
missing_guides = [relative for relative in shared_guides if not (root / relative).is_file()]
if missing_guides:
    raise SystemExit("Missing shared usage guides: " + ", ".join(missing_guides))
for relative in shared_guides:
    content = (root / relative).read_text(encoding="utf-8").lower()
    if "scikit-learn" in content or "nearest neighbor" in content or "knn" in content:
        raise SystemExit(f"Unrelated model discussion found in {relative}")

example_pages = [
    "docs/examples/index.md",
    "docs/examples/classification.md",
    "docs/examples/regression.md",
    "docs/examples/isolation.md",
    "docs/examples/multivariate.md",
    "docs/examples/unequal_length.md",
    "docs/examples/missing_values.md",
    "docs/examples/reference_retention.md",
]
missing_examples = [relative for relative in example_pages if not (root / relative).is_file()]
if missing_examples:
    raise SystemExit("Missing examples: " + ", ".join(missing_examples))
readme = (root / "README.md").read_text(encoding="utf-8")
if len(readme.splitlines()) > 100:
    raise SystemExit("README should remain a concise project landing page")
if "reference-backed training-data retention is not yet exposed" in readme:
    raise SystemExit("README contains a stale reference-retention limitation")

if not (root / "python/tests/test_documentation_examples.py").is_file():
    raise SystemExit("Missing executable documentation example tests")
if not (root / "scripts/audit_documentation.py").is_file():
    raise SystemExit("Missing final documentation audit")

