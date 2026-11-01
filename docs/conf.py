from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

project = "proximity-forests"
author = "The proximity-forests developers"
try:
    release = version("proximity-forests")
except PackageNotFoundError:
    release = "0.1.0.dev0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "_templates", "concurrency_future_work.md", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"proximity-forests {release}"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

nitpicky = False
