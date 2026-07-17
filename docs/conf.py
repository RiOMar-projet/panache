import os
import sys
from importlib.metadata import version as _version

sys.path.insert(0, os.path.abspath("../src"))

project = "panache"
copyright = "2026, Louis Terrats, Robert W. Schlegel"
author = "Louis Terrats, Robert W. Schlegel"
release = _version("panache-riomar")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True

myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/RiOMar-projet/panache",
    "show_toc_level": 2,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/panache-riomar/",
            "icon": "fa-solid fa-box",
        },
    ],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
