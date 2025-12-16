# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from unittest.mock import MagicMock

# Make sure the project root (containing the `mcdc/` package) is importable
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to import mcdc from the source tree / installed package
try:
    import mcdc  # noqa: F401
except ImportError:
    # As a last-resort for CI: mock mcdc so autosummary doesn't crash
    # (better to have slightly-empty API docs than a failing build)
    sys.modules["mcdc"] = MagicMock()

# On Read the Docs / CI, mock heavy optional dependencies
MOCK_MODULES = [
    "mpi4py",
    "mpi4py.util.dtlib",
    "colorama",
    "cvxpy",
    "sympy",
    "matplotlib.pyplot",
]
sys.modules.update((mod_name, MagicMock()) for mod_name in MOCK_MODULES)

# If you still need MPI.COMM_WORLD in docs build:
from mpi4py import MPI

MPI.COMM_WORLD.Get_size.return_value = 1
MPI.COMM_WORLD.Get_rank.return_value = 0


# -- Project information -----------------------------------------------------

project = "MC/DC"
copyright = "2025, Center for Exascale Monte Carlo Neutron Transport (CEMeNT)"
author = "Center for Exascale Monte Carlo Neutron Transport (CEMeNT)"

# The full version, including alpha/beta/rc tags
release = " "


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_toolbox.github",
    "sphinx_toolbox.sidebar_links",
    "sphinx.ext.autosectionlabel",
]
autosummary_generate = True


github_username = "CEMeNT-PSAAP"
github_repository = "MCDC"
github_url = "https://github.com/{github_username}/{github_repository}"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_logo = "images/home/mcdc.svg"

# html_permalinks = ['https://cement-psaap.github.io/', 'https://github.com/CEMeNT-PSAAP/MCDC']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
