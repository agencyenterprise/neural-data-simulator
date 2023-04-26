# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Neural Data Simulator"
copyright = "2023, AE Studio & Chadwick Boulay"
author = "AE Studio & Chadwick Boulay"
release = "0.1.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../src")))

spelling_lang = "en_US"
tokenizer_lang = "en_US"
spelling_word_list_filename = ["../../.github/allowed_word_list.txt"]

extensions = ["sphinx.ext.autodoc"]
extensions.append("sphinx.ext.mathjax")
extensions.append("sphinx.ext.viewcode")
extensions.append("myst_parser")
extensions.append("sphinxcontrib.spelling")
extensions.append("sphinx_gallery.gen_gallery")
extensions.append("sphinx.ext.autosectionlabel")
# Support for NumPy and Google style docstrings
extensions.append("sphinx.ext.napoleon")

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
}

suppress_warnings = ["autosectionlabel.*"]

myst_heading_anchors = 7
myst_enable_extensions = ["attrs_inline"]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}