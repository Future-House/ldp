# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinx_nameko_theme
import sys,os

project = 'ldp'
copyright = '2024, James Braza, Siddharth Narayanan'
author = 'James Braza, Siddharth Narayanan'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode'
]
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
sys.path.insert(0, os.path.abspath('../../'))


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'nameko'
html_theme_path = [sphinx_nameko_theme.get_html_theme_path()]