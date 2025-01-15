# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyvene'
copyright = '2024, Stanford NLP'
author = 'Stanford NLP'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinxcontrib.collections',
    'myst_nb',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for myst -------------------------------------------------------
# myst_enable_extensions = [
#     'amsmath',
#     'dollarmath',
# ]
# nb_execution_mode can be overridden on a notebook level via .ipynb metadata
nb_execution_mode = 'off'
nb_execution_allow_errors = False
# nb_execution_excludepatterns = ['notebooks/*']

# -- Options for autodoc / autosummary -----------------------------------------

# autoclass_content = 'class'
# autodoc_class_signature = 'separated'
# autodoc_default_options = {
#     'exclude-members': (
#         '__repr__, __str__, __weakref__, __hash__, __delattr__, __eq__,'
#         ' __setattr__, __subclasshook__'
#     ),
# }
# autodoc_inherit_docstrings = False

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
add_module_names = False # Remove namespaces from class/method signatures

# -- Options for sphinx-collections

collections_target = ''
collections = {
    'notebooks': {
        'driver': 'copy_folder',
        'source': os.path.abspath('../../tutorials/'),
        'target': 'tutorials/',
        'ignore': [],
    },
    'notebooks2': {
        'driver': 'copy_file',
        'source': os.path.abspath('../../pyvene_101.ipynb'),
        'target': 'tutorials/pyvene_101.ipynb',
        'ignore': [],
    },
    'contributing': {
        'driver': 'copy_file',
        'source': os.path.abspath('../../CONTRIBUTING.md'),
        'target': 'guides/contributing.md',
        'ignore': [],
    },
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
