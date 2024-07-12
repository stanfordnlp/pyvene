# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'pyvene'
copyright = '2024, Stanford University'
author = 'Stanford NLP'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_title = 'pyvene'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_book_theme'

html_theme_options = {
    'repository_url': 'https://github.com/stanfordnlp/pyvene',
    'use_repository_button': True,
    'use_issues_button': True,
    'show_toc_level': 4,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'