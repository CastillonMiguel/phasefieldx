# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
sys.path.insert(0, os.path.abspath('../../src/phasefieldx'))


# -- Project information -----------------------------------------------------
project = 'PhaseFieldX'
copyright = '2023, Miguel Castillón'
author = 'Miguel Castillón'

# The full version, including alpha/beta/rc tags
release = '0.1'
language = "en"


# -- pyvista configuration ---------------------------------------------------
import pyvista
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.docs import linkcode_resolve, pv_html_page_context  # noqa: F401

# Manage errors
pyvista.set_error_output_file("errors.txt")
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = [1024, 768]
pyvista.global_theme.font.size = 22
pyvista.global_theme.font.label_size = 22
pyvista.global_theme.font.title_size = 22
pyvista.global_theme.return_cpos = False
pyvista.set_jupyter_backend(None)


# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
os.environ['PYVISTA_BUILDING_GALLERY'] = 'true'

# SG warnings
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.",
)

# Prevent deprecated features from being used in examples
warnings.filterwarnings(
    "error",
    category=PyVistaDeprecationWarning,
)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

from sphinx_gallery.sorting import FileNameSortKey

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'sphinx_math_dollar',
    'sphinx_gallery.gen_gallery',
    'pyvista.ext.plot_directive',  #'matplotlib.sphinxext.plot_directive',
    "pyvista.ext.viewer_directive",
    "sphinx_design"
]
   
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

class ResetPyVista:
    """Reset pyvista module to default settings."""

    def __call__(self, gallery_conf, fname):
        """Reset pyvista module to default settings

        If default documentation settings are modified in any example, reset here.
        """
        import pyvista

        pyvista._wrappers['vtkPolyData'] = pyvista.PolyData
        pyvista.set_plot_theme('document')

    def __repr__(self):
        return 'ResetPyVista'


reset_pyvista = ResetPyVista()

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="trame.app")
nbsphinx_execute = 'auto'
sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    # 'image_scrapers': ("pyvista", "matplotlib"),
     #'filename_pattern': r'.*\.py',
     'filename_pattern': '/plot_',
     'ignore_pattern': r'__init__\.py',
     'ignore_pattern': r'main',
     # Remove sphinx configuration comments from code blocks
    "remove_config_comments": True,
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": FileNameSortKey,
    # directory where function granular galleries are stored
    "backreferences_dir": None,
    # Modules for which function level galleries are created.  In
    "doc_module": "pyvista",
    "image_scrapers": (DynamicScraper(), "matplotlib"),
    "first_notebook_cell": "%matplotlib inline",
    "reset_modules": (reset_pyvista,),
    "reset_modules_order": "both",
}

import re

# -- .. pyvista-plot:: directive ----------------------------------------------
from numpydoc.docscrape_sphinx import SphinxDocString

IMPORT_PYVISTA_RE = r'\b(import +pyvista|from +pyvista +import)\b'
IMPORT_MATPLOTLIB_RE = r'\b(import +matplotlib|from +matplotlib +import)\b'

plot_setup = """
from pyvista import set_plot_theme as __s_p_t
__s_p_t('document')
del __s_p_t
"""
plot_cleanup = plot_setup


def _str_examples(self):
    examples_str = "\n".join(self['Examples'])

    if (
        self.use_plots
        and re.search(IMPORT_MATPLOTLIB_RE, examples_str)
        and 'plot::' not in examples_str
    ):
        out = []
        out += self._str_header('Examples')
        out += ['.. plot::', '']
        out += self._str_indent(self['Examples'])
        out += ['']
        return out
    elif re.search(IMPORT_PYVISTA_RE, examples_str) and 'plot-pyvista::' not in examples_str:
        out = []
        out += self._str_header('Examples')
        out += ['.. pyvista-plot::', '']
        out += self._str_indent(self['Examples'])
        out += ['']
        return out
    else:
        return self._str_section('Examples')


SphinxDocString._str_examples = _str_examples

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_context = {
   "default_mode": "light"   # light dark auto
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = '_static/logo.png'

# Disable the "Show Source" link
html_show_sourcelink = False
html_theme_options = {
    "external_links": [],
    "footer_start": ["sphinx-version"],
    "github_url": "https://github.com/CastillonMiguel/phasefieldx",
    "navbar_align": "left",
    'icon_links': [
        {
            'name': 'Contributing',
            'url': 'https://github.com/CastillonMiguel/phasefieldx/blob/main/CONTRIBUTING.rst',
            'icon': 'fa fa-gavel fa-fw',
        },
        {
            'name': 'The Paper',
            'url': 'https://doi.org/10.21105/joss.07307',
            'icon': 'fa fa-file-text fa-fw',
        },
    ],
}
