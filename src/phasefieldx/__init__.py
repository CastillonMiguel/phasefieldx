# phasefieldx/__init__.py

__version__ = "0.1.0"
__author__ = "Miguel Castillón"
__email__ = "phasefieldx@gmail.com"
__license__ = "MIT"
__description__ = "PhaseFieldX: An Open-Source Framework for Advanced Phase-Field Simulations"
__url__ = "https://github.com/CastillonMiguel/phasefieldx"

# Import submodules to be included in the package namespace
from .Boundary import *
from .Element import *
from .Loading import *
from .Logger import *
from .Materials import *
from .Math import *
from .PostProcessing import *
from .solvers import *
from .errors_functions import *
from .files import *
from .norms import *

# Optionally, you can specify which symbols to export when using 'from phasefieldx import *'
__all__ = [
    'Boundary',
    'Element',
    'Loading',
    'Logger',
    'Materials',
    'Math',
    'PostProcessing',
    'solvers',
    'errors_functions',
    'files',
    'norms'
]
