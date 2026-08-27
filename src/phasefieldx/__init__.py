# phasefieldx/__init__.py

__version__ = "0.4.0"
__author__ = "Miguel Castillón"
__email__ = "miguel.research@pm.me"
__license__ = "MIT"
__title__ = "phasefieldx"
__description__ = "PhaseFieldX: An Open-Source Framework for Advanced Phase-Field Simulations"
__url__ = "https://github.com/CastillonMiguel/phasefieldx"
__documentation__ = "https://phasefieldx.readthedocs.io"
__copyright__ = "Copyright (c) 2023 Miguel Castillón"

# Import submodules to be included in the package namespace
from .Boundary import *
from .Element import *
from .Loading import *
from .Logger import *
from .Materials import *
from .Math import *
from .PostProcessing import *
from .Reactions import *
from .errors_functions import *
from .files import *
from .norms import *

# Optionally, you can specify which symbols to export when using 'from
# phasefieldx import *'
__all__ = [
    'Boundary',
    'Element',
    'Loading',
    'Logger',
    'Materials',
    'Math',
    'PostProcessing',
    'Reactions',
    'errors_functions',
    'files',
    'norms'
]
