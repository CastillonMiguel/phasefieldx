# phasefieldx/Element/Phase_Field/__init__.py

# Import submodules to be included in the package namespace

from .Input import *
from .solver import *

# Optionally, you can specify which symbols to export when using 'from phasefieldx import *'
__all__ = [
    'solver',
]