# phasefieldx/Element/__init__.py

# Import submodules to be included in the Element package namespace
from .Allen_Cahn import *
from .Elasticity import *
from .Phase_Field import *
from .Phase_Field_Fracture import *

# Optionally, you can specify which symbols to export when using 'from
# phasefieldx.Element import *'
__all__ = [
    'Allen_Cahn',
    'Elasticity',
    'Phase_Field',
    'Phase_Field_Fracture'
]
