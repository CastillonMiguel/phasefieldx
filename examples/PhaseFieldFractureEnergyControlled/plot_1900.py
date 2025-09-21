r"""
.. _ref_phase_field_central_cracked_simulation_3_a05_l1:

Simulation 3
------------
The model represents a square plate with a central crack, as shown in the figure below. The bottom part is fixed in all directions, while the upper part can slide vertically. A vertical displacement is applied at the top. The geometry and boundary conditions are depicted in the figure. We discretize the model with quadrilateral elements.

.. note::
   In this case, only one quarter of the model will be considered due to symmetry. Additionally, a regular mesh will be used.

.. code-block::

   #           u/\/\/\/\/\/\       u/\/\/\/\/\/\ 
   #            ||||||||||||        ||||||||||||
   #            *----------*    o|\ *----------*
   #            |          |    o|/ |          |
   #            | 2a=1.0   |    o|\ | a=a0     |
   #            |   ----   |    o|/ *----------*
   #            |          |             /_\/_\ 
   #            |          |            oo oo oo
   #            *----------*
   #            /_\/_\/_\/_\       
   #     |Y    /////////////
   #     |
   #     *---X


The Young's modulus, Poisson's ratio, and the critical energy release rate are given in the table :ref:`Properties <table_properties_label>`. Young's modulus $E$ and Poisson's ratio $\nu$ can be represented with the Lamé parameters as: $\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}$; $\mu=\frac{E}{2(1+\nu)}$.

.. _table_properties_label:

+----+---------+--------+
|    | VALUE   | UNITS  |
+====+=========+========+
| E  | 210     | kN/mm2 |
+----+---------+--------+
| nu | 0.3     | [-]    |
+----+---------+--------+
| Gc | 0.0027  | kN/mm  |
+----+---------+--------+
| l  | 0.015   | mm     |
+----+---------+--------+


"""

###############################################################################
# Import necessary libraries
# --------------------------
import numpy as np
import dolfinx
import mpi4py
import petsc4py
import os

###############################################################################
# Import from phasefieldx package
# -------------------------------
from phasefieldx.Element.Phase_Field_Fracture.Input import Input
from phasefieldx.Element.Phase_Field_Fracture.solver.solver_ener_non_variational import solve
from phasefieldx.Boundary.boundary_conditions import bc_y, bc_x, get_ds_bound_from_marker
from phasefieldx.PostProcessing.ReferenceResult import AllResults

print(1)
