"""
.. _ref_1101:

Force control
^^^^^^^^^^^^^

.. code-block::

   #           F/\/\/\/\/\/\       
   #            ||||||||||||  
   #            *----------*  
   #            |          | 
   #            |          |
   #            |          |
   #            |          |
   #            |          | 
   #            *----------*
   #            /_\/_\/_\/_\       
   #     |Y    /////////////
   #     |
   #      ---X
   #  Z /


+----+---------+--------+
|    | VALUE   | UNITS  |
+====+=========+========+
| E  | 210     | kN/mm2 |
+----+---------+--------+
| nu | 0.3     | [-]    |
+----+---------+--------+

"""


###############################################################################
# Import necessary libraries
# --------------------------
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import dolfinx
import mpi4py
import petsc4py
import os


###############################################################################
# Import from phasefieldx package
# -------------------------------
from phasefieldx.Element.Elasticity.Input import Input
from phasefieldx.Element.Elasticity.solver.solver import solve
from phasefieldx.Boundary.boundary_conditions import bc_x, bc_y, bc_xy, get_ds_bound_from_marker
from phasefieldx.Loading.loading_functions import loading_Txy
from phasefieldx.PostProcessing.ReferenceResult import AllResults


###############################################################################
# Parameters definition
# ---------------------
Data = Input(E=210.0,
             nu=0.3,
             save_solution_xdmf=False,
             save_solution_vtu=True,
             results_folder_name="1101_force_control")


###############################################################################
# Mesh definition
# ---------------
msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                                    [np.array([0, 0]),
                                     np.array([1, 1])],
                                    [10, 10],
                                    cell_type=dolfinx.mesh.CellType.quadrilateral)


def bottom(x):
    return np.isclose(x[1], 0)


def top(x):
    return np.isclose(x[1], 1)


fdim = msh.topology.dim - 1

bottom_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, bottom)
top_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, top)

ds_bottom = get_ds_bound_from_marker(top_facet_marker, msh, fdim)
ds_top = get_ds_bound_from_marker(top_facet_marker, msh, fdim)

ds_list = np.array([
                   [ds_bottom, "bottom"],
                   [ds_top, "top"]
                   ])


###############################################################################
# Function Space Definition
# -------------------------
# Define function spaces for the displacement field using Lagrange elements of
# degree 1.
V_u = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))


###############################################################################
# Boundary Conditions
bc_bottom = bc_xy(bottom_facet_marker, V_u, fdim)
bcs_list_u = [bc_bottom]


def update_boundary_conditions(bcs, time):
    return 0, 0, 0


###############################################################################
# External
T_top = loading_Txy(V_u, msh, ds_top)

T_list_u = [[T_top, ds_top]]


def update_loading(T_list_u, time):
    val = 0.1 * time
    T_list_u[0][0].value[1] = petsc4py.PETSc.ScalarType(val)
    return 0, val, 0


f = None

###############################################################################
# Call the solver
final_time = 10.0
dt = 1.0

solve(Data,
      msh,
      final_time,
      V_u,
      bcs_list_u,
      update_boundary_conditions,
      f,
      T_list_u,
      update_loading,
      ds_list,
      dt)


###############################################################################
# Load results
# ------------
# Once the simulation finishes, the results are loaded from the results folder.
# The AllResults class takes the folder path as an argument and stores all
# the results, including logs, energy, convergence, and DOF files.
# Note that it is possible to load results from other results folders to compare results.
# It is also possible to define a custom label and color to automate plot labels.
S = AllResults(Data.results_folder_name)
S.set_label('Simulation')
S.set_color('b')

###############################################################################
# Plot: displacement $\boldsymbol u$
# ----------------------------------
# The displacement result saved in the .vtu file is shown.
# For this, the file is loaded using PyVista.
pv.start_xvfb()
file_vtu = pv.read(os.path.join(Data.results_folder_name, "paraview-solutions_vtu", "phasefieldx_p0_000009.vtu"))
file_vtu.plot(scalars='u', cpos='xy', show_scalar_bar=True, show_edges=False)


###############################################################################
# Steps vs. Reaction Force
# ------------------------
# Plot the steps versus the reaction force.

steps = S.dof_files["top.dof"]["#step"]

###############################################################################
# Plot steps vs reaction force
fig, ax = plt.subplots()

ax.plot(steps, S.reaction_files['top.reaction']["Ry"], S.color + '.', linewidth=2.0, label=S.label)

ax.grid(color='k', linestyle='-', linewidth=0.3)
ax.set_xlabel('steps')
ax.set_ylabel('reaction force - F $[kN]$')
ax.legend()


plt.show()
