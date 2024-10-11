"""
.. _ref_1712:

Single edge notched shear test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A well-known benchmark simulation in fracture mechanics is performed, based on the work by [Miehe]_.

The model consists of a square plate with a notch located halfway up, extending from the left to the center, as shown in the figure below. The bottom part is fixed in all directions, while the upper part can slide vertically. A horizontal displacement is applied at the top. The geometry and boundary conditions are depicted in the figure. We discretize the model by refining the areas where crack evolution is expected, ensuring the element size h is sufficiently small to avoid mesh dependencies.

.. code-block::

   #            u
   #            =>=>=>=>=>=>
   #            *----------*
   #            |          |
   #            | a=0.5    |
   #            |---       |
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
| Gc | 0.0027  | kN/mm2 |
+----+---------+--------+
| l  | 0.015   | mm     |
+----+---------+--------+

.. [Miehe] A phase field model for rate-independent crack propagation: Robust algorithmic implementation based on operator splits, https://doi.org/10.1016/j.cma.2010.04.011.

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
from phasefieldx.Element.Phase_Field_Fracture.Input import Input
from phasefieldx.Element.Phase_Field_Fracture.solver.solver import solve
from phasefieldx.Boundary.boundary_conditions import bc_xy, bc_y, get_ds_bound_from_marker
from phasefieldx.PostProcessing.ReferenceResult import AllResults


###############################################################################
# Parameters Definition
# ---------------------
# Here, we define the class containing the input parameters. The material parameters are:
# Young's modulus $E = 210$, $\text{kN/mm}^2$, Poisson's ratio $\nu = 0.3$,
# critical energy release rate $G_c = 0.0027$, $\text{kN/mm}^2$, and length scale parameter $l = 0.015$, $\text{mm}$.
# We consider anisotropic degradation (spectral) and irreversibility as proposed by Miehe.
Data = Input(E=210.0,    # young modulus
             nu=0.3,     # poisson
             Gc=0.0027,  # critical energy release rate
             l=0.05,     # lenght scale parameter
             degradation="anisotropic",  # "isotropic" "anisotropic"
             split_energy="spectral",    # "spectral" "deviatoric"
             degradation_function="quadratic",
             irreversibility="miehe",  # "miehe"
             fatigue=False,
             fatigue_degradation_function="asymptotic",
             fatigue_val=0.05625,
             k=0.0,
             min_stagger_iter=2,
             max_stagger_iter=500,
             stagger_error_tol=1e-3,
             save_solution_xdmf=False,
             save_solution_vtu=True,
             results_folder_name="1712_Single_Edge_Notched_Shear_Test")


###############################################################################
# Mesh definition
msh_file = os.path.join("mesh", "mesh_shear.msh")
gdim = 2
gmsh_model_rank = 0
mesh_comm = mpi4py.MPI.COMM_WORLD

msh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(msh_file, mesh_comm, gmsh_model_rank, gdim)

fdim = msh.topology.dim - 1

bottom_facet_marker = facet_markers.find(9)
top_facet_marker = facet_markers.find(10)
right_facet_marker = facet_markers.find(11)
left_facet_marker = facet_markers.find(12)

ds_bottom = get_ds_bound_from_marker(bottom_facet_marker, msh, fdim)
ds_top = get_ds_bound_from_marker(top_facet_marker, msh, fdim)
ds_right = get_ds_bound_from_marker(right_facet_marker, msh, fdim)
ds_left = get_ds_bound_from_marker(left_facet_marker, msh, fdim)

ds_list = np.array([
                   [ds_bottom, "bottom"],
                   [ds_top, "top"],
                   [ds_right, "right"],
                   [ds_left, "left"]
                   ])


###############################################################################
# Function Space definition
V_u = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))
V_phi = dolfinx.fem.functionspace(msh, ("Lagrange", 1))


###############################################################################
# Boundary Conditions
# -------------------
# Boundary conditions applied: Bottom nodes fixed in both directions, while top nodes are allowed to slide horizontally.
bc_bottom = bc_xy(bottom_facet_marker, V_u, fdim)
bc_top = bc_xy(top_facet_marker, V_u, fdim)
bc_left = bc_y(left_facet_marker, V_u, fdim)
bc_right = bc_y(right_facet_marker, V_u, fdim)

bcs_list_u = [bc_top, bc_bottom, bc_left, bc_right]


def update_boundary_conditions(bcs, time):
    dt0 = 10**-4
    val = time * dt0
    bcs_list_u[0].g.value[0] = petsc4py.PETSc.ScalarType(val)
    return val, 0, 0


T_list_u = None
update_loading = None
f = None
T = dolfinx.fem.Constant(msh, petsc4py.PETSc.ScalarType((0.0, 0.0)))
f = dolfinx.fem.Constant(msh, petsc4py.PETSc.ScalarType((0.0, 0.0)))

###############################################################################
# Boundary Conditions four phase field
bcs_list_phi = []


###############################################################################
# Call the Solver
# ---------------
# The problem is solved for a final time of 200. The solver will handle the mesh, boundary conditions,
# and the given parameters to compute the solution.

dt = 1.0
final_time = 150.0

# Uncomment the following lines to run the solver with the specified parameters
# solve(Data,
#       msh,
#       final_time,
#       V_u,
#       V_phi,
#       bcs_list_u,
#       bcs_list_phi,
#       update_boundary_conditions,
#       f,
#       T_list_u,
#       update_loading,
#       ds_list,
#       dt,
#       path=None)


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
# Plot: phase-field $\phi$
# ------------------------
# The phase-field result saved in the .vtu file is shown.
# For this, the file is loaded using PyVista.
file_vtu = pv.read(os.path.join(Data.results_folder_name, "paraview-solutions_vtu", "phasefieldx_p0_000097.vtu"))
file_vtu.plot(scalars='phi', cpos='xy', show_scalar_bar=True, show_edges=False)


###############################################################################
# Plot: displacement $\boldsymbol u$
# ----------------------------------
# The displacements results saved in the .vtu file are shown.
# For this, the file is loaded using PyVista.
file_vtu = pv.read(os.path.join(Data.results_folder_name, "paraview-solutions_vtu", "phasefieldx_p0_000097.vtu"))
file_vtu.plot(scalars='u', cpos='xy', show_scalar_bar=True, show_edges=False)

plt.show()
