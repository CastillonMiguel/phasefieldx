"""
.. _ref_1714:

Three point bending test
^^^^^^^^^^^^^^^^^^^^^^^^

A well-known benchmark simulation in fracture mechanics is performed, relying on the simulation conducted by [Miehe]_. This simulation considers an anisotropic formulation with spectral energy decomposition.

A rectangular plate with an initial notch is located halfway down, extending from the left to the center, as shown in the figure below. This beam is supported at its ends, as shown in the figure. The bottom left part is fixed in all directions, while the bottom right part is fixed in the vertical direction. A vertical displacement is applied at the top. The geometry and boundary conditions are depicted in the figure. We discretize the model with triangular elements, refining the areas (element size h) where crack evolution is expected. The element size h must be sufficiently small to avoid mesh dependencies.

.. code-block::
      
   #                            u||||      
   #                             \/\/               
   #            ------------------------------------* 
   #            |                                   | 
   #            |                                   |
   #            |                                   |
   #            |                 /\                | 
   #            *-----------------  ----------------*
   #            /_\/_\        (0,0,0)          /_\/_\       
   #    |Y     ///////                         oo  oo
   #    |
   #    ---X
   # Z /


+----+---------+--------+
|    | VALUE   | UNITS  |
+====+=========+========+
| E  | 20.8    | kN/mm2 |
+----+---------+--------+
| nu | 0.3     | [-]    |
+----+---------+--------+
| Gc | 0.0005  | kN/mm2 |
+----+---------+--------+
| l  | 0.006   | mm     |
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
# Young's modulus $E = 20.8$, $\text{kN/mm}^2$, Poisson's ratio $\nu = 0.3$,
# critical energy release rate $G_c = 0.0005$, $\text{kN/mm}^2$, and length scale parameter $l = 0.06$, $\text{mm}$.
# We consider anisotropic degradation (spectral) and irreversibility as proposed by Miehe.
Data = Input(E=20.8,     # young modulus
             nu=0.3,     # poisson
             Gc=0.0005,  # critical energy release rate
             l=0.06,     # lenght scale parameter
             degradation="anisotropic",  # "isotropic" "anisotropic"
             split_energy="spectral",   # "spectral" "deviatoric"
             degradation_function="quadratic",
             irreversibility="miehe",  # "miehe"
             fatigue=False,
             fatigue_degradation_function="asymptotic",
             fatigue_val=0.05625,
             k=0.0,
             min_stagger_iter=2,
             max_stagger_iter=500,
             stagger_error_tol=1e-8,
             save_solution_xdmf=False,
             save_solution_vtu=True,
             results_folder_name="1714_Three_point_bending")


###############################################################################
# Mesh Definition
# ---------------
msh_file = os.path.join("mesh", "three_point_bending_test.msh")
gdim = 2
gmsh_model_rank = 0
mesh_comm = mpi4py.MPI.COMM_WORLD

msh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(msh_file, mesh_comm, gmsh_model_rank, gdim)

fdim = msh.topology.dim - 1


def bottom_left(x):
    return np.logical_and(np.isclose(x[1], 0), np.less(x[0], -3.9))


def bottom_right(x):
    return np.logical_and(np.isclose(x[1], 0), np.greater(x[0], 3.9))


def top(x):
    return np.logical_and(np.logical_and(np.isclose(x[1], 2), np.greater(x[0], -0.25)), np.less(x[0], 0.25))


fdim = msh.topology.dim - 1
bottom_left_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, bottom_left)
bottom_right_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, bottom_right)
top_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, top)

ds_bottom_left = get_ds_bound_from_marker(bottom_left_facet_marker, msh, fdim)
ds_bottom_right = get_ds_bound_from_marker(bottom_right_facet_marker, msh, fdim)
ds_top = get_ds_bound_from_marker(top_facet_marker, msh, fdim)

ds_list = np.array([
                   [ds_bottom_left, "bottom_left"],
                   [ds_bottom_right, "bottom_right"],
                   [ds_top, "top"]
                   ])


###############################################################################
# Function Space Definition
# -------------------------
# Define function spaces for displacement and phase-field using Lagrange elements.
V_u = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))
V_phi = dolfinx.fem.functionspace(msh, ("Lagrange", 1))


###############################################################################
# Boundary Conditions
# -------------------
# Apply boundary conditions: Bottom-left nodes fixed in both directions, bottom-right nodes fixed in the vertical direction, while top nodes can slide vertically.
fdim = msh.topology.dim - 1
bc_bottom_left = bc_xy(bottom_left_facet_marker, V_u, fdim)
bc_bottom_right = bc_y(bottom_right_facet_marker, V_u, fdim)
bc_top = bc_y(top_facet_marker, V_u, fdim)

bcs_list_u = [bc_top, bc_bottom_left, bc_bottom_right]


def update_boundary_conditions(bcs, time):
    dt0 = 10**-3
    if time <= 36:
        val = dt0 * time
    else:
        val = 36 * dt0 + dt0 / 10 * (time - 36)
    bcs[0].g.value[...] = petsc4py.PETSc.ScalarType(-val)
    return 0, val, 0


T_list_u = None
update_loading = None
f = None
T = dolfinx.fem.Constant(msh, petsc4py.PETSc.ScalarType((0.0, 0.0)))


###############################################################################
# Boundary Conditions four phase field
bcs_list_phi = []


###############################################################################
# Call the Solver
# ---------------
# The problem is solved for a final time of 200. The solver will handle the mesh, boundary conditions,
# and the given parameters to compute the solution.

final_time = 150
dt = 1

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
file_vtu = pv.read(os.path.join(Data.results_folder_name, "paraview-solutions_vtu", "phasefieldx_p0_000065.vtu"))
pv.start_xvfb()
file_vtu.plot(scalars='phi', cpos='xy', show_scalar_bar=True, show_edges=False)


###############################################################################
# Plot: displacement $\boldsymbol u$
# ----------------------------------
# The displacements results saved in the .vtu file are shown.
# For this, the file is loaded using PyVista.
file_vtu = pv.read(os.path.join(Data.results_folder_name, "paraview-solutions_vtu", "phasefieldx_p0_000065.vtu"))
pv.start_xvfb()
file_vtu.plot(scalars='u', cpos='xy', show_scalar_bar=True, show_edges=False)

plt.show()
