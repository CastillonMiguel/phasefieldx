"""
.. _ref_1713:

Symmetry: Single edge notched tension test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A well-known benchmark simulation in fracture mechanics is performed, relying on the simulation conducted by [Miehe]_. This simulation considers an anisotropic formulation with spectral energy decomposition, although we have also repeated the simulations with the isotropic formulation.

The model consists of a square plate with a notch located halfway up, extending from the left to the center, as shown in the figure below. The bottom part is fixed in all directions, while the upper part can slide vertically. A vertical displacement is applied at the top. The geometry and boundary conditions are depicted in the figure. We discretize the model with triangular elements, refining the areas (element size h) where crack evolution is expected. The element size h must be sufficiently small to avoid mesh dependencies.

.. note::
   In this case, only one half of the model will be considered due to symmetric conditions. Additionally, a regular mesh will be used.

.. code-block::

   #           u/\/\/\/\/\/\      u/\/\/\/\/\/\  
   #            ||||||||||||       ||||||||||||
   #            *----------*       *----------*
   #            |          |       |          |
   #            | a=0.5    |       | a=0.5    |
   #            |---       |       *----------* 
   #            |          |             /_\/_\ 
   #            |          |            ///////
   #            *----------*
   #            /_\/_\/_\/_\       
   #     |Y    /////////////
   #     |
   #     *---X


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
# We consider isotropic degradation where all the energy is degraded and irreversibility as proposed by Miehe.
Data = Input(E=210.0,   # young modulus
             nu=0.3,    # poisson
             Gc=0.0027,  # critical energy release rate
             l=0.015,   # lenght scale parameter
             degradation="isotropic",  # "isotropic" "anisotropic"
             split_energy="no",       # "spectral" "deviatoric"
             degradation_function="quadratic",
             irreversibility="miehe",  # "miehe"
             fatigue=False,
             fatigue_degradation_function="asymptotic",
             fatigue_val=0.05625,
             k=0.0,
             min_stagger_iter=2,
             max_stagger_iter=600,
             stagger_error_tol=1e-8,
             save_solution_xdmf=False,
             save_solution_vtu=True,
             results_folder_name="1713_Symmetry_Single_Edge_Notched_Tension_Test")


###############################################################################
# Mesh Definition
# ---------------
nx = 160
ny = 80
msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                                    [np.array([0, 0]),
                                     np.array([1, 0.5])],
                                    [nx, ny],
                                    cell_type=dolfinx.mesh.CellType.quadrilateral)

fdim = msh.topology.dim - 1


def bottom(x):
    return np.logical_and(np.isclose(x[1], 0), np.greater(x[0], 0.5))


def top(x):
    return np.isclose(x[1], 0.5)


bottom_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, bottom)
top_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, top)

ds_bottom = get_ds_bound_from_marker(bottom_facet_marker, msh, fdim)
ds_top = get_ds_bound_from_marker(top_facet_marker, msh, fdim)

ds_list = np.array([
                   [ds_bottom, "bottom"],
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
# Apply boundary conditions: bottom nodes fixed in both directions, top nodes can slide vertically.
bc_bottom = bc_xy(bottom_facet_marker, V_u, fdim)
bc_top = bc_y(top_facet_marker, V_u, fdim)
bcs_list_u = [bc_top, bc_bottom]


def update_boundary_conditions(bcs, time):
    dt0 = 0.5 * 10**-4
    if time <= 50:
        val = dt0 * time
    else:
        val = 50 * dt0 + dt0 / 10 * (time - 50)
    bcs[0].g.value[...] = petsc4py.PETSc.ScalarType(val)
    return 0, val, 0


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
file_vtu = pv.read(os.path.join(Data.results_folder_name, "paraview-solutions_vtu", "phasefieldx_p0_000149.vtu"))
pv.start_xvfb()
file_vtu.plot(scalars='phi', cpos='xy', show_scalar_bar=True, show_edges=False)


###############################################################################
# Plot: displacement $\boldsymbol u$
# ----------------------------------
# The displacements results saved in the .vtu file are shown.
# For this, the file is loaded using PyVista.
file_vtu = pv.read(os.path.join(Data.results_folder_name, "paraview-solutions_vtu", "phasefieldx_p0_000149.vtu"))
file_vtu.plot(scalars='u', cpos='xy', show_scalar_bar=True, show_edges=False)


###############################################################################
# Plot: Displacement vs Fracture Energy
# -------------------------------------
# The vertical displacement is saved in S.dof_files["top.dof"]["Uy"]. (Note: This simulation considers half of the model due to symmetry, so the results are multiplied by 2.)
displacement = 2 * S.dof_files["top.dof"]["Uy"]

fig, energyW = plt.subplots()

energyW.plot(displacement, 2 * S.energy_files['total.energy']["W"], 'b-', linewidth=2.0, label=r'$W$')
energyW.plot(displacement, 2 * S.energy_files['total.energy']["W_phi"], 'y-', linewidth=2.0, label=r'$W_{\phi}$')
energyW.plot(displacement, 2 * S.energy_files['total.energy']["W_gradphi"], 'g-', linewidth=2.0, label=r'$W_{\nabla \phi}$')

energyW.grid(color='k', linestyle='-', linewidth=0.3)
energyW.set_xlabel('displacement - u $[mm]$')
energyW.set_ylabel('Energy')
energyW.legend()


###############################################################################
# Plot: Force vs Vertical Displacement
# ------------------------------------
Miehe = np.loadtxt(os.path.join("reference_solutions", "miehe_solution.csv"))

fig, ax_reaction = plt.subplots()

ax_reaction.plot(Miehe[:, 0], Miehe[:, 1], 'g-', linewidth=2.0, label='Miehe')
ax_reaction.plot(displacement, S.reaction_files['bottom.reaction']["Ry"], 'k.', linewidth=2.0, label=S.label)

ax_reaction.grid(color='k', linestyle='-', linewidth=0.3)
ax_reaction.set_xlabel('displacement - u $[mm]$')
ax_reaction.set_ylabel('reaction force - F $[kN]$')
ax_reaction.set_title('Reaction Force vs Vertical Displacement')
ax_reaction.legend()


###############################################################################
# Plot: Staggered Iterations vs Vertical Displacement
# ---------------------------------------------------
fig, ax_convergence = plt.subplots()

ax_convergence.plot(displacement, S.convergence_files["phasefieldx.conv"]["stagger"], 'k.', linewidth=2.0, label='Stagger iterations')

ax_convergence.grid(color='k', linestyle='-', linewidth=0.3)
ax_convergence.set_xlabel('displacement - u $[mm]$')
ax_convergence.set_ylabel('stagger iterations - []')
ax_convergence.set_title('Stagger iterations vs vertical displacement')
ax_convergence.legend()

plt.show()
