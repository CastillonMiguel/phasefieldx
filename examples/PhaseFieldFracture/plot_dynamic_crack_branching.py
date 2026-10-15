r"""
.. _ref_dynamic_notched_tension:

Dynamic crack branching
^^^^^^^^^^^^^^^^^^^^^^^

A well-known benchmark simulation in dynamic fracture mechanics is performed, relying on the simulation conducted by :footcite:t:`phase_field_shear_boundary`. A pre-notched rectangular plate is loaded in tension by a traction applied at the initial time and held constant. The load is large enough to make the crack accelerate and branch. This simulation considers an anisotropic formulation with spectral energy decomposition, a Newmark time integration scheme and plane strain.

The irreversibility of the phase-field variable ($\dot\phi\geq0$, the crack cannot heal) is handled with a penalty approach instead of a history field.

The initial crack is handled **geometrically**. The notch is the part of the symmetry plane ($y=0$, $x<0$) where the vertical displacement is not constrained, so it behaves as a traction-free surface and the stress concentrates at the notch tip. Unlike the reference, which induces the notch with an initial strain-history field, no initial phase field or history field is imposed.

The full plate is $100$ mm x $40$ mm, with a $50$ mm notch at mid-height extending from the left edge to the center, and a traction $\sigma = 1$ MPa on the top and bottom edges. The problem is symmetric about the horizontal mid-plane, so only the upper half ($100$ mm x $20$ mm) is modeled, as shown in the figure below. Ahead of the notch tip the vertical displacement is fixed (symmetry), while the notch faces are traction-free. The element size $h$ must be sufficiently small to resolve the phase-field transition.

.. code-block::

   #             sigma = 1 MPa
   #        /\/\/\/\/\/\/\/\/\/\/\
   #        ||||||||||||||||||||||
   #        *--------------------*   y = 20
   #        |                    |
   #        |    upper half      |
   #        |                    |
   #        |====+---------------*   y = 0
   #        notch   u_y = 0 (symmetry)
   #     |Y
   #     |
   #      ---X
   #


.. image:: ../../../../examples/PhaseFieldFracture/results_dynamic_crack_branching/crack_branching.gif


The simulation uses the units $mm$, $kN$ and $\mu s$ (stress in $kN/mm^2$, density in $kN \mu s^2/mm^4$, velocity in $mm/\mu s$). The material properties are given in the table :ref:`Properties <table_properties_label>`.

.. _table_properties_label:

    +-------+--------+----------------+
    |       | VALUE  | UNITS          |
    +=======+========+================+
    | E     | 32     | kN/mm2         |
    +-------+--------+----------------+
    | nu    | 0.2    | [-]            |
    +-------+--------+----------------+
    | Gc    | 3.0e-6 | kN/mm          |
    +-------+--------+----------------+
    | rho   | 2.45   | kN us^2 / mm^4 |
    +-------+--------+----------------+
    | l     | 0.5    | mm             |
    +-------+--------+----------------+
    | sigma | 1.0e-3 | kN/mm2         |
    +-------+--------+----------------+

.. footbibliography::

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
from phasefieldx.Element.Phase_Field_Fracture.solver.solver_dynamic import solve
from phasefieldx.Boundary.boundary_conditions import bc_y, get_ds_bound_from_marker
from phasefieldx.PostProcessing.ReferenceResult import AllResults
from phasefieldx.Loading.loading_functions import loading_Txy


###############################################################################
# Parameters Definition
# ---------------------
# `Data` is an input object containing essential parameters for simulation setup
# and result storage:
#
# - `E`: Young's modulus, set to 32 $kN/mm^2$.
# - `nu`: Poisson's ratio, set to 0.2.
# - `Gc`: Critical energy release rate, set to 3.0e-6 $kN/mm$.
# - `l`: Length scale parameter, set to 0.5 $mm$.
# - `rho`: Mass density, set to 2.45 $kN \mu s^2/mm^4$.
# - `degradation`: Specifies the degradation type. Options are "isotropic" or "anisotropic".
# - `split_energy`: Controls how the energy is split; options include "no" (default), "spectral," or "deviatoric."
# - `degradation_function`: Specifies the degradation function; here, it is "quadratic."
# - `irreversibility`: Set to "no" because irreversibility is imposed with a penalty term in `solve`.
# - `fatigue`, `fatigue_degradation_function` and `fatigue_val`: Fatigue options, not used here.
# - `k`: Stiffness penalty parameter, set to 0.0.
# - `save_solution_xdmf` and `save_solution_vtu`: Specify the file formats to save results.
#   In this case, results are saved as `.vtu` files.
# - `results_folder_name`: Name of the folder for saving results. If it exists,
#   it will be replaced with a new empty folder.
Data = Input(E=32,    # young modulus
             nu=0.2,     # poisson
             Gc=3.0*10**-6, # critical energy release rate
             l=0.5,    # lenght scale parameter
             rho=2.45,  # density
             degradation="anisotropic",  # "isotropic" "anisotropic"
             split_energy="spectral",       # "spectral" "deviatoric"
             degradation_function="quadratic",
             irreversibility="no",  # "miehe"
             fatigue=False,
             fatigue_degradation_function="asymptotic",
             fatigue_val=0.0,
             k=0.0,
             save_solution_xdmf=False,
             save_solution_vtu=True,
             results_folder_name="results_dynamic_crack_branching",)


###############################################################################
# Mesh Definition
# ---------------
# Only the upper half of the plate is modeled, $x\in[-50, 50]$ mm and
# $y\in[0, 20]$ mm, with the notch tip at the origin. The mesh is uniform, with
# 400 x 80 quadrilaterals, so $h=0.25$ mm.
divx, divy = 400, 80
lx, ly = 100.0, 20.0
msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                                    [np.array([-lx/2, 0.0]),
                                     np.array([ lx/2, ly])],
                                    [divx, divy],
                                    cell_type=dolfinx.mesh.CellType.quadrilateral)

fdim = msh.topology.dim - 1 # Dimension of the mesh facets

# %%
# Boundaries are located with functions of the coordinates:
#
# - `bottom`: The ligament ahead of the notch tip ($y=0$, $x>0$), where symmetry is imposed.
# - `top`: The $y=ly$ boundary, where the traction is applied.
def bottom(x):
    return np.logical_and(np.isclose(x[1], 0.0), np.greater(x[0], 0.0))

def top(x):
    return np.isclose(x[1], ly)

bottom_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, bottom)
top_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, top)

# %%
# The `get_ds_bound_from_marker` function creates the measure of the top boundary.
# `ds_list` associates it with the name `"top"` used to save results.
ds_top = get_ds_bound_from_marker(top_facet_marker, msh, fdim)

ds_list = np.array([
                   [ds_top, "top"],
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
# The only Dirichlet condition is the symmetry condition, $u_y=0$, on the ligament
# (`bc_bottom`). The notch faces and the lateral boundaries are traction-free.
bc_bottom = bc_y(bottom_facet_marker, V_u, fdim)

# %%
# The bcs_list_u variable is a list that stores all boundary conditions for the displacement
# field $\boldsymbol u$. The reaction force is saved for each of them.
bcs_list_u = [bc_bottom]
bcs_list_u_names = ["bottom"]


###############################################################################
# Function: `update_boundary_conditions`
# --------------------------------------
# This function is called at each time step to update the Dirichlet conditions.
# The symmetry condition is constant, so nothing is updated and the imposed
# displacement `(0, 0, 0)` is returned.
def update_boundary_conditions(bcs, time):
    return 0, 0, 0


###############################################################################
# Loading
# -------
# The traction $\boldsymbol T=(0, \sigma)$ is applied on the top boundary. It is
# a step load: $\sigma=1$ MPa $=10^{-3}$ kN/mm² from $t=0$ and held constant.
# `update_loading` is called at each time step to set its value.
T_top = loading_Txy(msh, value_x=0.0, value_y=0.0)

T_list_u = [[T_top, ds_top],
           ]
f = None

def update_loading(T_list_u, time):
    val = 10**-3  # [kN/mm^2]
    T_list_u[0][0].value[1] = petsc4py.PETSc.ScalarType(val)
    return 0, 0, 0


###############################################################################
# Boundary Conditions for phase field
# -----------------------------------
# The initial crack is handled geometrically: the notch ($y=0$, $x<0$) is the part of the
# symmetry plane where the displacement is left free (see `bc_bottom`), so no
# initial phase field, history field or Dirichlet condition on $\phi$ is imposed.
bcs_list_phi = []


###############################################################################
# Solver Call for a Dynamic Phase-Field Fracture Problem
# ------------------------------------------------------
# This section sets up and calls the solver for a dynamic phase-field fracture problem.
#
# **Key Points:**
#
# - Time is integrated with the Newmark average-acceleration scheme. At each time step,
#   displacement and phase field are solved in a staggered way.
# - Irreversibility of the phase field is handled with a penalty approach 
#
# **Parameters:**
#
# - `dt`: Time step [µs]. The reference uses $\Delta t\approx h/v_R$, which is 0.1 µs
#   for $h=0.25$ mm. Here 0.05 µs is used.
# - `final_time`: Total simulation time [µs], set to 80.
# - `rho`: Penalty parameter of the irreversibility.
# - `stagger_error_tol`: With the "Residual" criterion the tolerance is absolute, so it
#   depends on the units. The "L2" criterion does not.
# - `path`: Optional parameter for specifying the folder where results will be saved;
#   here it is set to `None`, meaning results will be saved to the default location.
dt = 0.05  # [us]
final_time = 80  # [us]

# %%
# Uncomment the following lines to run the solver with the specified parameters.

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
#       path=None,
#       bcs_list_u_names=bcs_list_u_names,
#       min_stagger_iter=2,
#       max_stagger_iter=10000,
#       stagger_error_tol=1e-8,
#       case = 'AT2',
#       rho=1e3*Data.Gc,  # irreversibility penalty [kN/mm^2]
#       irreversibility=True,
#       stagger_conv_criterion="Residual")


###############################################################################
# Load results
# ------------
# Once the simulation finishes, the results are loaded from the results folder.
# The AllResults class takes the folder path as an argument and stores all
# the results, including logs, energy, convergence, and DOF files.
S = AllResults(Data.results_folder_name)
S.set_label('Simulation')
S.set_color('b')

# Time [us] of each step, and factor that converts the energy of the half plate in
# kN mm / mm to the full plate in J/m.
time_sol = S.energy_files['total.energy']["#step"]*dt
energy_factor = 2


###############################################################################
# Plot: phase-field $\phi$
# ------------------------
# The phase field at the last step (t = 80 µs) saved in the .vtu file is shown.
# For this, the file is loaded using PyVista.
last_step = len(time_sol) - 1
file_vtu = pv.read(os.path.join(Data.results_folder_name, "paraview-solutions_vtu", f"phasefieldx_p0_{last_step:06d}.vtu"))
file_vtu.plot(scalars='phi', cpos='xy', show_scalar_bar=True, show_edges=False)

###############################################################################
# Plot: Time vs Energy
# --------------------
# The elastic energy $E$.
fig, energy = plt.subplots()

energy.plot(time_sol, energy_factor*S.energy_files['total.energy']["E"], 'b-', linewidth=2.0, label=r'$E$')

energy.grid(color='k', linestyle='-', linewidth=0.3)
energy.set_xlabel('time - t $[u s]$')
energy.set_ylabel('Energy $[kN mm]$')


###############################################################################
# Plot: Time vs Gamma
# -------------------
fig, gamma = plt.subplots()

gamma.plot(time_sol, energy_factor*S.energy_files['total.energy']["gamma"], 'b-', linewidth=2.0, label=r'$\Gamma$')

gamma.grid(color='k', linestyle='-', linewidth=0.3)
gamma.set_xlabel('time - t $[\\mu s]$')
gamma.set_ylabel(r'$\Gamma$ [$mm$]')


plt.show()


###############################################################################
# Animation: phase-field $\phi$
# ----------------------------
# The saved .vtu files are used to create an animated GIF of the crack evolution. Each
# frame shows the phase field $\phi\in[0,1]$ on the deformed plate (displacement amplified
# by `factor`). Only the upper half is simulated, so the lower half is obtained by mirroring
# the deformed upper half with respect to the symmetry plane $y=0$. A frame is rendered every
# `frame_stride` steps. The images are rendered off screen and assembled with Pillow.

# from PIL import Image

# frame_stride = 50
# factor = 20.0
# fps = 3

# vtu_folder = os.path.join(Data.results_folder_name, "paraview-solutions_vtu")
# n_steps = len(S.dof_files["top.dof"]["#step"])
# frames = range(0, n_steps, frame_stride)


# def read_step(step):
#     return pv.read(os.path.join(vtu_folder, f"phasefieldx_p0_{step:06d}.vtu"))


# # The camera is fitted to the undeformed full plate (outline), so the whole specimen
# # is visible in every frame.
# plotter = pv.Plotter(off_screen=True, window_size=[1200, 600])
# plotter.add_mesh(pv.Box(bounds=(-lx/2, lx/2, -ly, ly, 0, 0)), style="wireframe", color="k", line_width=1)
# plotter.view_xy()
# plotter.camera.parallel_projection = True
# plotter.reset_camera()
# plotter.camera.zoom(1.4)

# images = []
# for step in frames:
#     upper = read_step(step).warp_by_vector("u", factor=factor)
#     lower = upper.reflect((0, 1, 0), point=(0, 0, 0))  # mirror about y = 0
#     for name, part in (("upper", upper), ("lower", lower)):
#         plotter.add_mesh(part, scalars="phi", clim=[0.0, 1.0], cmap="coolwarm", name=name,
#                          show_edges=False, show_scalar_bar=(name == "upper"),
#                          scalar_bar_args={"title": "phi"})
#     plotter.add_text(f"t = {(step + 1) * dt:.2f} us", name="time", font_size=12)
#     images.append(Image.fromarray(plotter.screenshot(return_img=True)))
# plotter.close()

# gif_path = os.path.join(Data.results_folder_name, "crack_branching.gif")
# images[0].save(gif_path, save_all=True, append_images=images[1:],
#                duration=int(1000 / fps), loop=0)
