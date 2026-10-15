r"""
.. _ref_dynamic_wall:

Dynamic problem: wall under a pulse load
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This script solves a linear elastodynamic problem with the implicit Newmark scheme, as described in
:ref:`theory_elasticity_dynamic` and :ref:`theory_elasticity_newmark`.

Model Overview
--------------
The model represents a 2D wall (plane strain) with:

- A fully fixed base.
- A horizontal pressure applied on a portion of its left side, near the top.
  The load is a short half-sine pulse in time. Once the pulse ends, the wall is free
  and vibrates freely.

.. code-block::

   #                  
   #           *-----------*
   #           |           |
   #           |           |
   #  p(t) --> |           |   } loaded patch
   #  p(t) --> |           |   }
   #           |           |
   #           |           |
   #           |           |
   #           *-----------*
   #          ///////////////    fixed base
   #     |Y
   #     |
   #      ---X


.. image:: ../../../../examples/Elasticity/results_dynamic/dynamic_wall_geometry.png


Units and Material Properties
-----------------------------
A consistent set of units is used: lengths in mm, time in ms, forces in N (per unit thickness
of the wall). The material properties are summarized in the table below:

+-----+-----------+-----------------+
|     | VALUE     | UNITS           |
+=====+===========+=================+
| E   | 210e3     | N/mm2           |
+-----+-----------+-----------------+
| nu  | 0.3       | [-]             |
+-----+-----------+-----------------+
| rho | 8.0e-3    | N ms2/mm4       |
+-----+-----------+-----------------+


.. image:: ../../../../examples/Elasticity/results_dynamic/wall_dynamic.gif

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
from phasefieldx.Element.Elasticity.solver.solver_dynamic import solve
from phasefieldx.Boundary.boundary_conditions import bc_xy, get_ds_bound_from_marker
from phasefieldx.Loading.loading_functions import loading_Txy
from phasefieldx.PostProcessing.ReferenceResult import AllResults


###############################################################################
# Define Simulation Parameters
# ----------------------------
# `Data` is an input object containing the material parameters and the result storage options:
#
# - `E`: Young's modulus, 210e3 N/mm².
# - `nu`: Poisson's ratio, 0.3.
# - `rho`: mass density, 8.0e-3 N ms²/mm⁴. In the dynamic problem it multiplies the
#   acceleration in the inertial term of the weak form.
# - `save_solution_xdmf` and `save_solution_vtu`: Set to `False` and `True`, respectively,
#   specifying the file format to save displacement results (.vtu here).
# - `results_folder_name`: Name of the folder for saving results. If it exists,
#   it will be replaced with a new empty folder.
Data = Input(E=210e3,
             nu=0.3,
             rho=8.0e-3,
             save_solution_xdmf=False,
             save_solution_vtu=True,
             results_folder_name="results_dynamic")


###############################################################################
# Mesh Definition
# ---------------
# The wall is meshed with a structured grid of quadrilateral elements:
#
# - `lx`, `ly`: width and height of the wall (1000 mm and 2000 mm).
# - `divx`, `divy`: number of elements along x and y (20 and 40, i.e. 50 mm elements).
lx, ly = 1000.0, 2000.0
divx, divy = 20, 40

msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                                    [np.array([0.0, 0.0]),
                                     np.array([lx, ly])],
                                    [divx, divy],
                                    cell_type=dolfinx.mesh.CellType.quadrilateral)


###############################################################################
# Boundary Identification
# -----------------------
# Two regions of the boundary are needed:
#
# - `base`: the bottom edge of the wall ($y=0$), where the wall is fixed.
# - `loaded_patch`: the part of the left edge ($x=0$) between $y = y_1$ and $y = y_2$,
#   where the pressure is applied.
#
# `fdim` is the dimension of the boundary facets (1 for a 2D mesh).
y1, y2 = 1500.0, 1800.0

np.logical_and
def base(x):
    return np.isclose(x[1], 0.0)


def loaded_patch(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.logical_and(x[1] >= y1, x[1] <= y2))


fdim = msh.topology.dim - 1

# %%
# The `locate_entities_boundary` function returns the indices of the facets that satisfy each condition.
base_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, base)
patch_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, loaded_patch)

# %%
# The `get_ds_bound_from_marker` function creates an integration measure restricted to the
# selected facets. `ds_patch` is used later to integrate the applied pressure.
ds_base = get_ds_bound_from_marker(base_facet_marker, msh, fdim)
ds_patch = get_ds_bound_from_marker(patch_facet_marker, msh, fdim)

# %%
# `ds_list` stores the measures used to compute reactions, together with a label for saving.
ds_list = np.array([
                   [ds_patch, "loaded_patch"]
                   ])


###############################################################################
# Function Space Definition
# -------------------------
# The displacement field is discretized with vector-valued Lagrange elements of degree 1.
V_u = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))


###############################################################################
# Boundary Conditions
# -------------------
# The base of the wall is fixed in both directions. In a dynamic analysis the boundary
# conditions can also vary in time, which is handled by `update_boundary_conditions`; here
# the base is fixed at all times, so it returns zero displacements.
bc_base = bc_xy(base_facet_marker, V_u, fdim)

# %%
# `bcs_list_u` collects all Dirichlet conditions for the displacement $\boldsymbol u$,
# and `bcs_list_u_names` the labels used to name the reaction files (`base.reaction`).
bcs_list_u = [bc_base]
bcs_list_u_names = ["base"]


def update_boundary_conditions(bcs, time):
    return 0, 0, 0


###############################################################################
# External Load Definition
# ------------------------
# The pressure is applied on the loaded patch, through `ds_patch`. `T_patch` is a constant
# vector with the traction components $(t_x, t_y)$ in N/mm². Its value is updated in time by
# `update_loading`.
T_patch = loading_Txy(msh, value_x=0.0, value_y=0.0)

# %%
# The list `T_list_u` pairs each traction with the boundary measure where it acts.
T_list_u = [[T_patch, ds_patch]]
f = None


###############################################################################
# Function: `update_loading`
# --------------------------
# In a dynamic analysis, the load history is a function of the physical time. Here, the
# horizontal pressure is a half-sine pulse of duration `t_pulse` and amplitude `p_max`,
#
# $p(t) = p_{max} \sin\left(\pi t / t_{pulse}\right), \quad 0 \le t \le t_{pulse},$
#
# and zero afterwards. The pulse is positive in the $x$ direction (the traction is applied on
# the left side of the wall, so it pushes the wall to the right). After the pulse, the wall
# vibrates freely.
#
# The function receives the list `T_list_u` and the current time, modifies the value of the
# traction in place, and returns three values that are not used here.
p_max = 10.0  # N/mm2
t_pulse = 2.0  # ms

def pressure(time):
    return p_max * np.sin(np.pi * time / t_pulse) if time <= t_pulse else 0.0


def update_loading(T_list_u, time):
    T_list_u[0][0].value[0] = petsc4py.PETSc.ScalarType(pressure(time))
    T_list_u[0][0].value[1] = petsc4py.PETSc.ScalarType(0.0)
    return 0, 0, 0


###############################################################################
# Solver Call
# -----------
# The time window is defined by:
#
# - `final_time`: end time of the simulation (20 ms).
# - `dt`: time step (0.05 ms), constant. Unlike in the quasi-static examples, it is a
#   physical time step: it must be small enough to resolve the load pulse and the vibration
#   of the wall.
#
# The `solve` function from the dynamic solver is called with the mesh, the function space,
# the boundary conditions, the loads and their update functions. The Newmark parameters are
# $\beta = 1/4$ and $\gamma = 1/2$ (see :ref:`theory_elasticity_newmark`), and the wall starts
# at rest in the undeformed configuration.
final_time = 20.0
dt = 0.05

# solve(Data,
#       msh,
#       final_time,
#       V_u,
#       bcs_list_u,
#       update_boundary_conditions,
#       f,
#       T_list_u,
#       update_loading,
#       ds_list,
#       dt,
#       path=None,
#       quadrature_degree=2,
#       bcs_list_u_names=bcs_list_u_names)


###############################################################################
# Load results
# ------------
# Once the simulation finishes, the results are loaded from the results folder.
# The AllResults class takes the folder path as an argument and stores all
# the results, including logs, energy, convergence, and DOF files.
# It is also possible to define a custom label and color to automate plot labels.
S = AllResults(Data.results_folder_name)
S.set_label('Simulation')
S.set_color('b')


###############################################################################
# Plot: displacement $\boldsymbol u$
# ----------------------------------
# The displacement saved in the .vtu file at a given step is shown over the deformed wall.
# Step 100 (t = 5 ms) is close to the maximum deflection, after the pulse has ended. The
# displacement is amplified (`factor`) to make the deformation visible.
plot_step = 100
file_vtu = pv.read(os.path.join(Data.results_folder_name,
                                "paraview-solutions_vtu",
                                f"phasefieldx_p0_{plot_step:06d}.vtu"))
file_vtu["u_magnitude"] = np.linalg.norm(file_vtu["u"], axis=1)
warped = file_vtu.warp_by_vector("u", factor=500.0)
warped.plot(scalars="u_magnitude", cpos="xy", show_scalar_bar=True, show_edges=False)


steps = S.dof_files["top.dof"]["#step"]
time = steps * dt


###############################################################################
# Plot: elastic energy
# --------------------
# Evolution of the elastic strain energy stored in the wall. It increases while the pulse is
# applied and then oscillates as the wall exchanges strain and kinetic energy during the
# free vibration.
fig, ax = plt.subplots()

ax.plot(time, S.energy_files['total.energy']["E"], S.color + '.', label=S.label)

ax.grid(color='k', linestyle='-', linewidth=0.3)
ax.set_xlabel('time $[ms]$')
ax.set_ylabel('elastic strain energy - E $[N\\,mm/mm]$')
ax.legend()

plt.show()




###############################################################################
# Animation: deformed wall
# ------------------------
# The saved .vtu files are used to create an animated GIF of the motion of the wall. Each
# frame shows the deformed wall (displacement amplified by `factor`) colored by the magnitude of
# the displacement, together with the undeformed mesh as a reference. A frame is rendered every
# `frame_stride` steps, and the color scale is fixed for all frames. The images are rendered off
# screen and assembled with Pillow.

# from PIL import Image
# frame_stride = 4
# factor = 500.0
# fps = 20

# vtu_folder = os.path.join(Data.results_folder_name, "paraview-solutions_vtu")
# n_steps = len(S.dof_files["top.dof"]["#step"])
# frames = range(0, n_steps, frame_stride)


# def read_step(step):
#     mesh = pv.read(os.path.join(vtu_folder, f"phasefieldx_p0_{step:06d}.vtu"))
#     mesh["u_magnitude"] = np.linalg.norm(mesh["u"], axis=1)
#     return mesh


# u_max = max(read_step(step)["u_magnitude"].max() for step in frames)

# plotter = pv.Plotter(off_screen=True, window_size=[500, 900])
# plotter.add_mesh(read_step(0), style="wireframe", color="gray", line_width=1)
# plotter.view_xy()
# plotter.camera.zoom(0.8)

# images = []
# for step in frames:
#     warped = read_step(step).warp_by_vector("u", factor=factor)
#     plotter.add_mesh(warped, scalars="u_magnitude", clim=[0.0, u_max], name="wall",
#                      show_edges=False, scalar_bar_args={"title": "|u| [mm]"})
#     plotter.add_text(f"t = {step * dt:.2f} ms", name="time", font_size=12)
#     images.append(Image.fromarray(plotter.screenshot(return_img=True)))
# plotter.close()

# gif_path = os.path.join(Data.results_folder_name, "wall_dynamic.gif")
# images[0].save(gif_path, save_all=True, append_images=images[1:],
#                duration=int(1000 / fps), loop=0)
