"""
.. _ref_2000:

Crack surface density functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following example, we consider the boundary value problem of the phase-field model, which represents the crack surface density functional, thus providing a continuous approximation of the discontinuous crack  (:ref:`theory_phase_field`). Due to the symmetry of the problem, only the left half of the bar is considered. Therefore, a boundary condition is applied at the left end of this half bar, as illustrated in the following diagrams.
For a one-dimensional simulation, the boundary condition $\phi=1$ is set at the point located on the left end.

.. code-block::
                  
   #                             
   #        
   #         x=0 
   #  phi= 1 *------------------------*
   #
   #         |<---------- lx --------->|
   #     |Y 
   #     |             
   #     *---X
   

If two or three dimensions are considered, the boundary condition $\phi=1$ is applied to the left surface.

.. code-block::
                  
   #         x=0                     
   #         *------------------------*
   #  phi= 1 |                        |
   #         *------------------------*
   #
   #         |<---------- a --------->|
   #     |Y 
   #     |             
   #     *---X
  
Three dimensions:

.. code-block::

   #           *------------------------*
   #          /                        /|
   #         *------------------------* |
   #  phi= 1 |                        |/
   #         *------------------------*
   #
   #         |<---------- a --------->|
   #     |Y 
   #     |             
   #     *---X
   #   Z/

"""

###############################################################################
# Import necessary libraries
# --------------------------
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import dolfinx
import mpi4py 
import os


###############################################################################
# Import from phasefieldx package
# -------------------------------
from phasefieldx.Element.Phase_Field.Input import Input
from phasefieldx.Element.Phase_Field.solver.solver import solve
from phasefieldx.Boundary.boundary_conditions import bc_phi, get_ds_bound_from_marker
from phasefieldx.PostProcessing.ReferenceResult import AllResults


###############################################################################
# Parameters definition
# ---------------------
# A length scale parameter of  $l = 4$  is defined. The phase field is saved in a VTU file. 
# All the results are stored in a folder named results_folder_name = name.
Data = Input(l=4.0,
             save_solution_xdmf=False,
             save_solution_vtu=True,
             results_folder_name="2000_General")


###############################################################################
# Mesh definition
# ---------------
divx, divy, divz = 100, 1, 1
lx, ly, lz = 5.0, 1.0, 1.0


###############################################################################
# A two-dimensional simulation is considered 
# (it is also possible to select ‘1D’ and ‘3D’ simulations) 
dimension = "2d"

if dimension=="1d":
    msh = dolfinx.mesh.create_interval(mpi4py.MPI.COMM_WORLD,
                                       divx,
                                       np.array([0.0,  lx]))
  
elif dimension=="2d":
    msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                                        [np.array([0.0, 0.0]),
                                        np.array([lx, ly])],
                                        [divx, divy],
                                        cell_type=dolfinx.mesh.CellType.quadrilateral)
      
elif dimension=="3d":
    msh = dolfinx.mesh.create_box(mpi4py.MPI.COMM_WORLD, 
                            [np.array([0.0, 0.0, 0.0]),
                            np.array([lx, ly, lz])],
                            [divx, divy, divz], 
                            cell_type=dolfinx.mesh.CellType.hexahedron)


###############################################################################
# The left part is denoted and shown to impose the boundary conditions
def left(x):
    return np.equal(x[0], 0)

fdim = msh.topology.dim - 1

left_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, left) 
ds_left = get_ds_bound_from_marker(left_facet_marker, msh , fdim)
ds_list = np.array([
                   [ds_left, "left"],
                   ])


###############################################################################
# Function Space Definition
# -------------------------
# Define function spaces for the phase-field using Lagrange elements of 
# degree 1.
V_phi = dolfinx.fem.functionspace(msh, ("Lagrange", 1))


###############################################################################
# Boundary Condition
# ------------------
# The boundary conditions of $\phi=1$ is set on the left part of the mesh.
bc_left = bc_phi(left_facet_marker, V_phi, fdim, value=1.0)
bcs_list_phi = [bc_left]
update_boundary_conditions = None
update_loading = None


###############################################################################
# Call the solver
# ---------------
# We set a final time of t=1 and a time step of \Delta t=1.
# Note that this is a linear boundary value problem.

final_time = 1.0
dt = 1.0

solve(Data, 
      msh, 
      final_time, 
      V_phi, 
      bcs_list_phi,
      update_boundary_conditions,  
      update_loading,
      ds_list,
      dt,
      path = None,
      quadrature_degree = 2)


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
file_vtu = pv.read(os.path.join(Data.results_folder_name, "paraview-solutions_vtu", "phasefieldx_p0_000000.vtu"))
file_vtu.plot(scalars='phi', cpos='xy', show_scalar_bar=True, show_edges=False)


###############################################################################
# Plot: Phase-field along the x-axis
# ----------------------------------
# The phase-field value along the x-axis is plotted and compared with the 
# analytic solution. The analytic solution is given by:
# $\phi(x) = e^{-|x|/l} + \frac{1}{e^{\frac{2a}{l}}+1} 2 \sinh \left( \frac{|x|}{l} \right)$
# Note: in this case a = lx 
xt = np.linspace(-lx, lx, 1000)
phi_theory = np.exp(-abs(xt)/Data.l) + 1/(np.exp(2*lx/Data.l)+1) * 2* np.sinh(np.abs(xt)/Data.l)

fig, ax_phi = plt.subplots() 

ax_phi.plot(xt, phi_theory, 'k-', label='Theory')
ax_phi.plot(file_vtu.points[:,0],  file_vtu['phi'],'r.', label=S.label)

ax_phi.grid(color='k', linestyle='-', linewidth=0.3)  
ax_phi.set_ylabel('phi(x)')
ax_phi.set_xlabel('x')
ax_phi.legend()


###############################################################################
# Plot: Energy values
# -------------------
# The energy values are compared with the analytic ones.
l_array = np.linspace(0.1, lx, 100)
a_div_l = lx/l_array
tanh_a_div_l = np.tanh(a_div_l)
W_phi= 0.5*tanh_a_div_l + 0.5*a_div_l*(1.0-tanh_a_div_l**2)
W_gradphi = 0.5*tanh_a_div_l - 0.5*a_div_l*(1.0-tanh_a_div_l**2)
W = np.tanh(a_div_l)


###############################################################################
# The next graph shows the theoretical energy versus the length scale parameter, 
# as well as those corresponding to the length scale parameter used in the 
# simulation. It is noted that the solution coincides with the theoretical one.
fig, energy = plt.subplots() 

energy.plot(l_array, W_phi, 'r-', label = r'$W_{\phi}$')
energy.plot(l_array, W_gradphi , 'b-', label = r'$W_{\nabla \phi}$')
energy.plot(l_array, W,'k-', label = r'$W$')

energy.plot(Data.l, 2*S.energy_files['total.energy']["gamma_phi"][0], 'k*', label=S.label)
energy.plot(Data.l, 2*S.energy_files['total.energy']["gamma_gradphi"][0], 'k*')
energy.plot(Data.l, 2*S.energy_files['total.energy']["gamma"][0], 'k*')

energy.set_xlabel('l' )  
energy.set_ylabel('energy') 
energy.grid(color='k', linestyle='-', linewidth=0.3)  
energy.legend() 

plt.show()
