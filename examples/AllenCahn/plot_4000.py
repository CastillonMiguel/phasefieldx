r"""
.. _ref_4000:

Free Energy: Allen-Cahn
^^^^^^^^^^^^^^^^^^^^^^^

In this example, we study the free energy potential of the Allen-Cahn equation. The model consists of a bar of length $2a$ with boundary conditions at $x=-a$ of $\phi=-1$ and $\phi=1$ at $x=a$.

.. code-block::

   #                       
   #                                      
   #         *------------------------* 
   #  phi=-1 |           * (0,0)      | phi=1
   #         *------------------------* 
   #
   #         |<----a----->|<-----a---->|    
   #     |Y   
   #     |                
   #     *---X  

The potential is the following:

.. math::
    W[\phi] = \int_\Omega \left( \frac{1}{l}f_{chem}(\phi) + \frac{l}{2} |\nabla \phi|^2 \right) dV 

with 

.. math::
    f_{chem}(\phi) = \frac{1}{4}(1-\phi^2)^2

as described in the theory section (:ref:`theory_allen_cahn`). The one-dimensional solution for an infinite domain is given by:

.. math::
    \phi(x) = \tanh\left(\frac{x}{\sqrt{2}l}\right)

In this case, the domain and the length scale parameter can be interpreted as an infinite domain (see the table values), as $tanh\left(\frac{a}{\sqrt{2}a}\right)=1$ and $tanh\left(\frac{-a}{\sqrt{2}a}\right)=-1$, so it is in concordance with the boundary conditions imposed.

.. _table_properties_label:

.. table:: Properties

   +----+---------+--------+
   |    | VALUE   | UNITS  |
   +====+=========+========+
   | l  | 10.0    | mm     |
   +----+---------+--------+
   | a  | 50.0    | mm     |
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
import os


###############################################################################
# Import from phasefieldx package
from phasefieldx.Element.Allen_Cahn.Input import Input
from phasefieldx.Element.Allen_Cahn.solver.solver_static import solve
from phasefieldx.Boundary.boundary_conditions import bc_phi, get_ds_bound_from_marker
from phasefieldx.PostProcessing.ReferenceResult import AllResults


###############################################################################
# Parameters definition
# ---------------------
# A length scale parameter of $l = 10.0$ is defined. The phase field is saved in a VTU file. 
# All the results are stored in a folder named results_folder_name = name.
Data = Input(l=10.0,
             save_solution_xdmf=False,
             save_solution_vtu=True,
             result_folder_name="4000_free_energy")


###############################################################################
# Mesh Definition
# ---------------
# A 2a mm x 1 mm rectangular mesh with quadrilateral elements is created using Dolfinx.
a = 50.0
divx, divy, divz = 100, 1, 1
lx, ly, lz = 100.0, 1.0, 1.0

msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                                    [np.array([-a, -0.5]),
                                     np.array([ a,  0.5])],
                                    [divx, divy],
                                     cell_type=dolfinx.mesh.CellType.quadrilateral)


###############################################################################
# The left and right parts are identified and used to impose
# the boundary conditions.
def left(x):
    return np.isclose(x[0], -a)

def right(x):
    return np.isclose(x[0], a)

fdim = msh.topology.dim - 1

left_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, left) 
right_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, right)         
ds_left = get_ds_bound_from_marker(left_facet_marker, msh , fdim)
ds_right = get_ds_bound_from_marker(right_facet_marker, msh , fdim)
ds_list = np.array([
                   [ds_left, "left"],
                   [ds_right, "right"]
                   ])


###############################################################################
# Function Space Definition
# -------------------------
# Define function spaces for the phase-field using Lagrange elements of 
# degree 1.
V_phi = dolfinx.fem.functionspace(msh, ("Lagrange", 1))


###############################################################################
# Boundary Conditions
# -------------------
# Apply boundary conditions:.
bc_left = bc_phi(left_facet_marker,  V_phi, fdim, value = -1.0)
bc_right = bc_phi(right_facet_marker, V_phi, fdim, value =  1.0)
bcs_list_phi = [bc_left, bc_right]
update_boundary_conditions = None
update_loading = None


###############################################################################
# Call the solver
# ---------------
# We set a final time of t=1 and a time step of \Delta t=1.

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
# $\phi(x) = \tanh\left(\frac{x}{\sqrt{2}l}\right)$
# Note: In this case, a = lx
xt = np.linspace(-a, a, 200)
phi_theory = np.tanh(xt/(Data.l*np.sqrt(2)))

fig, ax_phi = plt.subplots() 

ax_phi.plot(xt, phi_theory, 'k-',label='Theory')
ax_phi.plot(file_vtu.points[:,0], file_vtu['phi'],'r.', label=S.label)

ax_phi.grid(color='k', linestyle='-', linewidth=0.3)  
ax_phi.set_ylabel('phi(x)')
ax_phi.set_xlabel('x')
ax_phi.legend()

plt.show()


###############################################################################
# Energy values
# -------------
# The theoretical energy value is compared with the value calculated from the simulation.
energy_theory = np.sqrt(2) * np.tanh(a / (Data.l * np.sqrt(2))) - (1 / 3) * np.sqrt(2) * np.tanh(a / (Data.l * np.sqrt(2)))**3
energy_simulation = S.energy_files["total.energy"]["gamma"][0]
print(f"The theoretical energy is {energy_theory}")
print(f"The calculated numerical energy is {energy_simulation}")
