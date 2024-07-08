"""
.. _ref_2001:

Representation of a Cracked Plate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following example, we consider the boundary value problem of the phase-field model (:ref:`theory_phase_field`). A cracked plate will be represented using the phase field variable, following [Miehe]_, with the boundary condition \phi=1 and different length scale values. Due to the symmetry of the problem, only one half of the plate will be considered. The results will be shown for all the models by applying the reflection of the solution.”

.. code-block::
   
   #
   #        *---------------*  -
   #        |               |  |
   #        |               | 0.5
   #        | phi=1         |  |
   #        *........-------*  -
   #        |<-0.5->|   
   #     |Y          
   #     |                
   #     *---X          

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
# Length scale parameters of $l_{10} = 10$, $l_{05} = 0.5$, and $l_{01} = 0.1$ 
# are defined. The phase field is saved in a VTU file. All the results are 
# stored in a folder named results_folder_name = name.
Data1 = Input(l=1.0,
              save_solution_xdmf=False,
              save_solution_vtu=True,
              results_folder_name="2001_regularized_crack_surface_l1")

Data025 = Input(l=0.25,
                save_solution_xdmf=False,
                save_solution_vtu=True,
                results_folder_name="2001_regularized_crack_surface_l025")
 
Data005 = Input(l=0.05,
                save_solution_xdmf=False,
                save_solution_vtu=True,
                results_folder_name="2001_regularized_crack_surface_l005")


###############################################################################
# Mesh definition
# ---------------
divx,divy = 100, 50
lx, ly = 1.0, 0.5


###############################################################################
# A two-dimensional simulation is considered 
msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                                    [np.array([0, 0]),
                                     np.array([lx, ly])],
                                    [divx, divy],
                                    cell_type=dolfinx.mesh.CellType.quadrilateral)  


###############################################################################
# The bottom part is denoted and shown to impose the boundary conditions
def bottom(x):
    return np.logical_and(np.isclose(x[1], 0),  np.less(x[0], 0.5))

fdim = msh.topology.dim - 1

bottom_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, bottom) 
ds_bottom = get_ds_bound_from_marker(bottom_facet_marker, msh , fdim)
ds_list = np.array([
                   [ds_bottom, "bottom"],
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
# The boundary condition of $\phi=1$ is set on the bottom left part 
# of the mesh.
bc_bottom = bc_phi(bottom_facet_marker, V_phi, fdim, value=1.0)
bcs_list_phi = [bc_bottom]
update_boundary_conditions = None
update_loading = None


###############################################################################
# Call the solver
# ---------------
# We set a final time of t=1 and a time step of \Delta t=1.
# Note that this is a linear boundary value problem.
# And the three simulations for the different length scale parameters are run
final_time = 1.0
dt = 1.0


###############################################################################
# Simulation for $l=1$
solve(Data1, 
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

S1 = AllResults(Data1.results_folder_name)
S1.set_label('$l_1$')
   
###############################################################################
# Plot phase-field $\phi$ for $l=1$
file_vtu = pv.read(os.path.join(Data1.results_folder_name,"paraview-solutions_vtu","phasefieldx_p0_000000.vtu"))
file_vtu_reflected = file_vtu.reflect((0, 1, 0), point=(0, 0, 0))
p = pv.Plotter()
p.add_mesh(file_vtu, show_edges=False)
p.add_mesh(file_vtu_reflected, show_edges=False)
p.camera_position = 'xy'
p.show()

###############################################################################
# Simulation for $l=0.25$
solve(Data025, 
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

S025 = AllResults(Data025.results_folder_name)
S025.set_label('$l_025$')


###############################################################################
# Plot phase-field $\phi$ for $l=0.25$
file_vtu = pv.read(os.path.join(Data025.results_folder_name,"paraview-solutions_vtu","phasefieldx_p0_000000.vtu"))
file_vtu_reflected = file_vtu.reflect((0, 1, 0), point=(0, 0, 0))

p = pv.Plotter()
p.add_mesh(file_vtu, show_edges=False)
p.add_mesh(file_vtu_reflected, show_edges=False)
p.camera_position = 'xy'
p.show()


###############################################################################
# Simulation for $l=0.05$
solve(Data005, 
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
S005 = AllResults(Data005.results_folder_name)
S005.set_label('$l_005$')


###############################################################################
# Plot phase-field $\phi$ for $l=0.05$
file_vtu = pv.read(os.path.join(Data005.results_folder_name,"paraview-solutions_vtu","phasefieldx_p0_000000.vtu"))
file_vtu_reflected = file_vtu.reflect((0, 1, 0), point=(0, 0, 0))

p = pv.Plotter()
p.add_mesh(file_vtu, show_edges=False)
p.add_mesh(file_vtu_reflected, show_edges=False)
p.camera_position = 'xy'
p.show()
