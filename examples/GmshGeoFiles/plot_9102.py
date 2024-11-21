"""
.. _ref_9102:

.geo File: Single Shear Tesnion Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This example demonstrates the procedure for generating a mesh for a Single Shear Tesnion Test. In this case the geometry of the speciment is the same as the :ref:`ref_9101` example, but the refinement region of the mesh is different, in this case a quarter part of the mesh is refined.

Files with the .geo format define both the geometry and the mesh parameters, such as element types and sizes.
Gmsh can then generate the mesh based on the input from these .geo files.

Below is an example of the .geo file used for mesh generation.

.geo file
---------
.. include::  ../../../../examples/GmshGeoFiles/9102_SingleNotchedShearTest/file.geo
   :literal:

For more information about Gmsh and the `.geo` file format, please refer to the official `Gmsh website <https://gmsh.info>`_.

To generate the mesh in the `.msh` format from the terminal, use the following Gmsh command:

.. code-block::
   gmsh file.geo -3 -o mesh.msh

Here, `-3` specifies that the mesh is 3-dimensional, and `-o mesh.msh` tells Gmsh to save the mesh to a file named `mesh.msh` with the `.msh` extension.

Alternatively, you can generate the mesh using the Gmsh Python API, which allows for programmatic mesh generation within Python scripts.
"""

###############################################################################
# Mesh Visualization
# ------------------
# The purpose of this code is to visualize the mesh. The mesh is generated from
# the .geo file and saved as output_mesh_for_view.vtu. It is then loaded and
# visualized using PyVista.

import os
import gmsh
import pyvista as pv

folder = "9102_SingleNotchedShearTest"

# %%
# Initialize Gmsh
gmsh.initialize()

# %%
# Open the .geo file
geo_file = os.path.join(folder, "file.geo")
gmsh.open(geo_file)

# %%
# Generate the mesh (2D example, for 3D use generate(3))
gmsh.model.mesh.generate(3)

# %%
# Write the mesh to a .vtk file for visualization
# Note that the input mesh file for the *phasefieldx* simulation should have the .msh extension.
# Use "output_mesh_for_view.msh" to generate the mesh for the simulation input.
# In this case, the mesh is saved in .vtk format to facilitate visualization with PyVista.
vtu_file = os.path.join(folder, "output_mesh_for_view.vtk")
gmsh.write(vtu_file)

# %%
# Finalize Gmsh
gmsh.finalize()

print(f"Mesh successfully written to {vtu_file}")

pv.start_xvfb()
file_vtu = pv.read(vtu_file)
file_vtu.plot(cpos='xy', color='white', show_edges=True)
