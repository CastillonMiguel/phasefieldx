"""
.. _ref_9101:

.geo File: Single Notched Tension Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.geo file
---------

.. include::  ../../../../examples/GmshGeoFiles/9101_SingleNotchedTensionTest/file.geo
   :literal:

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

folder = "9101_SingleNotchedTensionTest"
# Initialize Gmsh
gmsh.initialize()

# Open the .geo file
geo_file = os.path.join(folder, "file.geo")
gmsh.open(geo_file)

# Generate the mesh (2D example, for 3D use generate(3))
gmsh.model.mesh.generate(3)

# Write the mesh to a .vtu file
vtu_file = os.path.join(folder, "output_mesh_for_view.vtk")
gmsh.write(vtu_file)

# Finalize Gmsh
gmsh.finalize()

print(f"Mesh successfully written to {vtu_file}")

pv.start_xvfb()
file_vtu = pv.read(vtu_file)
file_vtu.plot(cpos='xy', color='white', show_edges=True)
