# import os
# import pytest
# import gmsh
# import pyvista as pv

# # Define the folder containing the .geo files
# FOLDER = "geo_files"
# GEO_FILES = ["file_2d.geo", "file_3d.geo"]  # Replace with your actual .geo file names
# VTU_FILES = [os.path.join(FOLDER, "output_mesh_for_view_1.vtk"),
#              os.path.join(FOLDER, "output_mesh_for_view_2.vtk")]


# @pytest.fixture(scope="module", autouse=True)
# def setup_gmsh():
#     """Set up GMSH for tests."""
#     gmsh.initialize()
#     yield
#     gmsh.finalize()


# @pytest.mark.parametrize("geo_file, vtu_file", zip(GEO_FILES, VTU_FILES))
# def test_mesh_generation(geo_file, vtu_file):
#     """Test the mesh generation from .geo files."""
#     # Open the .geo file
#     full_geo_path = os.path.join(FOLDER, geo_file)
#     gmsh.open(full_geo_path)

#     # Generate the mesh (2D example, for 3D use generate(3))
#     gmsh.model.mesh.generate(3)

#     # Write the mesh to a .vtu file
#     gmsh.write(vtu_file)

#     # Check if the .vtu file is created
#     assert os.path.isfile(vtu_file), f"{vtu_file} was not created"

#     # Test reading the generated .vtu file with PyVista

#     # Attempt to read the .vtu file
#     mesh = pv.read(vtu_file)

#     # Check if the mesh is not empty
#     assert mesh.n_cells > 0, f"The mesh from {vtu_file} is empty or not read properly."
#     assert mesh.n_points > 0, f"The mesh from {vtu_file} has no points."

#     # Optional: Uncomment to visualize the mesh during manual testing
#     # mesh.plot(cpos='xy', color='white', show_edges=True)  # Uncomment for visualization
