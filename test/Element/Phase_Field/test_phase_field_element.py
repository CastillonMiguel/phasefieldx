import pytest
import numpy as np
import pyvista as pv
import os
import dolfinx
import mpi4py

# Import necessary components from the phasefieldx package
from phasefieldx.Element.Phase_Field.Input import Input
from phasefieldx.Element.Phase_Field.solver.solver import solve
from phasefieldx.Boundary.boundary_conditions import bc_phi, get_ds_bound_from_marker
from phasefieldx.PostProcessing.ReferenceResult import AllResults


@pytest.mark.parametrize("dimension", ["1d", "2d", "3d"])
def test_phase_field_simulation(dimension):
    """
    Test phase-field simulation for various dimensions (1D, 2D, 3D).

    Parameters:
    dimension (str): The dimension of the phase-field simulation. Options are '1d', '2d', '3d'.

    Asserts:
    - Phase-field values match theoretical predictions.
    - Computed energy values match theoretical values within a specified tolerance.
    """
    # Set parameters for each dimension
    lx, ly, lz = 5.0, 1.0, 1.0
    divx, divy, divz = 200, 1, 1

    # Create mesh depending on the dimension
    if dimension == "1d":
        msh = dolfinx.mesh.create_interval(mpi4py.MPI.COMM_WORLD,
                                           divx,
                                           np.array([0.0, lx]))
    elif dimension == "2d":
        msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                                            [np.array([0.0, 0.0]),
                                             np.array([lx, ly])],
                                            [divx, divy],
                                            cell_type=dolfinx.mesh.CellType.quadrilateral)
    elif dimension == "3d":
        msh = dolfinx.mesh.create_box(mpi4py.MPI.COMM_WORLD,
                                      [np.array([0.0, 0.0, 0.0]),
                                       np.array([lx, ly, lz])],
                                      [divx, divy, divz],
                                      cell_type=dolfinx.mesh.CellType.hexahedron)

    # Set phase-field input
    Data = Input(l=4.0, save_solution_xdmf=False,
                 save_solution_vtu=True, results_folder_name="test_results")

    # Define boundary condition and function space
    def left(x):
        return np.equal(x[0], 0)

    fdim = msh.topology.dim - 1
    left_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, left)
    ds_left = get_ds_bound_from_marker(left_facet_marker, msh, fdim)
    ds_list = np.array([[ds_left, "left"]])

    V_phi = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    bc_left = bc_phi(left_facet_marker, V_phi, fdim, value=1.0)
    bcs_list_phi = [bc_left]

    # Solve the problem
    solve(Data,
          msh,
          1.0,
          V_phi,
          bcs_list_phi,
          None,
          None,
          ds_list,
          1.0,
          path=None,
          quadrature_degree=2)

    # Load results using PyVista
    # pv.start_xvfb()
    file_vtu = pv.read(os.path.join(Data.results_folder_name,
                       "paraview-solutions_vtu", "phasefieldx_p0_000000.vtu"))

    # Theoretical phase-field values along the x-axis
    xt = file_vtu.points[:, 0]  # np.linspace(-lx, lx, divx)
    phi_theory = np.exp(-abs(xt) / Data.l) + 1 / \
        (np.exp(2 * lx / Data.l) + 1) * 2 * np.sinh(np.abs(xt) / Data.l)

    # Compare numerical results with theoretical values (tolerance is set to allow for slight deviations)
    assert np.allclose(file_vtu['phi'], phi_theory,
                       atol=1e-8), "Phase-field values do not match the theoretical values"

    # Assert the computed energy matches the theoretical one within a tolerance
    S = AllResults(Data.results_folder_name)
    a_div_l = lx / Data.l
    tanh_a_div_l = np.tanh(a_div_l)
    W_phi = 0.5 * tanh_a_div_l + 0.5 * a_div_l * (1.0 - tanh_a_div_l**2)
    W_gradphi = 0.5 * tanh_a_div_l - 0.5 * a_div_l * (1.0 - tanh_a_div_l**2)
    W = np.tanh(a_div_l)

    np.testing.assert_allclose(S.energy_files["total.energy"]["gamma_phi"][0], 0.5 * W_phi, rtol=1e-3, atol=1e-6,
                               err_msg=f"Computed phi energy {S.energy_files['total.energy']['gamma_phi'][0]} does not match theoretical energy {W_phi}")

    np.testing.assert_allclose(S.energy_files["total.energy"]["gamma_gradphi"][0], 0.5 * W_gradphi, rtol=1e-3, atol=1e-6,
                               err_msg=f"Computed gradphi energy {S.energy_files['total.energy']['gamma_gradphi'][0]} does not match theoretical energy {W_gradphi}")

    np.testing.assert_allclose(S.energy_files["total.energy"]["gamma"][0], 0.5 * W, rtol=1e-3, atol=1e-6,
                               err_msg=f"Computed energy {S.energy_files['total.energy']['gamma'][0]} does not match theoretical energy {W}")

    # Clean up: Remove generated files
    if os.path.exists(Data.results_folder_name):
        import shutil
        shutil.rmtree(Data.results_folder_name)
