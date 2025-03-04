import pytest
import numpy as np
import dolfinx
import mpi4py
import pyvista as pv
import os

# Import necessary components from the phasefieldx package
from phasefieldx.Element.Allen_Cahn.Input import Input
from phasefieldx.Element.Allen_Cahn.solver.solver_static import solve
from phasefieldx.Boundary.boundary_conditions import bc_phi, get_ds_bound_from_marker
from phasefieldx.PostProcessing.ReferenceResult import AllResults


@pytest.mark.parametrize("a, l, divx", [(50.0, 10.0, 100)])
def test_allen_cahn_free_energy(a, l, divx):
    """
    Test the Allen-Cahn free energy simulation.

    Parameters:
    a (float): Width of the simulation domain.
    l (float): Simulation parameter.
    divx (int): Number of divisions along x-axis.

    Asserts:
    - The numerical phase-field matches the theoretical solution.
    - The computed total energy matches the theoretical energy.
    """
    # Set parameters and mesh size
    divy = 1

    # Create mesh
    msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                                        [np.array([-a, -0.5]),
                                         np.array([a, 0.5])],
                                        [divx, divy],
                                        cell_type=dolfinx.mesh.CellType.quadrilateral)

    # Define the input parameters for the simulation
    Data = Input(l=l, save_solution_xdmf=False, save_solution_vtu=True,
                 result_folder_name="4000_free_energy_test")

    # Define boundary conditions
    def left(x): return np.isclose(x[0], -a)
    def right(x): return np.isclose(x[0], a)

    fdim = msh.topology.dim - 1
    left_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, left)
    right_facet_marker = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, right)
    ds_left = get_ds_bound_from_marker(left_facet_marker, msh, fdim)
    ds_right = get_ds_bound_from_marker(right_facet_marker, msh, fdim)
    ds_list = np.array([[ds_left, "left"], [ds_right, "right"]])

    # Define function space for the phase-field
    V_phi = dolfinx.fem.functionspace(msh, ("Lagrange", 1))

    # Apply boundary conditions
    bc_left = bc_phi(left_facet_marker, V_phi, fdim, value=-1.0)
    bc_right = bc_phi(right_facet_marker, V_phi, fdim, value=1.0)
    bcs_list_phi = [bc_left, bc_right]

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

    # Load results using AllResults class
    S = AllResults(Data.results_folder_name)

    # Load the phase-field results from the .vtu file
    file_vtu = pv.read(os.path.join(Data.results_folder_name,
                       "paraview-solutions_vtu", "phasefieldx_p0_000000.vtu"))

    # Theoretical solution for phi(x)
    xt = file_vtu.points[:, 0]  # np.linspace(-a, a, 200)
    phi_theory = np.tanh(xt / (Data.l * np.sqrt(2)))

    # Compare the numerical phase-field with the theoretical solution along the x-axis
    assert np.allclose(
        file_vtu.points[:, 0], xt, atol=1e-2), "X-axis values do not match"
    assert np.allclose(file_vtu['phi'], phi_theory,
                       atol=1e-2), "Phase-field values do not match the theoretical values"

    # Theoretical energy calculation
    energy_theory = np.sqrt(2) * np.tanh(a / (Data.l * np.sqrt(2))) - \
        (1 / 3) * np.sqrt(2) * np.tanh(a / (Data.l * np.sqrt(2)))**3

    # Compare the computed total energy with the theoretical energy
    energy_simulation = S.energy_files["total.energy"]["gamma"][0]

    np.testing.assert_allclose(energy_simulation, energy_theory, rtol=1e-3, atol=1e-3, err_msg=f"Simulated energy {energy_simulation} does not match theoretical energy {energy_theory}")

    # Clean up: remove the generated files
    if os.path.exists(Data.results_folder_name):
        import shutil
        shutil.rmtree(Data.results_folder_name)
