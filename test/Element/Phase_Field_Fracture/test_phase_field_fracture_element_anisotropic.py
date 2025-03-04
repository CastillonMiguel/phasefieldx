import pytest
import numpy as np
import pyvista as pv
import os
import dolfinx
import mpi4py
import petsc4py

# Import necessary components from the phasefieldx package
from phasefieldx.Element.Phase_Field_Fracture.Input import Input
from phasefieldx.Element.Phase_Field_Fracture.solver.solver import solve
from phasefieldx.Boundary.boundary_conditions import bc_xy, get_ds_bound_from_marker
from phasefieldx.PostProcessing.ReferenceResult import AllResults
from phasefieldx.Math import macaulay_bracket_positive, macaulay_bracket_negative


@pytest.mark.parametrize("split_energy_i", ["spectral"])
def test_phase_field_simulation_anisotropic(split_energy_i):
    """
    Test phase-field simulation under anisotropic conditions and compare results with theoretical values.
    """
    Data = Input(E=210.0,   # young modulus
                 nu=0.3,    # poisson
                 Gc=0.005,  # critical energy release rate
                 l=0.1,     # lenght scale parameter
                 degradation="anisotropic",  # "isotropic" "anisotropic"
                 split_energy=split_energy_i,       # "spectral" "deviatoric"
                 degradation_function="quadratic",
                 irreversibility="no",  # "miehe"
                 fatigue=False,
                 fatigue_degradation_function="asymptotic",
                 fatigue_val=0.05625,
                 k=0.0,
                 save_solution_xdmf=False,
                 save_solution_vtu=True,
                 results_folder_name="1700_One_element_isotropic_tension_test_anisotropic")

    ###############################################################################
    # Mesh Definition
    # ---------------
    # We create a 1x1 mm quadrilateral mesh with a single element using Dolfinx.
    msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                                        [np.array([0, 0]),
                                         np.array([1, 1])],
                                        [1, 1],
                                        cell_type=dolfinx.mesh.CellType.quadrilateral)

    def bottom(x):
        return np.isclose(x[1], 0)

    def top(x):
        return np.isclose(x[1], 1)

    fdim = msh.topology.dim - 1

    bottom_facet_marker = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, bottom)
    top_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, top)

    ds_bottom = get_ds_bound_from_marker(top_facet_marker, msh, fdim)
    ds_top = get_ds_bound_from_marker(top_facet_marker, msh, fdim)

    ds_list = np.array([
        [ds_top, "top"],
        [ds_bottom, "bottom"],
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
    # Apply boundary conditions: bottom nodes fixed in both directions, top nodes can slide vertically.
    bc_bottom = bc_xy(bottom_facet_marker, V_u, fdim)
    bc_top = bc_xy(top_facet_marker, V_u, fdim)
    bcs_list_u = [bc_top, bc_bottom]
    bcs_list_u_names = ["top", "bottom"]

    def update_boundary_conditions(bcs, time):
        if time <= 50:
            val = 0.0003 * time
        elif time <= 150:
            val = -0.0003 * (time - 50) + 0.015
        else:
            val = 0.0003 * (time - 150) - 0.015

        bcs[0].g.value[1] = petsc4py.PETSc.ScalarType(val)
        return 0, val, 0

    bcs_list_phi = []
    T_list_u = None
    update_loading = None
    f = None

    ###############################################################################
    # Call the Solver
    # ---------------
    # The problem is solved for a final time of 200. The solver will handle the mesh, boundary conditions,
    # and the given parameters to compute the solution.

    dt = 1.0
    final_time = 200.0

    solve(Data,
          msh,
          final_time,
          V_u,
          V_phi,
          bcs_list_u,
          bcs_list_phi,
          update_boundary_conditions,
          f,
          T_list_u,
          update_loading,
          ds_list,
          dt,
          path=None,
          bcs_list_u_names=bcs_list_u_names,
          min_stagger_iter=2,
          max_stagger_iter=500,
          stagger_error_tol=1e-8)

    S = AllResults(Data.results_folder_name)
    displacement = S.dof_files["top.dof"]["Uy"]

    if Data.split_energy == "spectral":
        ep = macaulay_bracket_positive(displacement)
        en = macaulay_bracket_negative(displacement)
        psi_t = 0.5 * ep**2 * (Data.lambda_ + 2 * Data.mu)
        phi_t = 2 * psi_t / (Data.Gc / Data.l + 2 * psi_t)
        sigma_t = ep * (Data.lambda_ + 2 * Data.mu) * (1 - phi_t)**2 + \
            en * (Data.lambda_ + 2 * Data.mu)

    elif Data.split_energy == "deviatoric":
        m = 3
        ep = macaulay_bracket_positive(displacement)
        en = macaulay_bracket_negative(displacement)
        k0 = Data.lambda_ + 2 / m * Data.mu
        psi_t = 0.5 * k0 * ep**2 + Data.mu * 2 / 3 * displacement**2
        phi_t = 2 * psi_t / (Data.Gc / Data.l + 2 * psi_t)
        sigma_t = k0 * ep + 2 * Data.mu * (2 / 3 * displacement) * (1 - phi_t)**2 + k0 * en

    else:
        raise ValueError(f"Invalid value for split_energy: {Data.split_energy}. Expected 'spectral' or 'deviatoric'.")

    np.testing.assert_allclose(S.reaction_files["top.reaction"]["Ry"], sigma_t, rtol=1e-3, atol=1e-8,
                               err_msg=f"Computed reaction force {S.reaction_files['top.reaction']['Ry']} does not match theoretical reaction {sigma_t} for {Data.split_energy}")

    # Clean up: Remove generated files
    if os.path.exists(Data.results_folder_name):
        import shutil
        shutil.rmtree(Data.results_folder_name)
