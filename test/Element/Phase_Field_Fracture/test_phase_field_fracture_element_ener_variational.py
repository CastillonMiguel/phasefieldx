import pytest
import numpy as np
import pyvista as pv
import os
import dolfinx
import mpi4py
import petsc4py

from phasefieldx.Element.Phase_Field_Fracture.Input import Input
from phasefieldx.Element.Phase_Field_Fracture.solver.solver_ener_variational import solve
from phasefieldx.Boundary.boundary_conditions import bc_xy, bc_x, get_ds_bound_from_marker
from phasefieldx.PostProcessing.ReferenceResult import AllResults

def test_phase_field_simulation_ener_variational():
    """
    Test phase-field fracrture energy variational solver and compare results with theoretical values.
    """

    Data = Input(E=210.0,   # young modulus
                 nu=0.3,    # poisson
                 Gc=0.005,  # critical energy release rate
                 l=0.1,     # lenght scale parameter
                 degradation="isotropic",  # "isotropic" "anisotropic"
                 split_energy="no",       # "spectral" "deviatoric"
                 degradation_function="quadratic",
                 irreversibility="no",  # "miehe"
                 fatigue=False,
                 fatigue_degradation_function="asymptotic",
                 fatigue_val=0.05625,
                 k=0.0,
                 save_solution_xdmf=False,
                 save_solution_vtu=True,
                 results_folder_name="results_test_ener_variational")

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

    bottom_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, bottom)
    top_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, top)


    ds_top = get_ds_bound_from_marker(top_facet_marker, msh, fdim)

    ds_list = np.array([
        [ds_top, "top"],
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
    bc_top = bc_x(top_facet_marker, V_u, fdim)
    bcs_list_u = [bc_top, bc_bottom]
    bcs_list_u_names = ["top", "bottom"]

    T_top = dolfinx.fem.Constant(msh, petsc4py.PETSc.ScalarType((0.0, 1.0)))

    T_list_u = [[T_top, ds_top]
            ]
    f = None

    bcs_list_phi = []

    ###############################################################################
    # Solver Call for a Phase-Field Fracture Problem
    # ----------------------------------------------
    final_gamma = 3.5

    # %%
    # Uncomment the following lines to run the solver with the specified parameters.
    c1 = 1.0
    c2 = 1.0

    solve(Data,
        msh,
        final_gamma,
        V_u,
        V_phi,
        bcs_list_u,
        bcs_list_phi,
        f,
        T_list_u,
        ds_list,
        dtau=0.0001,
        dtau_min=1e-6,
        dtau_max=0.1,
        path=None,
        bcs_list_u_names=bcs_list_u_names,
        c1=c1,
        c2=c2,
        threshold_gamma_save=0.1)

    S = AllResults(Data.results_folder_name)
    alpha = 1/np.sqrt(1+c1*S.dof_files["lambda.dof"]["lambda"]/Data.Gc)

    displacement = abs(2*S.energy_files['total.energy']["E"]/(S.reaction_files['bottom.reaction']["Ry"]))*alpha
    psi_t = 0.5 * displacement**2 * (Data.lambda_ + 2 * Data.mu)
    phi_t = 2 * psi_t / (Data.Gc / Data.l + 2 * psi_t)
    sigma_t = displacement * (Data.lambda_ + 2 * Data.mu) * (1 - phi_t)**2

    reaction_simulation = abs(S.reaction_files["bottom.reaction"]["Ry"]*alpha)


    np.testing.assert_allclose(reaction_simulation, sigma_t, rtol=1e-3, atol=1e-8,
                                err_msg=f"Computed reaction force {reaction_simulation} does not match theoretical reaction {sigma_t}")


    # Clean up: Remove generated files
    if os.path.exists(Data.results_folder_name):
        import shutil
        shutil.rmtree(Data.results_folder_name)
