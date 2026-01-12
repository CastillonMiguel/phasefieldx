import pytest
import numpy as np
import os
import dolfinx
import mpi4py
import petsc4py


# Import necessary components from the phasefieldx package
from phasefieldx.Element.Phase_Field_Fracture.Input import Input
from phasefieldx.Element.Phase_Field_Fracture.solver.solver_anisotropic import solve
from phasefieldx.Boundary.boundary_conditions import bc_xy, get_ds_bound_from_marker
from phasefieldx.Loading.loading_functions import loading_Txy
from phasefieldx.PostProcessing.ReferenceResult import AllResults


@pytest.mark.parametrize("split_energy_i", ["spectral", "deviatoric"])
def test_phase_field_simulation_anisotropic(split_energy_i):
    Data = Input(E=210.0,    # young modulus
                 nu=0.3,     # poisson
                 Gc=0.0027,  # critical energy release rate
                 l=0.06,     # lenght scale parameter
                 degradation="anisotropic",  # "isotropic" "anisotropic"
                 split_energy=split_energy_i,    # "spectral" "deviatoric"
                 degradation_function="quadratic",
                 irreversibility="miehe",  # "miehe"
                 fatigue=False,
                 fatigue_degradation_function="asymptotic",
                 fatigue_val=0.0,
                 k=0.0,
                 save_solution_xdmf=False,
                 save_solution_vtu=True,
                 results_folder_name="check")


    ###############################################################################
    # Mesh Definition
    # ---------------
    # The mesh is a structured grid with quadrilateral elements:
    #
    # - `divx`, `divy`: Number of elements along the x and y axes (10 each).
    # - `lx`, `ly`: Physical domain dimensions in x and y (1.0 units each).
    divx, divy = 10, 10
    lx, ly = 1.0, 1.0
    msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                                        [np.array([0, 0]),
                                        np.array([lx, ly])],
                                        [divx, divy],
                                        cell_type=dolfinx.mesh.CellType.quadrilateral)


    ###############################################################################
    # Boundary Identification
    # -----------------------
    # Boundary conditions and forces are applied to specific regions of the domain:
    #
    # - `bottom`: Identifies the $y=0$ boundary.
    # - `top`: Identifies the $y=ly$ boundary.
    # `fdim` is the dimension of boundary facets (1D for a 2D mesh).
    def bottom(x):
        return np.isclose(x[1], 0)

    def top(x):
        return np.isclose(x[1], ly)

    fdim = msh.topology.dim - 1

    # %%
    # Using the `bottom` and `top` functions, we locate the facets on the bottom and top sides of the mesh,
    # where $y = 0$ and $y = ly$, respectively. The `locate_entities_boundary` function returns an array of facet
    # indices representing these identified boundary entities.
    bottom_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, bottom)
    top_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, top)

    # %%
    # The `get_ds_bound_from_marker` function generates a measure for applying boundary conditions 
    # specifically to the facets identified by `top_facet_marker` and `bottom_facet_marker`, respectively. 
    # This measure is then assigned to `ds_bottom` and `ds_top`.
    ds_bottom = get_ds_bound_from_marker(bottom_facet_marker, msh, fdim)
    ds_top = get_ds_bound_from_marker(top_facet_marker, msh, fdim)

    # %%
    # `ds_list` is an array that stores boundary condition measures along with names 
    # for each boundary, simplifying result-saving processes. Each entry in `ds_list` 
    # is formatted as `[ds_, "name"]`, where `ds_` represents the boundary condition measure, 
    # and `"name"` is a label used for saving. Here, `ds_bottom` and `ds_top` are labeled 
    # as `"bottom"` and `"top"`, respectively, to ensure clarity when saving results.
    ds_list = np.array([
                    [ds_bottom, "bottom"],
                    ])

    ###############################################################################
    # Function Space Definition
    # -------------------------
    # Define function spaces for the displacement field using Lagrange elements of
    # degree 1.
    V_u = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))


    ###############################################################################
    # Boundary Conditions
    # -------------------
    # Dirichlet boundary conditions are applied as follows:
    #
    # - `bc_bottom`: Fixes x and y displacement to 0 on the bottom boundary.
    bc_bottom = bc_xy(bottom_facet_marker, V_u, fdim)

    # %%
    # The bcs_list_u variable is a list that stores all boundary conditions for the displacement
    # field $\boldsymbol u$. This list facilitates easy management of multiple boundary
    # conditions and can be expanded if additional conditions are needed.
    bcs_list_u = [bc_bottom]
    bcs_list_u_names = ["bottom"]

    def update_boundary_conditions(bcs, time):
        return 0, 0, 0


    ###############################################################################
    # External Load Definition
    # ------------------------
    # Here, we define the external load to be applied to the top boundary (`ds_top`). 
    # `T_top` represents the external force applied in the y-direction.
    T_top = loading_Txy(msh)

    # %%
    # The load is added to the list of external loads, `T_list_u`, which will be updated
    # incrementally in the `update_loading` function.
    T_list_u = [[T_top, ds_top]
            ]
    f = None

    def update_loading(T_list_u, time):
        if time <= 50:
            val = 0.3 * time
        elif time <= 100:
            val = -0.3 * (time - 50) + 0.3*50
        else:
            val = 0.3 * (time - 100)

        T_list_u[0][0].value[0] = petsc4py.PETSc.ScalarType(val)
        T_list_u[0][0].value[1] = petsc4py.PETSc.ScalarType(val)
        return val, val, 0


    f = None

    final_time = 200.0
    dt = 10.0

    solve(Data,
        msh,
        final_time,
        V_u,
        bcs_list_u,
        update_boundary_conditions,
        f,
        T_list_u,
        update_loading,
        ds_list,
        dt,
        path=None,
        quadrature_degree=2,
        bcs_list_u_names=bcs_list_u_names)


    S = AllResults(Data.results_folder_name)
    psi_isotropic = S.energy_files['total.energy']["E"]
    psi_a = S.energy_files['total.energy']["PSI_a"]
    psi_b = S.energy_files['total.energy']["PSI_b"]


    np.testing.assert_allclose(
        psi_a + psi_b,
        psi_isotropic,
        rtol=1e-3,
        atol=1e-8,
        err_msg=(
            f"Sum of anisotropic energies {psi_a + psi_b} does not match "
            f"isotropic energy {psi_isotropic}"
        )
    )

    # Clean up: Remove generated files
    if os.path.exists(Data.results_folder_name):
        import shutil
        shutil.rmtree(Data.results_folder_name)
