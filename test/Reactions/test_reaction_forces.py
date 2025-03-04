import pytest
import numpy as np
import dolfinx
import mpi4py
import petsc4py


###############################################################################
# Import from phasefieldx package
# -------------------------------
from phasefieldx.Element.Elasticity.Input import Input
from phasefieldx.Element.Elasticity.solver.solver import solve
from phasefieldx.Boundary.boundary_conditions import bc_x, bc_xy, bc_xyz, get_ds_bound_from_marker
from phasefieldx.PostProcessing.ReferenceResult import AllResults


def run_2d_simulation():
    Data = Input(E=210.0,
                 nu=0.3,
                 save_solution_xdmf=False,
                 save_solution_vtu=True,
                 results_folder_name="2_dimension")

    divx, divy = 1, 1
    lx, ly = 1.0, 1.0
    msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                                        [np.array([0, 0]),
                                         np.array([lx, ly])],
                                        [divx, divy],
                                        cell_type=dolfinx.mesh.CellType.quadrilateral)

    def bottom(x):
        return np.isclose(x[1], 0)

    def top(x):
        return np.isclose(x[1], ly)

    fdim = msh.topology.dim - 1

    bottom_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, bottom)
    top_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, top)

    ds_bottom = get_ds_bound_from_marker(bottom_facet_marker, msh, fdim)
    ds_top = get_ds_bound_from_marker(top_facet_marker, msh, fdim)

    ds_list = np.array([
        [ds_top, "top"],
        [ds_bottom, "bottom"],
    ])

    V_u = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    bc_bottom = bc_xy(bottom_facet_marker, V_u, fdim, value_x=0.0, value_y=0.0)
    bc_top = bc_xy(top_facet_marker, V_u, fdim, value_x=0.0, value_y=0.0)

    bcs_list_u = [bc_top, bc_bottom]
    bcs_list_u_names = ["top", "bottom"]

    def update_boundary_conditions(bcs, time):
        val = 0.0003 * time
        bcs[0].g.value[1] = petsc4py.PETSc.ScalarType(val)
        return 0, val, 0

    T_list_u = None
    update_loading = None
    f = None

    final_time = 10.0
    dt = 1.0

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


def run_3d_simulation():
    Data = Input(E=210.0,
                 nu=0.3,
                 save_solution_xdmf=False,
                 save_solution_vtu=True,
                 results_folder_name="3_dimension")

    divx, divy, divz = 1, 1, 1
    lx, ly, lz = 1.0, 1.0, 1.0
    msh = dolfinx.mesh.create_box(mpi4py.MPI.COMM_WORLD,
                                  [np.array([0, 0, 0]),
                                   np.array([lx, ly, lz])],
                                  [divx, divy, divz],
                                  cell_type=dolfinx.mesh.CellType.hexahedron)

    def bottom(x):
        return np.isclose(x[1], 0)

    def top(x):
        return np.isclose(x[1], ly)

    fdim = msh.topology.dim - 1

    bottom_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, bottom)
    top_facet_marker = dolfinx.mesh.locate_entities_boundary(msh, fdim, top)

    ds_bottom = get_ds_bound_from_marker(bottom_facet_marker, msh, fdim)
    ds_top = get_ds_bound_from_marker(top_facet_marker, msh, fdim)

    ds_list = np.array([
        [ds_top, "top"],
        [ds_bottom, "bottom"],
    ])

    V_u = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))

    bc_bottom = bc_xyz(bottom_facet_marker, V_u, fdim, value_x=0.0, value_y=0.0, value_z=0.0)
    bc_top = bc_xyz(top_facet_marker, V_u, fdim, value_x=0.0, value_y=0.0, value_z=0.0)

    bcs_list_u = [bc_top, bc_bottom]
    bcs_list_u_names = ["top", "bottom"]

    def update_boundary_conditions(bcs, time):
        val = 0.0003 * time
        bcs[0].g.value[1] = petsc4py.PETSc.ScalarType(val)
        return 0, val, 0

    T_list_u = None
    update_loading = None
    f = None

    final_time = 10.0
    dt = 1.0

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


@pytest.fixture(scope="module")
def run_simulations():
    # Run the simulations for 1D, 2D, and 3D cases
    run_2d_simulation()
    run_3d_simulation()
    # Load results for 1D, 2D, and 3D cases
    S_2d = AllResults("2_dimension")
    S_3d = AllResults("3_dimension")

    return S_2d, S_3d


def test_reaction_forces(run_simulations):

    S_2d, S_3d = run_simulations

    # Extract reaction forces
    reaction_2d = S_2d.reaction_files['bottom.reaction']["Ry"]
    reaction_3d = S_3d.reaction_files['bottom.reaction']["Ry"]

    # Check if the reaction forces are equal
    np.testing.assert_allclose(reaction_2d, reaction_3d, rtol=1e-5, atol=1e-8, err_msg="2D and 3D reaction forces do not match")
