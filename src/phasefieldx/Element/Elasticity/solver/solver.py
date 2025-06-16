'''
Solver: Elasticity
==================

'''

# Libraries ############################################################
########################################################################
import os
import time
import dolfinx
import ufl
from dolfinx.fem.petsc import NonlinearProblem


from phasefieldx.files import prepare_simulation, append_results_to_file
from phasefieldx.solvers.newton import NewtonSolver
from phasefieldx.Logger.library_versions import set_logger, log_library_versions, log_system_info, log_end_analysis, log_model_information

from phasefieldx.Materials.elastic_isotropic import epsilon, sigma, psi
from phasefieldx.Reactions import calculate_reaction_forces


def solve(Data,
          msh,
          final_time,
          V_u,
          bc_list_u=[],
          update_boundary_conditions=None,
          f_list_u=None,
          T_list_u=None,
          update_loading=None,
          ds_bound=None,
          dt=1.0,
          path=None,
          quadrature_degree=2,
          bcs_list_u_names=None):
    """
    Solver for elasticity problems.

    Parameters
    ----------
    Data : phasefieldx.Data
        Data object containing simulation parameters.
    msh : dolfinx.cpp.mesh.Mesh
        Mesh object defining the computational domain.
    final_time : float
        Final pseudo time for the simulation.
    V_u : dolfinx.fem.FunctionSpace
        Function space for the displacement field u.
    bc_list_u : list of dolfinx.fem.DirichletBC, optional
        List of Dirichlet boundary conditions for u (default is []).
    update_boundary_conditions : function, optional
        Function to update boundary conditions based on time (default is None).
    f_list_u : list of dolfinx.fem.Function, optional
        List of body forces (default is None).
    T_list_u : list of tuples, optional
        List of tuples (force, measure) for surface forces (default is None).
    update_loading : function, optional
        Function to update external loading conditions based on time (default is None).
    ds_bound : numpy.ndarray, optional
        Array containing boundary descriptions for reaction forces (default is None).
    dt : float, optional
        Time step size (default is 1.0).
    path : str, optional
        Path to store simulation results (default is the current working directory).

    Raises
    ------
    None

    Returns
    -------
    None

    Notes
    -----
    This function initializes and solves the elasticity problem for the displacement field u,
    updating it over the specified time period using a Newton-type solver. It logs simulation progress,
    saves results, and manages output using Paraview-compatible formats.

    Examples
    --------
    # Initialize Data, msh, V_u, bc_list_u, and optionally update functions
    solve(Data, msh, 10.0, V_u, bc_list_u, update_boundary_conditions, T_list_u=T_list_u, path='/path/to/results')

    # This will simulate the elasticity problem in time, saving results in the specified directory.
    """

    if path is None:
        path = os.getcwd()

    # Common #############################################################
    ######################################################################
    result_folder_name = Data.results_folder_name
    prepare_simulation(path, result_folder_name)
    logger = set_logger(result_folder_name)
    log_system_info(logger)  # log system imformation
    log_library_versions(logger)  # log Library versions
    Data.save_log_info(logger)  # log Simulation input data
    Data.save_parameters_to_csv(os.path.join(result_folder_name,"parameters.input"))
    log_model_information(msh, logger)

    # Dolfinx cpp logger
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    dolfinx.cpp.log.set_output_file(
        os.path.join(result_folder_name, "dolfinx.log"))

    if bcs_list_u_names is None:
        bcs_list_u_names = [f"bc_u_{i}" for i in range(len(bc_list_u))]

    # Formulation ##########################################################
    ########################################################################

    # Displacement ------------------------
    u = dolfinx.fem.Function(V_u, name="u")
    δu = ufl.TestFunction(V_u)

    metadata = {"quadrature_degree": quadrature_degree}
    # ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)
    dx = ufl.Measure("dx", domain=msh, metadata=metadata)

    # Displacement -----------------------------------------------------------
    F_u = ufl.inner(sigma(u, Data.lambda_, Data.mu), epsilon(δu)) * dx
    F_u_form = dolfinx.fem.form(F_u)

    # External forces --------------------------------------------------------
    if T_list_u is not None:
        L = ufl.inner(T_list_u[0][0], δu) * T_list_u[0][1]
        for i in range(1, len(T_list_u)):
            L += ufl.inner(T_list_u[i][0], δu) * T_list_u[i][1]

        F_u -= L

    J_u = ufl.derivative(F_u, u)
    J_u_form = dolfinx.fem.form(J_u)
    problem = dolfinx.fem.petsc.NonlinearProblem(F_u, u, bcs=bc_list_u, J=J_u)

    solver_u = NewtonSolver(problem)
    solver_u.save_log_info(logger)


    # Solve ################################################################
    ########################################################################
    start = time.perf_counter()

    logger.info(f" start time: {start}")

    # Paraview ------------------------
    if Data.save_solution_xdmf:
        paraview_solution_folder_name_xdmf = os.path.join(
            result_folder_name, "paraview-solutions_xdmf")
        xdmf_u = dolfinx.io.XDMFFile(msh.comm, os.path.join(
            paraview_solution_folder_name_xdmf, "u.xdmf"), "w")
        xdmf_u.write_mesh(msh)

    if Data.save_solution_vtu:
        paraview_solution_folder_name_vtu = os.path.join(
            result_folder_name, "paraview-solutions_vtu")
        vtk_sol = dolfinx.io.VTKFile(msh.comm, os.path.join(
            paraview_solution_folder_name_vtu, "phasefieldx.pvd"), "w")

    logger.info(f" S t a r t i n g    A n a l y s i s ")
    logger.info(f" ---------------------------------- ")
    logger.info(f" ---------------------------------- ")

    t = 0
    step = 0
    while t < final_time:
        logger.info(
            f"\n\nSolution at (pseudo) time = {t}, dt = {dt}, Step = {step} ")
        logger.info(
            f"===========================================================================")

        if bc_list_u is not None:
            bc_ux, bc_uy, bc_uz = update_boundary_conditions(bc_list_u, t)
        else:
            bc_ux, bc_uy, bc_uz = 0, 0, 0

        if T_list_u is not None:
            T_ux, T_uy, T_uz = update_loading(T_list_u, t)

        # Displacement --------------------------------------------
        logger.info(f">>> Solving phase for dofs: u ")
        u_iterations, _ = solver_u.solver.solve(u)
        # residuals = solver_u.ksp.getConvergenceHistory()
        logger.info(f" Newton iterations: {u_iterations}")
        resisual_u = solver_u.ksp.getResidualNorm()
        logger.info(f" Residual norm u: {resisual_u}")

        # Save results
        ######################################################################
        logger.info(f"\n\n Saving results: ")

        # conv ---------------------------------------------------------------
        append_results_to_file(os.path.join(
            result_folder_name, "phasefieldx.conv"), '#step\titerations', step, u_iterations)

        # Degree of freedom --------------------------------------------------
        append_results_to_file(os.path.join(
            result_folder_name, "top.dof"), '#step\tUx\tUy\tUz', step, bc_ux, bc_uy, bc_uz)

        # Reaction -----------------------------------------------------------
        for i in range(0, len(bc_list_u)):
            R = calculate_reaction_forces(J_u_form, F_u_form, [bc_list_u[i]], u, msh.topology.dim)
            append_results_to_file(os.path.join(result_folder_name, bcs_list_u_names[i] + ".reaction"), '#step\tRx\tRy\tRz', step, R[0], R[1], R[2])

        # Energy -------------------------------------------------------------
        E = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
            psi(u, Data.lambda_, Data.mu) * dx))
        append_results_to_file(os.path.join(
            result_folder_name, "total.energy"), '#step\tE', step, E)

        # Paraview -----------------------------------------------------------
        if Data.save_solution_xdmf:
            xdmf_u.write_function(u, step)

        if Data.save_solution_vtu:
            vtk_sol.write_function([u], step)

        t += dt
        step += 1

    if Data.save_solution_xdmf:
        xdmf_u.close()

    if Data.save_solution_vtu:
        vtk_sol.close()

    end = time.perf_counter()
    log_end_analysis(logger, end - start)
