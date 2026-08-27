'''
Solver: Allen-Cahn
==================


'''

# Libraries ############################################################
########################################################################
import os
import time
import dolfinx
import ufl

from phasefieldx.files import prepare_simulation, append_results_to_file
from phasefieldx.Logger.library_versions import set_logger, log_library_versions, log_system_info, log_end_analysis, log_model_information, log_header
from phasefieldx.Element.Allen_Cahn.potential import potential_function, potential_function_derivative, potential_coefficient
from phasefieldx.Element.Allen_Cahn.energy import calculate_potential_energy


def solve(Data,
          msh,
          final_time,
          V_Φ,
          bc_list_phi=[],
          update_boundary_conditions=None,
          update_loading=None,
          initial_condition=None,
          ds_bound=None,
          dt=1.0,
          path=None,
          quadrature_degree=2,
          case='DOUBLE',
          V_gradient_Φ=None):
    """
    Solver for the Allen-Cahn equation.

    Parameters
    ----------
    Data : phasefieldx.Data
        Data object containing simulation parameters.
    msh : dolfinx.cpp.mesh.Mesh
        Mesh object defining the computational domain.
    final_time : float
        Final pseudo time for the simulation.
    V_phi : dolfinx.fem.FunctionSpace
        Function space for the phase-field variable phi.
    bc_list_phi : list of dolfinx.fem.DirichletBC, optional
        List of Dirichlet boundary conditions for phi (default is []).
    update_boundary_conditions : function, optional
        Function to update boundary conditions based on time (default is None).
    update_loading : function, optional
        Function to update external loading conditions based on time (default is None).
    initial_condition : dolfinx.fem.Function, optional
        Initial condition for phi (default is None).
    ds_bound : numpy.ndarray, optional
        Array containing boundary descriptions for reaction forces (default is None).
    dt : float, optional
        Time step size (default is 1.0).
    path : str, optional
        Path to store simulation results (default is current working directory).

    Returns
    -------
    None

    Notes
    -----
    This function initializes and solves the Allen-Cahn equation for the phase-field variable phi,
    which represents a free energy minimization problem. It uses a Newton-type solver to update phi
    over the specified time period. Simulation progress is logged, and results are saved using
    Paraview-compatible formats.

    Examples
    --------
    # Initialize Data, msh, V_phi, and optionally update functions
    solve(Data, msh, 10.0, V_phi, bc_list_phi, update_boundary_conditions, update_loading)

    # This will simulate the free energy minimization problem using Allen-Cahn equation,
    # saving results in the specified directory.
    """
    M = Data.mobility

    # Get MPI communicator info
    comm = msh.comm
    rank = comm.Get_rank()

    if path is None:
        path = os.getcwd()
   
    # Common - Only rank 0 handles file operations
    ######################################################################
    result_folder_name = Data.results_folder_name

    if rank == 0:
        prepare_simulation(path, result_folder_name)
        logger = set_logger(result_folder_name)
        log_header(logger)
        log_system_info(logger)  # log system information
        log_library_versions(logger)  # log Library versions
        Data.save_log_info(logger)  # log Simulation input data
        Data.save_parameters_to_csv(os.path.join(result_folder_name, "parameters.input"))
        log_model_information(msh, logger)
    else:
        logger = None

    # Synchronize all processes
    comm.Barrier()

    # Dolfinx cpp logger - all processes
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    if rank == 0:
        dolfinx.cpp.log.set_output_file(
            os.path.join(result_folder_name, "dolfinx.log"))

    # Formulation ##########################################################
    ########################################################################

    # Phase-field -------------------------
    Φ = dolfinx.fem.Function(V_Φ, name="phi")
    δΦ = ufl.TestFunction(V_Φ)

    Φ0 = dolfinx.fem.Function(V_Φ, name="phi")

    metadata = {"quadrature_degree": quadrature_degree}
    dx = ufl.Measure("dx", domain=msh, metadata=metadata)

    x = ufl.SpatialCoordinate(msh)
    if initial_condition is not None:
        Φ.interpolate(initial_condition)
        Φ0.interpolate(initial_condition)
        Φ.x.array[:] = Φ0.x.array

    # Phase-field ------------------------------------------------------------
    c0 = potential_coefficient(case)
    
    F_phi = M*(1.0/(c0*Data.l)*potential_function_derivative(Φ, case)*δΦ + Data.l * 2/c0*ufl.inner(ufl.grad(Φ), ufl.grad(δΦ)))*dx
    F_phi += 1.0/dt*ufl.inner(Φ, δΦ) * dx 
    F_phi -= 1.0/dt*ufl.inner(Φ0, δΦ) * dx
    
    # x = ufl.SpatialCoordinate(msh)

    J_phi = ufl.derivative(F_phi, Φ)
    petsc_options_phi = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_linesearch_type": "none",
        "snes_max_it": 50000,
        "snes_rtol": 1e-8,
        "snes_atol": 1e-9,
    }

    problem_phi = dolfinx.fem.petsc.NonlinearProblem(
        F_phi,
        Φ,
        bcs=bc_list_phi,
        J=J_phi,
        petsc_options=petsc_options_phi,
        petsc_options_prefix="phase_field",
    )
    snes_phi = problem_phi.solver

    if rank == 0 and logger:
        logger.info(" SNES Settings:")
        for key, value in petsc_options_phi.items():
            logger.info(f"   {key}: {value}")

    # Solve ################################################################
    ########################################################################
    start = time.perf_counter()
    if rank == 0 and logger:
        logger.info(f" start time: {start}")

    # Paraview files - DOLFINx handles parallel I/O automatically
    if Data.save_solution_xdmf:
        paraview_solution_folder_name_xdmf = os.path.join(
            result_folder_name, "paraview-solutions_xdmf")
        # Create directory only on rank 0
        if rank == 0:
            os.makedirs(paraview_solution_folder_name_xdmf, exist_ok=True)
        comm.Barrier()  # Wait for directory creation
        
        xdmf_phi = dolfinx.io.XDMFFile(msh.comm, os.path.join(
            paraview_solution_folder_name_xdmf, "phi.xdmf"), "w")
        xdmf_phi.write_mesh(msh)

    if V_gradient_Φ is not None:
        gradient_Φ = dolfinx.fem.Function(V_gradient_Φ, name="gradient_phi")
    
    if Data.save_solution_vtu:
        paraview_solution_folder_name_vtu = os.path.join(
            result_folder_name, "paraview-solutions_vtu")
        # Create directory only on rank 0
        if rank == 0:
            os.makedirs(paraview_solution_folder_name_vtu, exist_ok=True)
        
        comm.Barrier()  # Wait for directory creation
        
        vtk_sol = dolfinx.io.VTKFile(msh.comm, os.path.join(
            paraview_solution_folder_name_vtu, "phasefieldx.pvd"), "w")

    if rank == 0 and logger:
        logger.info(f" S t a r t i n g    A n a l y s i s ")
        logger.info(f" ---------------------------------- ")
        logger.info(f" ---------------------------------- ")


    t = 0.0
    step = 0
    while t < final_time:
        if rank == 0 and logger:
            logger.info(
                f"\n\nSolution at (pseudo) time = {t}, dt = {dt}, Step = {step} ")
            logger.info(
                f"===========================================================================")

        if update_boundary_conditions is not None:
            bc_ux = update_boundary_conditions(bc_list_phi, t)

        if update_loading is not None:
            f, grad_f = update_loading(x, 0)

        # Phase-field solution - all processes participate
        if rank == 0 and logger:
            logger.info(f">>> Solving phase for dofs: Φ ")
        problem_phi.solve()
        converged = problem_phi.solver.getConvergedReason()
        phi_iterations = problem_phi.solver.getIterationNumber()
        
        residuals = snes_phi.getConvergenceHistory()
        residual_norm_phi = snes_phi.getFunctionNorm()

        if rank == 0 and logger:
            if converged <= 0:
                logger.error(f"Solver did not converge, got {converged}.")
                raise RuntimeError(f"Solver did not converge, got {converged}.")
            else:
                logger.info(f" Newton iterations: {phi_iterations}")
                logger.info(f" Residual norm Φ: {residual_norm_phi}")
                logger.info(f" Converged reason {converged}.")
                logger.info(f" Residual history Φ: {residuals}")

        # Update phi0
        Φ0.x.array[:] = Φ.x.array

        # Save results - Only rank 0 writes text files
        ######################################################################
        if rank == 0:
            if logger:
                logger.info(f"\n\n Saving results: ")

            # conv ---------------------------------------------------------------
            append_results_to_file(os.path.join(
                result_folder_name, "phasefieldx.conv"), '#step\titerations', step, phi_iterations)

        # Energy -------------------------------------------------------------

        
        # # Energy -------------------------------------------------------------
        gamma, gamma_phi, gamma_gradphi = calculate_potential_energy(Φ, Data.l, comm, case, dx)

        # Only rank 0 writes energy results
        if rank == 0:
            append_results_to_file(os.path.join(result_folder_name, "total.energy"),
                                   '#step\tgamma\tgamma_phi\tgamma_gradphi', step, gamma, gamma_phi, gamma_gradphi)

        if V_gradient_Φ is not None:
            # Compute gradient of Φ and save to gradient_Φ function
            gradient_expr = dolfinx.fem.Expression(ufl.grad(Φ), V_gradient_Φ.element.interpolation_points)
            gradient_Φ.interpolate(gradient_expr)
            
        # Paraview -----------------------------------------------------------
        if Data.save_solution_xdmf:
            xdmf_phi.write_function(Φ, step)
            if V_gradient_Φ is not None:
                xdmf_phi.write_function(gradient_Φ, step)

        if Data.save_solution_vtu:
            if V_gradient_Φ is not None:
                vtk_sol.write_function([Φ, gradient_Φ], step)
            else:
                vtk_sol.write_function([Φ], step)

        t += dt
        step += 1

    # Cleanup - all processes
    if Data.save_solution_xdmf:
        xdmf_phi.close()

    if Data.save_solution_vtu:
        vtk_sol.close()

    if rank == 0:
        end = time.perf_counter()
        if logger:
            log_end_analysis(logger, end - start)
