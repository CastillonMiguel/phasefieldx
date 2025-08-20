'''
Solver: Phase-field
===================

'''
# Libraries ############################################################
########################################################################
import os
import time
import dolfinx
import ufl
from mpi4py import MPI
from dolfinx.fem.petsc import NonlinearProblem

 
from phasefieldx.files import prepare_simulation, append_results_to_file
from phasefieldx.solvers.newton import NewtonSolver
from phasefieldx.Logger.library_versions import set_logger, log_library_versions, log_system_info, log_end_analysis, log_model_information
from phasefieldx.Element.Phase_Field.energy import calculate_crack_surface_energy

def solve(Data,
          msh,
          final_time,
          V_Φ,
          bc_list_phi=[],
          update_boundary_conditions=None,
          update_loading=None,
          ds_bound=None,
          dt=1.0,
          path=None,
          quadrature_degree=2):
    """
    Solver for phase-field problems.

    Parameters
    ----------
    Data : phasefieldx.Data
        Data object containing simulation parameters.
    msh : dolfinx.cpp.mesh.Mesh
        Mesh object defining the computational domain.
    final_time : float
        Final pseudo time for the simulation.
    V_Φ : dolfinx.fem.FunctionSpace
        Function space for the phase field variable Φ.
    bc_list_phi : list of dolfinx.fem.DirichletBC, optional
        List of Dirichlet boundary conditions for Φ (default is []).
    update_boundary_conditions : function, optional
        Function to update boundary conditions based on time (default is None).
    update_loading : function, optional
        Function to update external loading conditions based on time and spatial coordinates (default is None).
    ds_bound : numpy.ndarray, optional
        Array containing boundary descriptions for reaction forces (default is None).
    dt : float, optional
        Time step size (default is 1.0).
    path : str, optional
        Path to store simulation results (default is current working directory).
    quadrature_degree : int, optional
        Quadrature degree for numerical integration (default is 2).

    Returns
    -------
    None

    Notes
    -----
    This function initializes and solves the phase-field problem for the given phase-field variable Φ,
    updating it over the specified time period using a Newton-type solver. It logs simulation progress,
    saves results, and manages output using Paraview-compatible formats.

    Examples
    --------
    # Initialize Data, msh, V_Φ, bc_list_phi, and optionally update functions
    solve(Data, msh, 10.0, V_Φ, bc_list_phi, update_boundary_conditions, update_loading)

    # This will simulate the phase-field evolution in time, saving results in the current directory.
    """

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

    metadata = {"quadrature_degree": quadrature_degree}
    # ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)
    dx = ufl.Measure("dx", domain=msh, metadata=metadata)
    # Phase-field ------------------------------------------------------------
    F_phi = (1 / Data.l * ufl.inner(Φ, δΦ) + Data.l *
             ufl.inner(ufl.grad(Φ), ufl.grad(δΦ))) * dx

    x = ufl.SpatialCoordinate(msh)
    if update_loading is not None:
        f, grad_f = update_loading(x, 0)
        # F_phi-= Data.Gc*(1/Data.l*ufl.inner(f, delta_phi) + Data.l * ufl.inner(ufl.grad(f), ufl.grad(delta_phi))) * ufl.dx
        F_phi -= (1 / Data.l * ufl.inner(f, δΦ) + Data.l *
                  ufl.inner(grad_f, ufl.grad(δΦ))) * dx

    J_phi = ufl.derivative(F_phi, Φ)
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F_phi, Φ, bcs=bc_list_phi, J=J_phi)

    solver_phi = NewtonSolver(problem)
    if rank == 0 and logger:
        solver_phi.save_log_info(logger)

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

    # Main solution loop
    t = 0
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
        
        phi_iterations, _ = solver_phi.solver.solve(Φ)
        
        if rank == 0 and logger:
            logger.info(f" Newton iterations: {phi_iterations}")
            residual_Φ = solver_phi.ksp.getResidualNorm()
            logger.info(f" Residual norm Φ: {residual_Φ}")

        # Save results - Only rank 0 writes text files
        ######################################################################
        if rank == 0:
            if logger:
                logger.info(f"\n\n Saving results: ")

            # conv ---------------------------------------------------------------
            append_results_to_file(os.path.join(
                result_folder_name, "phasefieldx.conv"), '#step\titerations', step, phi_iterations)

        # Energy -------------------------------------------------------------
        gamma, gamma_phi, gamma_gradphi = calculate_crack_surface_energy(Φ, Data.l, comm, "AT2", dx)

        # Only rank 0 writes energy results
        if rank == 0:
            append_results_to_file(os.path.join(result_folder_name, "total.energy"),
                                   '#step\tgamma\tgamma_phi\tgamma_gradphi', step, gamma, gamma_phi, gamma_gradphi)

        # Paraview -----------------------------------------------------------
        if Data.save_solution_xdmf:
            xdmf_phi.write_function(Φ, step)

        if Data.save_solution_vtu:
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
