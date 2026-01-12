'''
Solver: Elasticity (Anisotropic Decomposition)
==============================================

'''

# Libraries ############################################################
########################################################################
import os
import time
import dolfinx
import ufl
from mpi4py import MPI

from phasefieldx.files import prepare_simulation, append_results_to_file
from phasefieldx.solvers.newton import NewtonSolver
from phasefieldx.Logger.library_versions import set_logger, log_library_versions, log_system_info, log_end_analysis, log_model_information

from phasefieldx.Materials.elastic_isotropic import epsilon, sigma, psi
from phasefieldx.Element.Phase_Field_Fracture.split_energy_stress_tangent_functions import (psi_a, psi_b, sigma_a,
                                                                                            sigma_b)
from phasefieldx.Reactions import calculate_reaction_forces
from phasefieldx.Element.Elasticity.energy import calculate_elastic_energy
from phasefieldx.Element.Phase_Field_Fracture.energy import compute_elastic_energy_components
from phasefieldx.Math.projection import project

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

        if bcs_list_u_names is None:
            bcs_list_u_names = [f"bc_u_{i}" for i in range(len(bc_list_u))]

    # Formulation ##########################################################
    ########################################################################

    # Displacement ------------------------
    u = dolfinx.fem.Function(V_u, name="u")
    δu = ufl.TestFunction(V_u)
    
    V_energies = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    PSI_a = dolfinx.fem.Function(V_energies, name="PSI_a")
    PSI_b = dolfinx.fem.Function(V_energies, name="PSI_b")


    metadata = {"quadrature_degree": quadrature_degree}
    # ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)
    dx = ufl.Measure("dx", domain=msh, metadata=metadata)

    # Displacement -----------------------------------------------------------
    F_u = ufl.inner(sigma_a(u, Data) + sigma_b(u, Data), epsilon(δu)) * dx
    F_u_form = dolfinx.fem.form(F_u)

    # External forces --------------------------------------------------------
    if T_list_u is not None:
        L = ufl.inner(T_list_u[0][0], δu) * T_list_u[0][1]
        for i in range(1, len(T_list_u)):
            L += ufl.inner(T_list_u[i][0], δu) * T_list_u[i][1]

        F_u -= L

    J_u = ufl.derivative(F_u, u)
    J_u_form = dolfinx.fem.form(J_u)
    petsc_options_u = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_linesearch_type": "none",
        "snes_max_it": 50000,
        "snes_rtol": 1e-8,
        "snes_atol": 1e-9,
    }

    problem_u = dolfinx.fem.petsc.NonlinearProblem(
        F_u,
        u,
        bcs=bc_list_u,
        J=J_u,
        petsc_options=petsc_options_u,
        petsc_options_prefix="elasticity",
    )
    snes_u = problem_u.solver

    if rank == 0 and logger:
        logger.info(" SNES Settings:")
        for key, value in petsc_options_u.items():
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
        xdmf_u = dolfinx.io.XDMFFile(msh.comm, os.path.join(
            paraview_solution_folder_name_xdmf, "u.xdmf"), "w")
        xdmf_u.write_mesh(msh)

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

    t = 0
    step = 0
    while t < final_time:
        if rank == 0 and logger:
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
        if rank == 0 and logger:
            logger.info(f">>> Solving phase for dofs: u ")
            
        problem_u.solve()
        converged = problem_u.solver.getConvergedReason()
        u_iterations = problem_u.solver.getIterationNumber()
        
        residuals = snes_u.getConvergenceHistory()
        residual_norm_u = snes_u.getFunctionNorm()

        if rank == 0 and logger:
            if converged <= 0:
                logger.error(f"Solver did not converge, got {converged}.")
                raise RuntimeError(f"Solver did not converge, got {converged}.")
            else:
                logger.info(f" Newton iterations: {u_iterations}")
                logger.info(f" Residual norm u: {residual_norm_u}")
                logger.info(f" Converged reason {converged}.")
                logger.info(f" Residual history u: {residuals}")
            
        # Save results - Only rank 0 writes text files
        ######################################################################
        if rank == 0:
            if logger:
                logger.info(f"\n\n Saving results: ")

            # conv ---------------------------------------------------------------
            append_results_to_file(os.path.join(
                result_folder_name, "phasefieldx.conv"), '#step\titerations', step, u_iterations)


            # Degree of freedom --------------------------------------------------
            append_results_to_file(os.path.join(
                result_folder_name, "top.dof"), '#step\tUx\tUy\tUz', step, bc_ux, bc_uy, bc_uz)

        # Reaction -----------------------------------------------------------
        if comm.Get_size() == 1: # Only available for single process
            for i in range(0, len(bc_list_u)):
                R = calculate_reaction_forces(J_u_form, F_u_form, [bc_list_u[i]], u, V_u, msh.topology.dim)
                append_results_to_file(os.path.join(result_folder_name, bcs_list_u_names[i] + ".reaction"), '#step\tRx\tRy\tRz', step, R[0], R[1], R[2])

        # Energy -------------------------------------------------------------        
        E = calculate_elastic_energy(u, Data, comm, dx)
        
        psi_a_, psi_b_ = compute_elastic_energy_components(u, Data, comm, dx=dx)
        
        project(psi_a(u, Data), PSI_a)
        project(psi_b(u, Data), PSI_b)


        # Only rank 0 writes energy results
        if rank == 0:
            append_results_to_file(os.path.join(
                result_folder_name, "total.energy"), '#step\tE\tPSI_a\tPSI_b', step, E, psi_a_, psi_b_)

        # Paraview -----------------------------------------------------------
        if Data.save_solution_xdmf:
            xdmf_u.write_function(u, step)

        if Data.save_solution_vtu:
            vtk_sol.write_function([u, PSI_a, PSI_b], step)

        t += dt
        step += 1

    # Cleanup - all processes
    if Data.save_solution_xdmf:
        xdmf_u.close()

    if Data.save_solution_vtu:
        vtk_sol.close()

    if rank == 0:
        end = time.perf_counter()
        if logger:
            log_end_analysis(logger, end - start)
