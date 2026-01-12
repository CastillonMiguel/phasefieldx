'''
Solver: Phase-field fracture/fatigue
====================================

This module provides a solver for phase-field fracture and fatigue simulations
using the Finite Element Method (FEM).

'''

# Libraries ############################################################
########################################################################
import os
import time
import dolfinx
import ufl
from mpi4py import MPI
import numpy as np

from phasefieldx.files import prepare_simulation, append_results_to_file
from phasefieldx.solvers.newton import NewtonSolver
from phasefieldx.Logger.library_versions import set_logger, log_system_info, log_library_versions, log_end_analysis, log_model_information


from phasefieldx.Materials.elastic_isotropic import epsilon
from phasefieldx.Reactions import calculate_reaction_forces


from phasefieldx.Element.Phase_Field_Fracture.g_degradation_functions import dg, g
from phasefieldx.Element.Phase_Field_Fracture.split_energy_stress_tangent_functions import (psi_a, psi_b, sigma_a,
                                                                                            sigma_b)
from phasefieldx.Element.Phase_Field_Fracture.fatigue_degradation_functions import fatigue_degradation

from phasefieldx.Element.Phase_Field.geometric_crack import geometric_crack_function_derivative, geometric_crack_coefficient
from phasefieldx.Element.Phase_Field.energy import calculate_crack_surface_energy
from phasefieldx.Element.Phase_Field_Fracture.energy import compute_total_energies

from phasefieldx.Math.projection import project
from phasefieldx.errors_functions import eval_error_L2, eval_error_L2_normalized
from phasefieldx.files import prepare_simulation
from phasefieldx.solvers.newton import NewtonSolver


def solve(Data,
          msh,
          final_time,
          V_u,
          V_Φ,
          bc_list_u=[],
          bc_list_phi=[],
          update_boundary_conditions=None,
          f_list_u=None,
          T_list_u=None,
          update_loading=None,
          ds_bound=None,
          dt=1.0,
          path=None,
          bcs_list_u_names=None,
          min_stagger_iter=2,
          max_stagger_iter=10000,
          stagger_error_tol=1e-8):
    """
    Solve the phase-field fracture and fatigue problem.

    Parameters
    ----------
    Data : object
        An object containing the simulation parameters and settings.
    msh : dolfinx.mesh
        The computational mesh for the simulation.
    final_time : float
        The final simulation time.
    V_u : dolfinx.FunctionSpace
        The function space for the displacement field.
    V_Φ : dolfinx.FunctionSpace
        The function space for the phase-field.
    bc_list_u : list of dolfinx.DirichletBC, optional
        List of Dirichlet boundary conditions for the displacement field.
    bc_list_phi : list of dolfinx.DirichletBC, optional
        List of Dirichlet boundary conditions for the phase-field.
    update_boundary_conditions : callable, optional
        A function to update boundary conditions dynamically.
    f_list_u : list, optional
        List of body forces.
    T_list_u : list of tuple, optional
        List of tuples containing traction forces and corresponding measures.
    update_loading : callable, optional
        A function to update loading conditions dynamically.
    ds_bound : numpy.ndarray, optional
        Array of boundary measures for calculating reaction forces.
    dt : float, optional
        Time step size.
    path : str, optional
        Directory path for saving simulation results. Defaults to current working directory.
    min_stagger_iter: int
        Minimum stagger iterations for numerical simulations.
    max_stagger_iter: int
        Maximum stagger iterations for numerical simulations.
    stagger_error_tol: float
        Tolerance for stagger error in simulations.

    Returns
    -------
    None
    """
    case = 'AT2'
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
        logger.info("========== Stagger settings ===========")
        logger.info(f"  minimum stagger iterations: {min_stagger_iter}")
        logger.info(f"  maximum stagger iterations: {max_stagger_iter}")
        logger.info(f"  stagger error tolerance: {stagger_error_tol}")
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
    u_new = dolfinx.fem.Function(V_u, name="u")
    u_old = dolfinx.fem.Function(V_u, name="u_old")
    δu = ufl.TestFunction(V_u)

    # Phase-field -------------------------
    Φ_new = dolfinx.fem.Function(V_Φ, name="phi")
    Φ_old = dolfinx.fem.Function(V_Φ, name="phi_old")
    δΦ = ufl.TestFunction(V_Φ)

    # History------------------------------
    H = dolfinx.fem.Function(V_Φ, name="H")
    V_c = dolfinx.fem.Function(V_Φ, name="V_c")
    V_n = dolfinx.fem.Function(V_Φ, name="V_n")

    # Fatigue------------------------------
    if Data.fatigue:
        Fatigue = dolfinx.fem.Function(V_Φ, name="fatigue")

        alpha_c = dolfinx.fem.Function(V_Φ, name="alpha_c")
        alpha_n = dolfinx.fem.Function(V_Φ, name="alpha_n")
        alpha_cum_bar_c = dolfinx.fem.Function(V_Φ, name="alpha_cum_bar_c")
        alpha_cum_bar_n = dolfinx.fem.Function(V_Φ, name="alpha_cum_bar_n")
        delta_alpha = dolfinx.fem.Function(V_Φ, name="delta_alpha")

    # Displacement -----------------------------------------------------------
    F_u = ufl.inner((g(Φ_new, Data.degradation_function) + Data.k) *
                    sigma_a(u_new, Data) + sigma_b(u_new, Data), epsilon(δu)) * ufl.dx
    #F_u_form = dolfinx.fem.form(F_u)
    F_u_form = dolfinx.fem.form(ufl.inner((g(Φ_new, Data.degradation_function) + Data.k) *
                    sigma_a(u_new, Data) + sigma_b(u_new, Data), epsilon(δu)) * ufl.dx)
    # External forces --------------------------------------------------------
    if T_list_u is not None:
        L = ufl.inner(T_list_u[0][0], δu) * T_list_u[0][1]
        for i in range(1, len(T_list_u)):
            L += ufl.inner(T_list_u[i][0], δu) * T_list_u[i][1]

        F_u -= L

    J_u = ufl.derivative(F_u, u_new)
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
        u_new,
        bcs=bc_list_u,
        J=J_u,
        petsc_options=petsc_options_u,
        petsc_options_prefix="elasticity",
    )
    snes_u = problem_u.solver

    if rank == 0 and logger:
        logger.info(" SNES Settings u:")
        for key, value in petsc_options_u.items():
            logger.info(f"   {key}: {value}")

    # Phase-field ------------------------------------------------------------
    F_phi = dg(Φ_new, Data.degradation_function) * H * δΦ * ufl.dx
    c0 = geometric_crack_coefficient(case)
    
    if Data.fatigue:
        F_phi += Fatigue * Data.Gc * (1 / Data.l * ufl.inner(Φ_new, δΦ) + Data.l *
                                      ufl.inner(ufl.grad(Φ_new), ufl.grad(δΦ))) * ufl.dx
    else:
        F_phi += Data.Gc*(1.0/(c0*Data.l)*geometric_crack_function_derivative(Φ_new, case)*δΦ + Data.l * 2/c0*ufl.inner(ufl.grad(Φ_new), ufl.grad(δΦ)))*ufl.dx

    J_phi = ufl.derivative(F_phi, Φ_new)
    problem_phi = dolfinx.fem.petsc.NewtonSolverNonlinearProblem(
        F_phi, Φ_new, bcs=bc_list_phi, J=J_phi)

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
        Φ_new,
        bcs=bc_list_phi,
        J=J_phi,
        petsc_options=petsc_options_phi,
        petsc_options_prefix="phase_field",
    )
    snes_phi = problem_phi.solver

    if rank == 0 and logger:
        logger.info(" SNES Settings phi:")
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
        xdmf_phi = dolfinx.io.XDMFFile(msh.comm, os.path.join(
            paraview_solution_folder_name_xdmf, "phi.xdmf"), "w")
        xdmf_phi.write_mesh(msh)

        xdmf_u = dolfinx.io.XDMFFile(msh.comm, os.path.join(
            paraview_solution_folder_name_xdmf, "u.xdmf"), "w")
        
        # Create directory only on rank 0
        if rank == 0:
            os.makedirs(paraview_solution_folder_name_xdmf, exist_ok=True)
        comm.Barrier()  # Wait for directory creation
        xdmf_phi = dolfinx.io.XDMFFile(msh.comm, os.path.join(
            paraview_solution_folder_name_xdmf, "phi.xdmf"), "w")
        xdmf_phi.write_mesh(msh)
        
        xdmf_u = dolfinx.io.XDMFFile(msh.comm, os.path.join(
            paraview_solution_folder_name_xdmf, "u.xdmf"), "w")

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

        if update_boundary_conditions is not None:
            bc_ux, bc_uy, bc_uz = update_boundary_conditions(bc_list_u, t)
        else:
            bc_ux, bc_uy, bc_uz = 0, 0, 0

        if update_loading is not None:
            T_ux, T_uy, T_uz = update_loading(T_list_u, t)

        # if Data.fatigue:
        #    delta_alpha = np.abs(alpha_c.x.array - alpha_n.x.array) * np.heaviside((alpha_c.x.array - alpha_n.x.array) / dt, 1)
        #    alpha_cum_bar_c.x.array[:] = alpha_cum_bar_n.x.array + delta_alpha
        #    Fatigue.x.array[:] = fatigue_degradation(alpha_cum_bar_c.x.array, Data)

        error_L2_phi = 1
        error_L2_u = 1
        stagger_iter = 0
        while (error_L2_phi > stagger_error_tol or error_L2_u > stagger_error_tol or stagger_iter <
               min_stagger_iter) and (stagger_iter < max_stagger_iter):
            stagger_iter += 1
            
            if rank == 0 and logger:
                logger.info(f" Stagger Iteration: {stagger_iter}")
                logger.info(f" ---------------------- ")

            # Displacement --------------------------------------------
            if rank == 0 and logger:
                logger.info(f">>> Solving phase for dofs: u ")
            problem_u.solve()
            converged = problem_u.solver.getConvergedReason()
            u_iterations = problem_u.solver.getIterationNumber()
            
            residuals = snes_u.getConvergenceHistory()
            residual_norm_u = snes_u.getFunctionNorm()
            error_L2_u = eval_error_L2_normalized(u_new, u_old, msh)

            if rank == 0 and logger:
                if converged <= 0:
                    logger.error(f"Solver did not converge, got {converged}.")
                    raise RuntimeError(f"Solver did not converge, got {converged}.")
                else:
                    logger.info(f" Newton iterations: {u_iterations}")
                    logger.info(f" Residual norm u: {residual_norm_u}")
                    logger.info(f" Converged reason {converged}.")
                    logger.info(f" Residual history u: {residuals}")
                    logger.info(f" L2 error in u   direction:  {error_L2_u}")

            u_old.x.array[:] = u_new.x.array

            #  Irreversibility ....
            project(psi_a(u_new, Data), V_c)

            if Data.irreversibility == "miehe":
                H.x.array[:] = np.maximum(V_c.x.array, V_n.x.array)
            else:
                H.x.array[:] = V_c.x.array

            if Data.fatigue:
                project(g(Φ_old, Data.degradation_function) * V_c, alpha_c)

                accelerated = False
                if accelerated:
                    N = 100
                    delta_alpha = N * alpha_c.x.array
                else:
                    delta_alpha = np.abs(alpha_c.x.array - alpha_n.x.array) * \
                        np.heaviside(
                            (alpha_c.x.array - alpha_n.x.array) / dt, 1)
                alpha_cum_bar_c.x.array[:] = alpha_cum_bar_n.x.array + delta_alpha
                Fatigue.x.array[:] = fatigue_degradation(
                    alpha_cum_bar_c.x.array, Data)

            # Phase-field ---------------------------------------------
            if rank == 0 and logger:
                logger.info(f">>> Solving phase for dofs: Φ ")
            problem_phi.solve()
            converged = problem_phi.solver.getConvergedReason()
            phi_iterations = problem_phi.solver.getIterationNumber()
        
            residuals = snes_phi.getConvergenceHistory()
            residual_norm_phi = snes_phi.getFunctionNorm()
            error_L2_phi = eval_error_L2_normalized(Φ_new, Φ_old, msh)
            if rank == 0 and logger:
                if converged <= 0:
                    logger.error(f"Solver did not converge, got {converged}.")
                    raise RuntimeError(f"Solver did not converge, got {converged}.")
                else:
                    logger.info(f" Newton iterations: {phi_iterations}")
                    logger.info(f" Residual norm Φ: {residual_norm_phi}")
                    logger.info(f" Converged reason {converged}.")
                    logger.info(f" Residual history Φ: {residuals}")
                    logger.info(f" L2 error in Φ direction:  {error_L2_phi}")

            Φ_old.x.array[:] = Φ_new.x.array

        # Irreversibility ....
        if Data.irreversibility == "miehe":
            V_n.x.array[:] = np.maximum(V_c.x.array, V_n.x.array)

        if Data.fatigue:
            # project(g(Φ_new, Data.degradation_function) * V_c, alpha_c)
            # delta_alpha = np.abs(alpha_c.x.array - alpha_n.x.array) * np.heaviside((alpha_c.x.array - alpha_n.x.array) / dt, 1)

            # alpha_cum_bar_c.x.array[:] = alpha_cum_bar_c.x.array + delta_alpha
            # + delta_alpha
            alpha_cum_bar_n.x.array[:] = alpha_cum_bar_c.x.array
            alpha_n.x.array[:] = alpha_c.x.array

        # Save results
        ######################################################################
        if rank == 0 and logger:
            logger.info(f"\n\n Saving results: ")

            # conv ---------------------------------------------------------------
            append_results_to_file(os.path.join(result_folder_name, "phasefieldx.conv"),
                                '#step\tstagger\titerPhi\titerU', step, stagger_iter, phi_iterations, u_iterations)

            # Degree of freedom --------------------------------------------------
            append_results_to_file(os.path.join(result_folder_name, "top.dof"),
                                '#step\tUx\tUy\tUz\tphi', step, bc_ux, bc_uy, bc_uz, 0.0)

        # Reaction -----------------------------------------------------------
        if comm.Get_size() == 1: # Only available for single process
            for i in range(0, len(bc_list_u)):
                R = calculate_reaction_forces(J_u_form, F_u_form, [bc_list_u[i]], u_new, V_u, msh.topology.dim)
                append_results_to_file(os.path.join(
                    result_folder_name, bcs_list_u_names[i] + ".reaction"), '#step\tRx\tRy\tRz', step, R[0], R[1], R[2])

        # Energy -------------------------------------------------------------
        E, PSI_a, PSI_b = compute_total_energies(u_new, Φ_new, Data, comm, dx=ufl.dx)
        gamma, gamma_phi, gamma_gradphi = calculate_crack_surface_energy(Φ_new, Data.l, comm, case, ufl.dx)

        if Data.fatigue:
            W_phi = comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(
                Fatigue * Data.Gc * 1 / (2 * Data.l) * ufl.inner(Φ_new, Φ_new) * ufl.dx)),op=MPI.SUM)
            W_gradphi = comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(
                Fatigue * Data.Gc * Data.l / 2 * ufl.inner(ufl.grad(Φ_new), ufl.grad(Φ_new)) * ufl.dx)),op=MPI.SUM)
            alpha_acum = comm.allreduce(dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(alpha_cum_bar_c * ufl.dx)),op=MPI.SUM)
        else:
            W_phi = Data.Gc * gamma_phi
            W_gradphi = Data.Gc * gamma_gradphi
            alpha_acum = 0.0

        W = W_phi + W_gradphi

        EplusW = E + W

        # Only rank 0 writes energy results
        if rank == 0:
            header = '#step\tEplusW\tE\tW\tW_phi\tW_gradphi\tgamma\tgamma_phi\tgamma_gradphi\tPSI_a\tPSI_b\talpha_acum'
            append_results_to_file(os.path.join(result_folder_name, "total.energy"), header, step, EplusW,
                                E, W, W_phi, W_gradphi, gamma, gamma_phi, gamma_gradphi, PSI_a, PSI_b, alpha_acum)

        # Paraview -----------------------------------------------------------
        if Data.save_solution_xdmf:
            xdmf_phi.write_function(Φ_new, step)
            xdmf_u.write_function(u_new, step)

        if Data.save_solution_vtu:
            vtk_sol.write_function([Φ_new, u_new], step)

        t += dt
        step += 1

    if Data.save_solution_xdmf:
        xdmf_phi.close()
        xdmf_u.close()

    if Data.save_solution_vtu:
        vtk_sol.close()

    if rank == 0:
        end = time.perf_counter()
        if logger:
            log_end_analysis(logger, end - start)
