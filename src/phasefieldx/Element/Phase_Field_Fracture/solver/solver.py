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
import numpy as np

from phasefieldx.files import prepare_simulation, append_results_to_file
from phasefieldx.solvers.newton import NewtonSolver
from phasefieldx.Logger.library_versions import set_logger, log_system_info, log_library_versions, log_end_analysis, log_model_information


from phasefieldx.Materials.elastic_isotropic import epsilon
from phasefieldx.Element.Phase_Field_Fracture.reactions_forces_functions import calculate_reaction_forces_phi, calculate_reaction_forces



from phasefieldx.Element.Phase_Field_Fracture.g_degradation_functions import dg, g
from phasefieldx.Element.Phase_Field_Fracture.split_energy_stress_tangent_functions import (psi_a, psi_b, sigma_a,
                                                   sigma_b)
from phasefieldx.Element.Phase_Field_Fracture.fatigue_degradation_functions import fatigue_degradation

from phasefieldx.Math.projection import project
from phasefieldx.errors_functions import eval_error_L2, eval_error_L2_normalized
from phasefieldx.files import prepare_simulation
from phasefieldx.solvers.newton import NewtonSolver

def solve(Data, 
          msh, 
          final_time, 
          V_u,
          V_phi,
          bc_list_u=[],
          bc_list_phi=[],
          update_boundary_conditions=None, 
          f_list_u=None, 
          T_list_u=None, 
          update_loading=None, 
          ds_bound=None,
          dt=1.0,
          path = None):
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
    V_phi : dolfinx.FunctionSpace
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

    Returns
    -------
    None
    """
    
    if path==None:
        path = os.getcwd()
        
    # Common #############################################################
    ######################################################################
    result_folder_name = Data.results_folder_name
    prepare_simulation(path, result_folder_name)
    logger = set_logger(result_folder_name)
    log_system_info(logger)  # log system imformation
    log_library_versions(logger)  # log Library versions
    Data.save_log_info(logger)  # log Simulation input data
    log_model_information(msh, logger)
    
    # Dolfinx cpp logger
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO) 
    dolfinx.cpp.log.set_output_file(os.path.join(result_folder_name, "dolfinx.log"))
    
    
    # Formulation ##########################################################
    ########################################################################
    
    # Displacement ------------------------
    u_new = dolfinx.fem.Function(V_u, name="u")
    u_old = dolfinx.fem.Function(V_u, name="u_old")
    δu = ufl.TestFunction(V_u)

    # Phase-field -------------------------
    phi_new = dolfinx.fem.Function(V_phi, name="phi")
    phi_old = dolfinx.fem.Function(V_phi, name="phi_old")
    δphi = ufl.TestFunction(V_phi)

    # History------------------------------
    H = dolfinx.fem.Function(V_phi, name="H")
    V_c = dolfinx.fem.Function(V_phi, name="V_c")
    V_n = dolfinx.fem.Function(V_phi, name="V_n")

    # Fatigue------------------------------
    if Data.fatigue:
        Fatigue = dolfinx.fem.Function(V_phi, name="fatigue")

        alpha_c = dolfinx.fem.Function(V_phi, name="alpha_c")
        alpha_n = dolfinx.fem.Function(V_phi, name="alpha_n")
        alpha_cum_bar_c = dolfinx.fem.Function(V_phi, name="alpha_cum_bar_c")
        alpha_cum_bar_n = dolfinx.fem.Function(V_phi, name="alpha_cum_bar_n")
        delta_alpha = dolfinx.fem.Function(V_phi, name="delta_alpha")


    # Displacement -----------------------------------------------------------
    F_u = ufl.inner( (g(phi_new, Data.degradation_function) + Data.k)*sigma_a(u_new,Data) + sigma_b(u_new, Data), epsilon(δu)) * ufl.dx
    
    # External forces --------------------------------------------------------
    if T_list_u is not None:
        L =  ufl.inner(T_list_u[0][0], δu) * T_list_u[0][1]
        for i in range(1, len(T_list_u)):
            L +=  ufl.inner(T_list_u[i][0], δu) * T_list_u[i][1]
            
        F_u -= L

    J_u = ufl.derivative(F_u, u_new)
    U_problem = dolfinx.fem.petsc.NonlinearProblem(F_u, u_new, bcs=bc_list_u, J=J_u)

    solver_u = NewtonSolver(U_problem)
    logger.info(f" Newton solver parameters for: u  ")
    logger.info(f" ---------------------------------")
    solver_u.save_log_info(logger)

    # Phase-field ------------------------------------------------------------
    F_phi = dg(phi_new, Data.degradation_function) * H * δphi*ufl.dx

    if Data.fatigue:
        F_phi += Fatigue*Data.Gc*(1/Data.l*ufl.inner(phi_new, δphi) + Data.l *
                                  ufl.inner(ufl.grad(phi_new), ufl.grad(δphi))) * ufl.dx
    else:
        F_phi += Data.Gc*(1/Data.l*ufl.inner(phi_new, δphi) + Data.l *
                          ufl.inner(ufl.grad(phi_new), ufl.grad(δphi))) * ufl.dx

    J_phi = ufl.derivative(F_phi, phi_new)
    PHI_problem = dolfinx.fem.petsc.NonlinearProblem(
        F_phi, phi_new, bcs=bc_list_phi, J=J_phi)

    solver_phi = NewtonSolver(PHI_problem)
    logger.info(f" Newton solver parameters for: phi")
    logger.info(f" ---------------------------------")
    solver_phi.save_log_info(logger)

    # Solve ################################################################
    ########################################################################
    start = time.perf_counter()
    
    logger.info(f" start time: {start}")

    # Paraview ------------------------
    if Data.save_solution_xdmf:
        paraview_solution_folder_name_xdmf = os.path.join(
            result_folder_name, "paraview-solutions_xdmf")
        xdmf_phi = dolfinx.io.XDMFFile(msh.comm, os.path.join(
            paraview_solution_folder_name_xdmf, "phi.xdmf"), "w")
        xdmf_phi.write_mesh(msh)

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

        if update_boundary_conditions is not None:
            bc_ux, bc_uy, bc_uz = update_boundary_conditions(bc_list_u, t)
        else:
            bc_ux, bc_uy, bc_uz = 0 ,0 ,0 
            
        if update_loading is not None:
            T_ux, T_uy, T_uz = update_loading(T_list_u, t)

        #if Data.fatigue:
        #    delta_alpha = np.abs(alpha_c.x.array - alpha_n.x.array) * np.heaviside((alpha_c.x.array - alpha_n.x.array) / dt, 1)
        #    alpha_cum_bar_c.x.array[:] = alpha_cum_bar_n.x.array + delta_alpha
        #    Fatigue.x.array[:] = fatigue_degradation(alpha_cum_bar_c.x.array, Data)

        error_L2_phi = 1
        error_L2_u = 1
        stagger_iter = 0
        while (error_L2_phi > Data.stagger_error_tol or error_L2_u > Data.stagger_error_tol or stagger_iter < Data.min_stagger_iter) and (stagger_iter < Data.max_stagger_iter):
            stagger_iter += 1
            logger.info(f" Stagger Iteration : {stagger_iter}")
            logger.info(f" ---------------------- ")

            # Displacement --------------------------------------------
            logger.info(f">>> Solving phase for dofs: u ")
            u_iterations, _ = solver_u.solver.solve(u_new)
            # residuals = solver_u.ksp.getConvergenceHistory()
            logger.info(f" Newton iterations: {u_iterations}")
            #error_L2_u = eval_error_L2(u_new, u_old, msh)/1
            error_L2_u = eval_error_L2_normalized(u_new, u_old, msh)
                      
            resisual_u = solver_u.ksp.getResidualNorm()
            logger.info(f" Residual norm u: {resisual_u}")
            logger.info(f" L2 error in u   direction:  {error_L2_u}")
            u_old.x.array[:] = u_new.x.array

            #  Irreversibility ....
            project(psi_a(u_new, Data), V_c)

            if Data.irreversibility == "miehe":
                H.x.array[:] = np.maximum(V_c.x.array, V_n.x.array)
            else:
                H.x.array[:] = V_c.x.array
                
            if Data.fatigue:
                project(g(phi_old, Data.degradation_function) * V_c, alpha_c)
                
                accelerated = False
                if accelerated:
                    N = 100
                    delta_alpha = N*alpha_c.x.array
                else:
                    delta_alpha = np.abs(alpha_c.x.array - alpha_n.x.array) * np.heaviside((alpha_c.x.array - alpha_n.x.array) / dt, 1)
                alpha_cum_bar_c.x.array[:] = alpha_cum_bar_n.x.array + delta_alpha
                Fatigue.x.array[:] = fatigue_degradation(alpha_cum_bar_c.x.array, Data)

            # Phase-field ---------------------------------------------
            logger.info(f">>> Solving phase for dofs: phi ")
            phi_iterations, _ = solver_phi.solver.solve(phi_new)
            logger.info(f" Newton iterations: {phi_iterations}")
            #error_L2_phi = eval_error_L2(phi_new, phi_old, msh)/1
            error_L2_phi = eval_error_L2_normalized(phi_new, phi_old, msh)
            resisual_phi = solver_phi.ksp.getResidualNorm()
            logger.info(f" Residual norm phi: {resisual_phi}")
            logger.info(f" L2 error in phi direction:  {error_L2_phi}")
            phi_old.x.array[:] = phi_new.x.array

        # Irreversibility ....
        if Data.irreversibility == "miehe":
            V_n.x.array[:] = np.maximum(V_c.x.array, V_n.x.array)

        if Data.fatigue:
            #project(g(phi_new, Data.degradation_function) * V_c, alpha_c)
            #delta_alpha = np.abs(alpha_c.x.array - alpha_n.x.array) * np.heaviside((alpha_c.x.array - alpha_n.x.array) / dt, 1)
                
            #alpha_cum_bar_c.x.array[:] = alpha_cum_bar_c.x.array + delta_alpha
            alpha_cum_bar_n.x.array[:] = alpha_cum_bar_c.x.array #+ delta_alpha
            alpha_n.x.array[:] = alpha_c.x.array

        # Save results
        ######################################################################
        logger.info(f"\n\n Saving results: ")

        # conv ---------------------------------------------------------------
        append_results_to_file(os.path.join(result_folder_name, "phasefieldx.conv"), '#step\tstagger\titerPhi\titerU', step, stagger_iter, phi_iterations, u_iterations)
        
        # Degree of freedom --------------------------------------------------
        append_results_to_file(os.path.join(result_folder_name, "top.dof"), '#step\tUx\tUy\tUz\tphi', step, bc_ux, bc_uy, bc_uz ,0.0)
        

        # Reaction -----------------------------------------------------------
        for i in range(0, ds_bound.shape[0]):
            R_phi = calculate_reaction_forces_phi(
                u_new, phi_new, Data, ds_bound[i][0], msh.topology.dim)
            R = calculate_reaction_forces(
                u_new, Data, ds_bound[i][0], msh.topology.dim)

            append_results_to_file(os.path.join(
                result_folder_name, ds_bound[i][1]+".reaction"), '#step\tRx\tRy\tRz', step, R_phi[0], R_phi[1], R_phi[2])

            append_results_to_file(os.path.join(
                result_folder_name, ds_bound[i][1]+"_natural.reaction"), '#step\tRx\tRy\tRz', step, R[0], R[1], R[2])
            
        # Energy -------------------------------------------------------------
        E = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
            (g(phi_new, Data.degradation_function)*psi_a(u_new, Data) + psi_b(u_new, Data)) * ufl.dx))

        gamma_phi = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
            1/(2*Data.l) * ufl.inner(phi_new, phi_new) * ufl.dx))
        gamma_gradphi = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
            Data.l/2 * ufl.inner(ufl.grad(phi_new), ufl.grad(phi_new)) * ufl.dx))
        gamma = gamma_phi + gamma_gradphi

        if Data.fatigue:
            W_phi = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
                Fatigue*Data.Gc * 1/(2*Data.l) * ufl.inner(phi_new, phi_new) * ufl.dx))
            W_gradphi = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
                Fatigue*Data.Gc * Data.l/2 * ufl.inner(ufl.grad(phi_new), ufl.grad(phi_new)) * ufl.dx))
            alpha_acum = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(alpha_cum_bar_c * ufl.dx))
        else:
            W_phi = Data.Gc*gamma_phi
            W_gradphi = Data.Gc*gamma_gradphi
            alpha_acum = 0.0

        W = W_phi + W_gradphi

        PSI_a = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(psi_a(u_new, Data) * ufl.dx))
        PSI_b = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(psi_b(u_new, Data) * ufl.dx))
        EplusW = E + W

        
        header = '#step\tEplusW\tE\tW\tW_phi\tW_gradphi\tgamma\tgamma_phi\tgamma_gradphi\tPSI_a\tPSI_b\talpha_acum'
        append_results_to_file(os.path.join(result_folder_name, "total.energy"), header, step, EplusW,
                                      E, W, W_phi, W_gradphi, gamma, gamma_phi, gamma_gradphi, PSI_a, PSI_b, alpha_acum)

        # Paraview -----------------------------------------------------------
        if Data.save_solution_xdmf:
            xdmf_phi.write_function(phi_new, step)
            xdmf_u.write_function(u_new, step)

        if Data.save_solution_vtu:
            vtk_sol.write_function([phi_new, u_new], step)

        t += dt
        step += 1

    if Data.save_solution_xdmf:
        xdmf_phi.close()
        xdmf_u.close()

    if Data.save_solution_vtu:
        vtk_sol.close()

    end = time.perf_counter()
    log_end_analysis(logger, end-start)
