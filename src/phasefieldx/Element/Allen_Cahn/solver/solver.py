'''
Phase-field Problem
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

def solve(path, 
          Data, 
          msh, 
          final_time, 
          V_phi, 
          bc_list_phi=[],
          update_boundary_conditions=None, 
          update_loading=None, 
          initial_condition=None,
          ds_bound=None,
          dt = 1.0):
    
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

    # Phase-field -------------------------
    phi0 = dolfinx.fem.Function(V_phi, name="phi")
    phi = dolfinx.fem.Function(V_phi, name="phi")
    δphi = ufl.TestFunction(V_phi)
    
    #x = ufl.SpatialCoordinate(msh)
    if initial_condition is not None:
        phi.interpolate(initial_condition)
        phi0.interpolate(initial_condition)
        
    f = 0.25*(phi**2 -1)**2
    dfdc = ufl.diff(f,phi)
    #metadata = {"quadrature_degree": 8}
    #ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)
    #dx = ufl.Measure("dx", domain=msh, metadata=metadata)
    # Phase-field ------------------------------------------------------------
    F_phi  = ufl.inner(phi,δphi) * ufl.dx
    F_phi -= ufl.inner(phi0,δphi)* ufl.dx 
    F_phi += Data.kappa*dt*ufl.inner(ufl.grad(phi),ufl.grad(δphi))*ufl.dx 
    F_phi += dt*ufl.inner(dfdc,δphi)*ufl.dx
    
    
    # x = ufl.SpatialCoordinate(msh)
    # if update_loading != None:  
    #     f, grad_f = update_loading(x, 0)
    #     #F_phi-= Data.Gc*(1/Data.l*ufl.inner(f, delta_phi) + Data.l * ufl.inner(ufl.grad(f), ufl.grad(delta_phi))) * ufl.dx
    #     F_phi -= Data.Gc*(1/Data.l*ufl.inner(f, delta_phi) + Data.l * ufl.inner(grad_f, ufl.grad(delta_phi))) * ufl.dx
        
    J_phi = ufl.derivative(F_phi, phi)
    problem = dolfinx.fem.petsc.NonlinearProblem(F_phi, phi, bcs=bc_list_phi, J=J_phi)

    solver_phi = NewtonSolver(problem)
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
            bc_ux = update_boundary_conditions(bc_list_phi, t)
            
        if update_loading is not None:
            f, grad_f = update_loading(x, 0)


        # Phase-field ---------------------------------------------
        logger.info(f">>> Solving phase for dofs: phi ")
        phi_iterations, _ = solver_phi.solver.solve(phi)
        # residuals = solver_u.ksp.getConvergenceHistory()
        logger.info(f" Newton iterations: {phi_iterations}")
        resisual_phi = solver_phi.ksp.getResidualNorm()
        logger.info(f" Residual norm phi: {resisual_phi}")
        
        #Update phi0
        phi0.x.array[:] = phi.x.array

        # Save results
        ######################################################################
        logger.info(f"\n\n Saving results: ")
        
        # conv ---------------------------------------------------------------
        append_results_to_file(os.path.join(result_folder_name, "phasefieldx.conv"), '#step\titerations', step, phi_iterations)


        #Energy -------------------------------------------------------------
        
        gamma_phi = dolfinx.fem.assemble_scalar(dolfinx.fem.form(0.25*(ufl.inner(phi, phi) -1)**2 * ufl.dx))
        gamma_gradphi = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Data.kappa**2 / 2.0 * ufl.inner(ufl.grad(phi), ufl.grad(phi)) * ufl.dx))
        gamma = gamma_phi + gamma_gradphi
        
        append_results_to_file(os.path.join(result_folder_name, "total.energy"), '#step\tgamma\tgamma_phi\tgamma_gradphi', step, gamma, gamma_phi, gamma_gradphi)
        
            
        # Paraview -----------------------------------------------------------
        if Data.save_solution_xdmf:
            xdmf_phi.write_function(phi, step)

        if Data.save_solution_vtu:
            vtk_sol.write_function([phi], step)

        t += dt
        step += 1

    if Data.save_solution_xdmf:
        xdmf_phi.close()

    if Data.save_solution_vtu:
        vtk_sol.close()

    end = time.perf_counter()
    log_end_analysis(logger, end-start)
