r"""
Energy-controlled PFF solver: Non-variational scheme
====================================================

This module implements the non-variational solver for the phase-field fracture
problem presented in :footcite:t:`Castillon2025_arxiv`.

.. footbibliography::

"""

# Libraries ############################################################
########################################################################
import os
import time
import dolfinx
import ufl
from petsc4py import PETSc
import dolfinx.fem.petsc
import scifem
import ufl

from phasefieldx.files import prepare_simulation, append_results_to_file
from phasefieldx.Logger.library_versions import set_logger, log_system_info, log_library_versions, log_end_analysis, log_model_information

from phasefieldx.Materials.elastic_isotropic import epsilon
from phasefieldx.Reactions import calculate_reaction_forces

from phasefieldx.Element.Phase_Field_Fracture.g_degradation_functions import dg, g
from phasefieldx.Element.Phase_Field_Fracture.split_energy_stress_tangent_functions import (psi_a, psi_b, sigma_a,
                                                                                            sigma_b)

from phasefieldx.solvers.newton import NewtonSolver
from phasefieldx.Element.Phase_Field.geometric_crack import geometric_crack_function, geometric_crack_function_derivative, geometric_crack_coefficient
from phasefieldx.Element.Phase_Field.energy import calculate_crack_surface_energy
from phasefieldx.Element.Phase_Field_Fracture.energy import compute_total_energies

from phasefieldx.files import prepare_simulation

def solve(Data,
          msh,
          final_gamma,
          V_u,
          V_Φ,
          bc_list_u=[],
          bc_list_phi=[],
          f_list_u=None,
          T_list_u=None,
          ds_bound=None,
          dtau=0.0005,
          dtau_min=1e-12,
          dtau_max=0.1,
          path=None,
          bcs_list_u_names=None,
          c1=1.0,
          c2=1.0,
          threshold_gamma_save=None):
    """
    Solve the phase-field fracture problem.

    Parameters
    ----------
    Data : object
        Container for simulation parameters (material constants, Gc, l, degradation_function,
        save flags, results folder name, logging helpers, and methods like save_log_info).
    msh : dolfinx.mesh.Mesh
        The computational mesh.
    final_gamma : float
        Target crack surface measure at which the simulation stops.
    V_u : dolfinx.FunctionSpace
        Function space for the displacement field.
    V_Φ : dolfinx.FunctionSpace
        Function space for the phase-field.
    bc_list_u : list, optional
        List of dolfinx DirichletBC objects for the displacement DOFs (default: []).
    bc_list_phi : list, optional
        List of dolfinx DirichletBC objects for the phase-field DOFs (default: []).
    f_list_u : list or None, optional
        List of body forces (not required by this solver; default: None).
    T_list_u : list of tuple or None, optional
        List of traction definitions as tuples (traction_vector, measure). The solver
        currently uses T_list_u[0] for external work and Lagrange coupling (default: None).
    ds_bound : array-like or None, optional
        Boundary measure(s) for reaction force calculation (default: None).
    dtau : float, optional
        Initial pseudo-time increment / load parameter increment (default: 5e-4).
    dtau_min : float, optional
        Minimum allowed dtau before aborting (default: 1e-12).
    dtau_max : float, optional
        Maximum allowed dtau (default: 0.1).
    path : str or None, optional
        Path where results folder will be created; if None uses current working dir (default: None).
    bcs_list_u_names : list of str or None, optional
        Names used when saving reaction files for displacement boundary conditions (default: None).
    c1 : float, optional
        Coefficient multiplying the crack measure (default: 1.0).
    c2 : float, optional
        Coefficient multiplying the external energy (default: 1.0).
    threshold_gamma_save : float or None, optional
        Threshold on change in gamma to trigger Paraview output. If None, treated as 0.0
        (default: None).

    Notes
    -----
    The docstring parameters correspond to the function signature. Some inputs (like
    T_list_u) are accessed by index in the implementation and must be provided in that form.
    """
    # Get MPI communicator info
    comm = msh.comm
    rank = comm.Get_rank()

    if path is None:
        path = os.getcwd()

    # Common #############################################################
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

    # Dolfinx cpp logger
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    if rank == 0:
        dolfinx.cpp.log.set_output_file(
            os.path.join(result_folder_name, "dolfinx.log"))
        if bc_list_phi !=[]:
            if bcs_list_u_names is None:
                bcs_list_u_names = [f"bc_u_{i}" for i in range(len(bc_list_u)-1)]
        else:
            if bcs_list_u_names is None:
                bcs_list_u_names = [f"bc_u_{i}" for i in range(len(bc_list_u))]

    # Formulation ##########################################################
    ########################################################################
    V_λ = scifem.create_real_functionspace(msh)
 
    W = ufl.MixedFunctionSpace(V_u, V_Φ, V_λ)

    u = dolfinx.fem.Function(V_u, name="u")
    Φ = dolfinx.fem.Function(V_Φ, name="phi")
    λ = dolfinx.fem.Function(V_λ, name="lambda")

    # Previous converged solution 
    u_c = dolfinx.fem.Function(V_u, name="u")
    Φ_c = dolfinx.fem.Function(V_Φ, name="phi")
    λ_c = dolfinx.fem.Function(V_λ, name="lambda")

    δu, δΦ, δλ = ufl.TestFunctions(W)
    du, dΦ, dλ = ufl.TrialFunctions(W)

    ###############################################################################
    # Define constitutive equations
    # -----------------------------

    # Displacement -----------------------------------------------------------
    F_u = ufl.inner((g(Φ, Data.degradation_function) + Data.k) * sigma_a(u, Data) + sigma_b(u, Data), epsilon(δu)) * ufl.dx

    # For reaction forces calculation 
    F_u_form = dolfinx.fem.form(ufl.inner((g(Φ, Data.degradation_function) + Data.k) * sigma_a(u, Data) + sigma_b(u, Data), epsilon(δu)) * ufl.dx)
    J_u = ufl.derivative(F_u, u, du)
    J_u_form = dolfinx.fem.form(J_u)

    F_u -= ufl.inner(T_list_u[0][0], δu) * T_list_u[0][1]
    F_u += λ*c2*ufl.inner(T_list_u[0][0], δu) * T_list_u[0][1]
    # for i in range(1, len(T_list_u)):
    #     F_u -= ufl.inner(T_list_u[i][0], δu) * T_list_u[i][1]
    #     F_u += λ*π_2*ufl.inner(T_list_u[i][0], δu) * T_list_u[i][1]

    # Phase-field ------------------------------------------------------------
    case = 'AT2'  # AT2 model for geometric crack function
    c0 = geometric_crack_coefficient(case)

    F_Φ = dg(Φ, Data.degradation_function) * psi_a(u, Data) * δΦ * ufl.dx
    F_Φ += Data.Gc*(1.0/(c0*Data.l)*geometric_crack_function_derivative(Φ, case)*δΦ + Data.l * 2/c0*ufl.inner(ufl.grad(Φ), ufl.grad(δΦ)))*ufl.dx

    # Scalar field ------------------------------------------------------------
    F_λ = δλ*c1*(1 / (c0 * Data.l) * geometric_crack_function(Φ, case) + Data.l / c0 * ufl.inner(ufl.grad(Φ), ufl.grad(Φ))) * ufl.dx
    F_λ += δλ*c2*ufl.inner(T_list_u[0][0], u) * T_list_u[0][1]
    # for i in range(1, len(T_list_u)):
    #     F_λ += π_2*δλ*ufl.inner(T_list_u[i][0], δu) * T_list_u[i][1]
    # F_λ += π_2*δλ*ufl.inner(T, u) * ds_top

    F_blocked = dolfinx.fem.form([F_u, F_Φ, F_λ])

    J_blocked = [ [ufl.derivative(F_u, u, du), ufl.derivative(F_u, Φ, dΦ), ufl.derivative(F_u, λ, dλ)],
                  [ufl.derivative(F_Φ, u, du), ufl.derivative(F_Φ, Φ, dΦ), None                      ],
                  [ufl.derivative(F_λ, u, du), ufl.derivative(F_λ, Φ, dΦ), None                      ],
                 ]

    tau = 0.0
    class RealSpaceNewtonSolver(scifem.BlockedNewtonSolver):

        def _assemble_residual(self, x: PETSc.Vec, b: PETSc.Vec) -> None:
            """Assemble the residual F into the vector b.
            Args:
                x: The vector containing the latest solution
                b: Vector to assemble the residual into
            """
            # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
            with b.localForm() as b_local:
                b_local.set(0.0)
            try:
                dolfinx.fem.petsc.assemble_vector_block(
                    b,
                    self._F,
                    self._a,
                    bcs=self._bcs,
                    x0=x,
                    alpha=-1.0,
                )
                start_pos = 0
                for s in self._u:
                    num_sub_dofs = (
                        s.function_space.dofmap.index_map.size_local
                        * s.function_space.dofmap.index_map_bs
                    )
                    if s.function_space.dofmap.index_map.size_global == 1:
                        assert s.function_space.dofmap.index_map_bs == 1
                        if s.function_space.dofmap.index_map.size_local > 0:
                            b.array_w[start_pos:start_pos+num_sub_dofs] -= tau
                    start_pos += num_sub_dofs
            except AttributeError:
                dolfinx.fem.petsc.assemble_vector(b, self._F)
                bcs1 = dolfinx.fem.bcs_by_block(
                    dolfinx.fem.extract_function_spaces(self._a, 1), self._bcs
                )
                dolfinx.fem.petsc.apply_lifting(b, self._a, bcs=bcs1, x0=x, alpha=-1.0)
                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                bcs0 = dolfinx.fem.bcs_by_block(
                    dolfinx.fem.extract_function_spaces(self._F), self._bcs
                )
                dolfinx.fem.petsc.set_bc(b, bcs0, x0=x, alpha=-1.0)
                start_pos = 0
                for s in self._u:
                    num_sub_dofs = (
                        s.function_space.dofmap.index_map.size_local
                        * s.function_space.dofmap.index_map_bs
                    )
                    if s.function_space.dofmap.index_map.size_global == 1:
                        assert s.function_space.dofmap.index_map_bs == 1
                        if s.function_space.dofmap.index_map.size_local > 0:
                            b.array_w[start_pos:start_pos+num_sub_dofs] -= tau
                    start_pos += num_sub_dofs
            b.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)
    

    petsc_options={
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_type": "newtonls",                 # Use line search Newton
        "snes_max_it": 10,                       # Max Newton iterations
        "snes_rtol": 1e-8,                       # Relative tolerance
        "snes_atol": 1e-10,                      # Absolute tolerance
        "snes_monitor": None,                    # Print SNES progress
        "ksp_monitor": None,                     # Print KSP progress
    }


    solver_snap = RealSpaceNewtonSolver(F_blocked, [u, Φ, λ], J=J_blocked, bcs=bc_list_u,
                                petsc_options=petsc_options)

    # Initial crack ########################################################
    ########################################################################
    # In case the inital crack is defined by Dirichlet BCs on the phase-field.
    # Not considered in the paper, but useful for some benchmarks.
    gamma0 = 0.0
    if bc_list_phi !=[]:
        if rank == 0 and logger:
            logger.info(f"\n\nBoundary condition for phase-field ")

        problem = dolfinx.fem.petsc.NewtonSolverNonlinearProblem(Data.Gc*(1.0/(c0*Data.l)*geometric_crack_function_derivative(Φ, case)*δΦ + Data.l * 2/c0*ufl.inner(ufl.grad(Φ), ufl.grad(δΦ)))*ufl.dx, Φ, bcs=bc_list_phi)
        solver_phi = NewtonSolver(problem)

        if rank == 0 and logger:
            logger.info(f"\n\nSolving boundary value problem for phase-field ")
         
        phi_iterations, _ = solver_phi.solver.solve(Φ)
    
        gamma0, gamma_phi0, gamma_gradphi0 = calculate_crack_surface_energy(Φ, Data.l, comm, case, ufl.dx)

    if rank == 0 and logger:
        logger.info(f"\n\nGamma0 = {gamma0}")
     
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


    tau = gamma0 + dtau
    last_saved_gamma = gamma0

    if threshold_gamma_save is None:
        threshold_gamma_save = 0.0

    golden_ratio = 2 / (1 + 5 ** 0.5)  # ≈ 0.618
 
    step = 0
    gamma = 0.0
    while gamma < final_gamma:
        # Advance time and tau at the start of the step
        tau += dtau

        if rank == 0 and logger:
            logger.info(
                f"\n\nSolution at tau = {tau}, dtau = {dtau}, Step = {step} ")
            logger.info(
                f"===========================================================================")

            logger.info(f">>> Solving phase for dofs: u, phi, lambda ")
            logger.info(f">>> tau(t) = {tau}")
        try:
            n_iterations, converged = solver_snap.solve()

            # residuals = solver_u.ksp.getConvergenceHistory()
            ResidualNorm = solver_snap.krylov_solver.getResidualNorm()
            if rank == 0 and logger:
                logger.info(f" Newton iterations: {n_iterations}")
                logger.info(f" Residual norm: {ResidualNorm}")
      
            # Advance time and tau ONLY if successful
            if converged:
                u_c.x.array[:] = u.x.array
                Φ_c.x.array[:] = Φ.x.array
                λ_c.x.array[:] = λ.x.array

                # FIXED: Only increase dt when Newton iterations <= 2
                if n_iterations <= 2:
                    dtau = min(dtau * (1+golden_ratio), dtau_max)
                    if rank == 0 and logger:
                        logger.info(f"Converged in ({n_iterations} iterations), increasing dt to {dtau}")
                else:
                    if rank == 0 and logger:
                        logger.info(f"Slow convergence ({n_iterations} iterations), reducing dt to {dtau}")
         
                step += 1
             
             
            # Save results
            ######################################################################
            if rank == 0 and logger:
                logger.info(f"\n\n Saving results: ")

                # conv ---------------------------------------------------------------
                append_results_to_file(os.path.join(result_folder_name, "phasefieldx.conv"),
                                    '#step\titerations\tResidualNorm', step, n_iterations, ResidualNorm)

                # Lagrange multiplier -------------------------------------------------
                append_results_to_file(os.path.join(result_folder_name, "lambda.dof"),
                                    '#step\tlambda', step, λ.x.array[0])
            
                # tau() -------------------------------------------------
                append_results_to_file(os.path.join(result_folder_name, "tau.input"),
                                    '#step\ttau', step, tau)
            
            # Degree of freedom --------------------------------------------------
            if comm.Get_size() == 1: # Only available for single process
                ux_max = 0.0
                uy_max = 0.0
                uz_max = 0.0
                
                if msh.topology.dim == 1:
                    ux_max = max(u.x.array[0::1])
                
                elif msh.topology.dim == 2:
                    ux_max = max(u.x.array[0::2])
                    uy_max = max(u.x.array[1::2])

                elif msh.topology.dim == 3:
                    ux_max = max(u.x.array[0::3])
                    uy_max = max(u.x.array[1::3])
                    uz_max = max(u.x.array[2::3])
                
                if rank == 0 and logger:
                    append_results_to_file(os.path.join(result_folder_name, "max_u.dof"),
                                        '#step\tUx\tUy\tUz\tphi', step, ux_max, uy_max, uz_max, 0.0)

            # Reaction -----------------------------------------------------------
            if comm.Get_size() == 1: # Only available for single process
                if bc_list_phi !=[]:
                    for i in range(0, len(bc_list_u)-1):
                        R = calculate_reaction_forces(J_u_form, F_u_form, [bc_list_u[i]], u, V_u, msh.topology.dim)
                        append_results_to_file(os.path.join(
                            result_folder_name, bcs_list_u_names[i] + ".reaction"), '#step\tRx\tRy\tRz', step, R[0], R[1], R[2])
                else:
                    for i in range(0, len(bc_list_u)):
                        R = calculate_reaction_forces(J_u_form, F_u_form, [bc_list_u[i]], u, V_u,msh.topology.dim)
                        append_results_to_file(os.path.join(
                            result_folder_name, bcs_list_u_names[i] + ".reaction"), '#step\tRx\tRy\tRz', step, R[0], R[1], R[2])
            
            
            # Energy -------------------------------------------------------------
            E, PSI_a, PSI_b = compute_total_energies(u, Φ, Data, comm, dx=ufl.dx)
            gamma, gamma_phi, gamma_gradphi = calculate_crack_surface_energy(Φ, Data.l, comm, case, ufl.dx)
        
            W_phi = Data.Gc * gamma_phi
            W_gradphi = Data.Gc * gamma_gradphi
            W = W_phi + W_gradphi

            PSI_a = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(psi_a(u, Data) * ufl.dx))
            PSI_b = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(psi_b(u, Data) * ufl.dx))
            EplusW = E + W

            external_energy = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(T_list_u[0][0], u) * T_list_u[0][1]))
            check_lagrange_multiplier = c1*gamma + c2*external_energy - tau
            if rank == 0 and logger:
                logger.info(f"Check constraint: c1*gamma + c2*external_energy - tau = {c1}*{gamma:.6e} + {c2}*{external_energy:.6e} - {tau:.6e} = {check_lagrange_multiplier:.6e}")
            
            # Only rank 0 writes energy results
            if rank == 0:
                header = '#step\tEplusW\tE\tW\tW_phi\tW_gradphi\tgamma\tgamma_phi\tgamma_gradphi\tPSI_a\tPSI_b\texternal'
                append_results_to_file(os.path.join(result_folder_name, "total.energy"), header, step, EplusW,
                                    E, W, W_phi, W_gradphi, gamma, gamma_phi, gamma_gradphi, PSI_a, PSI_b, external_energy)
            
            if rank == 0 and logger:
                logger.info(f"gamma = {gamma:.6e}, gamma_phi = {gamma_phi:.6e}, gamma_gradphi = {gamma_gradphi:.6e}") 
                logger.info(f"Check gamma save condition: abs({gamma:.6e} - {last_saved_gamma:.6e}) = {abs(gamma - last_saved_gamma):.6e} >= {threshold_gamma_save:.6e} ? {'YES' if abs(gamma - last_saved_gamma) >= threshold_gamma_save else 'NO'}")
            
            # Paraview -----------------------------------------------------------
            if step == 1 or threshold_gamma_save == 0.0 or abs(gamma - last_saved_gamma) >= threshold_gamma_save:
                
                if Data.save_solution_xdmf:
                    if rank == 0 and logger:
                        logger.info(f"Saving XDMF Paraview files at step={step}, gamma={gamma:.6f}, tau={tau:.6f}")
                    xdmf_phi.write_function(Φ, step)
                    xdmf_u.write_function(u, step)

                if Data.save_solution_vtu:
                    if rank == 0 and logger:
                        logger.info(f"Saving VTU Paraview files at step={step}, gamma={gamma:.6f}, tau={tau:.6f}")
                    vtk_sol.write_function([Φ, u], step)
                last_saved_gamma = gamma
           
        except Exception as e:
            if rank == 0 and logger:
                logger.info(f"Solver failed at tau={tau} with error: {e}. Reducing dtau.")
          
            u.x.array[:] = u_c.x.array
            Φ.x.array[:] = Φ_c.x.array
            λ.x.array[:] = λ_c.x.array
        
            # Roll back tau since this step failed
            tau -= dtau
            
            dtau = max(dtau * golden_ratio, dtau_min)
            
            if dtau <= dtau_min:
                if rank == 0 and logger:
                    logger.info(f"Minimum dtau reached. Stopping.")
                
                if Data.save_solution_xdmf:
                    if rank == 0 and logger:
                        logger.info(f"Saving XDMF Paraview files at step={step}, gamma={gamma:.6f}, tau={tau:.6f}")
                    xdmf_phi.write_function(Φ_c, step)
                    xdmf_u.write_function(u_c, step)

                if Data.save_solution_vtu:
                    if rank == 0 and logger:
                        logger.info(f"Saving VTU Paraview files at step={step}, gamma={gamma:.6f}, tau={tau:.6f}")
                    vtk_sol.write_function([Φ_c, u_c], step)
                  
                break
            continue


    if Data.save_solution_xdmf:
        xdmf_phi.close()
        xdmf_u.close()

    if Data.save_solution_vtu:
        vtk_sol.close()

    if rank == 0:
        end = time.perf_counter()
        if logger:
            log_end_analysis(logger, end - start)
