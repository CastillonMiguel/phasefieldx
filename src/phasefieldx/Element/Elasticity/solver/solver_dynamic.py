r"""
Solver: Elastodynamic
=====================

Transient (elastodynamic) solver for linear isotropic elasticity, using the
implicit Newmark-$\beta$ time integration scheme.

See :ref:`theory_elasticity_dynamic` and :ref:`theory_elasticity_newmark` for the
formulation.

"""

# Libraries ############################################################
########################################################################
import os
import time
import dolfinx
import ufl
from mpi4py import MPI
import petsc4py
from dolfinx.fem.petsc import NonlinearProblem


from phasefieldx.files import prepare_simulation, append_results_to_file
from phasefieldx.Logger.library_versions import set_logger, log_library_versions, log_system_info, log_end_analysis, log_model_information, log_header

from phasefieldx.Materials.elastic_isotropic import epsilon, sigma, psi
from phasefieldx.Reactions import calculate_reaction_forces
from phasefieldx.Element.Elasticity.energy import calculate_elastic_energy

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
    r"""
    Dynamic (transient) solver for linear isotropic elasticity.

    Solves for the displacement :math:`\boldsymbol u` with the implicit
    Newmark-:math:`\beta` scheme (:math:`\beta = 1/4`, :math:`\gamma = 1/2`). See
    :ref:`theory_elasticity_dynamic` and :ref:`theory_elasticity_newmark` for the
    formulation and the time integration scheme.

    Parameters
    ----------
    Data : phasefieldx.Element.Elasticity.Input.Input
        Data object containing the simulation parameters. It provides the Lamé
        constants (``lambda_``, ``mu``), the mass density ``rho``, the name of the
        results folder and the output flags (``save_solution_xdmf``, ``save_solution_vtu``).
    msh : dolfinx.mesh.Mesh
        Mesh object defining the computational domain.
    final_time : float
        Final time of the simulation. The time loop runs while ``t < final_time``,
        i.e. ``ceil(final_time / dt)`` steps are computed.
    V_u : dolfinx.fem.FunctionSpace
        Vector function space for the displacement field :math:`\boldsymbol u`.
    bc_list_u : list of dolfinx.fem.DirichletBC, optional
        Dirichlet boundary conditions for :math:`\boldsymbol u` (default is ``[]``).
    update_boundary_conditions : callable, optional
        Function with signature ``update_boundary_conditions(bc_list_u, t)`` that updates
        the Dirichlet boundary conditions at time ``t`` and returns the three imposed
        displacement components ``(bc_ux, bc_uy, bc_uz)``, which are stored in ``top.dof``.
    f_list_u : list of dolfinx.fem.Function, optional
        Body forces. Currently not used by the dynamic solver.
    T_list_u : list of tuple, optional
        Surface tractions, given as tuples ``(traction, measure)``, where ``traction``
        is a UFL expression or ``dolfinx.fem.Function`` and ``measure`` is the boundary
        measure (e.g. ``ds(1)``) where it is applied.
    update_loading : callable, optional
        Function with signature ``update_loading(T_list_u, t)`` that updates the
        tractions in ``T_list_u`` at time ``t`` and returns three components. Required
        if ``T_list_u`` is given.
    ds_bound : numpy.ndarray, optional
        Boundary description for reaction forces. Currently not used by the dynamic solver.
    dt : float, optional
        Time step size :math:`\Delta t` (default is 1.0). It is a physical time, as
        the inertial term makes the response time dependent.
    path : str, optional
        Path where the results folder is created (default is the current working directory).
    quadrature_degree : int, optional
        Quadrature degree used to integrate the stiffness term (default is 2).
    bcs_list_u_names : list of str, optional
        Names of the Dirichlet boundary conditions, used to name the ``*.reaction`` files
        (default is ``bc_u_0``, ``bc_u_1``, ...).

    Raises
    ------
    RuntimeError
        If the Newton solver does not converge in a time step.

    Returns
    -------
    None


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

        if bcs_list_u_names is None:
            bcs_list_u_names = [f"bc_u_{i}" for i in range(len(bc_list_u))]

    # Formulation ##########################################################
    ########################################################################

    # Displacement ------------------------
    u_new = dolfinx.fem.Function(V_u, name="u")
    δu = ufl.TestFunction(V_u)

    metadata = {"quadrature_degree": quadrature_degree}
    # ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)
    dx = ufl.Measure("dx", domain=msh, metadata=metadata)

    # Displacement -----------------------------------------------------------
    F_u = ufl.inner(sigma(u_new, Data.lambda_, Data.mu), epsilon(δu)) * dx
    F_u_form = dolfinx.fem.form(F_u)

    # External forces --------------------------------------------------------
    if T_list_u is not None:
        L = ufl.inner(T_list_u[0][0], δu) * T_list_u[0][1]
        for i in range(1, len(T_list_u)):
            L += ufl.inner(T_list_u[i][0], δu) * T_list_u[i][1]

        F_u -= L

    # Dynamic------------------------------
    beta = dolfinx.fem.Constant(msh, 0.25)
    gamma = dolfinx.fem.Constant(msh, 0.5)

    v_new = dolfinx.fem.Function(V_u, name="v_new")
    a_new = dolfinx.fem.Function(V_u, name="a_new")

    u_old = dolfinx.fem.Function(V_u, name="u_old")
    v_old = dolfinx.fem.Function(V_u, name="v_old")
    a_old = dolfinx.fem.Function(V_u, name="a_old")

    a = 1 / beta / dt**2 * (u_new - u_old - dt * v_old) + a_old * (1 - 1 / 2 / beta)
    v = v_old + dt * ((1 - gamma) * a_old + gamma * a_new)
    
    a_expr = dolfinx.fem.Expression(a, V_u.element.interpolation_points)
    v_expr = dolfinx.fem.Expression(v, V_u.element.interpolation_points)

    F_u += Data.rho * ufl.inner(a, δu) * ufl.dx

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
            
        a_new.interpolate(a_expr)
        v_new.interpolate(v_expr)

        u_old.x.array[:] = u_new.x.array
        v_old.x.array[:] = v_new.x.array
        a_old.x.array[:] = a_new.x.array
        
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
                R = calculate_reaction_forces(J_u_form, F_u_form, [bc_list_u[i]], u_new, V_u, msh.topology.dim)
                append_results_to_file(os.path.join(result_folder_name, bcs_list_u_names[i] + ".reaction"), '#step\tRx\tRy\tRz', step, R[0], R[1], R[2])

        # Energy -------------------------------------------------------------        
        E = calculate_elastic_energy(u_new, Data, comm, dx)
     
        # Only rank 0 writes energy results
        if rank == 0:
            append_results_to_file(os.path.join(
                result_folder_name, "total.energy"), '#step\tE', step, E)

        # Paraview -----------------------------------------------------------
        if Data.save_solution_xdmf:
            xdmf_u.write_function(u_new, step)

        if Data.save_solution_vtu:
            vtk_sol.write_function([u_new], step)

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