"""
Newton solver
=============

This module provides a class for configuring a Newton solver with PETSc options, designed for solving
nonlinear problems in finite element analysis.

"""

import dolfinx
import mpi4py
import petsc4py
import dolfinx.nls.petsc 

class NewtonSolver:
    """
    A class that encapsulates the configuration of a Newton solver with PETSc options.
    """

    def __init__(self, problem):
        """
        Initialize the NewtonSolver with a given problem.

        Parameters
        ----------
        problem : dolfinx.cpp.nls.Problem
            The problem to be solved using the NewtonSolver.
        """
        self.solver = dolfinx.nls.petsc.NewtonSolver(mpi4py.MPI.COMM_WORLD, problem)

        # Newton solver parameters
        self.solver.max_it = 500   # Maximum number of Newton iterations per step
        self.solver.rtol = 1e-8    # Relative tolerance for convergence
        self.solver.atol = 1e-9    # Absolute tolerance for convergence
        self.solver.convergence_criterion = "residual"  # Convergence criterion: "residual" or "incremental"
        self.solver.report = True  # Monitor convergence
        self.solver.relaxation_parameter = 1.0  # Relaxation parameter

        # Krylov solver parameters
        self.ksp = self.solver.krylov_solver
        self.opts = petsc4py.PETSc.Options()
        self.option_prefix = self.ksp.getOptionsPrefix()
        #self.opts[f"{self.option_prefix}ksp_type"] = "gmres"  # Krylov solver type: "cg" or "gmres" Chebyshev
        #self.opts[f"{self.option_prefix}pc_type"] = "gamg"  # Preconditioner type "gamg" ilu lu 
        #self.opts[f"{self.option_prefix}pc_factor_mat_solver_type"] = "mumps"  # Mat solver type for factorization
        self.ksp.setFromOptions()
        self.ksp.setConvergenceHistory()
        #self.ksp.setResidualNorm()

    def save_log_info(self, logger):
        """
        Save the configuration settings of the NewtonSolver and Krylov solver to a logger.

        Parameters
        ----------
        logger : logging.Logger
            The logger to save the information to.
        """
        logger.info("Newton Solver settings:")
        logger.info(f"  Convergence criterion: {self.solver.convergence_criterion}")
        logger.info(f"  Absolute tolerance: {self.solver.atol}")
        logger.info(f"  Relative tolerance: {self.solver.rtol}")
        logger.info(f"  Maximum iterations: {self.solver.max_it}")

        logger.info("KSP settings:")
        #logger.info(f"  KSP type: {self.opts[f'{self.option_prefix}ksp_type']}")
        #logger.info(f"  PC type: {self.opts[f'{self.option_prefix}pc_type']}")
        #logger.info(f"  PC factor mat solver type: {self.opts[f'{self.option_prefix}pc_factor_mat_solver_type']}")

    def __str__(self):
        """
        Get a human-readable representation of the NewtonSolver's configuration.

        Returns
        -------
        str
            A string representing the configuration settings.
        """
        config_str = "Newton Solver Configuration:\n"
        config_str += f"  Convergence criterion: {self.solver.convergence_criterion}\n"
        config_str += f"  Absolute tolerance: {self.solver.atol}\n"
        config_str += f"  Relative tolerance: {self.solver.rtol}\n"
        config_str += f"  Maximum iterations: {self.solver.max_it}\n\n"
        config_str += "KSP settings:\n"
        config_str += f"  KSP type: {self.opts[f'{self.option_prefix}ksp_type']}\n"
        config_str += f"  PC type: {self.opts[f'{self.option_prefix}pc_type']}\n"
        config_str += f"  PC factor mat solver type: {self.opts[f'{self.option_prefix}pc_factor_mat_solver_type']}"
        
        return config_str
    