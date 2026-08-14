r"""
Staggered convergence monitor
=============================

This module provides the :class:`ConvergenceMonitor` class, used to check
the convergence of the staggered scheme in phase-field fracture solvers.
It supports checking convergence based on the Newton residual, or on the
L2 or L-infinity norms of the displacement and phase-field increments.

"""
class ConvergenceMonitor:
    def __init__(self, criterion, tolerance, min_iter, logger=None):
        self.criterion = criterion
        self.tolerance = tolerance
        self.min_iter = min_iter
        self.logger = logger
        
    def check_residual(self, iteration, residual_norm, field_name):
        """Checks convergence based on the initial residual of the Newton solver."""
        if self.criterion != "Residual":
            return False
            
        if iteration < self.min_iter:
            return False

        if residual_norm < self.tolerance:
            if self.logger:
                self.logger.info(f"    <- Converged: {field_name} residual {residual_norm:.4e} < {self.tolerance}")
            return True
        return False

    def check_l2(self, iteration, error_u, error_phi):
        """Checks convergence based on increment L2 norms."""
        if self.criterion != "L2":
            return False

        if iteration < self.min_iter:
            return False

        val_u = float(error_u)
        val_phi = float(error_phi)

        if val_u < self.tolerance and val_phi < self.tolerance:
            if self.logger:
                self.logger.info(f"    <- Converged: L2 errors u={val_u:.4e}, phi={val_phi:.4e} < {self.tolerance}")
            return True
        return False
    

    def check_linf(self, iteration, error_u, error_phi):
        """Checks convergence based on increment L-infinity norms."""
        if self.criterion != "Linf":
            return False

        if iteration < self.min_iter:
            return False

        val_u = abs(float(error_u))
        val_phi = abs(float(error_phi))

        if val_u < self.tolerance and val_phi < self.tolerance:
            if self.logger:
                self.logger.info(
                    f"    <- Converged: Linf errors u={val_u:.4e}, phi={val_phi:.4e} < {self.tolerance}"
                )
            return True
        return False