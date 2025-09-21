r"""
Solver: Phase-field fracture energy controlled non-variational
==============================================================


"""

# Libraries ############################################################
########################################################################
import os
import time
import dolfinx
import ufl
from petsc4py import PETSc
import dolfinx.fem.petsc
import numpy as np
import scifem
import ufl


from phasefieldx.files import prepare_simulation, append_results_to_file
from phasefieldx.Logger.library_versions import set_logger, log_system_info, log_library_versions, log_end_analysis, log_model_information


from phasefieldx.Materials.elastic_isotropic import epsilon
from phasefieldx.Reactions import calculate_reaction_forces


from phasefieldx.Element.Phase_Field_Fracture.g_degradation_functions import dg, g
from phasefieldx.Element.Phase_Field_Fracture.split_energy_stress_tangent_functions import (psi_a, psi_b, sigma_a,
                                                                                            sigma_b)

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
          dt=0.0005,
          dt_min=1e-12,
          dt_max=0.1,
          path=None,
          bcs_list_u_names=None,
          c1=1.0,
          c2=1.0,
          threshold_gamma_save=None,
          continue_simulation=False,
          step_continue=10):
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
    threshold_h_save : float or None, optional
        If set, Paraview files are saved only when h increases by more than this value. If None or 0, files are saved every step.

    Returns
    -------
    None
    """
    print("Solving with variational phase-field fracture model")
