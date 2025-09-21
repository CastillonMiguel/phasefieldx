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

import adios4dolfinx

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
      solve _summary_

      _extended_summary_

      Parameters
      ----------
      Data : _type_
          _description_
      msh : _type_
          _description_
      final_gamma : _type_
          _description_
      V_u : _type_
          _description_
      V_ : _type_
          _description_
      bc_list_u : list, optional
          _description_, by default []
      bc_list_phi : list, optional
          _description_, by default []
      f_list_u : _type_, optional
          _description_, by default None
      T_list_u : _type_, optional
          _description_, by default None
      ds_bound : _type_, optional
          _description_, by default None
      dt : float, optional
          _description_, by default 0.0005
      dt_min : _type_, optional
          _description_, by default 1e-12
      dt_max : float, optional
          _description_, by default 0.1
      path : _type_, optional
          _description_, by default None
      bcs_list_u_names : _type_, optional
          _description_, by default None
      c1 : float, optional
          _description_, by default 1.0
      c2 : float, optional
          _description_, by default 1.0
      threshold_gamma_save : _type_, optional
          _description_, by default None
      continue_simulation : bool, optional
          _description_, by default False
      step_continue : int, optional
          _description_, by default 10
      """
    print("Solving with non-variational phase-field fracture model")
