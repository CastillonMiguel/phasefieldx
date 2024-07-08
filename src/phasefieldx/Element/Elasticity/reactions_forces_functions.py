"""
Reaction Forces
===============

This module provides functions to calculate reaction forces based on displacement fields and material data.
These functions are currently provisional, and a virtual work principle calculation will be applied in the future.

"""

import dolfinx
import numpy as np

from phasefieldx.Materials.elastic_isotropic import sigma

def calculate_reaction_forces(u, Data, ds_bound, dimension):
    """
    Calculate reaction forces based on the provided displacement field and data.

    Parameters
    ----------
    u : dolfinx.fem.Function
        The displacement field.
    Data : object
        An object containing material data and properties.
    ds_bound : dolfinx.fem.Measure
        The boundary measure for integration.
    dimension : int
        The spatial dimension of the problem (e.g., 2 or 3).

    Returns
    -------
    reaction_forces : numpy.ndarray
        An array containing the reaction forces in each dimension.
    """
    reaction_forces = np.array([0.0, 0.0, 0.0])
    for i in range(0, dimension):
        for j in range(0, dimension):
            reaction_forces[i] += dolfinx.fem.assemble_scalar(dolfinx.fem.form((sigma(u, Data.lambda_, Data.mu)[i, j])*ds_bound))
    return reaction_forces
