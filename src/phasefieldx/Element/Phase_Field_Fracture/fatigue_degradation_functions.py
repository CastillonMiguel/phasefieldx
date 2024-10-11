r"""
Fatigue Degradation Functions $f(\bar{\alpha})$
===============================================

This module provides functions to evaluate fatigue degradation.
Fatigue degradation functions describe how the degradation evolves
over time or under repeated loading conditions. The degradation is typically
quantified using a parameter $\bar{\alpha}$, which represents the accumulated
variable.

"""

import numpy as np

##############################################################################
##############################################################################


def asymptotic(alpha_bar, Data):
    """
    Calculate the asymptotic fatigue degradation function.

    Parameters
    ----------
    alpha_bar : numpy.ndarray
        The array of alpha_bar values, representing some measure of material degradation.
    Data : object
        An object containing the fatigue degradation properties, specifically `fatigue_val`.

    Returns
    -------
    numpy.ndarray
        An array containing the calculated degradation values based on the asymptotic function.
    """
    alpha_critica = np.ones(len(alpha_bar)) * Data.fatigue_val
    aux = (2.0 * alpha_critica / (alpha_bar + alpha_critica))**2
    f_val = np.where(alpha_bar >= alpha_critica, aux, 1.0)
    return f_val


def fatigue_degradation(alpha_bar, Data):
    """
    Select and evaluate the appropriate fatigue degradation function.

    Parameters
    ----------
    alpha_bar : numpy.ndarray
        The array of alpha_bar values, representing some measure of material degradation.
    Data : object
        An object containing the fatigue degradation properties, including the type of degradation function.

    Returns
    -------
    numpy.ndarray
        An array containing the calculated degradation values based on the selected degradation function.

    Raises
    ------
    ValueError
        If an invalid degradation type is provided in `Data.fatigue_degradation_function`.
    """
    if Data.fatigue_degradation_function == "asymptotic":
        return asymptotic(alpha_bar, Data)
    elif Data.fatigue_degradation_function == "logarithmic":
        return ValueError("Not implemented yet")
    else:
        raise ValueError("Invalid degradation_type.")
