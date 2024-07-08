r"""
Degradation Functions $g(\phi)$
===============================

This module provides various degradation functions and their derivatives, which are used in phase field fracture models.

"""

from math import exp


def quadratic_degradation_function(phi):
    """
    Evaluate the quadratic degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The degradation function value.
    """
    return (1.0 - phi) ** 2.0


def quadratic_degradation_derivative(phi):
    """
    Evaluate the derivative of the quadratic degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The derivative value.
    """
    return -2.0 * (1.0 - phi)


def borden_degradation_function(phi):
    """
    Evaluate the Borden degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The degradation function value.
    """
    s = 1.0
    return (3.0 - s) * (1.0 - phi) * (1.0 - phi) - (2.0 - s) * (1.0 - phi) * (1.0 - phi) * (1.0 - phi)


def borden_degradation_derivative(phi):
    """
    Evaluate the derivative of the Borden degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The derivative value.
    """
    s = 1.0
    return -2.0 * s * (1.0 - phi) + 3.0 * s * (1.0 - phi) * (1.0 - phi)


def alessi_degradation_function(phi):
    """
    Evaluate the Alessi degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The degradation function value.
    """
    k = 100.0
    Q = 1.0 - ((1.0 - phi) * (1.0 - phi))
    return (1.0 - phi) * (1.0 - phi) / (1.0 + (k - 1.0) * Q)


def alessi_degradation_derivative(phi):
    """
    Evaluate the derivative of the Alessi degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The derivative value.
    """
    k = 100.0
    return (2.0 * k * (phi - 1.0) * (k * (phi - 1.0) * phi - 1.0)) / ((k * phi - 1.0) ** 2.0)


def sargado_degradation_function(phi):
    """
    Evaluate the Sargado degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The degradation function value.
    """
    k = 100.0
    Q = 1.0 - ((1.0 - phi) * (1.0 - phi))
    return (1.0 - phi) * (1.0 - phi) / (1.0 + (k - 1.0) * Q)


def sargado_degradation_derivative(phi):
    """
    Evaluate the derivative of the Sargado degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The derivative value.
    """
    k = 100.0
    return -2.0 * k * (phi - 1.0) * k * (k * (phi - 2.0) * phi - 1.0) / ((k * phi - 1.0) ** 2.0 * exp(k * k))


def g(phi, degradation_type):
    """
    Evaluate the degradation function for a given phi and degradation type.

    Parameters:
        phi (float): The phi value.
        degradation_type (str): The type of degradation function ('quadratic', 'borden', 'alessi', or 'sargado').

    Returns:
        float: The degradation function value.
    """
    if degradation_type == "quadratic":
        return quadratic_degradation_function(phi)
    elif degradation_type == "borden":
        return borden_degradation_function(phi)
    elif degradation_type == "alessi":
        return alessi_degradation_function(phi)
    elif degradation_type == "sargado":
        return sargado_degradation_function(phi)
    else:
        raise ValueError("Invalid degradation_type. Please choose from 'quadratic', 'borden', 'alessi', or 'sargado'.")

# General degradation function derivative selector
def dg(phi, degradation_type):
    """
    Evaluate the derivative of the degradation function for a given phi and degradation type.

    Parameters:
        phi (float): The phi value.
        degradation_type (str): The type of degradation function ('quadratic', 'borden', 'alessi', or 'sargado').

    Returns:
        float: The derivative value.
    """
    if degradation_type == "quadratic":
        return quadratic_degradation_derivative(phi)
    elif degradation_type == "borden":
        return borden_degradation_derivative(phi)
    elif degradation_type == "alessi":
        return alessi_degradation_derivative(phi)
    elif degradation_type == "sargado":
        return sargado_degradation_derivative(phi)
    else:
        raise ValueError("Invalid degradation_derivative_type. Please choose from 'quadratic', 'borden', 'alessi', or 'sargado'.")
