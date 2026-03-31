r"""
Degradation Functions $g(\phi)$
===============================

This module provides various degradation functions and their derivatives, which are used in phase field fracture models.

Examples
--------
Plot the available degradation functions for illustrative purposes.

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from phasefieldx.Element.Phase_Field_Fracture.g_degradation_functions import g, dg

Define the range for the phase field variable phi.

>>> phi = np.linspace(0.0, 1.0, 100)

Evaluate the degradation functions and their derivatives.

>>> g_quadratic = [g(p, "quadratic") for p in phi]
>>> g_cubic = [g(p, "cubic") for p in phi]
>>> g_quartic = [g(p, "quartic") for p in phi]
>>> g_alessi = [g(p, "alessi") for p in phi]
>>> g_sargado = [g(p, "sargado") for p in phi]
>>> dg_quadratic = [dg(p, "quadratic") for p in phi]
>>> dg_cubic = [dg(p, "cubic") for p in phi]
>>> dg_quartic = [dg(p, "quartic") for p in phi]
>>> dg_alessi = [dg(p, "alessi") for p in phi]
>>> dg_sargado = [dg(p, "sargado") for p in phi]

Plot the evaluated functions and their derivatives.

>>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
>>> _ = ax1.plot(phi, g_quadratic, label="Quadratic")
>>> _ = ax1.plot(phi, g_cubic, label="Cubic")
>>> _ = ax1.plot(phi, g_quartic, label="Quartic")
>>> ax1.set_xlabel(r"$\phi$")
>>> ax1.set_ylabel(r"$g(\phi)$")
>>> ax1.set_title("Phase Field Degradation Functions")
>>> ax1.legend()
>>> ax1.grid(True)
>>> _ = ax2.plot(phi, dg_quadratic, label="Quadratic")
>>> _ = ax2.plot(phi, dg_cubic, label="Cubic")
>>> _ = ax2.plot(phi, dg_quartic, label="Quartic")
>>> ax2.set_xlabel(r"$\phi$")
>>> ax2.set_ylabel(r"$g'(\phi)$")
>>> ax2.set_title("Derivatives of Degradation Functions")
>>> ax2.legend()
>>> ax2.grid(True)
>>> plt.tight_layout()
>>> plt.show()
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


def cubic_degradation_function(phi):
    """
    Evaluate the Borden degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The degradation function value.
    """
    return 3.0 * (1.0 - phi)**2 - 2.0 * (1.0 - phi)**3


def cubic_degradation_derivative(phi):
    """
    Evaluate the derivative of the cubic degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The derivative value.
    """
    return -6.0 * phi * (1.0 - phi) 


def quartic_degradation_function(phi):
    """
    Evaluate the quartic degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The degradation function value.
    """
    return 4.0 * (1.0 - phi)**3 - 3.0 * (1.0 - phi)**4


def quartic_degradation_derivative(phi):
    """
    Evaluate the derivative of the quartic degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The derivative value.
    """
    return -12.0 * (1.0 - phi) ** 2 + 12.0 * (1.0 - phi)**3


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
    return (2.0 * k * (phi - 1.0) * (k * (phi - 1.0) * phi - 1.0)) / \
        ((k * phi - 1.0) ** 2.0)


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
    return -2.0 * k * (phi - 1.0) * k * (k * (phi - 2.0) *
                                         phi - 1.0) / ((k * phi - 1.0) ** 2.0 * exp(k * k))


def wu_degradation_function(phi):
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


def wu_degradation_derivative(phi):
    """
    Evaluate the derivative of the Sargado degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The derivative value.
    """
    k = 100.0
    return -2.0 * k * (phi - 1.0) * k * (k * (phi - 2.0) *
                                         phi - 1.0) / ((k * phi - 1.0) ** 2.0 * exp(k * k))


def g(phi, degradation_type):
    """
    Evaluate the degradation function for a given phi and degradation type.

    Parameters:
        phi (float): The phi value.
        degradation_type (str): The type of degradation function ('quadratic', 'cubic', 'quartic', 'alessi', or 'sargado').

    Returns:
        float: The degradation function value.
    """
    if degradation_type == "quadratic":
        return quadratic_degradation_function(phi)
    elif degradation_type == "cubic":
        return cubic_degradation_function(phi)
    elif degradation_type == "quartic":
        return quartic_degradation_function(phi)
    elif degradation_type == "alessi":
        return alessi_degradation_function(phi)
    elif degradation_type == "sargado":
        return sargado_degradation_function(phi)
    else:
        raise ValueError(
            "Invalid degradation_type. Please choose from 'quadratic', 'cubic', 'quartic', 'alessi', or 'sargado'.")



def dg(phi, degradation_type):
    """
    Evaluate the derivative of the degradation function for a given phi and degradation type.

    Parameters:
        phi (float): The phi value.
        degradation_type (str): The type of degradation function ('quadratic', 'cubic', 'quartic', 'alessi', or 'sargado').

    Returns:
        float: The derivative value.
    """
    if degradation_type == "quadratic":
        return quadratic_degradation_derivative(phi)
    elif degradation_type == "cubic":
        return cubic_degradation_derivative(phi)
    elif degradation_type == "quartic":
        return quartic_degradation_derivative(phi)
    elif degradation_type == "alessi":
        return alessi_degradation_derivative(phi)
    elif degradation_type == "sargado":
        return sargado_degradation_derivative(phi)
    else:
        raise ValueError(
            "Invalid degradation_derivative_type. Please choose from 'quadratic', 'cubic', 'quartic', 'alessi', or 'sargado'.")
