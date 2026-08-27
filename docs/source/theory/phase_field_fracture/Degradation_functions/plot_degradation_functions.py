###############################################################################
# Import necessary libraries
# --------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


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
    s = 0.0
    return (3.0 - s) * (1.0 - phi) * (1.0 - phi) - \
        (2.0 - s) * (1.0 - phi) * (1.0 - phi) * (1.0 - phi)


def borden_degradation_derivative(phi):
    """
    Evaluate the derivative of the Borden degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The derivative value.
    """
    s = 0.0
    return -2.0 * (3.0 - s) * (1.0 - phi) + 3.0 * (2.0 - s) * (1.0 - phi) * (1.0 - phi)


def quartic_degradation_function(phi):
    """
    Evaluate the quadratic degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The degradation function value.
    """
    return 4.0*(1.0 - phi)**3 - 3*(1.0 - phi)**4


def quartic_degradation_derivative(phi):
    """
    Evaluate the derivative of the quadratic degradation function for a given phi.

    Parameters:
        phi (float): The phi value.

    Returns:
        float: The derivative value.
    """
    return -12.0*(1.0 - phi)**2 + 12*(1.0 - phi)**3


phi = np.linspace(0, 1, 1000)


results_folder = "results_degradation_functions"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


color_theory = "black"
color_spectral = "green"
color_volumetric = "blue"
color_lvpp = "red"


# Plot degradation functions and their derivatives
fig, axs = plt.subplots()

axs.plot(phi, quadratic_degradation_function(phi),color=color_theory, label="Quadratic")
axs.plot(phi, borden_degradation_function(phi),color=color_spectral, label="Cubic")
axs.plot(phi, quartic_degradation_function(phi),color=color_volumetric, label="Quartic")

axs.set_xlabel(r"$\phi$")
axs.set_ylabel(r"$g(\phi)$")
axs.legend()
plt.savefig(os.path.join(results_folder, "phi_vs_g_phi"))

# Plot degradation functions and their derivatives
fig, axs = plt.subplots()

axs.plot(phi, quadratic_degradation_derivative(phi),color=color_theory, label="Quadratic")
axs.plot(phi, borden_degradation_derivative(phi),color=color_spectral, label="Cubic")
axs.plot(phi, quartic_degradation_derivative(phi),color=color_volumetric, label="Quartic")

axs.set_xlabel(r"$\phi$")
axs.set_ylabel(r"$g'(\phi)$")
# axs.legend()
plt.savefig(os.path.join(results_folder, "phi_vs_gp_phi"))

plt.show()
