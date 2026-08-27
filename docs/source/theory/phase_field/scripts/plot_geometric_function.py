r"""
.. _ref_TheoryPhaseField_plot_geometric_function:

Geometric functions
-------------------

This script provides functions and plots for the geometric crack functions
$\alpha(\phi)$ used in phase-field fracture models. The geometric functions
are defined for different models (AT1, AT2, Wu, and Double well potential).

"""

###############################################################################
# Import necessary libraries
# --------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil

###############################################################################
# Output folder setup
# -------------------
# Define the folder to save the results and ensure it is clean.
results_folder = "results_plot_geometric_function"
if os.path.exists(results_folder):
    shutil.rmtree(results_folder)
os.makedirs(results_folder)

###############################################################################
# Geometric functions
# -------------------
# These functions define the geometric crack functions $\alpha(\phi)$ for
# different phase-field models.

def evaluate_expression(alpha_function, phi, l0=1.0):
    """
    Evaluate the expression:
    x(phi) = l0 * integral_phi_to_1(alpha(phi)^(-1/2) dphi).

    Parameters:
    - alpha_function: Function representing alpha(phi).
    - phi: Phase field variable (array).
    - l0: Scaling factor (default is 1.0).

    Returns:
    - x: Evaluated values of the expression.
    """
    profile = alpha_function(phi)**(-0.5)
    integral = np.array([
        np.trapz(profile[i:], phi[i:]) for i in range(len(phi))
    ])
    return l0 * integral


def geometric_at2(phi):
    """
    Geometric function for the AT2 model.

    Parameters:
    - phi: Phase field variable.

    Returns:
    - alpha: Geometric function value.
    """
    alpha = phi**2

    return alpha

def geometric_prime_at2(phi):
    """
    Geometric function for the AT2 model.

    Parameters:
    - phi: Phase field variable.

    Returns:
    - alpha: Geometric function value.
    """
    alpha = 2*phi

    return alpha


def geometric_at1(phi):
    """
    Geometric function for the AT1 model.

    Parameters:
    - phi: Phase field variable.

    Returns:
    - alpha: Geometric function value.
    """
    alpha = phi

    return alpha


def geometric_prime_at1(phi):
    """
    Geometric function for the AT1 model.

    Parameters:
    - phi: Phase field variable.

    Returns:
    - alpha: Geometric function value.
    """
    alpha = 1.0+phi*0

    return alpha

def geometric_wu(phi):
    """
    Geometric function for the Wu model.

    Parameters:
    - phi: Phase field variable.

    Returns:
    - alpha: Geometric function value.
    """
    alpha = 2 * phi - phi**2

    return alpha


def geometric_prime_wu(phi):
    """
    Geometric function for the Wu model.

    Parameters:
    - phi: Phase field variable.

    Returns:
    - alpha: Geometric function value.
    """
    alpha = 2* (1- phi)

    return alpha

def geometric_double_well(phi):
    """
    Geometric function for the Check model.

    Parameters:
    - phi: Phase field variable.

    Returns:
    - alpha: Geometric function value.
    """
    alpha = 16 * phi**2 * (1 - phi)**2
    return alpha

def geometric_prime_double_well(phi):
    """
    Geometric function for the Check model.

    Parameters:
    - phi: Phase field variable.

    Returns:
    - alpha: Geometric function value.
    """
    alpha = 32 * phi * (1 - phi) * (1 - 2*phi)
    return alpha

label_at1 = r"AT1"
color_at1 = "green"
linestyle_at1 = '--'

marker_at1 = 'x'

label_at2 = r"AT2"
color_at2 = "red"
linestyle_at2 = '-'
marker_at2 = 'h'

label_wu = r"Wu"
color_wu = "blue"
linestyle_wu = ':'
marker_wu = 'o'


label_double_well = r"Double Well Potential"
color_double_well = "green"

###############################################################################
# Plot geometric functions $\alpha(\phi)$
# ---------------------------------------
# Plot the geometric functions for different models and save the results.

phi = np.linspace(-0.25, 1.25, 1000)

markevery_at1 = max(1, len(phi)//20)
markevery_at2 = max(1, len(phi)//20)
markevery_wu  = max(1, len(phi)//20)
markevery_double_well = max(1, len(phi)//20)


# %%
# AT1, AT2, WU 
fig, ax_reaction = plt.subplots()

ax_reaction.plot(
    phi, geometric_at2(phi), color=color_at2, linestyle=linestyle_at2, label=label_at2, markevery=markevery_at2, marker=marker_at2
)
ax_reaction.plot(
    phi, geometric_at1(phi), color=color_at1, linestyle=linestyle_at1, label=label_at1, markevery=markevery_at1, marker=marker_at1
)
ax_reaction.plot(
    phi, geometric_wu(phi), color=color_wu, linestyle=linestyle_wu, label=label_wu, markevery=markevery_wu, marker=marker_wu
)

ax_reaction.set_xlabel(r"$\phi$")
ax_reaction.set_ylabel(r"$\alpha(\phi)$")
ax_reaction.legend()

# Get limits from the first plot
xlims = ax_reaction.get_xlim()
ylims = ax_reaction.get_ylim()

plt.savefig(os.path.join(results_folder, "geometric_functions_at1_at2_wu"))



# %%
# Double Well Potential
fig, ax_check = plt.subplots()

# Plot the Check geometric function
ax_check.plot(
    phi, geometric_double_well(phi), '-', label=label_double_well, color=color_double_well, markevery=markevery_double_well, marker='o'
)

ax_check.set_xlabel(r"$\phi$")
ax_check.set_ylabel(r"$\alpha(\phi)$")
# ax_check.legend()

ax_check.set_xlim(xlims)
ax_check.set_ylim(ylims)

plt.savefig(os.path.join(results_folder, "geometric_function_double_well"))





plt.show()
