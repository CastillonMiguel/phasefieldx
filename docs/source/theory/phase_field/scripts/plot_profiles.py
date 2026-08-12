r"""

.. _ref_TheoryPhaseField_plot_profiles:

Phase-Field Profiles for a 1D Bar
---------------------------------

This script calculates and plots theoretical phase-field profiles for a crack
in a one-dimensional bar. The crack is centered at x=0 within a domain of [-a, a].

The profiles represent the analytical solutions to the ordinary differential
equations (ODEs) that govern the phase-field variable for different models (AT1, AT2, and Wu).
The script demonstrates how varying the length-scale parameter `l` affects the
solution, including boundary effects.

When the length-scale parameter is small relative to the domain size, the
phase-field solution approximates the sharp crack profile.

"""

###############################################################################
# Import necessary libraries
# --------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

save_figures = True
if save_figures:
    results_folder = "results_plot_profiles"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

###############################################################################
# Phase field profile functions
# ------------------------------
# These functions define the phase field profiles and their gradients for different formulations.
import sys
import numpy as np
def phi_at2(x, length_scale, a):
    """
    Phase field profile AT2

    Parameters:
    - x: Position along the bar.
    - length_scale: Length scale parameter controlling the width of the transition zone.
    - a: Half-length of the bar [-a, a].

    Returns:
    - Phase field value at position x.
    """
    one_div_exp2adivl_one = 1 / (np.exp(2 * a / length_scale) + 1)
    return np.exp(-abs(x) / length_scale) + one_div_exp2adivl_one * 2 * np.sinh(np.abs(x) / length_scale)


def gradphi_at2(x, length_scale, a):
    """
    Gradient of the phase field for the phi_at2 formulation.

    Parameters:
    - x: Position along the bar.
    - length_scale: Length scale parameter controlling the width of the transition zone.
    - a: Half-length of the bar [-a, a].

    Returns:
    - Gradient of the phase field at position x.
    """
    one_div_exp2adivl_one = 1 / (np.exp(2 * a / length_scale) + 1)
    return -np.sign(x) / length_scale * np.exp(-abs(x) / length_scale) \
        + one_div_exp2adivl_one * np.sign(x) / length_scale * 2 * np.cosh(np.abs(x) / length_scale)


def phi_at1(x, length_scale, a):
    """
    Phase field profile AT1

    Parameters:
    - x: Position along the bar.
    - length_scale: Length scale parameter controlling the width of the transition zone.
    - a: Half-length of the bar [-a, a].

    Returns:
    - Phase field value at position x.
    """
    control1 = np.heaviside(2 * length_scale - abs(x), 0)
    control2 = np.heaviside(2 * length_scale - a, 0)
    phi = (abs(x**2) / (4 * length_scale**2) - abs(x) / (length_scale) + 1) * control1
    phi += abs(x) / length_scale * (1 - a / (2 * length_scale)) * control2
    return phi


def gradphi_at1(x, length_scale, a):
    """
    Gradient of the phase field for the phi_at1 formulation.

    Parameters:
    - x: Position along the bar.
    - length_scale: Length scale parameter controlling the width of the transition zone.
    - a: Half-length of the bar [-a, a].

    Returns:
    - Gradient of the phase field at position x.
    """
    control1 = np.heaviside(2 * length_scale - abs(x), 0)
    control2 = np.heaviside(2 * length_scale - a, 0)
    gradphi = ((x) / (2 * length_scale**2) - np.sign(x) / length_scale) * control1
    gradphi += np.sign(x) / length_scale * (1 - a / (2 * length_scale)) * control2
    return gradphi


def phi_wu(x, length_scale, a):
    """
    Phase field profile Wu

    Parameters:
    - x: Position along the bar.
    - length_scale: Length scale parameter controlling the width of the transition zone.
    - a: Half-length of the bar [-a, a].

    Returns:
    - Phase field value at position x.
    """
    control1 = np.heaviside(length_scale * np.pi / 2 - abs(x), 0)
    control2 = np.heaviside(a - length_scale * np.pi / 2 , 0)
    # phi = (1.0 - np.sin(abs(x) / length_scale)) * control1 * (1-control2) + control2
    
    phi = (1.0 - np.sin(abs(x) / length_scale)) * control1 * control2
    return phi


def gradphi_wu(x, length_scale, a):
    """
    Gradient of the phase field for the phi_wu formulation.

    Parameters:
    - x: Position along the bar.
    - length_scale: Length scale parameter controlling the width of the transition zone.
    - a: Half-length of the bar [-a, a].

    Returns:
    - Gradient of the phase field at position x.
    """
    control1 = np.heaviside(length_scale * np.pi / 2 - abs(x), 0)
    gradphi = (-np.cos(abs(x) / length_scale)) * np.sign(x) / length_scale * control1
    return gradphi



###############################################################################
# Parameters definitions
# ----------------------
# These parameters define the length scale and the half-length of the bar for the phase field profiles.

a = 1.0
x = np.linspace(-a, a, 10000)

l = 0.1*a
l1 = 0.1*a
l2 = 0.5*a
l3 = 2.2*a

l1_label = r"$l/a=0.1$"
l2_label = r"$l/a=0.5$"
l3_label = r"$l/a=1.2$"

color_l1 = "blue"
color_l2 = "orangered"
color_l3 = "purple"

###############################################################################
# AT1 Phase-field model
# ---------------------
label_1 = r"AT1"
color_1 = "blue"
markevery_1 = max(1, len(x)//20)

phi_at1_profile_l1 = phi_at1(x, l1, a)
gradphi_at1_profile_l1 = gradphi_at1(x, l1, a)
label_at1_l1 = l1_label

phi_at1_profile_l2 = phi_at1(x, l2, a)
gradphi_at1_profile_l2 = gradphi_at1(x, l2, a)
label_at1_l2 = l1_label

phi_at1_profile_l3 = phi_at1(x, l3, a)
gradphi_at1_profile_l3 = gradphi_at1(x, l3, a)
label_at1_l3 = l3_label


    
###############################################################################
# AT2 Phase-field model
# ---------------------
label_2 = r"AT2"
color_2 = "red"
markevery_2 = max(1, len(x)//20)

phi_at2_profile_l1 = phi_at2(x, l, a)
gradphi_at2_profile_l1 = gradphi_at2(x, l, a)
label_at2_l1 = l1_label

phi_at2_profile_l2 = phi_at2(x, l2, a)
gradphi_at2_profile_l2 = gradphi_at2(x, l2, a)
label_at2_l2 = l2_label

phi_at2_profile_l3 = phi_at2(x, l3, a)
gradphi_at2_profile_l3 = gradphi_at2(x, l3, a)
label_at2_l3 = l3_label


    
###############################################################################
# WU Phase-field model
# --------------------
label_3 = r"Wu"
color_3 = "green"
markevery_3 = max(1, len(x)//20)

phi_wu_profile_l1 = phi_wu(x, l1, a)
gradphi_wu_profile_l1 = gradphi_wu(x, l1, a)
label_wu_l1 = l1_label

phi_wu_profile_l2 = phi_wu(x, l2, a)
gradphi_wu_profile_l2 = gradphi_wu(x, l2, a)
label_wu_l2 = l2_label

phi_wu_profile_l3 = phi_wu(x, l3, a)
gradphi_wu_profile_l3 = gradphi_wu(x, l3, a)
label_wu_l3 = l3_label





###############################################################################
# Compare models
# --------------

# %%
# Compare phase field profiles
fig, ax_compare_phi = plt.subplots(figsize=(11.69, 5.85))

ax_compare_phi.plot(x, phi_at1_profile_l1, color=color_1, linestyle='-', label=label_1)
ax_compare_phi.plot(x, phi_at2_profile_l1, color=color_2, linestyle='--', label=label_2)
ax_compare_phi.plot(x, phi_wu_profile_l1, color=color_3, linestyle='-.', label=label_3)
ax_compare_phi.set_xlabel(r"$x$")
ax_compare_phi.set_ylabel(r"$\phi(x)$")
ax_compare_phi.legend()
if save_figures:
    plt.savefig(os.path.join(results_folder, "compare_phi_profiles.png"))

# %%
# Compare phase field gradient profiles
fig, ax_compare_gradphi = plt.subplots(figsize=(11.69, 5.85))

ax_compare_gradphi.plot(x[0:int(len(x)/2)], gradphi_at1_profile_l1[0:int(len(x)/2)], color=color_1, linestyle='-', label=label_1)
ax_compare_gradphi.plot(x[1+int(len(x)/2):], gradphi_at1_profile_l1[1+int(len(x)/2):], color=color_1, linestyle='-')

ax_compare_gradphi.plot(x[0:int(len(x)/2)], gradphi_at2_profile_l1[0:int(len(x)/2)], color=color_2, linestyle='--', label=label_2)

ax_compare_gradphi.plot(x[0:int(len(x)/2)], gradphi_wu_profile_l1[0:int(len(x)/2)], color=color_3, linestyle='-.', label=label_3)
ax_compare_gradphi.plot(x[1+int(len(x)/2):], gradphi_wu_profile_l1[1+int(len(x)/2):], color=color_3, linestyle='-.')

ax_compare_gradphi.set_xlabel(r"$x$")
ax_compare_gradphi.set_ylabel(r"$\phi'(x)$")
ax_compare_gradphi.legend()
if save_figures:
    plt.savefig(os.path.join(results_folder, "compare_gradphi_profiles.png"))

plt.show()
