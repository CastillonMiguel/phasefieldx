r"""
Energy Decomposition
====================

Regarding energy degradation, there are two main descriptions of the constitutive free energy functional.
The isotropic model degrades all the energy; in the other case, the anisotropic model only degrades the energy related to tension efforts, trying to account that cracks only are generated under tension efforts.

In the Isotropic constitutive free energy functional,

.. math::

   E(u,\phi)=\int \left[g(\phi)+k\right] \psi(\boldsymbol \epsilon(\boldsymbol u)) dV,


the degradation function $g(\phi)$ degrades all the energy $\psi(\boldsymbol \epsilon(\boldsymbol u))$. In the case of the anisotropic constitutive free energy functional,

.. math::

   E(u,\phi)=\int \left(\left[g(\phi)+k\right] \psi_a(\boldsymbol \epsilon(\boldsymbol u)) + \psi_b(\boldsymbol \epsilon(\boldsymbol u))\right) dV,

the energy $\psi(\boldsymbol \epsilon(\boldsymbol u))$ is split in two components, $\psi=\psi_a+\psi_b$, in which the component $\psi_a$ attains to the energy relative to tensions, so the degradation function only applied to it, avoiding the creation of crack due to compression effort.

There are several methods to carry out the energy slip. PhaseFieldX code implements the Spectral [Miehe]_ and volumetric-deviatoric [Amor]_ decomposition.


"""

import ufl
from phasefieldx.Math.functions import macaulay_bracket_positive, macaulay_bracket_negative
from phasefieldx.Materials.elastic_isotropic import epsilon, sigma, psi
from phasefieldx.Math.tensor_decomposition import deviatoric_part, spectral_positive_part, spectral_negative_part


def psi_a_spectral(u, lambda_, mu):
    """
    Evaluate the spectral positive part of the strain energy density function.

    Parameters
    ----------
    u : ufl.Function
        The displacement field.
    lambda_ : float
        The first Lamé parameter.
    mu : float
        The second Lamé parameter.

    Returns
    -------
    float
        The spectral positive part of the strain energy density.
    """
    eps = epsilon(u)
    eps_P = spectral_positive_part(eps)
    return 0.5 * lambda_ * \
        macaulay_bracket_positive(ufl.tr(eps))**2 + \
        mu * ufl.inner(eps_P, eps_P)


def psi_b_spectral(u, lambda_, mu):
    """
    Evaluate the spectral negative part of the strain energy density function.

    Parameters
    ----------
    u : ufl.Function
        The displacement field.
    lambda_ : float
        The first Lamé parameter.
    mu : float
        The second Lamé parameter.

    Returns
    -------
    float
        The spectral negative part of the strain energy density.
    """
    eps = epsilon(u)
    eps_N = spectral_negative_part(eps)
    return 0.5 * lambda_ * \
        macaulay_bracket_negative(ufl.tr(eps))**2 + \
        mu * ufl.inner(eps_N, eps_N)


def sigma_a_spectral(u, lambda_, mu):
    """
    Evaluate the spectral positive part of the stress tensor.

    Parameters
    ----------
    u : ufl.Function
        The displacement field.
    lambda_ : float
        The first Lamé parameter.
    mu : float
        The second Lamé parameter.

    Returns
    -------
    ufl.Expr
        The spectral positive part of the stress tensor.
    """
    eps = epsilon(u)
    eps_P = spectral_positive_part(eps)
    return lambda_ * \
        macaulay_bracket_positive(ufl.tr(eps)) * \
        ufl.Identity(len(u)) + 2 * mu * eps_P


def sigma_b_spectral(u, lambda_, mu):
    """
    Evaluate the spectral negative part of the stress tensor.

    Parameters
    ----------
    u : ufl.Function
        The displacement field.
    lambda_ : float
        The first Lamé parameter.
    mu : float
        The second Lamé parameter.

    Returns
    -------
    ufl.Expr
        The spectral negative part of the stress tensor.
    """
    # eps = epsilon(u)
    # eps_N = spectral_negative_part(eps)
    # return lambda_ *
    # macaulay_bracket_negative(ufl.tr(eps))*ufl.Identity(len(u)) + 2*mu*eps_N
    return sigma(u, lambda_, mu) - sigma_a_spectral(u, lambda_, mu)


def psi_a_deviatoric(u, lambda_, mu):
    """
    Evaluate the deviatoric positive part of the strain energy density function.

    Parameters
    ----------
    u : ufl.Function
        The displacement field.
    lambda_ : float
        The first Lamé parameter.
    mu : float
        The second Lamé parameter.

    Returns
    -------
    float
        The deviatoric positive part of the strain energy density.
    """
    eps = epsilon(u)
    epsD = deviatoric_part(eps)
    # epsD = ufl.dev(eps)
    k0 = lambda_ + 2 / len(u) * mu
    return 0.5 * k0 * \
        macaulay_bracket_positive(ufl.tr(eps))**2 + mu * ufl.inner(epsD, epsD)


def psi_b_deviatoric(u, lambda_, mu):
    """
    Evaluate the deviatoric negative part of the strain energy density function.

    Parameters
    ----------
    u : ufl.Function
        The displacement field.
    lambda_ : float
        The first Lamé parameter.
    mu : float
        The second Lamé parameter.

    Returns
    -------
    float
        The deviatoric negative part of the strain energy density.
    """
    eps = epsilon(u)
    k0 = lambda_ + 2 / len(u) * mu
    return 0.5 * k0 * macaulay_bracket_negative(ufl.tr(eps))**2


def sigma_a_deviatoric(u, lambda_, mu):
    """
    Evaluate the deviatoric positive part of the stress tensor.

    Parameters
    ----------
    u : ufl.Function
        The displacement field.
    lambda_ : float
        The first Lamé parameter.
    mu : float
        The second Lamé parameter.

    Returns
    -------
    ufl.Expr
        The deviatoric positive part of the stress tensor.
    """
    eps = epsilon(u)
    epsD = deviatoric_part(eps)
    # epsD= ufl.dev(eps)
    k0 = lambda_ + 2 / len(u) * mu
    return k0 * \
        macaulay_bracket_positive(ufl.tr(eps)) * \
        ufl.Identity(len(u)) + 2 * mu * epsD


def sigma_b_deviatoric(u, lambda_, mu):
    """
    Evaluate the deviatoric negative part of the stress tensor.

    Parameters
    ----------
    u : ufl.Function
        The displacement field.
    lambda_ : float
        The first Lamé parameter.
    mu : float
        The second Lamé parameter.

    Returns
    -------
    ufl.Expr
        The deviatoric negative part of the stress tensor.
    """
    # eps = epsilon(u)
    # k0 = lambda_+2/len(u)*mu
    # return k0 * macaulay_bracket_negative(ufl.tr(eps))*ufl.Identity(len(u))
    return sigma(u, lambda_, mu) - sigma_a_deviatoric(u, lambda_, mu)


def psi_a(u, DataSimulation):
    """
    Evaluate the positive part of the strain energy density function based on the specified degradation type.

    Parameters
    ----------
    u : ufl.Function
        The displacement field.
    DataSimulation : object
        An object containing simulation parameters, including degradation type and split energy type.

    Returns
    -------
    float
        The positive part of the strain energy density.
    """
    if DataSimulation.degradation == "isotropic":
        return psi(u, DataSimulation.lambda_, DataSimulation.mu)

    elif DataSimulation.degradation == "anisotropic" or DataSimulation.degradation == "hybrid":
        if DataSimulation.split_energy == "deviatoric":
            return psi_a_deviatoric(
                u, DataSimulation.lambda_, DataSimulation.mu)
        elif DataSimulation.split_energy == "spectral":
            return psi_a_spectral(u, DataSimulation.lambda_, DataSimulation.mu)

    else:
        raise ValueError("Invalid")


def psi_b(u, DataSimulation):
    """
    Evaluate the negative part of the strain energy density function based on the specified degradation type.

    Parameters
    ----------
    u : ufl.Function
        The displacement field.
    DataSimulation : object
        An object containing simulation parameters, including degradation type and split energy type.

    Returns
    -------
    float
        The negative part of the strain energy density.
    """
    if DataSimulation.degradation == "isotropic":
        return psi(u, DataSimulation.lambda_, DataSimulation.mu) - \
            psi(u, DataSimulation.lambda_, DataSimulation.mu)

    elif DataSimulation.degradation == "anisotropic" or DataSimulation.degradation == "hybrid":
        if DataSimulation.split_energy == "deviatoric":
            return psi_b_deviatoric(
                u, DataSimulation.lambda_, DataSimulation.mu)
        elif DataSimulation.split_energy == "spectral":
            return psi_b_spectral(u, DataSimulation.lambda_, DataSimulation.mu)

    else:
        raise ValueError("Invalid")


def sigma_a(u, DataSimulation):
    """
    Evaluate the positive part of the stress tensor based on the specified degradation type.

    Parameters
    ----------
    u : ufl.Function
        The displacement field.
    DataSimulation : object
        An object containing simulation parameters, including degradation type and split energy type.

    Returns
    -------
    ufl.Expr
        The positive part of the stress tensor.
    """
    if DataSimulation.degradation == "isotropic" or DataSimulation.degradation == "hybrid":
        return sigma(u, DataSimulation.lambda_, DataSimulation.mu)

    elif DataSimulation.degradation == "anisotropic":
        if DataSimulation.split_energy == "deviatoric":
            return sigma_a_deviatoric(
                u, DataSimulation.lambda_, DataSimulation.mu)
        elif DataSimulation.split_energy == "spectral":
            return sigma_a_spectral(
                u, DataSimulation.lambda_, DataSimulation.mu)

    else:
        raise ValueError("Invalid")


def sigma_b(u, DataSimulation):
    """
    Evaluate the negative part of the stress tensor based on the specified degradation type.

    Parameters
    ----------
    u : ufl.Function
        The displacement field.
    DataSimulation : object
        An object containing simulation parameters, including degradation type and split energy type.

    Returns
    -------
    ufl.Expr
        The negative part of the stress tensor.
    """
    if DataSimulation.degradation == "isotropic" or DataSimulation.degradation == "hybrid":
        return 0 * sigma(u, DataSimulation.lambda_, DataSimulation.mu)

    elif DataSimulation.degradation == "anisotropic":
        if DataSimulation.split_energy == "deviatoric":
            return sigma_b_deviatoric(
                u, DataSimulation.lambda_, DataSimulation.mu)
        elif DataSimulation.split_energy == "spectral":
            return sigma_b_spectral(
                u, DataSimulation.lambda_, DataSimulation.mu)

    else:
        raise ValueError("Invalid")
