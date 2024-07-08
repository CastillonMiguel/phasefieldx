"""
Conversion
==========

This module provides functions to convert between different material properties used in continuum mechanics. The conversions involve Young’s modulus (E), Poisson’s ratio (ν), Lame’s parameters (λ and μ), and the bulk modulus (K). These functions are essential for understanding the mechanical behavior of materials and performing various engineering calculations. Each function includes validation to ensure the input parameters are within valid ranges, raising appropriate errors if they are not.

"""

def get_lambda_lame(E, nu):
    """
    Calculate the Lame's first parameter (lambda) using Young's modulus (E) and Poisson's ratio (nu).

    Lame's first parameter (lambda) is calculated using the formula:
        lambda = E * nu / ((1.0 - 2.0 * nu) * (1.0 + nu))

    Parameters:
        E (float): Young's modulus of the material.
        nu (float): Poisson's ratio of the material.

    Returns:
        float: Lame's first parameter (lambda).

    Raises:
        ValueError: If the value of Poisson's ratio (nu) is outside the valid range (-1 < nu < 0.5).
    """
    if nu <= -1 or nu >= 0.5:
        raise ValueError("Poisson's ratio (nu) must be within the range -1 < nu < 0.5")
    return E * nu / ((1.0 - 2.0 * nu) * (1.0 + nu))


def get_mu_lame(E, nu):
    """
    Calculate the Lame's second parameter (mu) using Young's modulus (E) and Poisson's ratio (nu).

    Lame's second parameter (mu) is calculated using the formula:
        mu = E / (2.0 * (1.0 + nu))

    Parameters:
        E (float): Young's modulus of the material.
        nu (float): Poisson's ratio of the material.

    Returns:
        float: Lame's second parameter (mu).

    Raises:
        ValueError: If the value of Poisson's ratio (nu) is outside the valid range (-1 < nu < 0.5).
    """
    if nu <= -1 or nu >= 0.5:
        raise ValueError("Poisson's ratio (nu) must be within the range -1 < nu < 0.5")
    return E / (2.0 * (1.0 + nu))


def get_bulk_modulus(E, nu):
    """
    Calculate the bulk modulus using Young's modulus (E) and Poisson's ratio (nu).

    The bulk modulus (K) is calculated using the formula:
        K = E / (3 * (1 - 2 * nu))

    Parameters:
        E (float): Young's modulus of the material.
        nu (float): Poisson's ratio of the material.

    Returns:
        float: Bulk modulus (K).

    Raises:
        ValueError: If the value of Poisson's ratio (nu) is outside the valid range (-1 < nu < 0.5).
    """
    if nu <= -1 or nu >= 0.5:
        raise ValueError("Poisson's ratio (nu) must be within the range -1 < nu < 0.5")
    return E / (3.0 * (1.0 - 2.0 * nu))



def get_youngs_modulus(lambda_, mu):
    """
    Calculate Young's modulus (E) using Lame's parameters (lambda and mu).

    Young's modulus (E) is calculated using the formula:
        E = mu * (3 * lambda + 2 * mu) / (lambda + mu)

    Parameters:
        lambda_ (float): Lame's first parameter.
        mu (float): Lame's second parameter.

    Returns:
        float: Young's modulus (E).

    Raises:
        ValueError: If the value of Lame's second parameter (mu) is not positive.
    """
    if mu <= 0:
        raise ValueError("Lame's second parameter (mu) must be positive")
    return mu * (3 * lambda_ + 2 * mu) / (lambda_ + mu)


def get_poissons_ratio(lambda_, mu):
    """
    Calculate Poisson's ratio (nu) using Lame's parameters (lambda and mu).

    Poisson's ratio (nu) is calculated using the formula:
        nu = lambda / (2 * (lambda + mu))

    Parameters:
        lambda_ (float): Lame's first parameter.
        mu (float): Lame's second parameter.

    Returns:
        float: Poisson's ratio (nu).

    Raises:
        ValueError: If the combination of Lame's parameters results in an invalid value for Poisson's ratio.
    """
    if lambda_ + 2 * mu == 0:
        raise ValueError("Invalid combination of Lame's parameters (lambda and mu)")
    return lambda_ / (2 * (lambda_ + mu))



def get_bulk_modulus_lame(lambda_, mu):
    """
    Calculate the bulk modulus (K) using Lame's parameters (lambda and mu).

    The bulk modulus (K) is calculated using the formula:
        K = lambda + (2/3) * mu

    Parameters:
        lambda_ (float): Lame's first parameter.
        mu (float): Lame's second parameter.

    Returns:
        float: Bulk modulus (K).

    Raises:
        ValueError: If the combination of Lame's parameters results in an invalid value for bulk modulus.
    """
    if lambda_ + 2 * mu == 0:
        raise ValueError("Invalid combination of Lame's parameters (lambda and mu)")
    return lambda_ + (2/3) * mu
