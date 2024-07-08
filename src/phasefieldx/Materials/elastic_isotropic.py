"""
Elactic isotropic
=================

This module provides functions for calculating various mechanical properties of materials based on displacement vectors and material parameters. The focus is on determining strain, strain energy density, and stress, which are fundamental to understanding the mechanical behavior of materials under load. These calculations are essential for simulations and analyses in computational mechanics.

"""

import ufl

def epsilon(u):
    """
    Calculate the symmetric strain tensor.

    Parameters:
        u (ufl.Expr): The displacement vector.

    Returns:
        ufl.Expr: The symmetric strain tensor calculated from the displacement vector.
    """
    return ufl.sym(ufl.grad(u))


def psi(u, lambda_, mu):
    """
    Calculate the strain energy density.

    This function calculates the strain energy density of a material using the given displacement vector,
    Lame's first parameter (lambda_), and shear modulus (mu).

    Parameters:
        u (ufl.Expr): The displacement vector.
        lambda_ (ufl.Expr): Lame's first parameter.
        mu (ufl.Expr): Shear modulus.

    Returns:
        ufl.Expr: The strain energy density calculated from the given parameters.
    """
    eps = epsilon(u)
    return 0.5 * lambda_ * ufl.tr(eps)**2 + mu * ufl.inner(eps, eps)


def sigma(u, lambda_, mu):
    """
    Calculate the stress tensor.

    This function calculates the stress tensor of a material using the given displacement vector,
    Lame's first parameter (lambda_), and shear modulus (mu).

    Parameters:
        u (ufl.Expr): The displacement vector.
        lambda_ (ufl.Expr): Lame's first parameter.
        mu (ufl.Expr): Shear modulus.

    Returns:
        ufl.Expr: The stress tensor calculated from the given parameters.
    """
    eps = epsilon(u)
    return lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * eps
