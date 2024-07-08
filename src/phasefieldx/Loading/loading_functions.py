"""
Finite Element Loads
====================

This module provides functions for creating loads. 

Notes
-----
These functions are useful for defining loads and subdomains in finite element simulations.

Examples
--------
See individual function documentation for usage examples.

"""

import dolfinx
import petsc4py

def loading_Tx(V_u, msh, ds):
    """
    Create a Dirichlet boundary condition for the scalar field `phi` on a specified facet.

    Parameters
    ----------
    facet : int
        The topological index of the facet on which the boundary condition is applied.
    V_phi : dolfinx.FunctionSpace
        The function space associated with the scalar field `phi`.
    fdim : int
        The topological dimension of the facets (e.g., 2 for triangles).

    Returns
    -------
    dolfinx.fem.dirichletbc
        A Dirichlet boundary condition for the scalar field `phi`.

    Notes
    -----
    This function is useful for defining Dirichlet boundary conditions for scalar fields in finite element simulations.
    The boundary condition enforces `phi` to have a constant value (`phi_D`) on the specified facet.

    Example
    -------
    >>> facet_index = 0
    >>> V_phi_space = dolfinx.FunctionSpace(mesh, "CG", 1)
    >>> bc_phi_condition = bc_phi(facet_index, V_phi_space, 2)
    """
    T = dolfinx.fem.Constant(msh, petsc4py.PETSc.ScalarType((0.0)))
    return T


def loading_Txy(V_u, msh, ds):
    """
    Create a Dirichlet boundary condition for the scalar field `phi` on a specified facet.

    Parameters
    ----------
    facet : int
        The topological index of the facet on which the boundary condition is applied.
    V_phi : dolfinx.FunctionSpace
        The function space associated with the scalar field `phi`.
    fdim : int
        The topological dimension of the facets (e.g., 2 for triangles).

    Returns
    -------
    dolfinx.fem.dirichletbc
        A Dirichlet boundary condition for the scalar field `phi`.

    Notes
    -----
    This function is useful for defining Dirichlet boundary conditions for scalar fields in finite element simulations.
    The boundary condition enforces `phi` to have a constant value (`phi_D`) on the specified facet.

    Example
    -------
    >>> facet_index = 0
    >>> V_phi_space = dolfinx.FunctionSpace(mesh, "CG", 1)
    >>> bc_phi_condition = bc_phi(facet_index, V_phi_space, 2)
    """
    T = dolfinx.fem.Constant(msh, petsc4py.PETSc.ScalarType((0.0, 0.0)))
    return T


def loading_Txyz(V_u, msh, ds):
    """
    t xyz
    """
    T = dolfinx.fem.Constant(msh, petsc4py.PETSc.ScalarType((0.0, 0.0, 0.0)))
    return
