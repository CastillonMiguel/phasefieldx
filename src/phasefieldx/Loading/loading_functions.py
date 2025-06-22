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


def loading_Tx(msh, value=0.0):
    """
    Create a constant scalar load (e.g., traction or body force) in the x-direction.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
        The computational mesh.
    value : float, optional
        The value of the load in the x-direction (default is 0.0).

    Returns
    -------
    dolfinx.fem.Constant
        A constant scalar value for use in variational forms.

    Notes
    -----
    This function is useful for defining constant scalar loads in 1D or as a component in higher dimensions.

    Example
    -------
    >>> T = loading_Tx(mesh, value=1.0)
    >>> L = ufl.inner(T, v) * ds(boundary_id)
    """
    T = dolfinx.fem.Constant(msh, petsc4py.PETSc.ScalarType((value)))
    return T


def loading_Txy(msh, value_x=0.0, value_y=0.0):
    """
    Create a constant vector load (e.g., traction or body force) in 2D.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
        The computational mesh.
    value_x : float, optional
        The value of the load in the x-direction (default is 0.0).
    value_y : float, optional
        The value of the load in the y-direction (default is 0.0).

    Returns
    -------
    dolfinx.fem.Constant
        A constant 2D vector for use in variational forms.

    Notes
    -----
    This function is useful for defining constant vector loads in 2D finite element simulations.

    Example
    -------
    >>> T = loading_Txy(mesh, value_x=1.0, value_y=0.0)
    >>> L = ufl.inner(T, v) * ds(boundary_id)
    """
    T = dolfinx.fem.Constant(msh, petsc4py.PETSc.ScalarType((value_x, value_y)))
    return T


def loading_Txyz(msh, value_x=0.0, value_y=0.0, value_z=0.0):
    """
    Create a constant vector load (e.g., traction or body force) in 3D.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
        The computational mesh.
    value_x : float, optional
        The value of the load in the x-direction (default is 0.0).
    value_y : float, optional
        The value of the load in the y-direction (default is 0.0).
    value_z : float, optional
        The value of the load in the z-direction (default is 0.0).

    Returns
    -------
    dolfinx.fem.Constant
        A constant 3D vector for use in variational forms.

    Notes
    -----
    This function is useful for defining constant vector loads in 3D finite element simulations.

    Example
    -------
    >>> T = loading_Txyz(mesh, value_x=1.0, value_y=0.0, value_z=0.0)
    >>> L = ufl.inner(T, v) * ds(boundary_id)
    """
    T = dolfinx.fem.Constant(msh, petsc4py.PETSc.ScalarType((value_x, value_y, value_z)))
    return T
