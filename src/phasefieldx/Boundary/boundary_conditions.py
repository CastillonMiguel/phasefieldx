"""
Finite Element Boundary Conditions and Mesh Tagging Utilities
=============================================================

This module provides functions for creating boundary conditions and handling mesh tagging.

Notes
-----
These functions are useful for defining boundary conditions and subdomains in finite element simulations.

Examples
--------
See individual function documentation for usage examples.

"""

import numpy as np
import dolfinx
import ufl
import petsc4py


def bc_phi(facet, V_phi, fdim, value=0.0):
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

    phi_D = petsc4py.PETSc.ScalarType(value)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V_phi, fdim, facet)
    return dolfinx.fem.dirichletbc(phi_D, boundary_dofs, V_phi)


def bc_x(facet, V_u, fdim, value=0.0):
    """
    Create a Dirichlet boundary condition for the x-component of a vector field `u` on a specified facet.

    Parameters
    ----------
    facet : int
        The topological index of the facet on which the boundary condition is applied.
    V_u : dolfinx.FunctionSpace
        The function space associated with the vector field `u`.
    fdim : int
        The topological dimension of the facets (e.g., 2 for triangles).

    Returns
    -------
    dolfinx.fem.dirichletbc
        A Dirichlet boundary condition for the x-component of the vector field `u`.

    Notes
    -----
    This function is useful for defining Dirichlet boundary conditions for the x-component of vector fields in finite element simulations.
    The boundary condition enforces the x-component of `u` to have a constant value (`u_x_D`) on the specified facet.

    Example
    -------
    >>> facet_index = 0
    >>> V_u_space = dolfinx.VectorFunctionSpace(mesh, "CG", 1)
    >>> bc_x_condition = bc_x(facet_index, V_u_space, 2)
    """

    u_x_D = petsc4py.PETSc.ScalarType(value)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        V_u.sub(0), fdim, facet)
    return dolfinx.fem.dirichletbc(u_x_D, boundary_dofs, V_u.sub(0))


def bc_y(facet, V_u, fdim, value=0.0):
    """
    Create a Dirichlet boundary condition for the y-component of a vector field `u` on a specified facet.

    Parameters
    ----------
    facet : int
        The topological index of the facet on which the boundary condition is applied.
    V_u : dolfinx.FunctionSpace
        The function space associated with the vector field `u`.
    fdim : int
        The topological dimension of the facets (e.g., 2 for triangles).

    Returns
    -------
    dolfinx.fem.DirichletBC
        A Dirichlet boundary condition for the y-component of the vector field `u`.

    Notes
    -----
    This function is useful for defining Dirichlet boundary conditions for the y-component of vector fields in finite element simulations.
    The boundary condition enforces the y-component of `u` to have a constant value (`u_y_D`) on the specified facet.

    Example
    -------
    >>> facet_index = 0
    >>> V_u_space = dolfinx.VectorFunctionSpace(mesh, "CG", 1)
    >>> bc_y_condition = bc_y(facet_index, V_u_space, 2)
    """

    u_y_D = petsc4py.PETSc.ScalarType(value)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        V_u.sub(1), fdim, facet)
    return dolfinx.fem.dirichletbc(u_y_D, boundary_dofs, V_u.sub(1))


def bc_z(facet, V_u, fdim, value=0.0):
    """
    Create a Dirichlet boundary condition for the z-component of a vector field `u` on a specified facet.

    Parameters
    ----------
    facet : int
        The topological index of the facet on which the boundary condition is applied.
    V_u : dolfinx.FunctionSpace
        The function space associated with the vector field `u`.
    fdim : int
        The topological dimension of the facets (e.g., 2 for triangles).

    Returns
    -------
    dolfinx.fem.dirichletbc
        A Dirichlet boundary condition for the z-component of the vector field `u`.

    Notes
    -----
    This function is useful for defining Dirichlet boundary conditions for the z-component of vector fields in finite element simulations.
    The boundary condition enforces the z-component of `u` to have a constant value (`u_z_D`) on the specified facet.

    Example
    -------
    >>> facet_index = 0
    >>> V_u_space = dolfinx.VectorFunctionSpace(mesh, "CG", 1)
    >>> bc_z_condition = bc_z(facet_index, V_u_space, 2)
    """

    u_z_D = petsc4py.PETSc.ScalarType(value)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        V_u.sub(2), fdim, facet)
    return dolfinx.fem.dirichletbc(u_z_D, boundary_dofs, V_u.sub(2))


def bc_xy(facet, V_u, fdim, value_x=0.0, value_y=0.0):
    """
    Create Dirichlet boundary conditions for both x and y components of a vector field `u` on a specified facet.

    Parameters
    ----------
    facet : int
        The topological index of the facet on which the boundary conditions are applied.
    V_u : dolfinx.FunctionSpace
        The function space associated with the vector field `u`.
    fdim : int
        The topological dimension of the facets (e.g., 2 for triangles).

    Returns
    -------
    dolfinx.fem.dirichletbc
        Dirichlet boundary conditions for both the x and y components of the vector field `u`.

    Notes
    -----
    This function is useful for defining Dirichlet boundary conditions for both the x and y components of vector fields in finite element simulations.
    The boundary conditions enforce constant values (`u_xy_D`) for both the x and y components on the specified facet.

    Example
    -------
    >>> facet_index = 0
    >>> V_u_space = dolfinx.VectorFunctionSpace(mesh, "CG", 1)
    >>> bc_xy_conditions = bc_xy(facet_index, V_u_space, 2)
    """

    u_xy_D = np.array([value_x, value_y], dtype=petsc4py.PETSc.ScalarType)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V_u, fdim, facet)
    return dolfinx.fem.dirichletbc(u_xy_D, boundary_dofs, V_u)


def bc_xyz(facet, V_u, fdim, value_x=0.0, value_y=0.0, value_z=0.0):
    """
    Create Dirichlet boundary conditions for all three components (x, y, and z) of a vector field `u` on a specified facet.

    Parameters
    ----------
    facet : int
        The topological index of the facet on which the boundary conditions are applied.
    V_u : dolfinx.FunctionSpace
        The function space associated with the vector field `u`.
    fdim : int
        The topological dimension of the facets (e.g., 2 for triangles).

    Returns
    -------
    dolfinx.fem.dirichletbc
        Dirichlet boundary conditions for all three components (x, y, and z) of the vector field `u`.

    Notes
    -----
    This function is useful for defining Dirichlet boundary conditions for all three components (x, y, and z) of vector fields in finite element simulations.
    The boundary conditions enforce constant values (`u_xyz_D`) for all three components on the specified facet.

    Example
    -------
    >>> facet_index = 0
    >>> V_u_space = dolfinx.VectorFunctionSpace(mesh, "CG", 1)
    >>> bc_xyz_conditions = bc_xyz(facet_index, V_u_space, 2)
    """

    u_xyz_D = np.array([value_x, value_y, value_z],
                       dtype=petsc4py.PETSc.ScalarType)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V_u, fdim, facet)
    return dolfinx.fem.dirichletbc(u_xyz_D, boundary_dofs, V_u)


def get_ds_bound_from_marker(facet_marker, msh, fdim):
    """
    Create a `ufl.Measure` representing the boundary of a specific subdomain based on facet markers.

    Parameters
    ----------
    facet_marker : numpy.ndarray
        An array of facet markers indicating the desired subdomain.
    msh : dolfinx.Mesh
        The mesh on which to define the boundary measure.
    fdim : int
        The topological dimension of the facets (e.g., 2 for triangles).

    Returns
    -------
    ufl.Measure
        A UFL Measure object representing the boundary of the specified subdomain.

    Notes
    -----
    This function is useful for defining a UFL Measure representing the boundary of a specific subdomain based on facet markers.
    It allows you to create a boundary measure for a subdomain with a specific facet marker value.

    Parameters:
    - `facet_marker` is an array that specifies which facets belong to the desired subdomain. Facets with marker value 1 are considered part of the subdomain.
    - `msh` is the mesh on which the boundary measure is defined.
    - `fdim` is the topological dimension of the facets (e.g., 2 for triangles).

    Example
    -------
    >>> facet_markers = np.array([0, 1, 1, 0, 1], dtype=int)
    >>> mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 2, 2)
    >>> boundary_measure = get_ds_bound_from_marker(facet_markers, mesh, 1)
    """

    marked_facets = np.hstack([facet_marker])
    marked_values = np.hstack([np.full_like(facet_marker, 1)])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(
        msh, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])
    ds_bound = ufl.Measure(
        "ds", domain=msh, subdomain_data=facet_tag, subdomain_id=1)
    return ds_bound
