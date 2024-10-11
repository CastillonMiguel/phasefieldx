"""
Errors
======
Module for calculating errors.

This module provides functions to calculate error norms, including L2 error and H1 semi-norm, between fields defined over a given mesh.
Additional error calculation functions can be added as needed.

"""

import dolfinx
import ufl
import numpy as np
import mpi4py
from phasefieldx.norms import norm_semiH1, norm_H1, norm_LP, norm_L2


def error_L2_higher_order_space(uh, u_ex, degree_raise=3, dx=ufl.dx):
    """
    error_L2_higher_order_space _summary_

    _extended_summary_

    Parameters
    ----------
    uh : _type_
        _description_
    u_ex : _type_
        _description_
    degree_raise : int, optional
        _description_, by default 3
    dx : _type_, optional
        _description_, by default ufl.dx

    Returns
    -------
    _type_
        _description_
    """
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = dolfinx.fem.FunctionSpace(mesh, (family, degree + degree_raise))

    # Interpolate approximate solution
    u_W = dolfinx.fem.Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = dolfinx.fem.Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = dolfinx.fem.Expression(u_ex, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = dolfinx.fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = dolfinx.fem.form(ufl.inner(e_W, e_W) * dx)
    error_local = dolfinx.fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=mpi4py.MPI.SUM)
    return np.sqrt(error_global)


def compute_error_semiH1(field_a, field_b, msh, dx=ufl.dx):
    """
    Compute the H1 semi norm error between two fields over a given mesh.

    Parameters
    ----------
    field_a : dolfinx.Function
        The first field for the H1 norm error calculation.
    field_b : dolfinx.Function
        The second field for the H1 norm error calculation.
    msh : dolfinx.Mesh
        The mesh over which the H1 norm error is computed.

    Returns
    -------
    error_semiH1 : float
        The H1 norm error between field_a and field_b over the mesh.

    Notes
    -----
    The H1 semi norm error is computed using the formula:
    error_H1 = sqrt( |∇(field_a - field_b)|^2 dx)

    where '∇' represents the gradient operator, and 'dx' represents the integration
    over the entire mesh.
    """
    error = field_a - field_b
    return norm_semiH1(error, msh, dx)


def compute_error_H1(field_a, field_b, msh, dx=ufl.dx):
    """
    Compute the H1 norm error between two fields over a given mesh.

    Parameters
    ----------
    field_a : dolfinx.Function
        The first field for the H1 norm error calculation.
    field_b : dolfinx.Function
        The second field for the H1 norm error calculation.
    msh : dolfinx.Mesh
        The mesh over which the H1 norm error is computed.

    Returns
    -------
    error_H1 : float
        The H1 norm error between field_a and field_b over the mesh.

    Notes
    -----
    The H1 norm error is computed using the formula:
    error_H1 = sqrt(∫|field_a - field_b|^2 dx + ∫|∇(field_a - field_b)|^2 dx)

    where '∇' represents the gradient operator, and 'dx' represents the integration
    over the entire mesh.
    """
    error = field_a - field_b
    return norm_H1(error, msh, dx)


def eval_error_LP(field_a, field_b, msh, p=2, dx=ufl.dx):
    """
    Compute the Lp error norm between two fields over a given mesh.

    Parameters
    ----------
    field_a : dolfinx.Function
        The first field for the Lp error norm calculation.
    field_b : dolfinx.Function
        The second field for the Lp error norm calculation.
    msh : dolfinx.Mesh
        The mesh over which the Lp error norm is computed.
    p : float, optional
        The exponent for the Lp norm calculation. Default is 2.

    Returns
    -------
    error_Lp : float
        The Lp error norm between field_a and field_b over the mesh.

    Notes
    -----
    The Lp error norm is computed using the formula:
    error_Lp = ( ∫|field_a - field_b|^p dx )^(1/p)

    where 'dx' represents the integration over the entire mesh.
    """
    error = field_a - field_b
    return norm_LP(error, msh, p, dx)


def eval_error_L2(field_a, field_b, msh, dx=ufl.dx):
    """
    Compute the L2 error norm between two fields over a given mesh.

    Parameters
    ----------
    field_a : dolfinx.Function
        The first field for the L2 error norm calculation.
    field_b : dolfinx.Function
        The second field for the L2 error norm calculation.
    msh : dolfinx.Mesh
        The mesh over which the L2 error norm is computed.

    Returns
    -------
    error_L2 : float
        The L2 error norm between field_a and field_b over the mesh.

    Notes
    -----
    The L2 error norm is computed using the formula:
    error_L2 = sqrt(∫|field_a - field_b|^2 dx)

    where 'dx' represents the integration over the entire mesh.
    """
    error = field_a - field_b
    return norm_L2(error, msh, dx)


def eval_error_L2_normalized(field_a, field_b, msh, dx=ufl.dx):
    """
    Compute the normalized L2 error norm between two fields over a given mesh.

    Parameters
    ----------
    field_a : dolfinx.Function
        The first field for the normalized L2 error norm calculation.
    field_b : dolfinx.Function
        The second field for the normalized L2 error norm calculation.
    msh : dolfinx.Mesh
        The mesh over which the normalized L2 error norm is computed.

    Returns
    -------
    normalized_error_L2 : float
        The normalized L2 error norm between field_a and field_b over the mesh.

    Notes
    -----
    The normalized L2 error norm is computed using the formula:
    normalized_error_L2 = sqrt(∫|field_a - field_b|^2 dx) / sqrt(∫|field_a|^2 dx)

    where 'dx' represents the integration over the entire mesh.
    """
    error = field_a - field_b
    error_form = dolfinx.fem.form(ufl.dot(error, error) * dx)
    local_error = dolfinx.fem.assemble_scalar(error_form)
    error_L2_phi = np.sqrt(msh.comm.allreduce(local_error, op=mpi4py.MPI.SUM))
    error_L2_phi_normalized = np.sqrt(msh.comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.dot(field_a, field_a) * ufl.dx)), op=mpi4py.MPI.SUM))

    # Check for division by zero
    if error_L2_phi_normalized == 0:
        # You can return a small number or handle this case differently if
        # needed.
        return 0.0

    return error_L2_phi / error_L2_phi_normalized
