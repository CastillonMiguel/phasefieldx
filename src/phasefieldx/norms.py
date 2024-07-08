"""
Norms
=====

Module for calculating norms over a mesh.
This module provides functions to calculate norms, including L2 and H1 semi-norm, over a given mesh.

"""

import dolfinx
import ufl
import numpy as np
import mpi4py


def norm_semiH1(field, msh, dx=ufl.dx):
    """
    Compute the H1 semi-norm of a field over a given mesh.

    Parameters
    ----------
    field : dolfinx.Function
        The field for the H1 semi-norm calculation.
    msh : dolfinx.Mesh
        The mesh over which the H1 semi-norm is computed.
    dx : ufl.Measure, optional
        The measure for integration over the mesh. Default is ufl.dx.

    Returns
    -------
    norm_semiH1 : float
        The H1 semi-norm of the field over the mesh.

    Notes
    -----
    The H1 semi-norm is computed using the formula:
    norm_semiH1 = sqrt( ∫ |∇(field)|^2 dx )

    where '∇' represents the gradient operator, and 'dx' represents the integration over the entire mesh.
    """
    field_form = dolfinx.fem.form(ufl.inner(ufl.grad(field), ufl.grad(field)) * dx)
    local_field = dolfinx.fem.assemble_scalar(field_form)
    norm_semiH1 = np.sqrt(msh.comm.allreduce(local_field, op=mpi4py.MPI.SUM))
    return norm_semiH1


def norm_H1(field, msh, dx=ufl.dx):
    """
    Compute the H1 norm of a field over a given mesh.

    Parameters
    ----------
    field : dolfinx.Function
        The field for the H1 norm calculation.
    msh : dolfinx.Mesh
        The mesh over which the H1 norm is computed.
    dx : ufl.Measure, optional
        The measure for integration over the mesh. Default is ufl.dx.

    Returns
    -------
    norm_H1 : float
        The H1 norm of the field over the mesh.

    Notes
    -----
    The H1 norm is computed using the formula:
    norm_H1 = sqrt( ∫ |field|^2 dx + ∫ |∇(field)|^2 dx )

    where '∇' represents the gradient operator, and 'dx' represents the integration over the entire mesh.
    """
    field_form = dolfinx.fem.form(ufl.dot(field, field) * dx + ufl.inner(ufl.grad(field), ufl.grad(field)) * dx)
    local_error = dolfinx.fem.assemble_scalar(field_form)
    norm_H1 = np.sqrt(msh.comm.allreduce(local_error, op=mpi4py.MPI.SUM))
    return norm_H1


def norm_LP(field, msh, p=2, dx=ufl.dx):
    """
    Compute the Lp norm of a field over a given mesh.

    Parameters
    ----------
    field : dolfinx.Function
        The field for the Lp norm calculation.
    msh : dolfinx.Mesh
        The mesh over which the Lp norm is computed.
    p : float, optional
        The exponent for the Lp norm calculation. Default is 2.
    dx : ufl.Measure, optional
        The measure for integration over the mesh. Default is ufl.dx.

    Returns
    -------
    norm_LP : float
        The Lp norm of the field over the mesh.

    Notes
    -----
    The Lp norm is computed using the formula:
    norm_LP = ( ∫ |field|^p dx )^(1/p)

    where 'dx' represents the integration over the entire mesh.
    """
    field_form = dolfinx.fem.form(ufl.pow(ufl.abs(field), p) * dx)
    local_error = dolfinx.fem.assemble_scalar(field_form)
    norm_LP = (msh.comm.allreduce(local_error, op=mpi4py.MPI.SUM))**(1/p)
    return norm_LP


def norm_L2(field, msh, dx=ufl.dx):
    """
    Compute the L2 norm of a field over a given mesh.

    Parameters
    ----------
    field : dolfinx.Function
        The field for the L2 norm calculation.
    msh : dolfinx.Mesh
        The mesh over which the L2 norm is computed.
    dx : ufl.Measure, optional
        The measure for integration over the mesh. Default is ufl.dx.

    Returns
    -------
    norm_L2 : float
        The L2 norm of the field over the mesh.

    Notes
    -----
    The L2 norm is computed using the formula:
    norm_L2 = sqrt( ∫ |field|^2 dx )

    where 'dx' represents the integration over the entire mesh.
    """
    field_form = dolfinx.fem.form(ufl.dot(field, field) * dx)
    local_field = dolfinx.fem.assemble_scalar(field_form)
    norm_L2 = np.sqrt(msh.comm.allreduce(local_field, op=mpi4py.MPI.SUM))
    return norm_L2
