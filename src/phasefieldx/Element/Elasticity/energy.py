"""
Energy
======

"""
import dolfinx
from mpi4py import MPI
import ufl
from phasefieldx.Materials.elastic_isotropic import psi

def calculate_elastic_energy(u, Data, comm, dx=ufl.dx):
    """
    Calculate the total elastic energy by integrating the isotropic elastic energy density over the entire domain.

    Parameters
    ----------
    u : dolfinx.fem.Function
        Displacement field function.
    Data : object
        Material properties container with attributes `lambda_` and `mu` representing Lamé parameters.
    comm : MPI.Comm
        MPI communicator for parallel reduction.
    dx : ufl.Measure, optional
        Integration measure over the domain (default is `ufl.dx`).

    Returns
    -------
    float
        The total elastic energy integrated over the domain.

    Notes
    -----
    This function computes the isotropic elastic energy density using the provided displacement field and material properties,
    then integrates it over the whole domain. The result is summed across all MPI processes to obtain the global elastic energy.

    """
    return comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(
            psi(u, Data.lambda_, Data.mu) * dx)), op=MPI.SUM)
