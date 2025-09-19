"""
Energy
======

"""
import dolfinx
from mpi4py import MPI
import ufl
from phasefieldx.Materials.elastic_isotropic import psi
from phasefieldx.Element.Phase_Field_Fracture.split_energy_stress_tangent_functions import psi_a, psi_b
from phasefieldx.Element.Phase_Field_Fracture.g_degradation_functions import g

def compute_elastic_energy_components(u, Data, comm, dx=ufl.dx):
    """
    Computes the split elastic energy components for a given displacement field.

    This function evaluates the elastic energy components `psi_a` and `psi_b` over the domain,
    using the provided displacement field and material data. The results are reduced across
    all MPI processes to obtain global values.

    Parameters
    ----------
    u : dolfinx.fem.Function
        Displacement field function.
    Data : dict or custom data structure
        Material and model data required for energy computation.
    comm : MPI.Comm
        MPI communicator for parallel reduction.
    dx : ufl.Measure, optional
        Integration measure over the domain (default is `ufl.dx`).

    Returns
    -------
    psi_a_val : float
        The global value of the first split elastic energy component.
    psi_b_val : float
        The global value of the second split elastic energy component.

    Notes
    -----
    The function assumes that `psi_a` and `psi_b` are callable functions that compute
    the respective energy densities, and that the domain is properly set up for integration.
    """
    psi_a_val = comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(psi_a(u, Data) * dx)), op=MPI.SUM)
    psi_b_val = comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(psi_b(u, Data) * dx)), op=MPI.SUM)
    return psi_a_val, psi_b_val

def compute_degraded_elastic_energy(u, phi, Data, comm, dx=ufl.dx):
    """
    Compute the degraded elastic energy for a phase-field fracture model.

    This function calculates the total degraded elastic energy by integrating the product of the degradation function,
    the elastic energy density, and the domain measure over the computational domain. The result is summed across all
    MPI processes.

    Parameters
    ----------
    u : dolfinx.fem.Function
        Displacement field function.
    phi : dolfinx.fem.Function
        Phase-field variable representing damage.
    Data : object
        Data container with model parameters, including the degradation function.
    comm : MPI.Comm
        MPI communicator for parallel reduction.
    dx : ufl.Measure, optional
        Domain integration measure (default is `ufl.dx`).

    Returns
    -------
    degraded_energy : float
        The total degraded elastic energy, summed across all MPI processes.

    Notes
    -----
    - The function assumes that `g` is the degradation function and `psi_a` is the elastic energy density function.
    - The energy is computed using Dolfinx's finite element assembly and MPI reduction.
    """
    degraded_energy = comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(g(phi, Data.degradation_function) * psi_a(u, Data) * dx)), op=MPI.SUM)
    return degraded_energy

def compute_total_energies(u, phi, Data, comm, dx=ufl.dx):
    """
    Computes the total energies for phase field fracture simulations.

    This function calculates the degraded elastic energy and the two components
    of the elastic energy, returning the total energy and its components.

    Parameters
    ----------
    u : Function or ndarray
        Displacement field.
    phi : Function or ndarray
        Phase field variable representing fracture.
    Data : dict or object
        Material and simulation data required for energy computations.
    comm : MPI communicator
        MPI communicator for parallel computations.
    dx : ufl.Measure, optional
        Integration measure (default is ufl.dx).

    Returns
    -------
    E : float or ndarray
        Total energy (degraded elastic energy plus psi_b).
    psi_a : float or ndarray
        First component of the elastic energy.
    psi_b : float or ndarray
        Second component of the elastic energy.
    """
    psi_a, psi_b = compute_elastic_energy_components(u, Data, comm, dx=dx)
    degraded_energy = compute_degraded_elastic_energy(u, phi, Data, comm, dx=dx)
    E = degraded_energy + psi_b
    return E, psi_a, psi_b
