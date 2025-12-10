"""
Energy
======

"""
import dolfinx
from mpi4py import MPI
import ufl
from phasefieldx.Element.Allen_Cahn.potential import potential_function, potential_coefficient

def calculate_potential_energy(Φ, l, comm, case='DOUBLE', dx=ufl.dx):
    """
    Calculates the total crack surface energy in a phase-field model.

    This function computes the integral of the crack surface energy density
    functional, which regularizes a sharp crack over a finite width. The
    functional is expressed as:

    Ψ_c = ∫_Ω [ (1 / (c₀ * l)) * w(Φ) + (l / c₀) * |∇Φ|² ] dΩ
   
    where `w(Φ)` is the geometric crack function (e.g., `Φ²` for the AT2 model,
    `Φ` for the AT1 model), `l` is the length scale parameter, and `Φ` is the
    phase-field variable.
    The calculation is performed in parallel using the provided MPI communicator.

    Φ : dolfinx.fem.Function
        The phase-field variable, representing the crack state.
    l : float
        The length scale parameter, controlling the regularization width.
    comm : mpi4py.MPI.Comm
        The MPI communicator for parallel computation.
    case : str, optional
        The phase-field model type, which determines the geometric crack
        function. Common values are 'AT1' or 'AT2'. Defaults to 'AT2'.
    dx : ufl.Measure, optional
        The UFL integration measure over the domain. Defaults to `ufl.dx`.

    tuple[float, float, float]
        A tuple containing:
        - The total crack surface energy (scalar value).
        - The contribution from the geometric crack function term.
        - The contribution from the phase-field gradient term.
    """
    c0 = potential_coefficient(case)
    gamma_phi = comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(1 / (c0 * l) * potential_function(Φ, case) * dx)),op=MPI.SUM)
    gamma_gradphi = comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        l / c0 * ufl.inner(ufl.grad(Φ), ufl.grad(Φ)) * dx)),op=MPI.SUM)
    gamma = gamma_phi + gamma_gradphi
    return gamma, gamma_phi, gamma_gradphi
