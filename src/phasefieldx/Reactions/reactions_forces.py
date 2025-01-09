"""
Reaction Forces
===============

This module provides functions to calculate reaction forces based on the residual vector and boundary conditions.
"""

import dolfinx
import numpy as np
import petsc4py

def calculate_reaction_forces(J_form, F_form, bcs, u, dimension):
    """
    Compute the reaction forces at the constrained degrees of freedom (DOFs).

    Parameters
    ----------
    F_form : dolfinx.fem.Form
        The residual form.
    J_form : dolfinx.fem.Form
        The Jacobian form.
    bcs : list of dolfinx.fem.DirichletBC
        List of Dirichlet boundary conditions.
    u : dolfinx.fem.Function
        The solution function.
    dimension : int
        The spatial dimension of the problem (1, 2, or 3).

    Returns
    -------
    reaction_forces : numpy.ndarray
        Array containing the reaction forces at the constrained DOFs.
    """
    
    # Assemble the residual vector F
    residual_vector = dolfinx.fem.petsc.create_vector(F_form)
    with residual_vector.localForm() as loc_L:
        loc_L.set(0.0)
    dolfinx.fem.petsc.assemble_vector(residual_vector, F_form)

    # Apply lifting of the boundary conditions to include contributions at constrained DOFs
    dolfinx.fem.petsc.apply_lifting(residual_vector, [J_form], [bcs], x0=[u.x.petsc_vec], alpha=1.0)
    dolfinx.fem.petsc.set_bc(residual_vector, bcs, u.x.petsc_vec, alpha=1.0)

    # Synchronize ghost values (necessary for parallel runs)
    residual_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD,
                                mode=petsc4py.PETSc.ScatterMode.REVERSE)

    # The reaction forces are the negative of the residual forces at the constrained DOFs
    residual_vector.scale(-1.0)
    reaction_forces = np.array([0.0, 0.0, 0.0])

    if dimension == 1:
        reaction_forces[0] = np.sum(residual_vector.array[0::1]) # rx
        
    elif dimension == 2:
        reaction_forces[0] = np.sum(residual_vector.array[0::2]) # rx
        reaction_forces[1] = np.sum(residual_vector.array[1::2]) # ry

    elif dimension == 3:
        reaction_forces[0] = np.sum(residual_vector.array[0::3]) # rx
        reaction_forces[1] = np.sum(residual_vector.array[1::3]) # ry
        reaction_forces[2] = np.sum(residual_vector.array[2::3]) # rz

    return reaction_forces
