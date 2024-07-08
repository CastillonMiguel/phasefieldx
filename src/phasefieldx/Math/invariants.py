"""
Invarians
=========

File from dolfiny: 
https://github.com/michalhabera/dolfiny/blob/master/src/dolfiny/invariants.py
dolfiny is free software: you can redistribute it and/or modify it under the terms of the 
GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

"""

import ufl

def invariants_principal(A):
    """Principal invariants of (real-valued) tensor A.
    https://doi.org/10.1007/978-3-7091-0174-2_3
    """
    i1 = ufl.tr(A)
    i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
    i3 = ufl.det(A)
    return i1, i2, i3


def invariants_main(A):
    """Main invariants of (real-valued) tensor A.
    https://doi.org/10.1007/978-3-7091-0174-2_3
    """
    j1 = ufl.tr(A)
    j2 = ufl.tr(A * A)
    j3 = ufl.tr(A * A * A)
    return j1, j2, j3


def eigenstate3_legacy(A):
    """Eigenvalues and eigenprojectors of the 3x3 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{2} λ_a * E_a
    with eigenvalues λ_a and their associated eigenprojectors E_a = n_a^R x n_a^L
    ordered by magnitude.
    The eigenprojectors of eigenvalues with multiplicity n are returned as 1/n-fold projector.

    Note: Tensor A must not have complex eigenvalues!
    """
    if ufl.shape(A) != (3, 3):
        raise RuntimeError(f"Tensor A of shape {ufl.shape(A)} != (3, 3) is not supported!")
    #
    eps = 1.0e-10
    #
    A = ufl.variable(A)
    #
    # --- determine eigenvalues λ0, λ1, λ2
    #
    # additively decompose: A = tr(A) / 3 * I + dev(A) = q * I + B
    q = ufl.tr(A) / 3
    B = A - q * ufl.Identity(3)
    # observe: det(λI - A) = 0  with shift  λ = q + ω --> det(ωI - B) = 0 = ω**3 - j * ω - b
    j = ufl.tr(B * B) / 2  # == -I2(B) for trace-free B, j < 0 indicates A has complex eigenvalues
    b = ufl.tr(B * B * B) / 3  # == I3(B) for trace-free B
    # solve: 0 = ω**3 - j * ω - b  by substitution  ω = p * cos(phi)
    #        0 = p**3 * cos**3(phi) - j * p * cos(phi) - b  | * 4 / p**3
    #        0 = 4 * cos**3(phi) - 3 * cos(phi) - 4 * b / p**3  | --> p := sqrt(j * 4 / 3)
    #        0 = cos(3 * phi) - 4 * b / p**3
    #        0 = cos(3 * phi) - r                  with  -1 <= r <= +1
    #    phi_k = [acos(r) + (k + 1) * 2 * pi] / 3  for  k = 0, 1, 2
    p = 2 / ufl.sqrt(3) * ufl.sqrt(j + eps ** 2)  # eps: MMM
    r = 4 * b / p ** 3
    r = ufl.max_value(ufl.min_value(r, +1 - eps), -1 + eps)  # eps: LMM, MMH
    phi = ufl.acos(r) / 3
    # sorted eigenvalues: λ0 <= λ1 <= λ2
    λ0 = q + p * ufl.cos(phi + 2 / 3 * ufl.pi)  # low
    λ1 = q + p * ufl.cos(phi + 4 / 3 * ufl.pi)  # middle
    λ2 = q + p * ufl.cos(phi)  # high
    #
    # --- determine eigenprojectors E0, E1, E2
    #
    E0 = ufl.diff(λ0, A).T
    E1 = ufl.diff(λ1, A).T
    E2 = ufl.diff(λ2, A).T
    #
    return [λ0, λ1, λ2], [E0, E1, E2]


def eigenstate3(A):
    """Eigenvalues and eigenprojectors of the 3x3 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{2} λ_a * E_a
    with (ordered) eigenvalues λ_a and their associated eigenprojectors E_a = n_a^R x n_a^L.

    Note: Tensor A must not have complex eigenvalues!
    """
    if ufl.shape(A) != (3, 3):
        raise RuntimeError(f"Tensor A of shape {ufl.shape(A)} != (3, 3) is not supported!")
    #
    eps = 3.0e-16  # slightly above 2**-(53 - 1), see https://en.wikipedia.org/wiki/IEEE_754
    #
    A = ufl.variable(A)
    #
    # --- determine eigenvalues λ0, λ1, λ2
    #
    I1, I2, I3 = invariants_principal(A)
    dq = 2 * I1**3 - 9 * I1 * I2 + 27 * I3
    #
    Δx = [
        A[0, 1] * A[1, 2] * A[2, 0] - A[0, 2] * A[1, 0] * A[2, 1],
        A[0, 1]**2 * A[1, 2] - A[0, 1] * A[0, 2] * A[1, 1] + A[0, 1] * A[0, 2] * A[2, 2] - A[0, 2]**2 * A[2, 1],
        A[0, 0] * A[0, 1] * A[2, 1] - A[0, 1]**2 * A[2, 0] - A[0, 1] * A[2, 1] * A[2, 2] + A[0, 2] * A[2, 1]**2,
        A[0, 0] * A[0, 2] * A[1, 2] + A[0, 1] * A[1, 2]**2 - A[0, 2]**2 * A[1, 0] - A[0, 2] * A[1, 1] * A[1, 2],
        A[0, 0] * A[0, 1] * A[1, 2] - A[0, 1] * A[0, 2] * A[1, 0] - A[0, 1] * A[1, 2] * A[2, 2] + A[0, 2] * A[1, 2] * A[2, 1],  # noqa: E501
        A[0, 0] * A[0, 2] * A[2, 1] - A[0, 1] * A[0, 2] * A[2, 0] + A[0, 1] * A[1, 2] * A[2, 1] - A[0, 2] * A[1, 1] * A[2, 1],  # noqa: E501
        A[0, 1] * A[1, 0] * A[1, 2] - A[0, 2] * A[1, 0] * A[1, 1] + A[0, 2] * A[1, 0] * A[2, 2] - A[0, 2] * A[1, 2] * A[2, 0],  # noqa: E501
        A[0, 0]**2 * A[1, 2] - A[0, 0] * A[0, 2] * A[1, 0] - A[0, 0] * A[1, 1] * A[1, 2] - A[0, 0] * A[1, 2] * A[2, 2] + A[0, 1] * A[1, 0] * A[1, 2] + A[0, 2] * A[1, 0] * A[2, 2] + A[1, 1] * A[1, 2] * A[2, 2] - A[1, 2]**2 * A[2, 1],  # noqa: E501
        A[0, 0]**2 * A[1, 2] - A[0, 0] * A[0, 2] * A[1, 0] - A[0, 0] * A[1, 1] * A[1, 2] - A[0, 0] * A[1, 2] * A[2, 2] + A[0, 2] * A[1, 0] * A[1, 1] + A[0, 2] * A[1, 2] * A[2, 0] + A[1, 1] * A[1, 2] * A[2, 2] - A[1, 2]**2 * A[2, 1],  # noqa: E501
        A[0, 0] * A[0, 1] * A[1, 1] - A[0, 0] * A[0, 1] * A[2, 2] - A[0, 1]**2 * A[1, 0] + A[0, 1] * A[0, 2] * A[2, 0] - A[0, 1] * A[1, 1] * A[2, 2] + A[0, 1] * A[2, 2]**2 + A[0, 2] * A[1, 1] * A[2, 1] - A[0, 2] * A[2, 1] * A[2, 2],  # noqa: E501
        A[0, 0] * A[0, 1] * A[1, 1] - A[0, 0] * A[0, 1] * A[2, 2] + A[0, 0] * A[0, 2] * A[2, 1] - A[0, 1]**2 * A[1, 0] - A[0, 1] * A[1, 1] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 1] + A[0, 1] * A[2, 2]**2 - A[0, 2] * A[2, 1] * A[2, 2],  # noqa: E501
        A[0, 0] * A[0, 1] * A[1, 2] - A[0, 0] * A[0, 2] * A[1, 1] + A[0, 0] * A[0, 2] * A[2, 2] - A[0, 1] * A[1, 1] * A[1, 2] - A[0, 2]**2 * A[2, 0] + A[0, 2] * A[1, 1]**2 - A[0, 2] * A[1, 1] * A[2, 2] + A[0, 2] * A[1, 2] * A[2, 1],  # noqa: E501
        A[0, 0] * A[0, 2] * A[1, 1] - A[0, 0] * A[0, 2] * A[2, 2] - A[0, 1] * A[0, 2] * A[1, 0] + A[0, 1] * A[1, 1] * A[1, 2] - A[0, 1] * A[1, 2] * A[2, 2] + A[0, 2]**2 * A[2, 0] - A[0, 2] * A[1, 1]**2 + A[0, 2] * A[1, 1] * A[2, 2],  # noqa: E501
        A[0, 0]**2 * A[1, 1] - A[0, 0]**2 * A[2, 2] - A[0, 0] * A[0, 1] * A[1, 0] + A[0, 0] * A[0, 2] * A[2, 0] - A[0, 0] * A[1, 1]**2 + A[0, 0] * A[2, 2]**2 + A[0, 1] * A[1, 0] * A[1, 1] - A[0, 2] * A[2, 0] * A[2, 2] + A[1, 1]**2 * A[2, 2] - A[1, 1] * A[1, 2] * A[2, 1] - A[1, 1] * A[2, 2]**2 + A[1, 2] * A[2, 1] * A[2, 2]]  # noqa: E501
    Δy = [
        A[0, 2] * A[1, 0] * A[2, 1] - A[0, 1] * A[1, 2] * A[2, 0],
        A[1, 0]**2 * A[2, 1] - A[1, 0] * A[1, 1] * A[2, 0] + A[1, 0] * A[2, 0] * A[2, 2] - A[1, 2] * A[2, 0]**2,
        A[0, 0] * A[1, 0] * A[1, 2] - A[0, 2] * A[1, 0]**2 - A[1, 0] * A[1, 2] * A[2, 2] + A[1, 2]**2 * A[2, 0],
        A[0, 0] * A[2, 0] * A[2, 1] - A[0, 1] * A[2, 0]**2 + A[1, 0] * A[2, 1]**2 - A[1, 1] * A[2, 0] * A[2, 1],
        A[0, 0] * A[1, 0] * A[2, 1] - A[0, 1] * A[1, 0] * A[2, 0] - A[1, 0] * A[2, 1] * A[2, 2] + A[1, 2] * A[2, 0] * A[2, 1],  # noqa: E501
        A[0, 0] * A[1, 2] * A[2, 0] - A[0, 2] * A[1, 0] * A[2, 0] + A[1, 0] * A[1, 2] * A[2, 1] - A[1, 1] * A[1, 2] * A[2, 0],  # noqa: E501
        A[0, 1] * A[1, 0] * A[2, 1] - A[0, 1] * A[1, 1] * A[2, 0] + A[0, 1] * A[2, 0] * A[2, 2] - A[0, 2] * A[2, 0] * A[2, 1],  # noqa: E501
        A[0, 0]**2 * A[2, 1] - A[0, 0] * A[0, 1] * A[2, 0] - A[0, 0] * A[1, 1] * A[2, 1] - A[0, 0] * A[2, 1] * A[2, 2] + A[0, 1] * A[1, 0] * A[2, 1] + A[0, 1] * A[2, 0] * A[2, 2] + A[1, 1] * A[2, 1] * A[2, 2] - A[1, 2] * A[2, 1]**2,  # noqa: E501
        A[0, 0]**2 * A[2, 1] - A[0, 0] * A[0, 1] * A[2, 0] - A[0, 0] * A[1, 1] * A[2, 1] - A[0, 0] * A[2, 1] * A[2, 2] + A[0, 1] * A[1, 1] * A[2, 0] + A[0, 2] * A[2, 0] * A[2, 1] + A[1, 1] * A[2, 1] * A[2, 2] - A[1, 2] * A[2, 1]**2,  # noqa: E501
        A[0, 0] * A[1, 0] * A[1, 1] - A[0, 0] * A[1, 0] * A[2, 2] - A[0, 1] * A[1, 0]**2 + A[0, 2] * A[1, 0] * A[2, 0] - A[1, 0] * A[1, 1] * A[2, 2] + A[1, 0] * A[2, 2]**2 + A[1, 1] * A[1, 2] * A[2, 0] - A[1, 2] * A[2, 0] * A[2, 2],  # noqa: E501
        A[0, 0] * A[1, 0] * A[1, 1] - A[0, 0] * A[1, 0] * A[2, 2] + A[0, 0] * A[1, 2] * A[2, 0] - A[0, 1] * A[1, 0]**2 - A[1, 0] * A[1, 1] * A[2, 2] + A[1, 0] * A[1, 2] * A[2, 1] + A[1, 0] * A[2, 2]**2 - A[1, 2] * A[2, 0] * A[2, 2],  # noqa: E501
        A[0, 0] * A[1, 0] * A[2, 1] - A[0, 0] * A[1, 1] * A[2, 0] + A[0, 0] * A[2, 0] * A[2, 2] - A[0, 2] * A[2, 0]**2 - A[1, 0] * A[1, 1] * A[2, 1] + A[1, 1]**2 * A[2, 0] - A[1, 1] * A[2, 0] * A[2, 2] + A[1, 2] * A[2, 0] * A[2, 1],  # noqa: E501
        A[0, 0] * A[1, 1] * A[2, 0] - A[0, 0] * A[2, 0] * A[2, 2] - A[0, 1] * A[1, 0] * A[2, 0] + A[0, 2] * A[2, 0]**2 + A[1, 0] * A[1, 1] * A[2, 1] - A[1, 0] * A[2, 1] * A[2, 2] - A[1, 1]**2 * A[2, 0] + A[1, 1] * A[2, 0] * A[2, 2],  # noqa: E501
        A[0, 0]**2 * A[1, 1] - A[0, 0]**2 * A[2, 2] - A[0, 0] * A[0, 1] * A[1, 0] + A[0, 0] * A[0, 2] * A[2, 0] - A[0, 0] * A[1, 1]**2 + A[0, 0] * A[2, 2]**2 + A[0, 1] * A[1, 0] * A[1, 1] - A[0, 2] * A[2, 0] * A[2, 2] + A[1, 1]**2 * A[2, 2] - A[1, 1] * A[1, 2] * A[2, 1] - A[1, 1] * A[2, 2]**2 + A[1, 2] * A[2, 1] * A[2, 2]]  # noqa: E501
    Δd = [9, 6, 6, 6, 8, 8, 8, 2, 2, 2, 2, 2, 2, 1]
    Δ = 0
    for i in range(len(Δd)):
        Δ += Δx[i] * Δd[i] * Δy[i]

    Δxp = [A[1, 0], A[2, 0], A[2, 1], -A[0, 0] + A[1, 1], -A[0, 0] + A[2, 2], -A[1, 1] + A[2, 2]]
    Δyp = [A[0, 1], A[0, 2], A[1, 2], -A[0, 0] + A[1, 1], -A[0, 0] + A[2, 2], -A[1, 1] + A[2, 2]]
    Δdp = [6, 6, 6, 1, 1, 1]

    dp = 0
    for i in range(len(Δdp)):
        dp += 1 / 2 * Δxp[i] * Δdp[i] * Δyp[i]

    # Avoid dp = 0 and disc = 0, both are known with absolute error of ~eps**2
    # Required to avoid sqrt(0) derivatives and negative square roots
    dp += eps**2
    Δ += eps**2

    phi3 = ufl.atan2(ufl.sqrt(27) * ufl.sqrt(Δ), dq)

    # sorted eigenvalues: λ0 <= λ1 <= λ2
    λ = [(I1 + 2 * ufl.sqrt(dp) * ufl.cos((phi3 + 2 * ufl.pi * k) / 3)) / 3 for k in range(1, 4)]
    #
    # --- determine eigenprojectors E0, E1, E2
    #
    E = [ufl.diff(λk, A).T for λk in λ]

    return λ, E


def eigenstate2(A):
    """Eigenvalues and eigenprojectors of the 2x2 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{1} λ_a * E_a
    with (ordered) eigenvalues λ_a and their associated eigenprojectors E_a = n_a^R x n_a^L.

    Note: Tensor A must not have complex eigenvalues!
    """
    if ufl.shape(A) != (2, 2):
        raise RuntimeError(f"Tensor A of shape {ufl.shape(A)} != (2, 2) is not supported!")
    #
    eps = 3.0e-16  # slightly above 2**-(53 - 1), see https://en.wikipedia.org/wiki/IEEE_754
    #
    A = ufl.variable(A)
    #
    # --- determine eigenvalues λ0, λ1
    #
    I1, _, _ = invariants_principal(A)
    #
    Δ = (A[0, 0] - A[1, 1])**2 + 4 * A[0, 1] * A[1, 0]  # = I1**2 - 4 * I2
    # Avoid dp = 0 and disc = 0, both are known with absolute error of ~eps**2
    # Required to avoid sqrt(0) derivatives and negative square roots
    Δ += eps**2
    # sorted eigenvalues: λ0 <= λ1
    λ = (I1 - ufl.sqrt(Δ)) / 2, (I1 + ufl.sqrt(Δ)) / 2
    #
    # --- determine eigenprojectors E0, E1
    #
    E = [ufl.diff(λk, A).T for λk in λ]

    return λ, E


def eigenstate(A):
    """Eigenvalues and eigenprojectors of the (real-valued) tensor A of dimension m = 2, 3.
    Provides the spectral decomposition A = sum_{a=0}^{m} λ_a * E_a
    with (ordered) eigenvalues λ_a and their associated eigenprojectors E_a = n_a^R x n_a^L.

    Note: Tensor A must not have complex eigenvalues!
    """
    if ufl.shape(A) == (3, 3):
        return eigenstate3(A)
    elif ufl.shape(A) == (2, 2):
        return eigenstate2(A)
    else:
        raise RuntimeError(f"Tensor A of shape {ufl.shape(A)} is not supported!")


def matrix_function(A, fn=lambda A: A):
    """Evaluates A -> fn(A) : R^(m x m) -> R^(m x m) for the given (real-valued) tensor A and fn.
    Uses spectral decomposition and spectral synthesis fn(A) = sum_{a=0}^{m} fn(λ_a) * E_a.

    Parameters
    ----------
    A
        UFL tensor
    fn
        Functor providing the analytic function

        Examples: `fn=ufl.exp`, `fn=lambda A: A**2`
        Note: If differentiation through the matrix function is needed, consider
              eps-ification of expressions which are not differentiable at critical points,
              e.g. `fn=lambda A: ufl.sqrt(A + eps)`.
    """
    # obtain symbolic spectral decomposition
    λ, E = eigenstate(A)
    # instantiate zero matrix
    fn_A = ufl.zero(ufl.shape(A))
    # apply UFL function on eigenvalue and synthesise matrix function
    for λ_, E_ in zip(λ, E):
        fn_A += fn(λ_) * E_

    return fn_A
