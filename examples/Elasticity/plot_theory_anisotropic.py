r"""
.. _ref_TheoryAnisotropicModels_plot_anisotropic:

Energy Splitting in Anisotropic Models
--------------------------------------

In anisotropic phase-field models, the strain energy is split to ensure that only the energy contributing to crack formation is degraded. This script implements and validates several common energy splitting methods. The process involves two main steps: first, decomposing the strain tensor into active and inactive components, and second, defining the energy terms based on this split.

Strain Tensor Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The strain tensor :math:`\boldsymbol{\epsilon}` is generally decomposed as:

.. math::
    \boldsymbol{\epsilon} = \boldsymbol{\epsilon}_a(\boldsymbol{\epsilon}) + \boldsymbol{\epsilon}_b(\boldsymbol{\epsilon})

where :math:`\boldsymbol{\epsilon}_a` and :math:`\boldsymbol{\epsilon}_b` represent the split components. The derivatives of these components yield the fourth-order projection tensors:

.. math::
    \mathbb{P}_a = \frac{\partial \boldsymbol{\epsilon}_a(\boldsymbol{\epsilon})}{\partial \boldsymbol{\epsilon}}, \quad
    \mathbb{P}_b = \frac{\partial \boldsymbol{\epsilon}_b(\boldsymbol{\epsilon})}{\partial \boldsymbol{\epsilon}}

Two widely used decomposition methods are implemented:

1.  **Spectral Decomposition (Miehe)**: The strain tensor is split into positive and negative parts based on its eigenvalues (principal strains).

    .. math::
        \boldsymbol{\epsilon} = \sum_{i=1}^{d} \epsilon_i \boldsymbol{n}_i \otimes \boldsymbol{n}_i

    where :math:`\epsilon_i` are the principal strains and :math:`\boldsymbol{n}_i` are the principal strain directions. The positive and negative parts are:

    .. math::
        \boldsymbol{\epsilon}_+ = \sum_{i=1}^{d} \langle \epsilon_i \rangle^+ \boldsymbol{n}_i \otimes \boldsymbol{n}_i, \quad
        \boldsymbol{\epsilon}_- = \sum_{i=1}^{d} \langle \epsilon_i \rangle^- \boldsymbol{n}_i \otimes \boldsymbol{n}_i

    using the bracket operators :math:`\langle x \rangle^\pm = \frac{x \pm |x|}{2}`.

2.  **Volumetric-Deviatoric Decomposition (Amor)**: The strain tensor is separated into its volumetric (spherical) and deviatoric parts.

    .. math::
        \boldsymbol{\epsilon} = \boldsymbol{\epsilon}_{vol} + \boldsymbol{\epsilon}_{dev}

    where:
    
    .. math::
        \boldsymbol{\epsilon}_{vol} = \frac{1}{d} \text{tr}(\boldsymbol{\epsilon}) \boldsymbol{I}, \quad
        \boldsymbol{\epsilon}_{dev} = \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{vol}


Energy Splitting
^^^^^^^^^^^^^^^^

Once the strain tensor is decomposed, the strain energy density :math:`\psi` is split into an active part :math:`\psi_a` (which drives damage) and an inactive part :math:`\psi_b`.

.. math::
    \psi(\boldsymbol{\epsilon}) = \psi_a(\boldsymbol{\epsilon}) + \psi_b(\boldsymbol{\epsilon})

The stress :math:`\boldsymbol{\sigma}` and material tangent :math:`\mathbb{C}` are derived from the total energy.

1.  **Miehe Energy Split**:
    
    .. math::
        \psi_a = \frac{1}{2}\lambda{\langle \text{tr}(\boldsymbol{\epsilon}) \rangle^+}^2 + \mu \text{tr}(\boldsymbol{\epsilon}_+^2) \\
        \psi_b = \frac{1}{2}\lambda{\langle \text{tr}(\boldsymbol{\epsilon}) \rangle^-}^2 + \mu \text{tr}(\boldsymbol{\epsilon}_-^2)

2.  **Amor Energy Split**:

    .. math::
        \psi_a = \frac{1}{2} \kappa_0 {\langle \text{tr}(\boldsymbol{\epsilon}) \rangle^+}^2 + \mu \text{tr}(\boldsymbol{\epsilon}_{dev}^2) \\
        \psi_b = \frac{1}{2} \kappa_0 {\langle \text{tr}(\boldsymbol{\epsilon}) \rangle^-}^2

    where :math:`\kappa_0 = \lambda + \frac{2}{d}\mu` is the bulk modulus.

This script provides functions to perform these decompositions and validates that the sum of the split components correctly reconstructs the original tensors and energy.

"""

import numpy as np

###############################################################################
# General functions
# -----------------
# Here several functions that has to be used to perform the energetic decompositions are considered.


def diagonal(eps):
    t=np.zeros((3,3))
    t[0,0] = eps[0,0]
    t[1,1] = eps[1,1]
    t[2,2] = eps[2,2]
    return t


def leftContract(a,T4):
    ret=np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3): 
                    ##ret(i,j) += a[k][l]*T(k,l,i,j)
                    ret[i,j] += a[i,j]*T4[k,l,i,j]
    return ret

def DoubleContract(Cp,T):
    Tp=np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3): 
                    Tp[k,l]+= Cp[i,j,k,l]*T[i,j]
    return Tp
                

def contract(a,U): #np.tensordot(a, U, axes=2)   a:U
    d=0.0
    for i in range(0,3):
        for j in range(0,3):
            d+=a[i][j] * U[i][j]
    return d


def dyadic(a,b):   #np.tensordot(a, U, axes=0)  a otimes b
     itensor=np.array([[a[0]*b[0], a[0]*b[1], a[0]*b[2]],
                       [a[1]*b[0], a[1]*b[1], a[1]*b[2]],
                       [a[2]*b[0], a[2]*b[1], a[2]*b[2]]])
     return itensor
 

def identity():
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]]) 


def bracket(f):
    f_p=0.5*(f+abs(f))
    f_n=0.5*(f-abs(f))
    return f_p, f_n

def Heaviside(x):
    """
    Computes the Heaviside step function.
    H(x) = 1 if x > 0, 0.5 if x = 0, 0 if x < 0.
    """
    if(x<0.0):
        H=0.0
    elif(x==0.0):
        H=0.5
    elif(x>0.0):
        H=1.0
    return H


###############################################################################
# Spectral Tensor decomposition
# -----------------------------
# This section implements the spectral decomposition of a symmetric second-order tensor.
# The tensor is decomposed into its positive and negative parts based on its eigenvalues.
#
# The decomposition is given by:
# T = T_+ + T_-
# T_+ = sum_{i=1 to d} <lambda_i>^+ n_i otimes n_i
# T_- = sum_{i=1 to d} <lambda_i>^- n_i otimes n_i
#
# where lambda_i are the eigenvalues and n_i are the eigenvectors.

def Spectral(T2):
    """
    Performs the spectral decomposition of a symmetric 2nd-order tensor T2.
    Returns the positive (T2_P) and negative (T2_N) parts.
    """
    eigValues, eigVectors = np.linalg.eig(T2)
    eigValuesP, eigValuesN = bracket(eigValues)
    T2_P = np.zeros([3, 3])
    T2_N = np.zeros([3, 3])
    for i in range(0, 3):
        aux = dyadic(eigVectors[:, i], eigVectors[:, i])
        T2_P += eigValuesP[i] * aux
        T2_N += eigValuesN[i] * aux
    return T2_P, T2_N

def Projection_Spectral(epsilon):
    """
    Computes the 4th-order projection tensors for the spectral decomposition.
    These tensors (PP for positive, PN for negative) can project the strain
    tensor onto its positive and negative parts.
    
    P_+ = partial_epsilon(epsilon_+)
    P_- = partial_epsilon(epsilon_-)
    """
    eigValues,  eigVectors = np.linalg.eig(epsilon)
    eigValuesP, eigValuesN = bracket(eigValues)
    
    Vectors=eigVectors
    PP=np.zeros((3,3,3,3))
  
    for a in range(0,3):
        #nota: se puede optimizar bastante (lo dejo así por claridad)
        for b in range(0,3):
            if a==b:
                aa=np.tensordot(Vectors[:,a], Vectors[:,a], axes=0)
                bb=np.tensordot(Vectors[:,b], Vectors[:,b], axes=0)
                PP+= Heaviside(eigValues[a])*np.tensordot(aa, bb, axes=0)
            
            if a!=b:
                ab=np.tensordot(Vectors[:,a], Vectors[:,b], axes=0)
                ba=np.tensordot(Vectors[:,b], Vectors[:,a], axes=0)
                # For distinct eigenvalues, the formula is used.
                # For repeated eigenvalues, the derivative is the Heaviside function.
                if np.isclose(eigValues[a], eigValues[b]):
                    T = Heaviside(eigValues[a])
                else:
                    T=(eigValuesP[a]-eigValuesP[b])/(eigValues[a]-eigValues[b])
                PP += 0.5 * T * np.tensordot(ab, ab + ba, axes=0)

    I = np.identity(3)
    # The fourth-order symmetric identity tensor
    IS4 = 0.5 * (np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))
    # The sum of positive and negative projection tensors gives the symmetric identity
    PN = IS4 - PP
    return PP, PN

# %%
# So now that the functions for the spectral decomposition are defined.
# We consider a ramdom tensor to test the decomposition.
tensor_3x3 = np.random.rand(3, 3)

# The spectral decomposition is defined for symmetric tensors.
# Let's make the random tensor symmetric.
tensor_3x3 = (tensor_3x3 + tensor_3x3.T) / 2

print("Random Symmetric Tensor:\n", tensor_3x3)
tensor_3x3_a, tensor_3x3_b = Spectral(tensor_3x3)

print("\nPositive part (tensor_3x3_a) from Spectral:\n", tensor_3x3_a)
print("\nNegative part (tensor_3x3_b) from Spectral:\n", tensor_3x3_b)

# Check that the sum of the decomposed tensors equals the original tensor
print("\nSum of positive and negative parts:\n", tensor_3x3_a + tensor_3x3_b)
print("\nIs the sum of the parts equal to the original tensor?", np.allclose(tensor_3x3, tensor_3x3_a + tensor_3x3_b))

# Note that knowing the projection P, it is possible to perform the decompositions
print("\n--- Using Projection Tensors ---")
P_pos, P_neg = Projection_Spectral(tensor_3x3)

# Calculate positive and negative parts using projection tensors
# epsilon_+ = P_+ : epsilon
tensor_3x3_a_proj = np.einsum('ijkl,kl->ij', P_pos, tensor_3x3)
tensor_3x3_b_proj = np.einsum('ijkl,kl->ij', P_neg, tensor_3x3)

print("\nPositive part from Projection:\n", tensor_3x3_a_proj)
print("\nNegative part from Projection:\n", tensor_3x3_b_proj)

# Check that the results from both methods are the same
print("\nIs positive part from Projection same as from Spectral?", np.allclose(tensor_3x3_a, tensor_3x3_a_proj))
print("Is negative part from Projection same as from Spectral?", np.allclose(tensor_3x3_b, tensor_3x3_b_proj))

###############################################################################
# Volumetric-Deviatoric Tensor decomposition
# ------------------------------------------
# This section implements the decomposition of a tensor into its volumetric
# and deviatoric parts.
#
# epsilon = epsilon_vol + epsilon_dev
# epsilon_vol = (1/d) * tr(epsilon) * I
# epsilon_dev = epsilon - epsilon_vol

def VolDev(T2):
    """Decomposes a 2nd-order tensor into its volumetric and deviatoric parts."""
    trace_T2 = T2.trace()
    # Volumetric part: T2_Vol = (1/3) * tr(T2) * I
    T2_Vol = (1.0 / 3.0) * trace_T2 * identity()
    # Deviatoric part: T2_Dev = T2 - T2_Vol
    T2_Dev = T2 - T2_Vol
    return T2_Dev, T2_Vol


def Projection_VolDev():
    """
    Computes the volumetric and deviatoric 4th-order projection tensors.
    P_vol = (1/3) * I otimes I
    P_dev = I_sym - P_vol
    """
    I = identity()
    # Volumetric projection tensor: Pv = 1/3 * (I ⊗ I)
    Pv = (1.0/3.0) * np.einsum('ij,kl->ijkl', I, I)
    # Symmetric 4th-order identity tensor
    IS4 = 0.5 * (np.einsum("ik,jl->ijkl", I, I) + np.einsum("il,jk->ijkl", I, I))
    # Deviatoric projection tensor: Pd = Isym - Pv
    Pd = IS4 - Pv
    return Pd, Pv

# %%
# Now we test the volumetric-deviatoric decomposition.
# We use the same random tensor.

print("\n\n--- Volumetric-Deviatoric Decomposition ---")
tensor_3x3_dev, tensor_3x3_vol = VolDev(tensor_3x3)

print("\nDeviatoric part from VolDev:\n", tensor_3x3_dev)
print("\nVolumetric part from VolDev:\n", tensor_3x3_vol)

# Check that the sum of the decomposed tensors equals the original tensor
print("\nSum of deviatoric and volumetric parts:\n", tensor_3x3_dev + tensor_3x3_vol)
print("\nIs the sum of the parts equal to the original tensor?", np.allclose(tensor_3x3, tensor_3x3_dev + tensor_3x3_vol))

# Note that knowing the projection P, it is possible to perform the decompositions
print("\n--- Using Projection Tensors ---")
P_dev, P_vol = Projection_VolDev()

# Calculate deviatoric and volumetric parts using projection tensors
# epsilon_dev = P_dev : epsilon
tensor_3x3_dev_proj = np.einsum('ijkl,kl->ij', P_dev, tensor_3x3)
tensor_3x3_vol_proj = np.einsum('ijkl,kl->ij', P_vol, tensor_3x3)

print("\nDeviatoric part from Projection:\n", tensor_3x3_dev_proj)
print("\nVolumetric part from Projection:\n", tensor_3x3_vol_proj)

# Check that the results from both methods are the same
print("\nIs deviatoric part from Projection same as from VolDev?", np.allclose(tensor_3x3_dev, tensor_3x3_dev_proj))
print("Is volumetric part from Projection same as from VolDev?", np.allclose(tensor_3x3_vol, tensor_3x3_vol_proj))




###############################################################################
# Energy functions for different decomposition models
# ---------------------------------------------------
# Here, we define the energy, stress, and tangent stiffness tensor functions
# for the standard isotropic model, as well as for the Miehe (spectral) and
# Amor (volumetric-deviatoric) decomposition models.

# Standard Isotropic Model (Reference)
# ------------------------------------
def energy(eps, lamb, mu):
    """
    Calculates the strain energy for the standard isotropic model.
    psi = 0.5 * lambda * tr(eps)^2 + mu * tr(eps^2)
    """
    psi = 0.5 * lamb * eps.trace() ** 2 + mu * contract(eps, eps)
    return psi


def stress(eps, lamb, mu):
    """
    Calculates the stress tensor for the standard isotropic model.
    sigma = lambda * tr(eps) * I + 2 * mu * eps
    """
    sigma = lamb * eps.trace() * identity() + 2 * mu * eps
    return sigma


def tangent(lamb, mu):
    """
    Calculates the tangent stiffness tensor for the standard isotropic model.
    C = lambda * (I otimes I) + 2 * mu * I_sym
    """
    I = identity()
    # Using np.einsum for a more concise representation
    C = lamb * np.einsum('ij,kl->ijkl', I, I) + \
        mu * (np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))
    return C


# Miehe Model (Spectral Decomposition)
# ------------------------------------
def energy_Miehe(epsilon, lamb, mu):
    """
    Calculates the positive and negative parts of strain energy using Miehe's model.
    psi_+ = 0.5 * lambda * <tr(eps)>+^2 + mu * tr(eps_+^2)
    psi_- = 0.5 * lambda * <tr(eps)>-^2 + mu * tr(eps_-^2)
    """
    epsP, epsN = Spectral(epsilon)
    trP, trN = bracket(epsilon.trace())

    psiP = 0.5 * lamb * trP * trP + mu * contract(epsP, epsP)
    psiN = 0.5 * lamb * trN * trN + mu * contract(epsN, epsN)
    return psiP, psiN


def stress_Miehe(epsilon, lamb, mu):
    """
    Calculates the positive and negative parts of the stress tensor using Miehe's model.
    sigma_+ = lambda * <tr(eps)>+ * I + 2 * mu * eps_+
    sigma_- = lambda * <tr(eps)>- * I + 2 * mu * eps_-
    """
    epsP, epsN = Spectral(epsilon)
    trP, trN = bracket(epsilon.trace())

    sigmaP = lamb * trP * identity() + 2 * mu * epsP
    sigmaN = lamb * trN * identity() + 2 * mu * epsN
    return sigmaP, sigmaN


def tangent_Miehe(epsilon, lamb, mu):
    """
    Calculates the positive and negative parts of the tangent tensor using Miehe's model.
    C_+ = lambda * H(tr(eps)) * (I otimes I) + 2 * mu * P_+
    C_- = lambda * H(-tr(eps)) * (I otimes I) + 2 * mu * P_-
    """
    PP, PN = Projection_Spectral(epsilon)
    I = identity()
    J = np.einsum('ij,kl->ijkl', I, I)

    Cpos = lamb * Heaviside(epsilon.trace()) * J + 2 * mu * PP
    Cneg = lamb * Heaviside(-epsilon.trace()) * J + 2 * mu * PN
    return Cpos, Cneg


# Amor Model (Volumetric-Deviatoric Decomposition)
# ------------------------------------------------
def energy_Amor(epsilon, lamb, mu):
    """
    Calculates the positive and negative parts of strain energy using Amor's model.
    psi_+ = 0.5 * k0 * <tr(eps)>+^2 + mu * tr(eps_dev^2)
    psi_- = 0.5 * k0 * <tr(eps)>-^2
    """
    k0 = lamb + 2.0 / 3.0 * mu
    eD, _ = VolDev(epsilon)
    trP, trN = bracket(epsilon.trace())

    psiP = 0.5 * k0 * trP * trP + mu * contract(eD, eD)
    psiN = 0.5 * k0 * trN * trN
    return psiP, psiN


def stress_Amor(epsilon, lamb, mu):
    """
    Calculates the positive and negative parts of the stress tensor using Amor's model.
    sigma_+ = k0 * <tr(eps)>+ * I + 2 * mu * eps_dev
    sigma_- = k0 * <tr(eps)>- * I
    """
    k0 = lamb + 2.0 / 3.0 * mu
    eD, _ = VolDev(epsilon)
    trP, trN = bracket(epsilon.trace())

    sigmaP = k0 * trP * identity() + 2 * mu * eD
    sigmaN = k0 * trN * identity()
    return sigmaP, sigmaN


def tangent_Amor(epsilon, lamb, mu):
    """
    Calculates the positive and negative parts of the tangent tensor using Amor's model.
    C_+ = k0 * H(tr(eps)) * (I otimes I) + 2 * mu * P_dev
    C_- = k0 * H(-tr(eps)) * (I otimes I)
    """
    k0 = lamb + 2.0 / 3.0 * mu
    Pd, _ = Projection_VolDev()
    I = identity()
    J = np.einsum('ij,kl->ijkl', I, I)

    Cpos = k0 * Heaviside(epsilon.trace()) * J + 2 * mu * Pd
    Cneg = k0 * Heaviside(-epsilon.trace()) * J
    return Cpos, Cneg


# %%
# Now we test the energy decompositions and compare the models.
# We define a sample strain tensor and material parameters.
print("\n\n--- Energy Decomposition Model Checks ---")

# Define a sample non-symmetric tensor
eps_nonsym = np.array([[2.0, 8.0, 5.0],
                       [9.0, 3.0, -1.0],
                       [5.0, -1.0, -7.0]])

# Symmetrize the tensor, as the theory applies to symmetric strain tensors
eps = (eps_nonsym + eps_nonsym.T) / 2
print("\nSymmetric Strain Tensor (eps):\n", eps)

# Material parameters (Lame parameters)
lamb = 1.0
mu = 1.0

# --- Reference Model Calculations ---
print("\n--- Reference (Standard Isotropic) Model ---")
psi_ref = energy(eps, lamb, mu)
sigma_ref = stress(eps, lamb, mu)
C_ref = tangent(lamb, mu)

print(f"Energy (psi): {psi_ref:.4f}")
print("Stress (sigma):\n", sigma_ref)

# --- Miehe Model Calculations ---
print("\n--- Miehe (Spectral) Decomposition Model ---")
psi_P_miehe, psi_N_miehe = energy_Miehe(eps, lamb, mu)
sigma_P_miehe, sigma_N_miehe = stress_Miehe(eps, lamb, mu)
C_P_miehe, C_N_miehe = tangent_Miehe(eps, lamb, mu)

print(f"Positive Energy (psi+): {psi_P_miehe:.4f}")
print(f"Negative Energy (psi-): {psi_N_miehe:.4f}")
print(f"Total Energy (psi+ + psi-): {psi_P_miehe + psi_N_miehe:.4f}")
print("Positive Stress (sigma+):\n", sigma_P_miehe)
print("Negative Stress (sigma-):\n", sigma_N_miehe)
print("Total Stress (sigma+ + sigma-):\n", sigma_P_miehe + sigma_N_miehe)

# Check consistency
print("\nIs Miehe total energy equal to reference energy?", np.allclose(psi_ref, psi_P_miehe + psi_N_miehe))
print("Is Miehe total stress equal to reference stress?", np.allclose(sigma_ref, sigma_P_miehe + sigma_N_miehe))
# Check tangent consistency: 0.5 * eps : C : eps
energy_from_tangent_miehe = 0.5 * np.einsum('ij,ijkl,kl', eps, C_P_miehe + C_N_miehe, eps)
print("Is Miehe energy from tangent consistent?", np.allclose(psi_ref, energy_from_tangent_miehe))


# --- Amor Model Calculations ---
print("\n--- Amor (Volumetric-Deviatoric) Decomposition Model ---")
psi_P_amor, psi_N_amor = energy_Amor(eps, lamb, mu)
sigma_P_amor, sigma_N_amor = stress_Amor(eps, lamb, mu)
C_P_amor, C_N_amor = tangent_Amor(eps, lamb, mu)

print(f"Positive Energy (psi+): {psi_P_amor:.4f}")
print(f"Negative Energy (psi-): {psi_N_amor:.4f}")
print(f"Total Energy (psi+ + psi-): {psi_P_amor + psi_N_amor:.4f}")
print("Positive Stress (sigma+):\n", sigma_P_amor)
print("Negative Stress (sigma-):\n", sigma_N_amor)
print("Total Stress (sigma+ + sigma-):\n", sigma_P_amor + sigma_N_amor)

# Check consistency
print("\nIs Amor total energy equal to reference energy?", np.allclose(psi_ref, psi_P_amor + psi_N_amor))
print("Is Amor total stress equal to reference stress?", np.allclose(sigma_ref, sigma_P_amor + sigma_N_amor))
# Check tangent consistency: 0.5 * eps : C : eps
energy_from_tangent_amor = 0.5 * np.einsum('ij,ijkl,kl', eps, C_P_amor + C_N_amor, eps)
print("Is Amor energy from tangent consistent?", np.allclose(psi_ref, energy_from_tangent_amor))

