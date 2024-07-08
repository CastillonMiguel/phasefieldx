"""
Tensor Decomposition
====================

This module provides functions for decomposing tensors into their deviatoric, volumetric, positive spectral, and negative spectral parts. These decompositions are fundamental in continuum mechanics and material science for understanding the behavior of materials under various stress and strain conditions.

"""

import ufl
from phasefieldx.Math.functions import macaulay_bracket_positive, macaulay_bracket_negative
from phasefieldx.Math.invariants import eigenstate2, eigenstate3

def deviatoric_part(T):
    """
    Computes the deviatoric part of a given tensor.

    Parameters
    ----------
    T: Tensor
        The input tensor for which the deviatoric part needs to be computed.

    Returns
    -------
    Tensor
        The deviatoric part of the input tensor, computed as the tensor minus
        one-third of its trace times the identity tensor.
    
    Notes
    -----
    The deviatoric part of a tensor is defined as the part of the tensor that
    remains after subtracting the isotropic part. In 3D, for a second-order tensor
    T, the deviatoric part D is given by $D = T - (1/3)  tr(T)  I$, where tr(T) is the trace of T and I is the identity tensor.
    """
    return ufl.dev(T)


def volumetric_part(T):
    """
    Computes the volumetric part of a given tensor.

    Parameters
    ----------
    T: Tensor
        The input tensor for which the volumetric part needs to be computed.

    Returns
    -------
    Tensor
        The volumetric part of the input tensor, computed as the tensor minus
        its deviatoric part.

    Notes
    -----
    The volumetric part of a tensor is defined as the isotropic part of the tensor,
    i.e., the part that is invariant under rotations. In 3D, for a second-order tensor
    T, the volumetric part V is given by $V = T - dev(T)$ where dev(T) is the deviatoric part of T.
    """
    return T - ufl.dev(T)


def spectral_positive_part(T):
    r"""
    Computes the spectral possitive part of a given tensor.

    Parameters
    ----------
    T: Tensor
        The input tensor for which the spectral possitive part needs to be computed.

    Returns
    -------
    Tensor
        The spectral possitive part of the input tensor.

    Notes
    -----
    The tensor is split into its positive and negative parts through spectral decomposition, $\boldsymbol \epsilon=\sum_{i=1}^{\alpha} \epsilon^i \boldsymbol n^i \otimes \boldsymbol n^i$ where $\epsilon^i$ are the principal strains, and $\boldsymbol n^i$ are the principal strains directions. The positive and negative parts of the tensor $\boldsymbol \epsilon    = \boldsymbol \epsilon^+ + \boldsymbol \epsilon^-$,  are defined as:

    * $\boldsymbol \epsilon^+: = \sum_{i=1}^{\alpha} \langle \epsilon^i \rangle^+ \boldsymbol n^i \otimes \boldsymbol n^i$,
    * $\boldsymbol \epsilon^-: = \sum_{i=1}^{\alpha} \langle \epsilon^i \rangle^- \boldsymbol n^i \otimes \boldsymbol n^i$.
  
    In which $\langle\rangle^{\pm}$ are the bracket operators. $\langle x \rangle^+:=\frac{x+|x|}{2}$, and $\langle x \rangle^-:=\frac{x-|x|}{2}$.
 
    """
    if ufl.shape(T) == (2, 2):
        eig, eig_vec = eigenstate2(T)
        T_p  = macaulay_bracket_positive(eig[0]) * eig_vec[0]
        T_p += macaulay_bracket_positive(eig[1]) * eig_vec[1]
    else:
        eig, eig_vec = eigenstate3(T)
        T_p  = macaulay_bracket_positive(eig[0]) * eig_vec[0]
        T_p += macaulay_bracket_positive(eig[1]) * eig_vec[1]
        T_p += macaulay_bracket_positive(eig[2]) * eig_vec[2]
 
    return T_p

def spectral_negative_part(T):
    r"""
    Computes the spectral negative part of a given tensor.

    Parameters
    ----------
    T: Tensor
        The input tensor for which the spectral negative part needs to be computed.

    Returns
    -------
    Tensor
        The spectral negative part of the input tensor.

    Notes
    -----
    The tensor is split into its positive and negative parts through spectral decomposition, $\boldsymbol \epsilon=\sum_{i=1}^{\alpha} \epsilon^i \boldsymbol n^i \otimes \boldsymbol n^i$ where $\epsilon^i$ are the principal strains, and $\boldsymbol n^i$ are the principal strains directions. The positive and negative parts of the tensor $\boldsymbol \epsilon    = \boldsymbol \epsilon^+ + \boldsymbol \epsilon^-$,  are defined as:

    * $\boldsymbol \epsilon^+: = \sum_{i=1}^{\alpha} \langle \epsilon^i \rangle^+ \boldsymbol n^i \otimes \boldsymbol n^i$,
    * $\boldsymbol \epsilon^-: = \sum_{i=1}^{\alpha} \langle \epsilon^i \rangle^- \boldsymbol n^i \otimes \boldsymbol n^i$.

    In which $\langle\rangle^{\pm}$ are the bracket operators. $\langle x \rangle^+:=\frac{x+|x|}{2}$ and $\langle x \rangle^-:=\frac{x-|x|}{2}$.

    """
    if ufl.shape(T) == (2, 2):
        eig, eig_vec = eigenstate2(T)
        T_p  = macaulay_bracket_negative(eig[0]) * eig_vec[0]
        T_p += macaulay_bracket_negative(eig[1]) * eig_vec[1]
    else:
        eig, eig_vec = eigenstate3(T)
        T_p  = macaulay_bracket_negative(eig[0]) * eig_vec[0]
        T_p += macaulay_bracket_negative(eig[1]) * eig_vec[1]
        T_p += macaulay_bracket_negative(eig[2]) * eig_vec[2]
    return T_p
