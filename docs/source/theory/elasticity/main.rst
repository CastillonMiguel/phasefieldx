.. _theory_elasticity:

Elasticity
==========

Before studying phase-field fracture simulations—which couple displacement (elasticity) and phase-field evolution—it is recommended to understand each problem separately. Here, we analyze the linear elasticity problem independently.

.. note::
    Please view the examples related to elasticity in :ref:`ref_examples_elasticity`.


Variational approach
--------------------

The elasticity problem is stationary and derives from a potential energy minimization principle. The elasticity solution is given by the minimizer of the total potential energy functional:

.. math::
   :label: eq_elasticity_energy

   E[\boldsymbol u] = \int_\Omega \psi(\boldsymbol{\epsilon}(\boldsymbol u)) \, \mathrm{d}\boldsymbol{x} - E_{\text{ext}}[\boldsymbol u],

where $E_{\text{ext}}[\boldsymbol u]$ is the potential of the external forces:

.. math::
   :label: eq_elasticity_external_energy

   E_{\text{ext}}[\boldsymbol u] = \int_\Omega \boldsymbol f \cdot \boldsymbol u \, \mathrm{d}\boldsymbol{x} + \int_{\partial \Omega_t} \boldsymbol t \cdot \boldsymbol u \, \mathrm{d}S.

The equilibrium equations are recovered from the stationarity condition $\delta E = 0$.

For a linear elastic isotropic material, the strain energy density function is:

.. math::
   :label: eq_elasticity_strain_energy

   \psi(\boldsymbol{\epsilon}) = \frac{1}{2} \lambda \, \text{tr}^2(\boldsymbol{\epsilon}) + \mu \, \text{tr}(\boldsymbol{\epsilon}^2),

where $\lambda$ and $\mu$ are the Lamé constants, $\boldsymbol{\epsilon}$ is the symmetric strain tensor:

.. math::
   :label: eq_elasticity_strain_tensor

   \boldsymbol{\epsilon}(\boldsymbol u) = \frac{1}{2} \left( \nabla \boldsymbol u + (\nabla \boldsymbol u)^T \right),

and the Cauchy stress tensor is given by:

.. math::
   :label: eq_elasticity_stress_tensor

   \boldsymbol{\sigma}(\boldsymbol{\epsilon}) = \frac{\partial \psi}{\partial \boldsymbol{\epsilon}} = \lambda \, \text{tr}(\boldsymbol{\epsilon})\boldsymbol{I} + 2 \mu \boldsymbol{\epsilon}.

Applying the Gateaux derivative to the total energy functional yields the weak form of the elastic equilibrium equation:

.. math::
   :label: eq_elasticity_weak_form

   \int_\Omega \boldsymbol{\sigma}(\boldsymbol{\epsilon}(\boldsymbol u)) : \boldsymbol{\epsilon}(\delta \boldsymbol u) \, \mathrm{d}\boldsymbol{x} - \int_\Omega \boldsymbol f \cdot \delta \boldsymbol u \, \mathrm{d}\boldsymbol{x} - \int_{\partial \Omega_t} \boldsymbol t \cdot \delta \boldsymbol u \, \mathrm{d}S = 0,

where $\delta \boldsymbol u$ denotes an admissible displacement variation.


Anisotropic formulations (Energy splits)
----------------------------------------

The elasticity solver is a fundamental component of the PhaseFieldX framework. In phase-field fracture modeling, capturing tension–compression asymmetry is crucial: materials typically fracture under tensile loading but remain undamaged under compressive or hydrostatic pressure.

To incorporate this behavior, anisotropic elasticity models decompose the strain energy density $\psi(\boldsymbol{\epsilon})$ into an **active** (tensile/crack-driving) part $\psi_a$ and an **inactive** (compressive/undegraded) part $\psi_b$:

.. math::
   :label: eq_elasticity_energy_split

   \psi(\boldsymbol{\epsilon}) = \psi_a(\boldsymbol{\epsilon}) + \psi_b(\boldsymbol{\epsilon}).

Taking derivatives with respect to the strain tensor yields the corresponding active and inactive stress tensors:

.. math::
   :label: eq_elasticity_stress_split

   \boldsymbol{\sigma}_a(\boldsymbol{\epsilon}) = \frac{\partial \psi_a}{\partial \boldsymbol{\epsilon}}, \quad
   \boldsymbol{\sigma}_b(\boldsymbol{\epsilon}) = \frac{\partial \psi_b}{\partial \boldsymbol{\epsilon}}, \quad \text{such that} \quad
   \boldsymbol{\sigma}(\boldsymbol{\epsilon}) = \boldsymbol{\sigma}_a(\boldsymbol{\epsilon}) + \boldsymbol{\sigma}_b(\boldsymbol{\epsilon}).

The weak form for the split elasticity problem is expressed as:

.. math::
   :label: eq_elasticity_anisotropic_weak_form

   \int_\Omega \left[ \boldsymbol{\sigma}_a(\boldsymbol{\epsilon}(\boldsymbol{u})) + \boldsymbol{\sigma}_b(\boldsymbol{\epsilon}(\boldsymbol{u})) \right] : \boldsymbol{\epsilon}(\delta \boldsymbol{u}) \, \mathrm{d}\boldsymbol{x} - \int_\Omega \boldsymbol f \cdot \delta \boldsymbol u \, \mathrm{d}\boldsymbol{x} - \int_{\partial \Omega_t} \boldsymbol t \cdot \delta \boldsymbol u \, \mathrm{d}S = 0.


Common Energy Decomposition Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PhaseFieldX supports two primary strain energy split formulations:

1. **Spectral Decomposition** :footcite:t:`phase_field_Miehe2010`:
   Based on the spectral decomposition of the strain tensor $\boldsymbol{\epsilon} = \sum_{i=1}^3 \epsilon^i \boldsymbol{n}^i \otimes \boldsymbol{n}^i$:

   .. math::

      \psi_a^{\text{spectral}}(\boldsymbol{\epsilon}) &= \frac{1}{2}\lambda{\langle \text{tr}(\boldsymbol{\epsilon}) \rangle_+}^2 + \mu \text{tr}(\boldsymbol{\epsilon}_+^2), \\
      \psi_b^{\text{spectral}}(\boldsymbol{\epsilon}) &= \frac{1}{2}\lambda{\langle \text{tr}(\boldsymbol{\epsilon}) \rangle_-}^2 + \mu \text{tr}(\boldsymbol{\epsilon}_-^2),

   where $\boldsymbol{\epsilon}_\pm = \sum_{i=1}^3 \langle \epsilon^i \rangle_\pm \boldsymbol{n}^i \otimes \boldsymbol{n}^i$, and $\langle x \rangle_\pm = \frac{x \pm |x|}{2}$ are Macaulay brackets.

2. **Volumetric-Deviatoric Decomposition** :footcite:t:`phase_field_Amor2009`:
   Splits the strain tensor into volumetric $\boldsymbol{\epsilon}^S = \frac{1}{m}\text{tr}(\boldsymbol{\epsilon})\boldsymbol{I}$ and deviatoric $\boldsymbol{\epsilon}^D = \boldsymbol{\epsilon} - \boldsymbol{\epsilon}^S$ parts:

   .. math::

      \psi_a^{\text{vol-dev}}(\boldsymbol{\epsilon}) &= \frac{1}{2} \kappa {\langle \text{tr}(\boldsymbol{\epsilon}) \rangle_+}^2 + \mu \text{tr}({\boldsymbol{\epsilon}^D}^2), \\
      \psi_b^{\text{vol-dev}}(\boldsymbol{\epsilon}) &= \frac{1}{2} \kappa {\langle \text{tr}(\boldsymbol{\epsilon}) \rangle_-}^2,

   where $\kappa = \lambda + \frac{2}{m}\mu$ is the bulk modulus in spatial dimension $m$.

.. note::

   **Validation and Consistency Check:**
   Prior to coupling with damage, any anisotropic elasticity formulation must satisfy two consistency properties in the un-damaged state ($\phi = 0$):

   1. **Energy conservation:** $\psi_a(\boldsymbol{\epsilon}) + \psi_b(\boldsymbol{\epsilon}) = \psi(\boldsymbol{\epsilon})$
   2. **Stress recovery:** $\boldsymbol{\sigma}_a(\boldsymbol{\epsilon}) + \boldsymbol{\sigma}_b(\boldsymbol{\epsilon}) = \boldsymbol{\sigma}(\boldsymbol{\epsilon})$

   Checking that the stress responses under pure tension, compression, and shear match the standard isotropic response provides a fundamental benchmark to verify the implementation of anisotropic strain energy splits.