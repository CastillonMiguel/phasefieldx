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


.. _theory_elasticity_dynamic:

Dynamic Problem
---------------
The potential energy :math:`E[\boldsymbol u]`, given by :eq:`eq_elasticity_energy`, describes the quasi-static elastic response. However, when dynamic problems are considered, we must also take the kinetic energy of the solid into account. Denoting the time derivative of $\boldsymbol u$ by $\dot{\boldsymbol u}$, and assuming a solid mass density $\rho$, the kinetic energy may be written as

.. math::
   :label: eq_elasticity_kinetic_energy

   E_{\text{kin}}[\dot{\boldsymbol u}] = \frac{1}{2}\int_\Omega \rho \, \dot{\boldsymbol u}\cdot\dot{\boldsymbol u} \, \mathrm{d}\boldsymbol{x}.

Combining :eq:`eq_elasticity_energy` and :eq:`eq_elasticity_kinetic_energy`, we can form the Lagrangian of the dynamic elasticity problem as

.. math::
   :label: eq_elasticity_lagrangian

   \mathcal{L}[\boldsymbol u, \dot{\boldsymbol u}] = E_{\text{kin}}[\dot{\boldsymbol u}] - E[\boldsymbol u]
   = \frac{1}{2}\int_\Omega \rho \, \dot{\boldsymbol u}\cdot\dot{\boldsymbol u} \, \mathrm{d}\boldsymbol{x}
   - \int_\Omega \psi(\boldsymbol{\epsilon}(\boldsymbol u)) \, \mathrm{d}\boldsymbol{x}
   + \int_\Omega \boldsymbol f \cdot \boldsymbol u \, \mathrm{d}\boldsymbol{x}
   + \int_{\partial \Omega_t} \boldsymbol t \cdot \boldsymbol u \, \mathrm{d}S.

The equations of motion are obtained from the stationary points of the action $\int_{t_0}^{t_1} \mathcal{L} \, \mathrm{d}t$ (Hamilton's principle), for variations $\delta \boldsymbol u$ that vanish at $t_0$ and $t_1$ and on the Dirichlet boundary $\partial\Omega_u$. Integrating by parts in time (the boundary terms vanish because $\delta \boldsymbol u(t_0) = \delta \boldsymbol u(t_1) = \boldsymbol 0$) gives the weak form: find $\boldsymbol u$ such that, at every time,

.. math::
   :label: eq_elasticity_dynamic_weak_form

   \int_\Omega \rho \, \ddot{\boldsymbol u} \cdot \delta \boldsymbol u \, \mathrm{d}\boldsymbol{x}
   + \int_\Omega \boldsymbol{\sigma}(\boldsymbol{\epsilon}(\boldsymbol u)) : \boldsymbol{\epsilon}(\delta \boldsymbol u) \, \mathrm{d}\boldsymbol{x}
   - \int_\Omega \boldsymbol f \cdot \delta \boldsymbol u \, \mathrm{d}\boldsymbol{x}
   - \int_{\partial \Omega_t} \boldsymbol t \cdot \delta \boldsymbol u \, \mathrm{d}S = 0,

where $\ddot{\boldsymbol u} = \partial^2 \boldsymbol u / \partial t^2$ is the acceleration. The inertial term is the only addition with respect to the static weak form :eq:`eq_elasticity_weak_form`.

For reference, integrating by parts in space and using the arbitrariness of $\delta \boldsymbol u$ recovers the strong form (balance of linear momentum):

.. math::
   :label: eq_elasticity_dynamic_strong

   \begin{aligned}
   \rho \, \ddot{\boldsymbol u} - \nabla \cdot \boldsymbol{\sigma}(\boldsymbol{\epsilon}(\boldsymbol u)) &= \boldsymbol f && \text{in } \Omega, \\
   \boldsymbol{\sigma}\, \boldsymbol n &= \boldsymbol t && \text{on } \partial\Omega_t, \\
   \boldsymbol u &= \bar{\boldsymbol u} && \text{on } \partial\Omega_u,
   \end{aligned}

completed with the initial conditions $\boldsymbol u(\boldsymbol x, 0) = \boldsymbol u_0$ and $\dot{\boldsymbol u}(\boldsymbol x, 0) = \boldsymbol v_0$.

After spatial discretization the weak form becomes the semi-discrete system $\boldsymbol M \ddot{\boldsymbol u} + \boldsymbol K \boldsymbol u = \boldsymbol F_{\text{ext}}$, which still has to be integrated in time.


.. _theory_elasticity_newmark:

Newmark time integration
~~~~~~~~~~~~~~~~~~~~~~~~

Let $\Delta t$ be the time step, and let $\boldsymbol u_n, \boldsymbol v_n, \boldsymbol a_n$ be the known displacement, velocity and acceleration at time $t_n$. The Newmark-:math:`\beta` scheme approximates the state at $t_{n+1} = t_n + \Delta t$ as

.. math::
   :label: eq_newmark_displacement

   \boldsymbol u_{n+1} = \boldsymbol u_n + \Delta t \, \boldsymbol v_n + \Delta t^2 \left[ \left(\tfrac{1}{2} - \beta\right) \boldsymbol a_n + \beta \, \boldsymbol a_{n+1} \right],

.. math::
   :label: eq_newmark_velocity

   \boldsymbol v_{n+1} = \boldsymbol v_n + \Delta t \left[ (1 - \gamma) \, \boldsymbol a_n + \gamma \, \boldsymbol a_{n+1} \right],

where $\beta$ and $\gamma$ are parameters controlling accuracy and stability. The unknown at each step is the displacement $\boldsymbol u_{n+1}$. Solving :eq:`eq_newmark_displacement` for the acceleration expresses it in terms of $\boldsymbol u_{n+1}$ and known quantities only:

.. math::
   :label: eq_newmark_acceleration

   \boldsymbol a_{n+1} = \frac{1}{\beta \, \Delta t^2} \left( \boldsymbol u_{n+1} - \boldsymbol u_n - \Delta t \, \boldsymbol v_n \right) - \left( \frac{1}{2\beta} - 1 \right) \boldsymbol a_n.

Substituting :eq:`eq_newmark_acceleration` in the inertial term of :eq:`eq_elasticity_dynamic_weak_form` (the equilibrium is enforced at $t_{n+1}$) yields a weak form that depends only on $\boldsymbol u_{n+1}$, which is solved like the static problem (with an additional mass-like contribution to the tangent stiffness). Once $\boldsymbol u_{n+1}$ is known, $\boldsymbol a_{n+1}$ and $\boldsymbol v_{n+1}$ are updated with :eq:`eq_newmark_acceleration` and :eq:`eq_newmark_velocity`, and the old fields are overwritten for the next step.

**Parameters.** The scheme is implicit and unconditionally stable for linear problems if

.. math::

   \gamma \geq \tfrac{1}{2}, \qquad \beta \geq \tfrac{1}{4}\left(\gamma + \tfrac{1}{2}\right)^2,

and it is second-order accurate only for $\gamma = 1/2$ (values $\gamma > 1/2$ add numerical damping but reduce the accuracy to first order). PhaseFieldX uses the *average (constant) acceleration* choice $\beta = 1/4$, $\gamma = 1/2$, which is unconditionally stable, second-order accurate and has no numerical dissipation.
