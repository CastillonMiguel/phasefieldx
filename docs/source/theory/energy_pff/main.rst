.. _theory_energy_pff:

Energy-Controlled PFF Solvers
=============================

This section describes energy-controlled phase-field fracture (PFF) solvers presented in :footcite:t:`Castillon2025_arxiv`. 

.. note::
    Please view the examples related to phase-field fracture :ref:`ref_examples_phase_field_fracture_energy_controlled`.

Traditional PFF strategies, such as displacement or force control, often fail to capture the complete equilibrium path during instabilities like snap-back or snap-through behavior. Displacement-controlled solvers may skip stable equilibrium points, while force-controlled schemes may not trace the path entirely.

To address these issues, :footcite:t:`Castillon2025_arxiv` proposes two energy-controlled schemes that robustly trace the equilibrium path during crack propagation by using a monotonically increasing energy-like parameter. These schemes ensure stable progress through instabilities and accurately characterize the system's response throughout the fracture process.

Variational scheme
------------------

The first approach is a variational scheme driven by an energetic control function, :math:`\tau(t)`. This is achieved by introducing a constraint that equates :math:`\tau(t)` to a weighted combination of crack surface energy and external work. A Lagrange multiplier, :math:`\lambda`, enforces this constraint:

.. math::
   c_1 \int_\Omega \left( \frac{1}{2l} \phi^2 + \frac{l}{2} |\nabla \phi|^2 \right) \,\mathrm{d}\Omega 
   + c_2 \int_{\partial_N\Omega} \boldsymbol{t} \cdot \boldsymbol{u} \,\mathrm{d}S = \tau(t)

Here, :math:`c_1` and :math:`c_2` are numerical parameters for solver convergence and dimensional consistency.

The augmented functional for this constrained system, :math:`V`, is defined as:

.. math::
   \begin{aligned}
   V(\boldsymbol{u}, \phi, \lambda) &= \int_\Omega g(\phi)\psi(\boldsymbol{\epsilon}(\boldsymbol{u})) \,\mathrm{d}\Omega 
   + G_c \int_\Omega \left( \frac{1}{2l} \phi^2 + \frac{l}{2} |\nabla \phi|^2 \right) \,\mathrm{d}\Omega 
   - \int_{\partial_N\Omega} \boldsymbol{t} \cdot \boldsymbol{u} \,\mathrm{d}S \\
   &\quad + \lambda \left[ c_1 \int_\Omega \left( \frac{1}{2l} \phi^2 + \frac{l}{2} |\nabla \phi|^2 \right) \,\mathrm{d}\Omega 
   + c_2 \int_{\partial_N\Omega} \boldsymbol{t} \cdot \boldsymbol{u} \,\mathrm{d}S - \tau(t) \right]
   \end{aligned}

The equilibrium equations are obtained by enforcing stationarity, :math:`\delta V = 0`. The resulting weak form is:

.. math::
   \int_\Omega g(\phi)\boldsymbol{\sigma}(\boldsymbol{\epsilon}(\boldsymbol{u})):\boldsymbol{\epsilon}(\delta \boldsymbol{u}) \,\mathrm{d}\Omega 
   - (1 - \lambda c_2) \int_{\partial_N\Omega} \boldsymbol{t} \cdot \delta\boldsymbol{u} \,\mathrm{d}S = 0

.. math::
   \int_\Omega g'(\phi) \delta\phi \, \psi(\boldsymbol{\epsilon}(\boldsymbol{u})) \,\mathrm{d}\Omega 
   + (G_c + \lambda c_1) \int_\Omega \left( \frac{1}{l} \phi \delta\phi + l \nabla\phi \cdot \nabla \delta \phi \right) \,\mathrm{d}\Omega = 0

.. math::
   \delta \lambda \left[ c_1 \int_\Omega \left( \frac{1}{2l} \phi^2 + \frac{l}{2} |\nabla \phi|^2 \right) \,\mathrm{d}\Omega 
   +  c_2 \int_{\partial_N\Omega} \boldsymbol{t} \cdot \boldsymbol{u} \,\mathrm{d}S - \tau(t) \right] = 0

From the solution :math:`(\boldsymbol{u}, \phi, \lambda)`, the Lagrange multiplier provides key insights. A correction factor :math:`\alpha` allows recovering the physical response for a constant :math:`G_c`:

.. math::
   \alpha = \sqrt{\frac{G_c}{G_c^{\text{eff}}}} = \frac{1}{\sqrt{1 + \frac{\lambda c_1}{G_c}}}

.. math::
   P_{G_c} = \alpha \cdot P_{\text{variational}}

.. math::
   u_{G_c} = \alpha \cdot u_{\text{variational}}

.. math::
   \psi_{G_c} = \alpha^2 \cdot \psi_{\text{variational}}


Non-variational scheme
----------------------

Alternatively, a non-variational energy-controlled scheme computes the physical equilibrium path for an effective constant :math:`G_c` without post-processing. The governing equations are:

.. math::
   \int_\Omega g(\phi)\boldsymbol{\sigma}(\boldsymbol{\epsilon}(\boldsymbol{u})):\boldsymbol{\epsilon}(\delta \boldsymbol{u}) \,\mathrm{d}\Omega 
   - (1 - \lambda c_2) \int_{\partial_N\Omega} \boldsymbol{t} \cdot \delta\boldsymbol{u} \,\mathrm{d}S = 0

.. math::
   \int_\Omega g'(\phi) \delta\phi \, \psi(\boldsymbol{\epsilon}(\boldsymbol{u})) \,\mathrm{d}\Omega 
   + G_c \int_\Omega \left( \frac{1}{l} \phi \delta\phi + l \nabla\phi \cdot \nabla \delta \phi \right) \,\mathrm{d}\Omega = 0

.. math::
   \delta \lambda \left[ c_1 \int_\Omega \left( \frac{1}{2l} \phi^2 + \frac{l}{2} |\nabla \phi|^2 \right) \,\mathrm{d}\Omega 
   +  c_2 \int_{\partial_N\Omega} \boldsymbol{t} \cdot \boldsymbol{u} \,\mathrm{d}S - \tau(t) \right] = 0

The scalar :math:`\lambda` in this scheme is not strictly a Lagrange multiplier but serves a similar role, ensuring that the material toughness :math:`G_c` remains constant and allowing the solver to trace the physical force-displacement curve.

Both energy-controlled solvers trace the same physical equilibrium path. The choice between variational and non-variational schemes depends on computational preference, while numerical parameters :math:`c_1` and :math:`c_2` are used solely for convergence and do not affect physical results.

.. footbibliography::
   