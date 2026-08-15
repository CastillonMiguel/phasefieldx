Theory
******

Welcome to the Theory section of the documentation. Here, we introduce the fundamental concepts and mathematical frameworks that underpin the models implemented in PhaseFieldX.

The following video presents the motivation behind phase-field methods and provides an overview of some of their most common applications, including fracture mechanics, solidification and melting processes, and topology optimization.

.. youtube:: q06rGJNSjAw
  :align: center

The main focus of the package is the **phase-field fracture problem**, which combines concepts from elasticity theory, variational methods, and phase-field modeling to describe crack initiation and propagation in a diffuse manner.

However, phase-field fracture is not an isolated theory. It is built upon several well-established ingredients that can be studied independently. Understanding these fundamental components is essential before approaching the complete coupled formulation. A clear understanding of the underlying theories facilitates the analysis of the governing equations, the implementation of numerical algorithms, and the identification of potential modeling or coding errors.

For this reason, this documentation is divided into two parts:

1. The **Problems of Application**, which represent the final coupled formulations of interest.

2. The **Fundamental Building Blocks**, which introduce the individual theories and concepts from which the phase-field models frameworks are constructed.


Problems of Application
=======================

1. **Phase-Field Fracture/Fatigue** Phase-Field Fracture/Fatigue (:ref:`theory_phase_field_fracture`, :ref:`theory_phase_field_fatigue`): Integrates continuum elasticity with phase-field regularization to model the initiation, propagation, branching, coalescence, and fatigue-driven evolution of cracks without the need for explicit crack tracking.

2. **Phase-Field Topology Optimization** *(Upcoming)*

3. **Phase-Field Solidification** *(Upcoming)*


Fundamental Building Blocks
===========================

Because phase-field applications like fracture or topology optimization combine several physical and mathematical ingredients, it is helpful to study each component independently. A clear understanding of the underlying building blocks simplifies analyzing the governing equations, implementing numerical algorithms, and detecting formulation or coding errors.

1. **Phase-Field Problem** (:ref:`theory_phase_field`): Introduces the variational regularization of discontinuities and the crack surface density functional (CSDF).

2. **Elasticity Problem** (:ref:`theory_elasticity`): Covers linear elastic stress analysis, strain measures, and energy densities under mechanical deformation.

3. **Allen-Cahn Equation** (:ref:`theory_allen_cahn`): Introduces phase-field order-parameter dynamics and gradient-energy functionals.

Through a systematic exploration of these interconnected domains, this documentation provides in-depth explanations and practical guidance for understanding and applying phase-field theoretical principles.


.. toctree::
   :hidden:

   phase_field_fracture/main
   phase_field_fatigue/main
   elasticity/main
   phase_field/main
   energy_pff/main
   crack_measurement/main
   Allen_Cahn/main
   errors/main
