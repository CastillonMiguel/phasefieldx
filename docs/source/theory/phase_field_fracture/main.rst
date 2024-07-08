.. _theory_phase_field_fracture:

Phase-field fracture
====================

.. note::
    Please view the examples related to phase-field fracture :ref:`ref_examples_phase_field_fracture`.


Variational approach
--------------------
The solution of a phase-field fracture simulation, in the absence of external forces, is given by the minimization of the functional:

.. math::

   V(u,\phi)= E(u,\phi) + W(\phi)

in which $E(u,\phi)$ is the constitutive free energy functional, and $W(\phi)$ is the dissipation functional due to fracture. Reference .


Constitutive free energy functional :math:`E(u,\phi)`
-----------------------------------------------------
Regarding energy degradation, there are two main descriptions of the constitutive free energy functional.
The isotropic model degrades all the energy; in the other case, the anisotropic model only degrades the energy related to tension efforts, trying to account that cracks only are generated under tension efforts.

In the Isotropic constitutive free energy functional,

.. math::

   E(u,\phi)=\int \left[g(\phi)+k\right] \psi(\boldsymbol \epsilon(\boldsymbol u)) dV,

the degradation function :math:`g(\phi)` degrades all the energy :math:`\psi(\boldsymbol \epsilon(\boldsymbol u))`. In the case of the anisotropic constitutive free energy functional,

.. math::

   E(u,\phi)=\int \left(\left[g(\phi)+k\right] \psi_a(\boldsymbol \epsilon(\boldsymbol u)) + \psi_b(\boldsymbol \epsilon(\boldsymbol u))\right) dV,

the energy $\psi(\boldsymbol \epsilon(\boldsymbol u))$ is split in two components, $\psi=\psi_a+\psi_b$, in which the component :math:`\psi_a` attains to the energy relative to tensions, so the degradation function only applied to it, avoiding the creation of crack due to compression effort.

There are several methods to carry out the energy slip. PhaseFieldX code implements the Spectral [Miehe]_ and volumetric-deviatoric [Amor]_ decomposition.

.. note::

   * Spectral decomposition 
      The strain tensor is split into its positive and negative parts through spectral decomposition, :math:`\boldsymbol \epsilon=\sum_{i=1}^{\alpha} \epsilon^i \boldsymbol n^i \otimes \boldsymbol n^i` where :math:`{\epsilon^i}` are the principal strains, and :math:`{\boldsymbol n^i}` are the principal strains directions. the positive and negative parts of the tensor :math:`\boldsymbol \epsilon    = \boldsymbol \epsilon^+ + \boldsymbol \epsilon^-`,  are defined as:
   
      :math:`\boldsymbol \epsilon^+: = \sum_{i=1}^{\alpha} \langle \epsilon^i \rangle^+ \boldsymbol n^i \otimes \boldsymbol n^i`,
      :math:`\boldsymbol \epsilon^-: = \sum_{i=1}^{\alpha} \langle \epsilon^i \rangle^- \boldsymbol n^i \otimes \boldsymbol n^i`

      In which :math:`\langle\rangle^{\pm}` are the bracket operators:

      :math:`\langle x \rangle^+:=\frac{x+|x|}{2}`,  
      :math:`\langle x \rangle^-:=\frac{x-|x|}{2}`.

      With the previous definitions, the isotropic reference energy function is broken down into its "a" and "b" parts, $\psi(\boldsymbol \epsilon) = \psi_a(\boldsymbol \epsilon) + \psi_b(\boldsymbol \epsilon)$; with components:

      :math:`\psi_a(\boldsymbol \epsilon)=\frac{1}{2}\lambda{\langle tr(\boldsymbol\epsilon)\rangle^+}^2+\mu tr((\boldsymbol\epsilon^+)^2)`, 

      and

      :math:`\psi_b(\boldsymbol \epsilon)=\frac{1}{2}\lambda{\langle tr(\boldsymbol\epsilon)\rangle^-}^2+\mu tr((\boldsymbol\epsilon^-)^2)`


   * Volumetric-Deviatoric
      In this case, Amor represents the strain tensor in its positive (:math:`\boldsymbol \epsilon^D` deviatoric) and negative (:math:`\boldsymbol \epsilon^S` volumetric/spherical) parts.

      :math:`\boldsymbol \epsilon  =\boldsymbol \epsilon^S + \boldsymbol\epsilon^D`, 
      :math:`\boldsymbol \epsilon^S =\frac{1}{m}tr(\boldsymbol \epsilon)\boldsymbol I`, 
      :math:`\boldsymbol\epsilon^D  =\boldsymbol \epsilon-\frac{1}{m}tr(\boldsymbol \epsilon)\boldsymbol I`.

      Calculates the positive and negative parts of the strain energy by:

      :math:`\psi_a(\boldsymbol \epsilon) =\frac{1}{2} \kappa_0 {\langle tr(\boldsymbol \epsilon)\rangle^+}^2+\mu tr((\boldsymbol \epsilon^D)^2)`,

      and

      :math:`\psi_b(\boldsymbol \epsilon) =\frac{1}{2} \kappa_0 {\langle tr(\boldsymbol \epsilon)\rangle^+}^2`
      
      where :math:`\kappa_0=\lambda+\frac{2}{m}\mu` is the bulk modulus, and $m$ is the dimension.


In the case of the degradation functions, the most common one is the quadratic

:math:`g(\phi)=(1-\phi)^2`,

other types of degradation functions:

.. note::

   * Borden [Jian]_     :math:`g(\phi)=(3-s)(1-\phi)^2-(2-s)(1-\phi)^3`, (implemented with s=1)
   * Alessi [Jian]_     :math:`g(\phi)=\frac{(1-\phi)^2}{(1+(k-1)Q)}`, :math:`Q=1-(1-\phi)^2` (implemented with k=100)
   * Sargado [Sargado]_ Exponential-type degradation function  
   * limit

All the degradation functions have the properties $g(0)=1$, $g(1)=0$ and $g'(1)=0$. The fisrt two properties attains the unbroken and the fully broken situations respectivily.  The last ones ensures that the energetic fracture force converges to a finite value if $\phi=0$.

The parameter $k$ attains for numerical purposes.



Dissipation functional due to fracture :math:`W(\phi)`
------------------------------------------------------

Please check the :ref:`theory_phase_field`
The dissipation functional due to fracture

.. math::

   W(\phi)= G_c \int \gamma(\phi, \nabla \phi) dV,


have account of the dissipated energy due to fracture, in which :math:`G_c` is the critical energy release rate and

.. math::
   \gamma(\phi, \nabla \phi)=\frac{1}{2l}\phi^2+\frac{l}{2}|\nabla \phi|^2,

is the crack surface density function, in which :math:`l` is the length scale parameter.


Fatigue
-------

.. note::
   The major part of the fatigue model is taken from the paper by Lorenzis [Lorenzis_fatigue]_

It is possible to consider fatigue phenomena, by modifiying the critical energy release depending of the repeated applied loads.

So the dissipation functional takes this form with the new term:

.. math::

   W(\phi)= f(\bar{\alpha(t)}) G_c \int \gamma(\phi, \nabla \phi) dV,


Cumulated history variable :math:`\bar{\alpha}(t)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A cumulation of any scalar quantity which can exhaustively **describe the fatigue history** experienced by the material fulfilling the property. Is a history variable that can be cumulated using any quantity :math:`\alpha` able to account for the fatigue effects experienced by the material.

.. note::

   * Mean load independent: for materials whose fatigue life is not affected by the mean load of a cycle.

   .. math::
      \bar{\alpha} (\boldsymbol x, t) = \int_0^t H(\alpha \dot{\alpha}) |\dot{\alpha}| d \tau

   * Mean load dependent: the model can be enriched by introducing a history variable that weighs differently the rate of the cumulated variable depending on the load level achieved as

      .. math::
         \bar{\alpha} (\boldsymbol x, t) = \frac{1}{\alpha_n} \int_0^t H(\alpha \dot{\alpha}) \alpha \dot{\alpha} d \tau

   where :math:`\alpha_n` is a normalization parameter needed to achieve dimensional consistency.

Definition: :math:`\alpha` 
^^^^^^^^^^^^^^^^^^^^^^^^^^
Account for the active part of the elastic strain energy density

.. math::
   \alpha = (1-\phi)^2 \psi(\boldsymbol \epsilon(u))

Also, the fatigue effects are cumulated only during the loading phase.


Fatigue degradation function :math:`f(\bar{\alpha(t)})`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The fatigue degradation function :math:`f(\bar{\alpha(t)})` **describes how** fatigue effectively reduces the fracture toughness of the material. The following two fatigue degradation functions are considered here

.. note::
   
   a) asymptotic

   .. math::
      f(\bar{\alpha(t)})=
      \begin{cases}
      1, \bar{\alpha}(t)  \leq \alpha_T \\
      \left(\frac{2}{\bar{\alpha}(t)+\alpha_T}\right)^2, \bar{\alpha}(t) > \alpha_T  \\
      \end{cases}


   b) logarithmic

   .. math::
      f(\bar{\alpha(t)})=
      \begin{cases}
      1, \bar{\alpha}(t)  \leq \alpha_T \\
      \left[1 - k \log\left( \frac{\bar{\alpha}(t) }{\alpha_T}\right) \right]^2,  \alpha_T  \leq \bar{\alpha}(t) \leq \alpha_T 10^{1/k} \\
      0, \bar{\alpha}(t) > \alpha_T 10^{1/k}
      \end{cases}



   
   where :math:`k` is a material parameter, and :math:`\alpha_T` is a threshold controlling when the fatigue effect is triggered.

   


.. [Sargado] High-accuracy phase-field models for brittle fracture based on a new family of degradation functions. https://doi.org/10.1016/j.jmps.2017.10.015
.. [Jian] Phase-field modelling of fracture. Jian-Ying Wu , Vinh Phu Nguyen , Chi Thanh Nguyen , Danas Sutula , Sina Sinaie ,and Stephane Bordas
.. [Miehe] A phase field model for rate-independent crack propagation: Robust algorithmic implementation based on operator splits, https://doi.org/10.1016/j.cma.2010.04.011.
.. [Amor] Regularized formulation of the variational brittle fracture with unilateral contact: Numerical experiments, https://doi.org/10.1016/j.jmps.2009.04.011.
.. [Lorenzis_fatigue] A framework to model the fatigue behavior of brittle materials based on a variational phase-field approach. P. Carraraa, M. Ambati, R. Alessi, L. De Lorenzis. https://doi.org/10.1016/j.cma.2019.112731

Implementation
--------------