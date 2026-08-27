.. _theory_phase_field_fatigue:

PHASE-FIELD FATIGUE
===================

.. note::
   The fatigue modeling approach described here is primarily based on the framework presented by Carrara et al. :footcite:t:`phase_field_Carrara2020`, which extends the variational phase-field method to account for fatigue effects in brittle materials.

It is possible to consider fatigue phenomena, by modifiying the critical energy release depending of the repeated applied loads.

So the dissipation functional takes this form with the new term:

.. math::

   \Gamma(\phi)= f(\bar{\alpha(t)}) G_c \int \gamma(\phi, \nabla \phi) dV,


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

.. footbibliography::