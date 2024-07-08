.. _theory_elasticity:

Elasticity
==========
Before studying the phase-field fracture simulation, which is a coupled problem of displacement (elasticity) and phase-field, it is recommended to understand each of the problems separately. In this case, the elasticity problem will be studied alone.

.. note::
    Please view the examples related to elasticity in :ref:`ref_examples_elasticity`.


Variational approach
--------------------

For this case the problem that needs to be solve is stationary and derives from an optimization principle.

The elasticity solution is given by the minimizer of the following energy functional

.. math::
	E[\boldsymbol u] =  \int_\Omega \psi(\boldsymbol \epsilon (\boldsymbol u)) dV - E_{ext}[\boldsymbol u]

where $E_{ext}[\boldsymbol u]$ is the is the displacement external potential

.. math::
	E_{ext}[\boldsymbol u] = \int_\Omega \boldsymbol f \cdot \boldsymbol u \ dV + \int_{\partial \Omega} \boldsymbol t \cdot \boldsymbol u \ dS

. The equilibrium equations can recovered from the optimality condition $\delta E=0$.

The strain energy function is given by:

.. math::
   \psi(\boldsymbol{\epsilon}) = \frac{1}{2} \lambda tr^2(\boldsymbol{\epsilon}) + \mu tr (\boldsymbol{\epsilon}^2)

the stress is given as 

.. math::
   \sigma(\boldsymbol{\epsilon}) = \lambda tr(\boldsymbol{\epsilon})\boldsymbol I + 2 \mu  \boldsymbol{\epsilon}

and the strain tensor:

.. math::
   \boldsymbol{\epsilon} = \frac{1}{2}(\nabla \boldsymbol u + \nabla^T \boldsymbol u)


Next, we calculate the variations of the functional, applying the Gateaux derivative.

For the internal potential:

.. math::
   \delta_{\boldsymbol u} E_{internal}[\boldsymbol u] = \int_\Omega \boldsymbol \sigma (\boldsymbol \epsilon (\boldsymbol u)) : \boldsymbol \epsilon(\delta{\boldsymbol{u}}) dV


and for the external potential:

.. math::
    \begin{align}
    \delta_{\boldsymbol u} E_{ext}[\boldsymbol u] & = \frac{d}{d \epsilon} E_{ext} (\boldsymbol u+\alpha\delta{\boldsymbol u}) \bigg\rvert_{\alpha=0}                                                                                                       \\
                & = \frac{d}{d \alpha} \int_\Omega  \boldsymbol f \cdot (\boldsymbol u+\alpha\delta{\boldsymbol u})) \ dV  \bigg\rvert_{\alpha=0}    + \frac{d}{d \alpha} \int_\partial  \boldsymbol t \cdot (\boldsymbol u+\alpha\delta{\boldsymbol u}) \ dS  \bigg\rvert_{\alpha=0}              \\
                & = \int_\Omega  \boldsymbol f \cdot \delta{\boldsymbol u} \ dV      +  \int_\partial  \boldsymbol t \cdot \delta{\boldsymbol u} \ dS      
    \end{align}


So the weak form of the elactic problem  is given by: 

.. math::
    \int_\Omega \boldsymbol \sigma (\boldsymbol \epsilon (\boldsymbol u)) : \boldsymbol \epsilon(\delta{\boldsymbol{u}}) dV - \int_\Omega  \boldsymbol f \cdot \delta{\boldsymbol u} \ dV -  \int_\partial  \boldsymbol t \cdot \delta{\boldsymbol u} \ dS  = 0


Implementation
--------------

