.. _theory_allen_cahn:

Allen Cahn
==========

.. note::
    Please view the examples related to the crack surface density functional in :ref:`ref_examples_allen_cahn`.

The Allen-Cahn equation is given by:

.. math::
	\frac{\partial \phi}{\partial t} = -M \mu


where $\phi$ is the phase-field, $M$ is the mobility, and $\mu$ is the chemical potential.


Free energy $\mu$
-----------------
In this case, the problem that needs to be solved is stationary and derives from an optimization principle.
The phase field is given by the minimizer of the following energy-functional

.. math::
	W[\phi] = \int_\Omega \left( \frac{1}{l}f_{chem}(\phi) + \frac{l}{2} |\nabla \phi|^2 \right) dV 

with 

.. math::
	f_{chem}(\phi) = \frac{1}{4}(1-\phi^2)^2

.. note::
	.. math::
		f'_{chem}(\phi) = \phi^3 - \phi

	.. math::
		f''_{chem}(\phi) = 3\phi^2 - 1


The equilibrium equations can recovered from the optimality condition $\delta W=0$. Next, we calculate the variations of the funcional, applying the Gateaux derivative.
	
.. math::
	\delta_\phi W & = \frac{d}{d \epsilon} W (\phi+\epsilon\delta_\phi) \bigg\rvert_{\epsilon=0}                                                                                                       \\
				  & = \frac{d}{d \epsilon} \int \left( \frac{1}{l}f_{chem}(\phi +\epsilon \delta_\phi) + \frac{l}{2} |\nabla (\phi+\epsilon\delta_\phi)|^2  \right) dV \bigg\rvert_{\epsilon=0}               \\
				  & = \int \left( \frac{1}{l}f'_{chem}(\phi+\epsilon \delta_\phi) \delta_\phi + \frac{l}{2} 2(\nabla (\phi+\epsilon\delta_\phi))\cdot \nabla \delta_\phi   \right) dV \bigg\rvert_{\epsilon=0} \\
				  & = \int \left( \frac{1}{l}f'_{chem}(\phi+\epsilon \delta_\phi) \delta_\phi   +       l      (\nabla (\phi+\epsilon\delta_\phi))\cdot \nabla \delta_\phi   \right) dV \bigg\rvert_{\epsilon=0} \\
				  & = \int \left( \frac{1}{l}f'_{chem}(\phi) \delta_\phi                        +       l       \nabla  \phi                      \cdot \nabla \delta_\phi  \right) dV


So, the weak form of the phase-field problem in the absence of external potential is given by:

.. math::
	\int_\Omega \left( \frac{1}{l} f'_{chem}(\phi)\delta\phi  + l \nabla\phi \cdot \nabla \delta \phi \right) dV = 0



One dimension solution
----------------------

Consider a broken bar with length $L$, as depicted in fig |fig:bar_allen|, featuring a crack positioned at its center.w

.. |fig:bar_allen| image:: images/bar_graph.png


For the one-dimensional scenario, the functional reads:

.. math::
    W_{1D}[\phi] = \int_\Omega \left( \frac{1}{l} f'_{chem}(\phi) + \frac{l}{2} \phi'^2 \right) dx
	
subject to the boundary conditions $\phi(\pm \infty) = \pm 1$, $\phi'(\pm\infty= = 0)$.

The problem can be reduced to a simple ordinary differential equation, with boundary conditions $\phi(0) = 1$, and $\phi'(A) = 0$:

.. math::
	\frac{1}{l}f'_{chem}(\phi) - l \phi''=0

The solution to this equation is given by:

.. math::
	\phi(x) = tanh\left(\frac{x}{l \sqrt{2}}\right)

.. math::
	\phi'(x) = \frac{1}{l\sqrt{2}} sech^2 \left( \frac{x}{l \sqrt{2}}\right)
