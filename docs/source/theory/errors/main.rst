.. _theory_errors:

Error Estimates
===============

In numerical analysis, error estimates provide a quantitative measure of the difference between the exact solution $u_e$ and the approximate solution $u_h$ obtained using numerical methods. The error $e$ is defined as:

.. math::
    e = u_e - u_h

Different norms are used to evaluate the magnitude of this error, each focusing on different aspects of the solution.

$H_1$ Norm 
----------
The $H^1$ norm combines the $L^2$ norm of the error and the $L^2$ norm of the error's gradient. This norm measures both the error in the function values and the error in the derivatives.

.. math::
    ||e||_{H^1} = \left( \int_{\Omega} |e|^2 \, d\Omega + \int_{\Omega} |\nabla e|^2 \, d\Omega \right)^\frac{1}{2}

Here, $\Omega$ represents the domain over which the error is evaluated, and $\nabla e$ is the gradient of the error.

$H^1$ Seminorm 
--------------
The $H^1$ seminorm considers only the $L^2$ norm of the gradient of the error. It focuses on the error in the derivatives of the function.

.. math::
    |e|_{H^1} = \left( \int_{\Omega} |\nabla e|^2 \, d\Omega \right)^\frac{1}{2}

This seminorm is particularly useful when the primary concern is the accuracy of the derivatives of the approximate solution.

$L^P$ Norms
-----------
These norms measure only the error in the function values, not in the derivatives. The general $L^p$ norm for $1 \leq p < \infty$ is given by:

.. math::
    ||e||_{L^p} = \left( \int_{\Omega} |e|^p \, d\Omega \right)^\frac{1}{p}

A common $L^p$ norm is the $L^2$ norm, which corresponds to $p = 2$. The $L^2$ norm is widely used because it provides a good balance between sensitivity to large errors and ease of computation.

$L^\infty$ Norm
---------------
The $L^\infty$ norm measures the maximum absolute value of the error over the domain. It is defined as:

.. math::
    ||e||_{L^\infty} = \max_{\Omega} |e|

This norm is useful when the maximum error is of particular interest, such as in applications where large errors cannot be tolerated.

Summary
-------
- The $H^1$ norm evaluates both the function values and their gradients.
- The $H^1$ seminorm focuses on the gradients.
- The $L^p$ norms (including $L^2$) evaluate the function values.
- The $L^\infty$ norm measures the maximum error.

Each of these norms provides different insights into the accuracy of the numerical solution, and the choice of norm depends on the specific requirements of the problem being solved.
