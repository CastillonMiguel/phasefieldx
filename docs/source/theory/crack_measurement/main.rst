.. _theory_crack_measurement:

Crack Measurements
===================

Measuring the crack surface area in phase-field models is not trivial. The crack area is typically approximated by integrating the crack surface density functional, $\Gamma(\phi)$, over the domain. However, when the system of PDEs is solved using numerical methods such as the standard finite element formulation, this approach systematically overestimates the true physical crack area. As shown 
in :footcite:t:`phase_field_Castillon` and :footcite:t:`phase_field_castillon_dgcm2026`, this discrepancy also propagates to derived
quantities such as forces, displacements, and energy measures. This page summarizes the origin of this overestimation and presents several correction strategies, together with an image-based skeletonization procedure for measuring crack length directly from simulation results.

.. contents::
   :local:
   :depth: 2


Strain localization
--------------------

A common numerical artifact in phase-field fracture simulations is *strain localization*
(also discussed in :ref:`sec_pff_mesh_effects`), analyzed in
:footcite:t:`phase_field_effective_Gc_factor_2`. In this phenomenon the phase-field variable
artificially saturates to $\phi=1$ across an entire element of size $h$, which overestimates
the crack surface area when the latter is measured from the fracture energy functional.

.. tikz::
   :align: center

   \begin{tikzpicture}[scale=1.0]
       % Bar representation
       \draw[fill=gray!20, draw=black]
           (-4,-0.15) rectangle (4,0.15);

       % Element size h
       \def\h{0.25}

       % Half-length annotations
       \draw[<->] (-4,-0.32) -- (0,-0.32);
       \node[below] at (-2,-0.32) {$a$};

       \draw[<->] (0,-0.32) -- (4,-0.32);
       \node[below] at (2,-0.32) {$a$};

       % Ideal phase-field profile (dashed)
       \draw[
           thick,
           dashed,
           black,
           domain=-4:4,
           samples=200,
           smooth
       ]
       plot (\x,{1.5*exp(-abs(\x)/1.0) + \h});

       % Phase-field profile with strain localization
       % Left part
       \draw[
           thick,
           black,
           domain=-3.75:0,
           samples=200,
           smooth
       ]
       plot ({\x-\h},{1.5*exp(-abs(\x)/1.0) + \h});

       % Localized plateau
       \draw[
           thick,
           black
       ]
       plot coordinates {
           (-\h,1.75)
           (\h,1.75)
       };

       % Right part
       \draw[
           thick,
           black,
           domain=0:3.75,
           samples=200,
           smooth
       ]
       plot ({\x+\h},{1.5*exp(-abs(\x)/1.0) + \h});

       % Element size annotation
       \draw[<->] (-\h,2.0) -- (\h,2.0);
       \node[above] at (0,2.0) {$h$};

   \end{tikzpicture}

   Illustration of the phase-field profile distortion caused by strain localization. The ideal profile (dashed line) is artificially flattened to $\phi=1$ over an element of size $h$ (solid line), leading to an overestimation of the crack surface energy.

Within the localized element the phase-field is constant ($\phi=1$) and its gradient is zero.
This alters the energy calculation. The total energy including the strain-localization effect,
denoted $\Gamma_{\text{sl}}$, is obtained by adding the constant contribution of the localized
region to the general functional $\Gamma(\phi)$ (see :eq:`eq_csdf_general_functional`):

.. math::
   :label: eq_crack_measurement_sl_energy

   \Gamma_{\text{sl}}[\phi, \nabla \phi] = \int_{-a}^{+a}  \frac{1}{c_0}\left( \frac{\alpha(\phi)}{l} + l (\phi')^2 \right) \mathrm{d}x + \int_{0}^{h}  \frac{1}{c_0}\left( \frac{\alpha(1)}{l} + l (0)^2 \right) \mathrm{d}x.

As a result, the phase-field term $\Gamma_\phi$ increases due to the localized region, whereas
the gradient term $\Gamma_{\nabla\phi}$ is unchanged, since the gradient vanishes where $\phi$
is constant:

.. math::
   :label: eq_crack_measurement_sl_split

   \Gamma_{\phi, \text{sl}} =  \Gamma_{\phi} + \frac{h}{c_0 l}, \qquad
   \Gamma_{\nabla\phi, \text{sl}} = \Gamma_{\nabla\phi}.

so that the total energy with strain localization becomes:

.. math::
   :label: eq_crack_measurement_sl_total

   \Gamma_{\text{sl}} = \Gamma_{\phi, \text{sl}} + \Gamma_{\nabla\phi, \text{sl}} = \Gamma + \frac{h}{c_0 l}.

Strain localization therefore introduces an additive error of $h/(c_0 l)$ in the total crack
surface energy for the one-dimensional case, directly proportional to the element size $h$ and
inversely proportional to the length scale $l$. As the element size $h$ decreases, this
additional energy vanishes and the computed energy converges to the theoretical value.


Crack area correction factor
-----------------------------

To mitigate the overestimation described above, the crack area, $\Gamma$, corresponding to the case without strain localization is recovered from the crack area affected by strain localization, $\Gamma_\mathrm{sl}$, through the application of a correction factor, $\mathcal{F}$, following :footcite:t:`phase_field_castillon_dgcm2026`:

.. math::
   :label: eq_crack_measurement_area_correction

   \Gamma = \frac{\Gamma_\mathrm{sl}}{\mathcal{F}}.

Here $\mathcal{F}$ quantifies the overestimation caused by the diffuse crack representation and
numerical artifacts such as strain localization. The following sections present three methods
for estimating $\mathcal{F}$.

Physically consistent application of the correction factor
-------------------------------------------------------------

The crack surface overestimation can equivalently be interpreted as an effectively higher
energy release rate, so it is important to distinguish whether the crack surface is
overestimated or the critical energy release rate is effectively higher than expected. Three
alternative, mathematically consistent schemes for turning simulated ("sl") quantities into
physical quantities using $\mathcal{F}$ are summarized below. They also involve the degraded
strain energy $\Psi(\boldsymbol{u}, \phi) = \int_\Omega g(\phi) \, \psi(\boldsymbol \epsilon(\boldsymbol{u})) \, \mathrm{d}\Omega$
and the specimen stiffness $K = P/u$.

.. _tab_crack_measurement_schemes:
.. list-table:: Three mathematically consistent schemes for transforming simulated results into physical quantities using the correction factor $\mathcal{F}$.
   :header-rows: 1

   * - **Physical quantity**
     - **Scheme I**
     - **Scheme II**
     - **Scheme III**
   * - Critical energy release rate $G_{c,\mathrm{phys}}$
     - $G_{c,\mathrm{sl}}$
     - $G_{c,\mathrm{sl}}\mathcal{F}$
     - $G_{c,\mathrm{sl}}\mathcal{F}$
   * - Crack area $\Gamma_\mathrm{phys}$
     - $\Gamma_\mathrm{sl} / \mathcal{F}$
     - $\Gamma_\mathrm{sl}$
     - $\Gamma_\mathrm{sl} / \mathcal{F}$
   * - Degraded strain energy $\Psi_\mathrm{phys}$
     - $\Psi_\mathrm{sl} / \mathcal{F}$
     - $\Psi_\mathrm{sl} \mathcal{F}$
     - $\Psi_\mathrm{sl}$
   * - Force $P_\mathrm{phys}$
     - $P_\mathrm{sl} / \sqrt{\mathcal{F}}$
     - $P_\mathrm{sl}\sqrt{\mathcal{F}}$
     - $P_\mathrm{sl}$
   * - Displacement $u_\mathrm{phys}$
     - $u_\mathrm{sl} / \sqrt{\mathcal{F}}$
     - $u_\mathrm{sl} \sqrt{\mathcal{F}}$
     - $u_\mathrm{sl}$
   * - Stiffness $K$
     - $K_\mathrm{sl}$
     - $K_\mathrm{sl}$
     - $K_\mathrm{sl}$

All three schemes are consistent with the following relation of the critical energy release rate,

.. math::
   :label: eq_crack_measurement_energy_release_rate

   G_c = \frac{P^2}{2} \frac{\partial C}{\partial a},

where $P$ is the applied force, $C$ is the structural compliance, and $a$ is the crack area.
Substituting the transformations of each scheme reproduces the same relationship:

- **Scheme I**: $G_c = \dfrac{(P / \sqrt{\mathcal{F}})^2}{2} \left( \dfrac{\partial C}{\partial a} \cdot \mathcal{F} \right)$
- **Scheme II**: $G_c\mathcal{F} = \dfrac{(P\sqrt{\mathcal{F}})^2}{2} \dfrac{\partial C}{\partial a}$
- **Scheme III**: $G_c\mathcal{F} = \dfrac{P^2}{2} \left( \dfrac{\partial C}{\partial a} \cdot \mathcal{F} \right)$

confirming that the transformations are thermodynamically consistent, and since force and
displacement are scaled by the same factor, the stiffness of the system is unchanged in all
three schemes.

**Scheme I** is principally preferred because:

- The critical energy release rate $G_c$ is an intrinsic material property and should not be
  altered by a correction factor addressing a numerical artifact.
- Since the inaccuracy is directly related to the overestimation of the crack area, it is more
  intuitive to apply the correction to the crack length rather than to the energy release rate.
- Force and displacement are global quantities; it is preferable to correct only the simulation
  outputs rather than the material input parameters.
- This preserves the exact physical $G_c$ and applies the correction only to the simulation
  outputs (crack area, force, displacement).

In the general literature, the correction is typically applied to $G_c$ instead (Scheme II or
III), which can be confusing because modifying $G_c$ changes the force or the crack length, but
not both simultaneously, and the effect on displacement is rarely addressed. Mixing Scheme II
for force with Scheme III for crack area produces physically inconsistent results, so the
unambiguous Scheme I is preferred here. Under Scheme I, the relative error introduced by strain
localization is:

.. math::
   :label: eq_crack_measurement_relative_error

   \text{Relative Error} = \frac{|\Gamma - \Gamma_\mathrm{sl}|}{|\Gamma|} = \frac{|\Gamma - \mathcal{F} \, \Gamma_\mathrm{sl}|}{|\Gamma_\mathrm{sl}|} = |1 - \mathcal{F}|.


Element size-based correction method
------------------------------------

This correction is derived from the one-dimensional analytical solution. As shown above, strain
localization causes the computed total crack surface energy $\Gamma_{\text{sl}}$ to be
overestimated by an additive term proportional to the element size $h$
(:eq:`eq_crack_measurement_sl_total`). The correction factor scaling the simulated energy back
to the theoretical value $\Gamma$ is therefore:

.. math::
   :label: eq_crack_measurement_element_factor

   \mathcal{F}_\mathrm{elem} = \frac{\Gamma_{\text{sl}}}{\Gamma} = \frac{\Gamma + \frac{h}{c_0 l}}{\Gamma} = 1 + \frac{1}{\Gamma} \frac{h}{c_0 l}.

A key limitation is that $\Gamma$ itself is unknown in a simulation (only $\Gamma_\mathrm{sl}$
is computed). To proceed, it is standard to assume an idealized 1D crack in an infinite domain
(or where boundary effects are negligible, i.e., $l/a \to 0$), so that $\Gamma \to 1$ (see
:ref:`tab_pff_sharp_crack_conditions`). Under this assumption, the factor simplifies to the
correction proposed by Bourdin in :footcite:t:`phase_field_Bourdin2008`:

.. math::
   :label: eq_crack_measurement_bourdin_factor

   \mathcal{F}_\mathrm{Bourdin} = 1 + \frac{h}{c_0 l}.

This factor remains constant throughout the simulation, depending only on the mesh size $h$ and
the length scale $l$. Its effectiveness is highest for straight cracks propagating
perpendicular to mesh edges in a regular mesh — mirroring the idealized 1D case it is derived
from — and it does not account for boundary effects. As the mesh is refined ($h/l \to 0$), the
factor approaches 1, but the underlying assumption $\Gamma \approx 1$ only holds when the
boundaries are far compared to the length scale ($a/l \to \infty$).

For simulations using symmetry boundary conditions, where only half of the crack is modeled,
the theoretical energy in :eq:`eq_crack_measurement_element_factor` is effectively halved
($\Gamma \approx 0.5$), so the Bourdin factor for symmetric cases becomes:

.. math::
   :label: eq_crack_measurement_bourdin_sym_factor

   \mathcal{F}_\mathrm{Bourdin, sym} = 1 + \frac{2 h}{c_0 l}.

This is consistent with :footcite:t:`phase_field_effective_Gc_factor_2`, where the factor of 2
arises because strain localization occurs over a full element width $h$, but this contribution
is scaled relative to the energy of only half the crack.

Double Gradient Correction Method (DGCM)
----------------------------------------

The Double Gradient Correction Method (DGCM) exploits the fact that, while strain localization
overestimates the phase-field energy term $\Gamma_\phi$, the gradient energy term
$\Gamma_{\nabla\phi}$ remains largely unaffected (its gradient is simply zero within the
localized element, but the integral over the rest of the domain is essentially unchanged).

When the length scale $l$ is small relative to the domain, the phase-field and gradient energy
contributions become equal, $\Gamma_\phi = \Gamma_{\nabla\phi}$ — see the" AT2,AT1,WU" energy
expressions in :ref:`tab_pff_analytical_energies`, where both terms tend to $1/2$ as
$a/l \to \infty$. This *equipartition* is a core property of the regularized model in the
sharp-crack limit, so the true physical crack area can be approximated by doubling the
(unaffected) gradient term:

.. math::
   :label: eq_crack_measurement_dgcm_approx

   \Gamma \approx 2 \Gamma_{\nabla\phi} = 1.

Comparing the simulated area $\Gamma_\mathrm{sl} = \Gamma_{\mathrm{sl},\phi} + \Gamma_{\mathrm{sl},\nabla\phi}$
with this approximation, and using $\Gamma_{\mathrm{sl},\nabla\phi} = \Gamma_{\nabla\phi}$ (the
gradient term is unaffected by strain localization), gives the DGCM correction factor:

.. math::
   :label: eq_crack_measurement_dgcm_factor

   \mathcal{F}_\mathrm{DGCM} = \frac{\Gamma_\mathrm{sl}}{\Gamma} = \frac{\Gamma_{\mathrm{sl},\phi} + \Gamma_{\nabla\phi}}{2\Gamma_{\nabla\phi}} = \frac{1}{2} \left(1 + \frac{\Gamma_{\mathrm{sl},\phi}}{\Gamma_{\nabla\phi}}\right)
   = \frac{1}{2} \left(1 + \frac{\int_\Omega \alpha(\phi) \,\mathrm{d}\Omega}{l^2 \int_\Omega |\nabla \phi|^2 \,\mathrm{d}\Omega}\right).

Both the numerator and denominator are computed as integrals over the domain from the finite
element solution at each simulation step, so $\mathcal{F}_\mathrm{DGCM}$ is evaluated
dynamically rather than assumed constant. Unlike the Bourdin correction, it is independent of
the mesh size $h$, adapts to the evolving phase-field, and applies unmodified in 1D, 2D, or 3D.

Crack Length Measurement via Image Post-Processing and Skeletonization Algorithms
---------------------------------------------------------------------------------

The automated crack-length measurement workflow presented in :footcite:t:`phase_field_lvpp_Castillon`
is implemented using image post-processing and the skeletonization routines from scikit-image
(:footcite:t:`code_skeleton_algorithm`, :footcite:t:`code_scikit_image`). The pipeline converts
the phase-field variable into a binary mask, applies ``skeletonize`` to obtain a
single-pixel-wide centerline, extracts pixel coordinates and maps them to physical domain
coordinates, and fits a smooth spline to the centerline for accurate length measurement. The
skeletonization algorithm iteratively thins object boundaries until the medial axis (the
single-pixel-wide centerline) remains.

Below are the results of the simulations, for which we have images showing the crack evolution.

.. raw:: html

   <div style="padding: 30px 0; margin: 20px 0;">
         <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
            <div style="text-align: center;">
               <img src="../../_static/animations/crack_measurement_1.gif" width="180px" style="border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);" loop="infinite" autoplay />
            </div>
            <div style="text-align: center;">
               <img src="../../_static/animations/crack_measurement_2.gif" width="180px" style="border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);" loop="infinite" autoplay />
            </div>
            <div style="text-align: center;">
               <img src="../../_static/animations/crack_measurement_4.gif" width="180px" style="border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);" loop="infinite" autoplay />
            </div>
         </div>
   </div>

The algorithm extracts the line representing the crack path, saving for each frame a series of points that define the line. Once the line coordinates are known, the crack length can be computed from these points.

The animations below show the extracted line (for which all coordinates are known) for the results presented above. Note that the extracted line matches the crack obtained from the phase-field simulation very well, allowing the crack length to be computed with high accuracy.

.. raw:: html

   <div style="padding: 30px 0; margin: 20px 0;">
         <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
            <div style="text-align: center;">
               <img src="../../_static/animations/crack_measurement_1_sol.gif" width="180px" style="border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);" loop="infinite" autoplay />
            </div>
            <div style="text-align: center;">
               <img src="../../_static/animations/crack_measurement_2_sol.gif" width="180px" style="border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);" loop="infinite" autoplay />
            </div>
            <div style="text-align: center;">
               <img src="../../_static/animations/crack_measurement_4_sol.gif" width="180px" style="border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);" loop="infinite" autoplay />
            </div>
         </div>
   </div>


Procedure
---------

Crack length is extracted automatically from phase-field solutions stored in ``.vtu`` format
through the following steps: (1) identify cracked regions where $\phi > \phi_\text{th} = 0.95$;
(2) generate a binary image with red zones for $\phi > \phi_\text{th}$ and black zones for
$\phi < \phi_\text{th}$; (3) skeletonize to extract a single-pixel-wide crack path using
``skimage.morphology.skeletonize``; (4) map pixel coordinates to physical domain coordinates
using the known pixel-to-domain mapping; (5) fit a spline curve through these points to avoid
length mismeasurement when the crack follows a diagonal path; and (6) compute the crack length
from the fitted spline. Since the phase-field variable is not saved at every simulation step,
quadratic interpolation is used to obtain the crack length at all time steps.

1. The crack is identified in regions where the phase-field variable $\phi$ exceeds the
   threshold value $\phi_{th} = 0.95$.

   .. figure:: crack_evolution.gif
      :align: center
      :width: 400

      Crack evolution over time (animated GIF).

2. Using this threshold, the crack area is extracted from $\phi$. An image is generated,
   highlighting regions where $\phi > \phi_{th}$ in one color and $\phi < \phi_{th}$ in another.

   .. note::
      For the examples considered here, a rectangular domain is used, so the generated image
      is a rectangle matching the simulation domain size. This allows a direct mapping between
      image pixels and the physical dimensions of the domain, enabling accurate determination
      of the real coordinates of the crack in the images.

3. The skeleton of the crack is extracted from the binary image. Skeletonization reduces the
   crack region to a single-pixel-wide path representing the crack trajectory, using
   ``skimage.morphology.skeletonize`` for all time steps of the simulation. The underlying
   skeletonization algorithm is explained in detail in the
   `scikit-image documentation <https://scikit-image.org/docs/0.25.x/auto_examples/edges/plot_skeleton.html>`_.

   .. figure:: phasefieldx_p0_000140.png
      :align: center
      :width: 60%

      Example binary mask for a compact-tension specimen with holes: black/white regions mark
      $\phi < \phi_\text{th}$ / $\phi > \phi_\text{th}$, and the red line is the extracted
      skeleton (crack path) used for length measurement.

4. Since pixel-based measurements can introduce errors — especially where the crack path is
   diagonal or curved — the skeleton points are treated as coordinates, and a curve is fitted
   through them to better approximate the actual crack path.

5. Finally, the length of the fitted curve is measured to obtain an accurate estimate of the
   crack length.

   .. figure:: crack_evolution_pyvista.gif
      :align: center
      :width: 400

      Crack evolution over time (animated GIF).

Summary of correction methods
-----------------------------

This section reviewed three methods for correcting crack area overestimation: the element
size-based (Bourdin) correction, the Double Gradient Correction Method (DGCM), and the
skeletonization technique. The table below summarizes the correction factor $\mathcal{F}$ for
each method.

.. _tab_crack_measurement_corrections:
.. list-table:: Summary of correction factors for crack area overestimation.
   :header-rows: 1

   * - **Method**
     - **Correction factor** $\mathcal{F}$
     - **Type**
   * - Bourdin
     - $\mathcal{F}_\mathrm{Bourdin} = 1 + \dfrac{h}{c_0 l}$
     - Constant
   * - DGCM
     - $\mathcal{F}_\mathrm{DGCM} = \dfrac{1}{2} \left(1 + \dfrac{\int_\Omega \alpha(\phi) \,\mathrm{d}\Omega}{l^2 \int_\Omega |\nabla \phi|^2 \,\mathrm{d}\Omega}\right)$
     - Dynamic
   * - Skeletonization
     - $\mathcal{F}_\mathrm{skeleton} = \dfrac{\Gamma_\mathrm{sl}}{\Gamma_\mathrm{measured}}$
     - Dynamic

:ref:`tab_pff_sharp_crack_conditions` (in :ref:`theory_phase_field`) summarizes the conditions
on the ratio $a/l$ under which the one-dimensional energy functional is independent of boundary
effects and exactly equals the sharp-crack energy. Under these conditions, the following
equality holds for the 1D case:

.. math::
   :label: eq_crack_measurement_sharp_crack_equivalence

   G_c \int_{\Omega} \gamma(\phi,\nabla\phi)\,\mathrm{d}\boldsymbol{x} = G_c \int_{\Gamma_{\text{sharp}}} \mathrm{d}S.

The Bourdin correction requires $\Gamma=1$; the DGCM requires that the equipartition of the
phase-field and gradient energies holds. These two conditions are equivalent: if energy
equipartition holds, $\Gamma=1$ is satisfied, and vice versa. The relative error for the DGCM in
the 1D case is:

.. math::
   :label: eq_crack_measurement_dgcm_error

   \text{Relative Error} = \frac{|2 \Gamma_{\nabla\phi} - \Gamma|}{\Gamma}.

For the AT2 regularization, this can be expressed as a function of $a/l$:

.. math::
   :label: eq_crack_measurement_dgcm_error_at2

   \text{Relative Error AT2} = \frac{a}{l} \frac{1-\tanh^2(a/l)}{\tanh(a/l)}.

This error decreases rapidly as $a/l$ increases — e.g., at $a/l = 5$ the error is below $0.1\%$
— and is specific to the AT2 model, which has infinite support. For models with finite support
(like AT1), the error is exactly zero once the phase-field has enough space to fully develop
without boundary interference ($a/l \ge 2$, see :ref:`tab_pff_sharp_crack_conditions`), since
the compact support of AT1 is then entirely contained within the domain.

Because the Bourdin factor is constant, it can be applied either *a priori* or *a posteriori*,
as it acts purely as a scaling parameter — a fact easily verified in the dimensionless
formulation, or by comparing two problems with different critical energy release rates, as in
:footcite:t:`phase_field_castillon_dgcm2026`. The DGCM and skeletonization factors, in contrast,
are obtained dynamically at each simulation step and applied *a posteriori*.

Once determined, the correction factor $\mathcal{F}$ is applied to the raw ("sl") simulation
results to recover the physical quantities, using **Scheme I** as summarized in
:ref:`tab_crack_measurement_schemes`.

.. footbibliography::
