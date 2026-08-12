.. _theory_phase_field_fracture:

PHASE-FIELD FRACTURE
====================
Fracture mechanics plays a critical role in engineering, as structural failure can lead to
catastrophic economic and safety consequences. Among the various failure mechanisms, fracture and
fatigue---damage accumulation under cyclic loading---is responsible for the vast majority of
mechanical failures in service. Predicting the life of components under these conditions is
therefore a primary goal in design and maintenance.

Over the years, significant advancements have been made in modeling and simulating fracture
processes. While classical approaches like Linear Elastic Fracture Mechanics (LEFM) have been
successful for simple geometries, they struggle with complex crack topologies, branching, and
merging. In this context, the Phase-Field Fracture (PFF) method has emerged as a powerful and
versatile computational tool. By regularizing the sharp crack interface into a diffuse damage
band, PFF eliminates the need for explicit tracking of the crack path, offering a flexible
framework to simulate complex failure scenarios.

.. _sec_pff_background:

PFF background
--------------
Among the different models available to simulate failure, the PFF model has received
significant attention in the last decade. To understand its advantages, it is useful to first
consider the classical discrete approach. In classical mechanics, a crack is modeled as a
sharp geometric discontinuity, denoted as $\Gamma_{\text{sharp}}$, within the solid domain
$\Omega$. While physically intuitive, this sharp-interface representation poses significant
numerical challenges: the crack path is an unknown boundary that must be explicitly tracked
and updated as it propagates, making the handling of complex topologies like branching or
merging computationally difficult.

The phase-field method addresses these difficulties by regularizing the sharp discontinuity
into a diffuse band. The sharp interface is replaced by a region where a continuous scalar
variable, $\phi \in [0, 1]$, varies smoothly from intact material ($\phi=0$) to fully
damaged material ($\phi=1$). Rooted in the variational formulation proposed by
:footcite:t:`phase_field_FrancfortMarigo1998` and further developed by
:footcite:t:`phase_field_Bourdin2000` and :footcite:t:`phase_field_Miehe2010`, this framework
eliminates the need for explicit crack tracking. Consequently, the crack is treated not as a
geometric boundary but as a region of localized damage, allowing the fracture problem to be
solved as a coupled system of partial differential equations on a fixed mesh.

.. grid:: 2
   :gutter: 3

   .. grid-item::

      .. tikz::
         :align: center

         \begin{tikzpicture}[scale=1.0]
             \draw[thick] (0,0) rectangle (4,4);
             \node at (1.0, 2.2) {$\Gamma_\text{sharp}$};
             \draw[-] (0, 2.0) -- (2.0, 2.0);
             \node at (2.0, 1.0) {$\Omega$};
             \node at (2.0, 4.2) {$\partial \Omega$};
         \end{tikzpicture}

      **(a)** Idealized discrete crack model.

   .. grid-item::

      .. tikz::
         :align: center

         \begin{tikzpicture}[scale=1.0]
             \draw[thick] (0,0) -- (4,0) -- (4,4) -- (0,4)
                           -- (0,2.1) -- (2,2) -- (0,1.9) -- (0,0);
         \end{tikzpicture}

      **(b)** Real-world fractured plate.

   .. grid-item::

      .. tikz::
         :align: center

         \begin{tikzpicture}[scale=1.0]
             \draw[thick] (0,0) rectangle (4,4);
             \node at (1.0, 2.5) {$\Gamma(\phi,l)$};
             \draw[-] (0, 2.0) -- (2.0, 2.0);
             \draw[thick] (0, 2.1) -- (2, 2.1);
             \draw[thick] (0, 2.2) -- (2, 2.2);
             \draw[thick] (0, 1.9) -- (2, 1.9);
             \draw[thick] (0, 1.8) -- (2, 1.8);
             \draw[thick] (2, 2.2) arc[start angle=90,end angle=-90,radius=0.2];
             \draw[thick] (2, 2.1) arc[start angle=90,end angle=-90,radius=0.1];
             \node at (2.0, 1.0) {$\Omega$};
             \node at (2.0, 4.2) {$\partial \Omega$};
         \end{tikzpicture}

      **(c)** Schematic of regularized crack :math:`\Gamma(\phi,l)`.

   .. grid-item::

      .. figure:: images/screenshot.png
         :align: center
         :width: 100%

      **(d)** Numerical phase-field result.

Comparison of fracture representations. Top: discrete crack approach — (a) idealized
mathematical crack and (b) real-world fractured plate. Bottom: phase-field approach --- (c)
diffuse interface regularization $\Gamma(\phi, l)$ and (d) numerical phase-field profile
$\phi$.

The variational approach to fracture mechanics is fundamentally based on the principle of
minimizing the total potential energy of a solid body containing a crack. This total
potential energy comprises the elastic strain energy stored in the bulk material and the
surface energy associated with the crack. In this formulation, the crack discontinuity set
$\Gamma_{\text{sharp}}$ is treated as an unknown of the problem, determined such that it
minimizes the energy functional. This classical formulation is expressed in
:eq:`eq_pff_variational_discontinuity`:

.. math::
   :label: eq_pff_variational_discontinuity

   V(\boldsymbol{u},\Gamma_{\text{sharp}}) =
   \underbrace{\int_{\Omega \setminus \Gamma_{\text{sharp}}} \psi(\boldsymbol{\epsilon}(\boldsymbol{u})) \, \mathrm{d}\boldsymbol{x}}_{\text{Elastic strain energy}}
   + \underbrace{G_c \int_{\Gamma_{\text{sharp}}} \mathrm{d}S}_{\text{Fracture surface energy}}
   - \mathcal{E}_{\text{ext}}(\boldsymbol{u}),

where $\boldsymbol{u}$ is the displacement field, $\boldsymbol{\epsilon}$ is the strain
tensor, $\psi(\boldsymbol{\epsilon}(\boldsymbol{u}))$ represents the strain energy density,
$G_c$ denotes the critical energy release rate, $\Gamma_{\text{sharp}}$ represents the sharp
discontinuity or crack surface, and $\mathcal{E}_{\text{ext}}(\boldsymbol{u})$ refers to the
potential energy of the external forces.

A major challenge in the classical formulation is that the crack set $\Gamma_{\text{sharp}}$
is an unknown in the problem. To address this, :footcite:t:`phase_field_FrancfortMarigo1998`
and :footcite:t:`phase_field_Bourdin2000` reformulated the problem within a variational
framework where the sharp discontinuity is approximated by a continuous auxiliary variable,
the phase-field. This field varies smoothly from $\phi=0$ (not damaged) to $\phi=1$ (fully
damaged), representing the crack transition. This method was further developed by authors
such as :footcite:t:`phase_field_Miehe2010`. The resulting regularization permits solving the
fracture problem via a minimization principle that represents cracks using a continuous scalar
field $\phi$, eliminating the need for explicit crack-tracking algorithms and naturally
handling complex crack topologies such as branching and merging.

The PFF method models crack propagation through the minimization of a total potential energy
functional. Similar to :eq:`eq_pff_variational_discontinuity`, this functional is composed of
the effective elastic strain energy, the fracture surface energy, and the work done by external
forces. However, in this case, the unknown discontinuity is not explicitly present; instead, a
new continuous variable $\phi$ is introduced to represent the crack. The total potential energy
functional in the phase-field fracture framework is given by:

.. math::
   :label: eq_pff_variational_formulation

   \mathcal{E}(\boldsymbol{u}, \phi) =
   \underbrace{\int_\Omega \Big( g(\phi) \underbrace{\psi_a(\boldsymbol{\epsilon}(\boldsymbol{u})) + \psi_b(\boldsymbol{\epsilon}(\boldsymbol{u}))}_{\psi(\boldsymbol{\epsilon}(\boldsymbol{u}))}\Big) \, \mathrm{d}\boldsymbol{x}}_{\Psi(\boldsymbol \epsilon(\boldsymbol u), \phi)\,\text{Effective elastic strain energy}}
   + \underbrace{G_c \underbrace{\int_\Omega \gamma(\phi,\nabla \phi) \, \mathrm{d}\boldsymbol{x}}_{\Gamma(\phi)}}_{\text{Fracture surface energy}}
   - \mathcal{E}_{\text{ext}}(\boldsymbol{u}),

where $\boldsymbol{\epsilon}(\boldsymbol{u}) = \frac{1}{2}(\nabla \boldsymbol{u} + (\nabla \boldsymbol{u})^T)$
is the symmetric strain tensor and $\phi$ is the phase-field variable. Here,

.. math::
   :label: eq_pff_strain_energy_density

   \psi(\boldsymbol{\epsilon}) = \frac{1}{2}\lambda (\text{tr}(\boldsymbol{\epsilon}))^2 + \mu \text{tr}(\boldsymbol{\epsilon}^2)

represents the strain energy density, with $\lambda$ and $\mu$ denoting the Lamé parameters.
These are defined in terms of Young's modulus $E$ and Poisson's ratio $\nu$ as
$\lambda = \frac{E \nu}{(1+\nu)(1-2\nu)}$ and $\mu = \frac{E}{2(1+\nu)}$. The strain energy
is decomposed into an active part $\psi_a$, which is affected by the degradation function
$g(\phi)$, and an inactive component $\psi_b$, which remains undegraded. This decomposition
distinguishes between the energy that drives crack evolution ($\psi_a$) and the energy that
does not ($\psi_b$). This separation enables the model to capture physically realistic
behaviors, such as allowing damage under tensile loading while preventing it under compression.
Different types of energy splits exist in the literature; each is covered in detail in
:ref:`sec_pff_iso_aniso_models`.

The degradation function $g(\phi)$ quantifies the reduction in material stiffness as damage
evolves. In general, the most widely used function takes a quadratic form

.. math::
   :label: eq_pff_degradation_quadratic

   g(\phi) = (1 - \phi)^2,

though other functional forms have been proposed in the literature
:footcite:t:`phase_field_degradation_functions`.

The term $G_c \int_\Omega \gamma(\phi, \nabla \phi) \, \mathrm{d}\boldsymbol{x}$ represents
the fracture surface energy, where $G_c$ is the critical energy release rate and
$\gamma(\phi, \nabla \phi)$ is the crack surface density function. This term is directly
related to the diffuse representation of the crack, and several functional forms can be
considered, as detailed in :ref:`theory_phase_field`. Furthermore, this
component can be analyzed as a problem depending only on the phase-field variable. The integral
$\Gamma(\phi) = \int_\Omega \gamma(\phi,\nabla \phi) \, \mathrm{d}\boldsymbol{x}$ defines the
Crack Surface Density Functional (CSDF), which governs the geometric regularization. This
functional must not be confused with the local crack surface density function,
$\gamma(\phi, \nabla \phi)$, which refers to the integrand. It is important to note that, in
contrast to the previous sharp interface formulation, this energy depends on the phase-field
variable $\phi$ and its gradient $\nabla \phi$, and it is defined as an integral over the
whole domain, rather than over the crack surface as in :eq:`eq_pff_variational_discontinuity`.

The term $\mathcal{E}_{\text{ext}}(\boldsymbol{u})$ represents the potential energy of the
external forces given by

.. math::
   :label: eq_pff_external_energy

   \mathcal{E}_{\text{ext}}(\boldsymbol{u}) = \int_\Omega \boldsymbol{f} \cdot \boldsymbol{u} \,\mathrm{d}\boldsymbol{x} + \int_{\partial \Omega_t} \boldsymbol{t} \cdot \boldsymbol{u} \, \mathrm{d}S,

where $\boldsymbol{f}$ is the body force per unit volume and $\boldsymbol{t}$ is the traction
applied on the Neumann boundary $\partial \Omega_N$. This term remains consistent between both
the discrete and diffuse formulations, as it depends solely on the displacement field and
external forces, and does not involve the crack representation.

Given the fundamental differences between the discrete (sharp) and the diffuse (phase-field)
variational formulations, it is useful to establish a clear relationship between their
respective terms to clarify the concepts. :ref:`tab_pff_discrete_vs_diffuse` presents a
side-by-side comparison of the energetic components in both formulations. As seen in the table,
the effective elastic strain energy in the phase-field formulation is modulated by the
degradation function $g(\phi)$, which accounts for the damage state of the material. The
fracture surface energy in the phase-field approach is represented as an integral over the
entire domain, involving a crack surface density function $\gamma(\phi, \nabla \phi)$. The
external energy term remains consistent between both formulations, as it depends solely on the
displacement field and external forces.

.. _tab_pff_discrete_vs_diffuse:
.. list-table:: Comparison between PFF (diffuse) and discrete (sharp) variational formulations.
   :header-rows: 1

   * -
     - **Diffuse (phase-field)**
     - **Discrete (sharp)**
   * - **Unknowns**
     - $\boldsymbol{u},\ \phi$
     - $\boldsymbol{u},\ \Gamma_{\text{sharp}}$
   * - **Elastic strain energy**
     - $\int_{\Omega} g(\phi)\,\big(\psi_a(\boldsymbol{\epsilon}(\boldsymbol{u}))+\psi_b(\boldsymbol{\epsilon}(\boldsymbol{u}))\big)\,\mathrm{d}\boldsymbol{x}$
     - $\int_{\Omega\setminus\Gamma_{\text{sharp}}} \psi(\boldsymbol{\epsilon}(\boldsymbol{u}))\,\mathrm{d}\boldsymbol{x}$
   * - **Fracture surface energy**
     - $G_c \int_{\Omega} \gamma(\phi,\nabla\phi)\,\mathrm{d}\boldsymbol{x}$
     - $G_c \int_{\Gamma_{\text{sharp}}} \mathrm{d}S$
   * - **External energy**
     - $\mathcal{E}_{\text{ext}}(\boldsymbol{u})$
     - $\mathcal{E}_{\text{ext}}(\boldsymbol{u})$

To provide a clear roadmap of the PFF method, it is useful to view it as a modular framework
rather than a monolithic theory. The formulation is constructed around distinct pillars that
collectively define a specific model and its numerical implementation:

.. tikz::
   :align: center

   \begin{tikzpicture}[
        node distance=0.8cm,
        process/.style={
            rectangle,
            draw=black,
            thick,
            fill=gray!5,
            text width=16em,
            text centered,
            rounded corners,
            minimum height=4em,
            inner sep=5pt,
            font=\small
        },
        process_wide/.style={
            process,
            text width=34em
        },
        process_blue/.style={
            process_wide,
            fill=blue!5
        },
        process_green/.style={
            process,
            fill=green!5
        },
        process_red/.style={
            process,
            fill=red!5
        },
        subtag/.style={
            rectangle,
            draw=black!40,
            dashed,
            fill=white,
            rounded corners=2pt,
            font=\scriptsize,
            inner sep=2pt,
            align=center
        },
        line/.style={
            draw,
            -latex,
            thick,
            rounded corners=5pt
        }
    ]

    % -------------------------------------------------
    % Row 1: Ingredients
    % -------------------------------------------------
    \node [process_green] (elastic) at (-3.75,0) {
        \textbf{1. Effective elastic strain energy}\\[0.3em]
        \textit{\scriptsize Degradation:}\\
        \tikz[baseline]\node[subtag]{Quad$\cdot$Cubic$\cdot$Quartic};\\[0.3em]
        \textit{\scriptsize Energy Split:}\\
        \tikz[baseline]\node[subtag]{Isotropic, Anisotropic (Spectral, Vol-Dev)};
    };

    \node [process_green] (surface) at (3.75,0) {
        \textbf{2. Fracture surface energy}\\[0.3em]
        \textit{\scriptsize Surface Functional:}\\
        \tikz[baseline]\node[subtag]{AT1 $\cdot$ \textbf{AT2} $\cdot$ Wu $\cdot$ DW};\\[0.3em]
        \textit{\scriptsize Model Parameter:}\\
        \tikz[baseline]\node[subtag,draw=red]{Critical energy release rate $G_c$};
    };

    % -------------------------------------------------
    % Main PFF block
    % -------------------------------------------------
    \node[
        process_blue,
        below=0.8cm of elastic,
        xshift=3.75cm
    ] (pff) {
        \large\textbf{PFF Functional}
    };

    % -------------------------------------------------
    % Irreversibility
    % -------------------------------------------------
    \node[
        process_wide,
        below=0.8cm of pff
    ] (irrev) {
        \textbf{3. Interval constraint and irreversibility}\\[0.3em]
        \tikz[baseline]\node[subtag]{History Field};
        \quad
        \tikz[baseline]\node[subtag]{Penalty Approach};

    };

    % -------------------------------------------------
    % FEM
    % -------------------------------------------------
    \node[
        process_wide,
        below=0.8cm of irrev
    ] (fem) {
        \textbf{4. FEM Discretization}\\[0.3em]
        \textit{\scriptsize Function Spaces}
    };

    % -------------------------------------------------
    % Solvers
    % -------------------------------------------------
    \node[
        process_red,
        below=1cm of fem,
        xshift=-8.75em,
        minimum height=3.5em
    ] (staggered) {
        \textbf{Staggered Schemes}
    };

    \node[
        process_red,
        below=1cm of fem,
        xshift=8.75em,
        minimum height=3.5em
    ] (monolithic) {
        \textbf{Monolithic Schemes}
    };

    \node[
        process_blue,
        below=1.8cm of $(staggered)!0.5!(monolithic)$,
        text width=34em,
        minimum height=4em
    ] (solver) {
        \textbf{Solution Algorithms}\\[0.3em]
        \textit{\scriptsize Newton-Raphson \& Linear solvers}
    };

    % -------------------------------------------------
    % Main arrows
    % -------------------------------------------------
    \draw[line] (elastic.south) -- (pff.north-|elastic.south);
    \draw[line] (surface.south) -- (pff.north-|surface.south);
    \draw[line] (pff) -- (irrev);
    \draw[line] (irrev) -- (fem);


    \draw[line]
    (fem.south-|staggered.north)
    -- (staggered.north);
    \draw[line]
    (fem.south-|monolithic.north)
    -- (monolithic.north);
    \draw[line]
    (staggered.south)
    -- (solver.north-|staggered.south);
    \draw[line]
    (monolithic.south)
    -- (solver.north-|monolithic.south);

    % -------------------------------------------------
    % AT2 branch
    % -------------------------------------------------
    \draw[
        line,
        dashed,
        very thick,
        blue
    ]
    ([yshift=5pt]surface.east)
    -- ++(2cm,0)
    |- (fem.east);

    \draw[
        line,
        dashed,
        very thick,
        blue
    ]
    ([yshift=-5pt]fem.east)
    -- ++(2cm,0)
    |- ([yshift=5pt]monolithic.east);

    \draw[
        line,
        dashed,
        very thick,
        blue
    ]
    ([yshift=-5pt]monolithic.east)
    -- ++(2cm,0)
    |- (solver.east);

    \node[
        text=blue,
        font=\scriptsize,
        fill=white
    ]
    at ([xshift=2cm,yshift=2em]pff.east)
    {AT2};

    % -------------------------------------------------
    % AT1/Wu branch
    % -------------------------------------------------
    \draw[
        line,
        dashed,
        very thick,
        green!60!black
    ]
    ([yshift=-5pt]surface.east)
    -- ++(1cm,0)
    |- ([yshift=5pt]irrev.east);

    \draw[
        line,
        dashed,
        very thick,
        green!60!black
    ]
    ([yshift=-5pt]irrev.east)
    -- ++(1cm,0)
    |- ([yshift=5pt]fem.east);

    \draw[
        line,
        dashed,
        very thick,
        green!60!black
    ]
    ([yshift=-10pt]fem.east)
    -- ++(1cm,0)
    |- ([yshift=10pt]monolithic.east);

    \draw[
        line,
        dashed,
        very thick,
        green!60!black
    ]
    ([yshift=-10pt]monolithic.east)
    -- ++(1cm,0)
    |- ([yshift=5pt]solver.east);

    \node[
        text=green!60!black,
        font=\scriptsize,
        fill=white
    ]
    at ([xshift=1cm]pff.east)
    {AT1, Wu};

   \end{tikzpicture}

   PFF modular framework: overview of the key components and numerical strategies. The dashed lines highlight the AT2 branch (blue) and the AT1/Wu branches (green), which can be analyzed independently by considering the CSDF separately from the full PFF problem.

   
The functional :eq:`eq_pff_variational_formulation` is assembled from four independent
building blocks. Each may be chosen separately, giving rise to a wide range of model
combinations:

.. grid:: 2
   :gutter: 3

   .. grid-item-card::
      :class-header: sd-bg-success sd-text-white sd-font-weight-bold
      :class-card: sd-border-2 sd-rounded-3

      Component 1 — Degradation Function :math:`g(\phi)`
      ^^^

      Quantifies the progressive loss of material stiffness as damage evolves.
      Couples the elastic response to the phase-field variable :math:`\phi`.
      Must satisfy:

      .. math::

         g(0)=1,\quad g(1)=0,\quad g'(1)=0.

      - **Quadratic:** :math:`g(\phi)=(1-\phi)^2` *(default)*
      - Cubic, Quartic: alternative softening behaviors.

      *Details* :math:`\rightarrow` :ref:`sec_pff_degradation_functions`

   .. grid-item-card::
      :class-header: sd-bg-success sd-text-white sd-font-weight-bold
      :class-card: sd-border-2 sd-rounded-3

      Component 2 — Energy Split :math:`\psi = \psi_a + \psi_b`
      ^^^

      Decomposes the strain energy density into:

      - :math:`\psi_a` — **active** (degraded by :math:`g(\phi)`): drives crack growth.
      - :math:`\psi_b` — **inactive** (undegraded): resists damage under compression.

      Options: **Isotropic**, Spectral (Miehe 2010), Vol-Dev (Amor 2009).

      *Details* :math:`\rightarrow` :ref:`sec_pff_iso_aniso_models`

   .. grid-item-card::
      :class-header: sd-bg-info sd-text-white sd-font-weight-bold
      :class-card: sd-border-2 sd-rounded-3

      Component 3 — Crack Surface Density :math:`\gamma(\phi,\nabla\phi)`
      ^^^

      Regularizes the sharp crack into a diffuse band of width :math:`\sim l`.
      Approximates the sharp crack surface energy via :math:`\Gamma`-convergence:

      .. math::

         G_c\!\int_\Omega \gamma\,\mathrm{d}\boldsymbol{x}
         \;\xrightarrow{l\to 0}\;
         G_c\!\int_{\Gamma_\text{sharp}}\!\mathrm{d}S.

      Common forms: **AT2**, AT1, Wu, DW.

      *Details* :math:`\rightarrow` :ref:`theory_phase_field`

   .. grid-item-card::
      :class-header: sd-bg-warning sd-font-weight-bold
      :class-card: sd-border-2 sd-rounded-3

      Component 4 — Irreversibility & Bounds
      ^^^

      Two physical constraints must be strictly imposed:

      - **No crack healing:** :math:`\dot{\phi} \geq 0`
      - **Physical admissibility:** :math:`\phi \in [0,\,1]`

      Enforcement: **history-field variable**, penalty approach, LVPP.

      *Details* :math:`\rightarrow` :ref:`sec_pff_numerical_strategies`


.. _sec_pff_effective_elastic_energy:

Effective elastic strain energy
-------------------------------

The effective elastic strain energy describes the stored energy in the body, accounting for
stiffness degradation due to damage. As presented in :eq:`eq_pff_variational_formulation`, the
effective elastic strain energy is defined as:

.. math::
   :label: eq_pff_effective_elastic_energy

   \Psi(\boldsymbol \epsilon(\boldsymbol u), \phi) = \int_\Omega \Big( g(\phi) \underbrace{\psi_a(\boldsymbol{\epsilon}(\boldsymbol{u})) + \psi_b(\boldsymbol{\epsilon}(\boldsymbol{u}))}_{\psi(\boldsymbol{\epsilon}(\boldsymbol{u}))}\Big) \, \mathrm{d}\boldsymbol{x}.

Here, two main aspects are significant: first, the degradation function $g(\phi)$ modulates
the strain energy density $\psi(\boldsymbol{\epsilon})$; second, the energy is decomposed into
an active part $\psi_a$, which is affected by the degradation function $g(\phi)$, and an
inactive component $\psi_b$, which is not. The strain energy density
:eq:`eq_pff_strain_energy_density` is thus split as
$\psi(\boldsymbol{\epsilon}) = \psi_a(\boldsymbol{\epsilon}) + \psi_b(\boldsymbol{\epsilon})$.

Several formulations for both the degradation function and the energy split exist in the
literature; they are each addressed in the subsections below.


.. _sec_pff_degradation_functions:

1) Degradation functions
~~~~~~~~~~~~~~~~~~~~~~~~

The degradation function $g(\phi)$, as explained in detail in
:footcite:t:`phase_field_modeling_of_fracture`, plays a fundamental role in PFF models by
quantifying the loss of material stiffness as damage accumulates. By modulating the elastic
strain energy density, it effectively creates the coupling between the phase-field variable
and the mechanical response.

To ensure a physically consistent representation of the damage process, $g(\phi)$ is required
to satisfy the following properties, as outlined in :footcite:t:`phase_field_Miehe2010`:

.. math::

   g(0) = 1, \quad g(1) = 0, \quad g'(1) = 0.

The first two conditions correspond to the states of intact material and complete failure,
respectively. Thus, when damage is absent ($\phi=0$), the degradation function equals 1 and
does not affect the strain energy. Conversely, if damage is fully developed ($\phi=1$), the
degradation function equals 0, effectively eliminating the energy contribution. For
intermediate values of the phase-field ($0 \leq \phi \leq 1$), the degradation function scales the
energy by a factor between 0 and 1. The condition $g'(1)=0$ ensures that the driving force
for damage vanishes when the material is fully broken, preventing unphysical evolution in the
fully damaged state.

While the quadratic form :eq:`eq_pff_degradation_quadratic` is widely adopted due to its
simplicity and robustness, as shown in :footcite:t:`phase_field_Bourdin2000`, other functional
forms have been proposed to tailor the softening behavior or to approximate specific cohesive
laws. The standard quadratic degradation function is adopted as the primary choice for all
analyses. However, for completeness, other
common alternatives, including cubic :footcite:t:`phase_field_degradation_function_cubic` and
quartic :footcite:t:`phase_field_degradation_functions` polynomials, are also presented here.
:ref:`tab_pff_degradation_functions` summarizes these common degradation functions along
with their derivatives.

.. note::

   The **quadratic degradation function** :math:`g(\phi)=(1-\phi)^2` is used as the default
   throughout this documentation. It is computationally efficient, satisfies all required
   mathematical properties.

.. _tab_pff_degradation_functions:
.. list-table:: Summary of available degradation functions $g(\phi)$ and their derivatives.
   :header-rows: 1

   * - **Degradation Type**
     - **Function** $g(\phi)$
     - **Derivative** $g'(\phi)$
   * - Quadratic
     - $(1 - \phi)^2$
     - $-2(1 - \phi)$
   * - Cubic
     - $3(1-\phi)^2 - 2(1-\phi)^3$
     - $-6\phi(1 - \phi)$
   * - Quartic
     - $4(1-\phi)^3 - 3(1-\phi)^4$
     - $-12(1-\phi)^2 + 12(1-\phi)^3$

.. grid:: 2

   .. grid-item::

      .. figure:: Degradation_functions/results_degradation_functions/phi_vs_g_phi.png
         :width: 100%

         Degradation function :math:`g(\phi)` versus phase-field :math:`\phi`.

   .. grid-item::

      .. figure:: Degradation_functions/results_degradation_functions/phi_vs_gp_phi.png
         :width: 100%

         Derivative of degradation function :math:`g'(\phi)` versus phase-field :math:`\phi`.


Having discussed the degradation function, the decomposition of the strain energy density into
active and inactive parts is analyzed, which is crucial for controlling the conditions under
which damage evolves.


.. _sec_pff_iso_aniso_models:

2) Isotropic and anisotropic models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A central requirement in PFF modeling is to ensure that damage evolution is consistent with
physical observations. Specifically, cracks should propagate under certain conditions; for
example, they should propagate under tension but not under compression. This is achieved by
applying the degradation function exclusively to the energy contribution that can generate
cracks, $\psi_a$, while leaving the inactive compressive component, $\psi_b$, undegraded.
Various strategies have been developed to decompose the strain energy density into these
contributions; these are collectively referred to as anisotropic formulations. Conversely,
formulations that assume the degradation function acts on the total strain energy density are
classified as isotropic.

.. grid:: 2

   .. grid-item::

      .. figure:: Isotropic_anisotropic/isotropic.png
         :width: 100%

         Isotropic model

   .. grid-item::

      .. figure:: Isotropic_anisotropic/anisotropic.png
         :width: 100%

         Anisotropic model


.. _sec_pff_isotropic_model:

Isotropic model
^^^^^^^^^^^^^^^

The simplest approach is the isotropic model, where degradation affects the entire strain
energy density given in :eq:`eq_pff_strain_energy_density`. In this framework, the entire
strain energy density :eq:`eq_pff_strain_energy_density` is subject to degradation, so for
the general representation presented in :eq:`eq_pff_variational_formulation`, the following
terms are considered: $\psi_a = \psi(\boldsymbol{\epsilon})$ and $\psi_b = 0$. While
computationally efficient, this model does not distinguish between tensile and compressive
stress states, often leading to unrealistic crack patterns in compression-dominated regimes.
To address the limitations of the isotropic model, anisotropic formulations aim to handle
this aspect.


.. _sec_pff_anisotropic_models:

Anisotropic models
^^^^^^^^^^^^^^^^^^

To accurately capture fracture behavior under complex loading conditions, anisotropic models
decompose the strain energy density into active (energy capable of driving fracture) and
inactive (energy not capable of driving fracture) parts. These energy decompositions are based
on two main considerations: 1) a kinematic decomposition of the strain tensor, and 2) the
formulation of energy components based on the strain tensor as well as the split strain
components. Two widely adopted methods in the literature are the spectral decomposition and
the volumetric-deviatoric decomposition.


Spectral decomposition
""""""""""""""""""""""

Proposed in :footcite:t:`phase_field_Miehe2010`, this method is based on the spectral
decomposition of the strain tensor. The strain tensor $\boldsymbol{\epsilon}$ is diagonalized
as $\boldsymbol{\epsilon} = \sum_{i=1}^{3} \epsilon^i \boldsymbol{n}^i \otimes \boldsymbol{n}^i$,
where $\epsilon^i$ are the principal strains and $\boldsymbol{n}^i$ are the corresponding
principal directions. Using Macaulay brackets,

.. math::
   :label: eq_pff_macaulay_brackets

   \langle x \rangle_\pm = \frac{x \pm |x|}{2},

the strain tensor is split into positive and negative contributions:

.. math::
   :label: eq_pff_spectral_strain

   \boldsymbol{\epsilon}_\pm := \sum_{i=1}^{3} \langle \epsilon^i \rangle_\pm \boldsymbol{n}^i \otimes \boldsymbol{n}^i.

Accordingly, the strain energy density is decomposed into tensile ($\psi_a$) and compressive
($\psi_b$) parts:

.. math::

   \psi_a^{\text{spectral}}(\boldsymbol{\epsilon}) &= \frac{1}{2}\lambda{\langle \text{tr}(\boldsymbol{\epsilon}) \rangle_+}^2 + \mu \text{tr}(\boldsymbol{\epsilon}_+^2), \\
   \psi_b^{\text{spectral}}(\boldsymbol{\epsilon}) &= \frac{1}{2}\lambda{\langle \text{tr}(\boldsymbol{\epsilon}) \rangle_-}^2 + \mu \text{tr}(\boldsymbol{\epsilon}_-^2).

This formulation provides a physically rigorous separation of tensile and compressive modes.
Note that the term multiplying the Lamé parameter $\lambda$ contributes entirely to either the
active or the inactive part, depending on the sign of the strain tensor trace. In contrast,
the term multiplying the shear modulus $\mu$ is split based on the positive and negative
spectral components of the strain tensor, distributing energy between the degraded and
undegraded parts.


Volumetric-Deviatoric decomposition
"""""""""""""""""""""""""""""""""""

Proposed in :footcite:t:`phase_field_Amor2009`, this method is based on the volumetric (or
spherical) and deviatoric parts of the strain tensor. The strain tensor is decomposed as:

.. math::

   \boldsymbol{\epsilon} &= \boldsymbol{\epsilon}^S + \boldsymbol{\epsilon}^D, \\
   \boldsymbol{\epsilon}^S &= \frac{1}{m} \text{tr}(\boldsymbol{\epsilon}) \boldsymbol{I}, \\
   \boldsymbol{\epsilon}^D &= \boldsymbol{\epsilon} - \boldsymbol{\epsilon}^S,

where $m$ is the spatial dimension (1, 2, or 3). The energy split assumes that cracks form
due to positive volumetric expansion and deviatoric deformations (shear). The energy
components are defined as:

.. math::

   \psi_a^{\text{vol-dev}}(\boldsymbol{\epsilon}) &= \frac{1}{2} \kappa {\langle \text{tr}(\boldsymbol{\epsilon}) \rangle_+}^2 + \mu \text{tr}({\boldsymbol{\epsilon}^D}^2), \\
   \psi_b^{\text{vol-dev}}(\boldsymbol{\epsilon}) &= \frac{1}{2} \kappa {\langle \text{tr}(\boldsymbol{\epsilon}) \rangle_-}^2,

where $\kappa = \lambda + \frac{2}{m}\mu$ is the bulk modulus.

Here again, the term multiplying the bulk modulus $\kappa$ contributes entirely to either the
active or the inactive part, depending on the sign of the trace of the strain tensor. In
contrast, the energy term multiplying the shear modulus $\mu$ is fully degraded, meaning it
belongs entirely to the active energy component.

It is crucial to emphasize that while the nomenclature of anisotropic models typically refers
to the kinematic decomposition of the strain tensor (the first ingredient), the physical
behavior is ultimately determined by the definition of the active and inactive energy densities
based on these tensor components (the second ingredient). Consequently, distinct models may
share the same strain tensor decomposition but employ different energy splits. To illustrate
this distinction, consider an alternative formulation based on the volumetric-deviatoric strain
tensor split. Unlike the standard model presented above, this alternative defines the active
energy solely through the deviatoric component, leaving the entire volumetric energy
undegraded:

.. math::

   \psi_a^{\text{alternative vol-dev}}(\boldsymbol{\epsilon}) &= \mu \text{tr}({\boldsymbol{\epsilon}^D}^2), \\
   \psi_b^{\text{alternative vol-dev}}(\boldsymbol{\epsilon}) &= \frac{1}{2} \kappa \,\text{tr}(\boldsymbol{\epsilon})^2.


Summary of energy decompositions
""""""""""""""""""""""""""""""""

As seen before, both models rely on two main components: a kinematic decomposition of the
strain tensor and a subsequent decomposition of the strain energy density. First, the strain
tensor decompositions are outlined in :numref:`tab_pff_strain_tensor_decomposition`.
Subsequently, :numref:`tab_pff_energy_decompositions` summarizes the split components for
both isotropic and anisotropic models. This includes the strain energy densities ($\psi_a$ and
$\psi_b$), the active and inactive stress tensors defined as the first derivatives of the
energy with respect to the strain
($\boldsymbol{\sigma}_a = \frac{\partial \psi_a}{\partial \boldsymbol{\epsilon}}$ and
$\boldsymbol{\sigma}_b = \frac{\partial \psi_b}{\partial \boldsymbol{\epsilon}}$), and the
material tangent stiffness tensors computed as the second derivatives
($\mathbb{C}_a = \frac{\partial^2 \psi_a}{\partial \boldsymbol{\epsilon}^2}$ and
$\mathbb{C}_b = \frac{\partial^2 \psi_b}{\partial \boldsymbol{\epsilon}^2}$). The split stress
tensors will appear directly in the weak form of the governing equations of the PFF problem,
while the material tangent stiffness tensors will be used to define the numerical framework
and linearize the problem within a Newton-Raphson solution scheme.

.. _tab_pff_strain_tensor_decomposition:
.. list-table:: Strain tensor decomposition methods.
   :header-rows: 1

   * - **Spectral**
     - **Volumetric-Deviatoric**
   * - $\boldsymbol{\epsilon}_+=\sum_{i=1}^{3} \langle \epsilon^i \rangle^+ \boldsymbol{n}^i \otimes \boldsymbol{n}^i$
     - $\boldsymbol{\epsilon}_S=\frac{1}{m} \text{tr}(\boldsymbol{\epsilon}) \boldsymbol{I}$
   * - $\boldsymbol{\epsilon}_-=\sum_{i=1}^{3} \langle \epsilon^i \rangle^- \boldsymbol{n}^i \otimes \boldsymbol{n}^i$
     - $\boldsymbol{\epsilon}_D=\boldsymbol{\epsilon} - \frac{1}{m} \text{tr}(\boldsymbol{\epsilon}) \boldsymbol{I}$

.. _tab_pff_energy_decompositions:
.. list-table:: Energy densities, stress tensors, and tangent tensors ($\mathbb{C}$) decompositions for isotropic and anisotropic models. Here, $\mathbb{J} = \boldsymbol{I} \otimes \boldsymbol{I}$, $\mathbb{I}^{\text{sym}}$ is the symmetric fourth-order identity, $\mathbb{Q} = \mathbb{I}^{\text{sym}} - \frac{1}{m}\mathbb{J}$ is the deviatoric projector, $\mathbb{P}_\pm$ are spectral projectors defined in :footcite:t:`phase_field_spectral_decomposition_projection`, and $H(\cdot)$ is the Heaviside step function.
   :header-rows: 1

   * -
     - **Isotropic**
     - **Spectral**
     - **Volumetric-Deviatoric**
   * - $\psi_{a}$
     - $\frac{1}{2}\lambda{(\text{tr}(\boldsymbol{\epsilon}))}^2+\mu \text{tr}(\boldsymbol{\epsilon}^2)$
     - $\frac{1}{2}\lambda{\langle \text{tr}(\boldsymbol{\epsilon})\rangle_+}^2+\mu \text{tr}({\boldsymbol{\epsilon}_+}^2)$
     - $\frac{1}{2}\kappa{\langle \text{tr}(\boldsymbol{\epsilon})\rangle_+}^2+\mu \text{tr}({\boldsymbol{\epsilon}^D}^2)$
   * - $\psi_{b}$
     - $0$
     - $\frac{1}{2}\lambda{\langle \text{tr}(\boldsymbol{\epsilon})\rangle_-}^2+\mu \text{tr}({\boldsymbol{\epsilon}_-}^2)$
     - $\frac{1}{2}\kappa{\langle \text{tr}(\boldsymbol{\epsilon})\rangle_-}^2$
   * - $\boldsymbol{\sigma}_{a}$
     - $\lambda\,\text{tr}(\boldsymbol{\epsilon})\,\boldsymbol{I}+ 2\mu\,\boldsymbol{\epsilon}$
     - $\lambda{\langle \text{tr}(\boldsymbol{\epsilon})\rangle_+}\boldsymbol{I}+ 2\mu\,\boldsymbol{\epsilon}_+$
     - $\kappa{\langle \text{tr}(\boldsymbol{\epsilon})\rangle_+}\boldsymbol{I}+ 2\mu\,\boldsymbol{\epsilon}^D$
   * - $\boldsymbol{\sigma}_{b}$
     - $\boldsymbol{0}$
     - $\lambda{\langle \text{tr}(\boldsymbol{\epsilon})\rangle_-}\boldsymbol{I}+ 2\mu\,\boldsymbol{\epsilon}_-$
     - $\kappa{\langle \text{tr}(\boldsymbol{\epsilon})\rangle_-}\boldsymbol{I}$
   * - $\mathbb{C}_{a}$
     - $\lambda \mathbb{J} + 2\mu \mathbb{I}^{\text{sym}}$
     - $\lambda H(\text{tr}(\boldsymbol{\epsilon})) \mathbb{J} + 2\mu \mathbb{P}_+(\boldsymbol{\epsilon})$
     - $\kappa H(\text{tr}(\boldsymbol{\epsilon})) \mathbb{J} + 2\mu \mathbb{Q}$
   * - $\mathbb{C}_{b}$
     - $\boldsymbol{0}$
     - $\lambda H(-\text{tr}(\boldsymbol{\epsilon})) \mathbb{J} + 2\mu \mathbb{P}_-(\boldsymbol{\epsilon})$
     - $\kappa H(-\text{tr}(\boldsymbol{\epsilon})) \mathbb{J}$

Other anisotropic formulations have also been proposed in the literature. In particular,
directional splits :footcite:t:`phase_field_steinke` introduce a crack-oriented decomposition
of the driving energy based on a local crack orientation vector, enabling a consistent
distinction between tensile and shear contributions and allowing for mode-dependent fracture
behavior. Additionally, recent multi-cohesive anisotropic models
:footcite:t:`phase_field_anisotropic_fajardo` allow different cohesive lengths along the
material directions, enabling independent control of the critical stresses governing crack
nucleation in each direction. These formulations are relevant because they provide additional
flexibility to represent crack nucleation and propagation under anisotropic conditions.

3) Fracture surface energy
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _sec_pff_csdf_general:

The fracture surface energy, as introduced in :eq:`eq_pff_variational_formulation`, is defined as the product of the critical energy release rate, $G_c$, and the Crack Surface Density Functional (CSDF), $\Gamma(\phi)$. Since $G_c$ is a material constant, the primary component governing crack regularization is $\Gamma(\phi)$, which approximates the sharp crack surface through a continuous diffuse representation.


General Formulation of the CSDF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In its general form, the Crack Surface Density Functional is defined as:

.. math::
   :label: eq_pff_csdf_general_functional

   \Gamma(\phi) = \int_\Omega \gamma(\phi,\nabla \phi) \, \mathrm{d}\boldsymbol{x},

where $\gamma(\phi,\nabla \phi)$ is the crack surface density function per unit volume. A unified geometric formulation :footcite:p:`phase_field_Wu, phase_field_modeling_of_fracture` is given by:

.. math::
   :label: eq_pff_csdf_general_function

   \gamma(\phi,\nabla \phi) = \frac{1}{c_0} \left( \frac{\alpha(\phi)}{l} + l |\nabla \phi|^2 \right),

where $l$ is the length-scale parameter governing the width of the diffuse crack band, $\alpha(\phi)$ is the geometric crack function satisfying $\alpha(0) = 0$ (intact) and $\alpha(1) = 1$ (fully damaged), and $c_0$ is a normalization constant:

.. math::
   :label: eq_pff_csdf_c0_definition

   c_0 := 4 \int_0^1 \sqrt{\alpha(\eta)} \, \mathrm{d}\eta.

This normalization ensures that $\Gamma(\phi)$ recovers the exact sharp crack surface area as $l \to 0$ ($\Gamma$-convergence).

The CSDF can be decomposed into a local phase-field energy $\Gamma_\phi$ and a gradient energy $\Gamma_{\nabla \phi}$:

.. math::
   :label: eq_pff_csdf_decomposed_energies

   \Gamma_\phi(\phi) := \frac{1}{c_0} \int_\Omega \frac{\alpha(\phi)}{l} \, \mathrm{d}\boldsymbol{x}, \quad
   \Gamma_{\nabla \phi}(\phi) := \frac{1}{c_0} \int_\Omega l |\nabla \phi|^2 \, \mathrm{d}\boldsymbol{x}, \quad \text{such that} \quad
   \Gamma(\phi) = \Gamma_\phi(\phi) + \Gamma_{\nabla \phi}(\phi).

Enforcing stationarity ($\delta \Gamma = 0$) leads to the variational weak form:

.. math::
   :label: eq_pff_csdf_variational_weak_form

   \Gamma'(\phi) = \int_\Omega \frac{1}{c_0} \left( \frac{\alpha'(\phi)}{l} \delta\phi + 2l \nabla\phi \cdot \nabla\delta\phi \right) \mathrm{d}\boldsymbol{x} = 0,

with corresponding strong form PDE and Neumann boundary condition:

.. math::
   :label: eq_pff_csdf_strong_form

   \frac{1}{c_0} \left( \frac{\alpha'(\phi)}{l} - 2 l \Delta \phi \right) = 0 \quad \text{in } \Omega, \quad \text{with} \quad \nabla \phi \cdot \boldsymbol{n} = 0 \quad \text{on } \partial \Omega.


Specific Regularization Models (AT2, AT1, Wu, Double-Well)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different choices of the geometric function $\alpha(\phi)$ yield distinct regularization models:

1. **AT2 Model** (Ambrosio–Tortorelli type 2) :footcite:p:`phase_field_Bourdin2000`: $\alpha(\phi) = \phi^2$, $\alpha'(\phi) = 2\phi$, $c_0 = 2$.
2. **AT1 Model** (Ambrosio–Tortorelli type 1) :footcite:p:`introduction_ambrosio_tortorelli`: $\alpha(\phi) = \phi$, $\alpha'(\phi) = 1$, $c_0 = 8/3$.
3. **Wu Model** :footcite:p:`phase_field_Wu`: $\alpha(\phi) = 2\phi - \phi^2$, $\alpha'(\phi) = 2 - 2\phi$, $c_0 = \pi$.
4. **Double-Well Potential Model**: $\alpha(\phi) = 16\phi^2(1-\phi)^2$, $\alpha'(\phi) = 32\phi(1-\phi)(1-2\phi)$, $c_0 = 8/3$.

.. _tab_pff_geometric_functions:
.. list-table:: Geometric crack functions $\alpha(\phi)$, derivatives $\alpha'(\phi)$, and normalization constants $c_0$.
   :header-rows: 1

   * - **Model**
     - **Geometric Function** $\alpha(\phi)$
     - **Derivative** $\alpha'(\phi)$
     - **Constant** $c_0$
   * - **AT2**
     - $\phi^2$
     - $2\phi$
     - $2$
   * - **AT1**
     - $\phi$
     - $1$
     - $8/3 \approx 2.667$
   * - **Wu**
     - $2\phi - \phi^2$
     - $2 - 2\phi$
     - $\pi \approx 3.1416$
   * - **Double-Well**
     - $16\phi^2 (1-\phi)^2$
     - $32\phi(1-\phi)(1-2\phi)$
     - $8/3 \approx 2.667$

.. note::
   Note that this part, relative to the regularization of the discontinuity, can be analyzed as an isolated problem, defined as the crack surface density functional. IIt is recommended to analyze and understand this isolated problem prior to handling the coupled phase-field fracture problem. All the details of this isolated problem are presented in the :ref:`theory_phase_field` section.




.. _sec_pff_governing_equations:

Governing equations for the PFF problem
-----------------------------------------

The governing equations are derived by enforcing the stationarity of the total potential
energy functional :eq:`eq_pff_variational_formulation` with respect to independent variations
of the displacement field $\boldsymbol{u}$ and the phase-field $\phi$. This variational
procedure yields the following coupled weak forms:

.. math::
   :label: eq_pff_weak_forms

   \mathcal{E}'_{\boldsymbol{u}}(\boldsymbol{u}, \phi) &:= \int_\Omega \big[g(\phi)\boldsymbol{\sigma}_a(\boldsymbol{\epsilon}(\boldsymbol{u})) + \boldsymbol{\sigma}_b(\boldsymbol{\epsilon}(\boldsymbol{u}))\big] : \boldsymbol{\epsilon}(\delta \boldsymbol{u}) \, \mathrm{d}\boldsymbol{x} - \int_\Omega \boldsymbol{f} \cdot \delta \boldsymbol{u} \, \mathrm{d}\boldsymbol{x} - \int_{\partial \Omega_t} \boldsymbol{t} \cdot \delta \boldsymbol{u} \, \mathrm{d}S = \boldsymbol{0}, \\
   \mathcal{E}'_{\phi}(\boldsymbol{u}, \phi) &:= \int_\Omega g'(\phi) \psi_a(\boldsymbol{\epsilon}) \delta \phi \, \mathrm{d}\boldsymbol{x} + \frac{G_c}{c_0} \int_\Omega \left(\frac{\alpha'(\phi)}{l} \delta\phi + 2l \nabla\phi \cdot \nabla\delta\phi \right) \, \mathrm{d}\boldsymbol{x} = 0,

where $\delta\boldsymbol{u}$ and $\delta\phi$ denote admissible variations, $g'(\phi)$ is the
derivative of the degradation function with respect to $\phi$ detailed in
:numref:`tab_pff_degradation_functions`, $\boldsymbol{\sigma}_a = \frac{\partial \psi_a}{\partial \boldsymbol{\epsilon}}$
and $\boldsymbol{\sigma}_b = \frac{\partial \psi_b}{\partial \boldsymbol{\epsilon}}$ are the
active and inactive stress tensors corresponding to the energy split, as detailed in
:numref:`tab_pff_energy_decompositions`.

.. note::

   The weak form :eq:`eq_pff_weak_forms` integrates all three modular ingredients:

   - **Component 1 (Degradation function):** :math:`g(\phi)` and :math:`g'(\phi)` appear
     explicitly in both the mechanical and phase-field equations.
   - **Component 2 (Energy split):** The split stress tensors :math:`\boldsymbol{\sigma}_a`
     and :math:`\boldsymbol{\sigma}_b` enter the mechanical equilibrium equation.
   - **Component 3 (Crack surface density):** The function :math:`\alpha(\phi)` and the
     normalization constant :math:`c_0` from the CSDF choice govern the regularization term
     in the phase-field equation.

The corresponding strong form equations are given by:

.. math::
   :label: eq_pff_strong_forms

   \nabla \cdot \left[g(\phi) \boldsymbol{\sigma}_a(\boldsymbol{\epsilon}(\boldsymbol{u})) + \boldsymbol{\sigma}_b(\boldsymbol{\epsilon}(\boldsymbol{u}))\right] + \boldsymbol{f} &= \boldsymbol{0} \quad \text{in } \Omega, \\
   g'(\phi)\, \psi_a(\boldsymbol{\epsilon}(\boldsymbol{u})) + \frac{G_c}{c_0} \left( \frac{\alpha'(\phi)}{l} - 2 l \Delta \phi \right) &= 0 \quad \text{in } \Omega, \\
   \left[g(\phi) \boldsymbol{\sigma}_a(\boldsymbol{\epsilon}(\boldsymbol{u})) + \boldsymbol{\sigma}_b(\boldsymbol{\epsilon}(\boldsymbol{u}))\right] \cdot \boldsymbol{n} &= \boldsymbol{t} \quad \text{on } \partial \Omega_{\boldsymbol{t}}, \\
   \nabla \phi \cdot \boldsymbol{n} &= \boldsymbol{0} \quad \text{on } \partial \Omega.

Solving the problem defined by these governing equations requires addressing several critical
aspects. First, the physical constraints---specifically the boundedness of the phase-field
variable and the irreversibility of crack growth---must be enforced. Common strategies include
penalty methods :footcite:t:`phase_field_Gerasimov`, which convert the inequality problem into
an equality problem via a penalization term, and history variable approaches
:footcite:t:`phase_field_Miehe2010`, which ensure irreversibility by driving the phase-field
evolution with a non-decreasing energy quantity.

Second, once the approximated model is defined, the PDEs are solved numerically within a
finite element framework. Due to the non-convexity of the total energy functional, the
resulting complex coupled system requires robust numerical strategies. In the literature,
solution methods are generally categorized into staggered (operator-split) and monolithic
approaches.



.. _sec_pff_approximated_continuous_models:

Approximated continuous models: irreversibility and boundedness
---------------------------------------------------------------

To ensure the thermodynamic consistency of the fracture process, the irreversibility condition---preventing crack healing ($\dot{\phi} \ge 0$)---must be strictly enforced. Additionally, the phase-field variable $\phi$ must remain within the physical bounds $[0, 1]$.

Solving the governing equations with these inequality constraints transforms the original constrained optimization problem into an approximated model suitable for numerical implementation. Two of the most common approaches are penalty formulations and history field methods.

.. tikz:: Overview of the numerical solution strategy.
   :align: center

   \begin{tikzpicture}[
        process/.style={
            rectangle, 
            draw=black, 
            thick, 
            fill=gray!5,
            text width=14em, 
            text centered, 
            rounded corners, 
            minimum height=3em,
            inner sep=5pt,
            font=\small
        },
        process_wide/.style={
            process,
            text width=24em
        },
        line/.style={
            draw, 
            -latex, 
            thick,
            rounded corners=5pt    
        }
   ]

        \node [process_wide] (p1) at (0, 0) {\textbf{Variational Problem} \\ with Inequality Constraints};
        \node [process] (p2) at (0, -2.2) {\textbf{Unconstrained Nonlinear} \\ System of PDEs $(\phi, \boldsymbol{u})$};
        \node [process] (p3) at (0, -4.4) {\textbf{Discretization} \\ Galerkin FEM $(\phi_h, \boldsymbol{u}_h)$};
        \node [process] (p4) at (3.5, -6.6) {\textbf{Staggered Schemes}};
        \node [process] (p4_mono) at (-3.5, -6.6) {\textbf{Monolithic Schemes}};
        \node [process_wide] (p5) at (0, -8.8) {\textbf{Linear Solver} \\ Newton-Raphson};

        \draw [line] (p1) -- (p2);
        \draw [line] (p2) -- (p3);
        \draw [line] (p3) -- (p4);
        \draw [line] (p3) -- (p4_mono);
        \draw [line] (p4) -- (p5);
        \draw [line] (p4_mono) -- (p5);

   \end{tikzpicture}


.. _sec_pff_penalty_model:

Penalty formulations
~~~~~~~~~~~~~~~~~~~~

The penalty method transforms a constrained optimization problem into an unconstrained one by augmenting the potential energy functional with a penalization term scaled by a penalty parameter $\rho > 0$ :footcite:t:`phase_field_Gerasimov`.

For the full PFF problem, the irreversibility condition requires that damage does not decrease over time, i.e., $\phi \ge \phi_{\text{prev}}$. Enforcing this condition implicitly satisfies the non-negativity constraint ($\phi \ge 0$), provided that the initial state is non-negative ($\phi_{\text{prev}} \ge 0$). Taking $\phi_{\text{prev}}$ from the previous converged step as the reference, the total energy functional is defined by adding the penalization term to :eq:`eq_pff_variational_formulation`:

.. math::
   :label: eq_pff_penalty_functional

   \mathcal{E}_\text{penalty}(\boldsymbol{u}, \phi) := \mathcal{E}(\boldsymbol{u}, \phi) + \frac{\rho}{2} \int_\Omega \langle \phi - \phi_{\text{prev}} \rangle_-^2 \, \mathrm{d}\boldsymbol{x}.

Minimizing this functional with respect to the independent fields $\boldsymbol{u}$ and $\phi$ yields the coupled system of weak equations:

.. math::
   :label: eq_pff_penalty_weak_forms

   {\mathcal{E}_\text{penalty}}'_{\boldsymbol{u}}(\boldsymbol{u}, \phi) &:= \mathcal{E}'_{\boldsymbol{u}}(\boldsymbol{u}, \phi) = \boldsymbol{0}, \\
   {\mathcal{E}_\text{penalty}}'_{\phi}(\boldsymbol{u}, \phi) &:= \mathcal{E}'_{\phi}(\boldsymbol{u}, \phi) + \rho \int_\Omega \langle \phi - \phi_{\text{prev}} \rangle_- \, \delta\phi \, \mathrm{d}\boldsymbol{x} = 0.

Because the penalty term depends exclusively on $\phi$, the equilibrium equation for the displacement field :eq:`eq_pff_weak_forms` remains unaffected by this modification. Consequently, the modified phase-field equation is obtained by adding the variation of the penalty term to :eq:`eq_pff_weak_forms`.


.. _sec_pff_history_field_method:

History field method
~~~~~~~~~~~~~~~~~~~~

Another widely used alternative is the history field method proposed by :footcite:t:`phase_field_Miehe2010`. This approach introduces a local history variable $\mathcal{H}$ that captures the maximum tensile strain energy density experienced by the material at a point $\boldsymbol{x}$ over the entire loading history $t$:

.. math::
   :label: eq_pff_history_variable

   \mathcal{H}(\boldsymbol{x}, t) = \max_{s \in [0, t]} \psi_a(\boldsymbol{\epsilon}(\boldsymbol{x}, s)).

In a time-discrete setting at time step $t_n$, $\mathcal{H}$ is updated pointwise as:

.. math::
   :label: eq_pff_history_update

   \mathcal{H}(\boldsymbol{x}, t_{n}) = \begin{cases}
       \psi_a(\boldsymbol{\epsilon}(\boldsymbol{u})), & \text{if } \psi_a(\boldsymbol{\epsilon}(\boldsymbol{u})) > \mathcal{H}_{n-1}(\boldsymbol{x}), \\
       \mathcal{H}_{n-1}(\boldsymbol{x}), & \text{otherwise},
   \end{cases}

where $\mathcal{H}_{n-1}$ is the history field from the previous converged step.

In this formulation, phase-field evolution is driven by the history field $\mathcal{H}$ rather than the instantaneous active strain energy $\psi_a$. Since $\mathcal{H}$ is non-decreasing by definition ($\dot{\mathcal{H}} \ge 0$), the driving force for damage cannot decrease, thereby ensuring that $\phi$ does not decrease ($\dot{\phi} \ge 0$).

The governing weak forms for the coupled problem are modified accordingly. The momentum equation remains dependent on the current stress state, while the phase-field weak form replaces $\psi_a$ with $\mathcal{H}$. For an AT2 regularization, the system of equations reads:

.. math::
   :label: eq_pff_miehe_weak_forms

   {\mathcal{E}_\text{Miehe}}'_{\boldsymbol{u}}(\boldsymbol{u}, \phi) &:= \mathcal{E}'_{\boldsymbol{u}}(\boldsymbol{u}, \phi) = \boldsymbol{0}, \\
   {\mathcal{E}_\text{Miehe}}'_{\phi}(\boldsymbol{u}, \phi) &:= \int_\Omega g'(\phi) \delta\phi \, \mathcal{H} \,\mathrm{d}\boldsymbol{x} + G_c \int_\Omega \left( \frac{1}{l} \phi \delta\phi + l \nabla\phi \cdot \nabla \delta \phi \right) \,\mathrm{d}\boldsymbol{x} = 0.

Note that this approach loses the strict variational structure of the original problem, as the history field $\mathcal{H}$ is not derived directly from an energy functional but is introduced into the phase-field weak form as a driving force.

This formulation handles the irreversibility condition but not the upper bound $\phi \le 1$. However, for AT2 regularization with a quadratic degradation function, the phase-field variable naturally satisfies the lower bound $\phi \ge 0$. To enforce both bounds simultaneously, a common strategy combines the history field for irreversibility with a penalty term for upper/lower bounds.


.. _sec_pff_hybrid_formulation:

Hybrid formulation
^^^^^^^^^^^^^^^^^^

A related approach is the hybrid formulation introduced by :footcite:t:`phase_field_Ambati2015`. Building upon :footcite:t:`phase_field_Miehe2010`, the momentum equation is modified to incorporate the degradation function $g(\phi)$ applied to the total stress tensor $\boldsymbol{\sigma}(\boldsymbol{\epsilon})$ (from the isotropic model):

.. math::
   :label: eq_pff_hybrid_weak_forms

   {\mathcal{E}_\text{Hybrid}}'_{\boldsymbol{u}}(\boldsymbol{u}, \phi) &:= \int_\Omega g(\phi)\boldsymbol{\sigma}(\boldsymbol{\epsilon}(\boldsymbol{u})) : \boldsymbol{\epsilon}(\delta \boldsymbol{u}) \, \mathrm{d}\boldsymbol{x} - \int_\Omega \boldsymbol{f} \cdot \delta \boldsymbol{u} \, \mathrm{d}\boldsymbol{x} - \int_{\partial \Omega_t} \boldsymbol{t} \cdot \delta \boldsymbol{u} \, \mathrm{d}S = \boldsymbol{0}, \\
   {\mathcal{E}_\text{Hybrid}}'_{\phi}(\boldsymbol{u}, \phi) &:= {\mathcal{E}_\text{Miehe}}'_{\phi}(\boldsymbol{u}, \phi) = 0.

The primary motivation for this approach is computational efficiency: when solving via Newton-Raphson iterations, it avoids computing and assembling the tangent stiffness contributions associated with anisotropic stress splits. Only the isotropic stiffness tensor needs to be assembled, significantly reducing computational cost.



.. _sec_pff_numerical_models:

Numerical models
----------------

This section outlines the numerical framework employed to solve the PFF problem. The solution methodology consists of two primary stages: the spatial discretization of the governing weak forms via the FEM, followed by the application of an iterative scheme to solve the resulting non-linear coupled system.


.. _sec_pff_fem_phase_field_fracture:

Finite element discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The governing equations are discretized using the FEM. Both the displacement field $\boldsymbol u$ and the phase-field $\phi$ are approximated within a standard Galerkin framework using the same set of nodal shape functions $N_a$. The function space for the unknowns is defined as:

.. math::
   :label: eq_pff_fem_function_space

   V = \left\{ \nu: \Omega \rightarrow \mathbb{R},\ \nu(\boldsymbol x) = \sum_{a=1}^{n_{\text{node}}} N_a(\boldsymbol x)\, \nu_a \right\}

The continuous fields and their variations are then expressed as:

.. math::
   :label: eq_pff_fem_approximations

   \boldsymbol u_h(\boldsymbol x) = \sum_{a=1}^{n_{\text{node}}} N_a(\boldsymbol x) \boldsymbol u_a, \quad \delta \boldsymbol u_h(\boldsymbol x) = \sum_{a=1}^{n_{\text{node}}} N_a(\boldsymbol x) \delta \boldsymbol u_a, \\
   \phi_h(\boldsymbol x) = \sum_{a=1}^{n_{\text{node}}} N_a(\boldsymbol x) \phi_a, \quad \delta \phi_h(\boldsymbol x) = \sum_{a=1}^{n_{\text{node}}} N_a(\boldsymbol x) \delta \phi_a,

where $\boldsymbol u_a$ and $\phi_a$ denote the nodal displacement vector and nodal phase-field value, respectively, and $n_{\text{node}}$ is the total number of nodes in the discretization.

Substituting these approximations into the weak form equations yields a system of non-linear algebraic equations. The requirement that the weak forms vanish for arbitrary variations $\delta \boldsymbol u_a$ and $\delta \phi_a$ leads to the definition of the global residual vectors at each node $a$:

.. math::
   :label: eq_pff_residuals

   \boldsymbol{R}^{\boldsymbol u}_a = \underset{e}{\mathbf{A}} \left( \boldsymbol{R}^{\boldsymbol u, e}_a \right) = \boldsymbol{0} \quad \text{and} \quad
   R^{\phi}_a = \underset{e}{\mathbf{A}} \left( R^{\phi, e}_a \right) = 0,

where $\mathbf{A}$ represents the global assembly operator, and $\boldsymbol{R}^{\boldsymbol u, e}_a$ and $R^{\phi, e}_a$ are the elemental contributions to the displacement and phase-field residuals, respectively.

The detailed definitions of the residuals and tangent stiffness matrices for the different formulations considered in this chapter are provided in the corresponding subsections below. Resolving this coupled non-linear system requires robust numerical strategies. Two main categories of solution schemes are commonly employed in the literature: monolithic schemes and staggered (or operator-split) schemes.


.. _sec_pff_monolithic_schemes:

Monolithic schemes
~~~~~~~~~~~~~~~~~~

In monolithic schemes, the displacement field $\boldsymbol{u}$ and the phase-field variable $\phi$ are solved simultaneously as a single coupled system. While this approach can be computationally efficient, it is generally less robust, as discussed in :footcite:t:`phase_field_Ambati2015`. The non-convexity of the total energy functional renders the discrete system difficult to solve, and the standard Newton-Raphson method often fails to converge. Consequently, globalization strategies, such as line-search algorithms, are typically required to aid convergence.


Reference case
^^^^^^^^^^^^^^

As a baseline for comparison, the monolithic solution of the standard phase-field model without explicitly enforcing the irreversibility constraint is first considered. In this case, assuming the AT2 regularization, the boundedness of the phase-field variable ($\phi \in [0,1]$) is naturally satisfied by the model structure. This simplified formulation is applicable mainly to problems with monotonic loading paths where the crack driving force does not decrease, effectively satisfying irreversibility implicitly.

The elemental residual contributions are given by:

.. math::
   :label: eq_pff_fem_residuals

   \mathbf{R}^{\boldsymbol{u}}_a &= \int_{\Omega_e} \left[ g(\phi_h)\, \boldsymbol{\sigma}_a(\boldsymbol{\epsilon}(\boldsymbol{u}_h)) + \boldsymbol{\sigma}_b(\boldsymbol{\epsilon}(\boldsymbol{u}_h)) \right] : \boldsymbol{\epsilon}(N_a) \, \mathrm{d}\boldsymbol{x} - \int_{\Omega_e} \boldsymbol{f} N_a \, \mathrm{d}\boldsymbol{x} - \int_{\partial \Omega_{t,e}} \boldsymbol{t} N_a \, \mathrm{d}S, \\
   R^{\phi}_a &= \int_{\Omega_e} g'(\phi_h) \psi_a(\boldsymbol{\epsilon}(\boldsymbol{u}_h)) N_a \, \mathrm{d}\boldsymbol{x} + \frac{G_c}{c_0} \int_{\Omega_e} \left( \frac{\alpha'(\phi_h)}{l} N_a + 2l \nabla\phi_h \cdot \nabla N_a \right) \mathrm{d}\boldsymbol{x}.

Note that for the standard AT2 model, $\alpha'(\phi)=2\phi$ and $c_0=2$, recovering the familiar terms.

The global residual vector for the monolithic coupled system is obtained by assembling these elemental contributions:

.. math::
   :label: eq_pff_fem_residual_assembly

   \boldsymbol{R} = \underset{e}{\mathbf{A}} 
   \begin{bmatrix}
       \mathbf{R}^{\boldsymbol{u}}_a \\
       R^{\phi}_a
   \end{bmatrix}.

The corresponding components of the elemental tangent stiffness matrices are given by:

.. math::
   :label: eq_pff_fem_tangents

   \mathbf{K}^{\boldsymbol{u}\boldsymbol{u}}_{ab} &= \int_{\Omega_e} \boldsymbol{\epsilon}(N_a) : \left[ g(\phi_h) \mathbb{C}_a(\boldsymbol{\epsilon}(\boldsymbol{u}_h)) + \mathbb{C}_b(\boldsymbol{\epsilon}(\boldsymbol{u}_h)) \right] : \boldsymbol{\epsilon}(N_b) \, \mathrm{d}\boldsymbol{x}, \\
   \mathbf{K}^{\boldsymbol{u}\phi}_{ab} &= \int_{\Omega_e} [g'(\phi_h) \boldsymbol{\sigma}_a(\boldsymbol{\epsilon}(\boldsymbol{u}_h)) : \boldsymbol{\epsilon}(N_a)] N_b \, \mathrm{d}\boldsymbol{x}, \\
   \mathbf{K}^{\phi\boldsymbol{u}}_{ab} &= \int_{\Omega_e} [g'(\phi_h) \boldsymbol{\sigma}_a(\boldsymbol{\epsilon}(\boldsymbol{u}_h)) : \boldsymbol{\epsilon}(N_b)] N_a \, \mathrm{d}\boldsymbol{x}, \\
   K^{\phi\phi}_{ab} &= \int_{\Omega_e} g''(\phi_h) \psi_a(\boldsymbol{\epsilon}(\boldsymbol{u}_h)) N_a N_b \, \mathrm{d}\boldsymbol{x} + \frac{G_c}{c_0} \int_{\Omega_e} \left( \frac{\alpha''(\phi_h)}{l} N_a N_b + 2l \nabla N_a \cdot \nabla N_b \right) \mathrm{d}\boldsymbol{x}.

Similar to the residual vector, the global tangent stiffness matrix is obtained by assembling the elemental contributions:

.. math::
   :label: eq_pff_fem_tangent_assembly

   \boldsymbol{K} = \underset{e}{\mathbf{A}} 
   \begin{bmatrix}
       \mathbf{K}^{\boldsymbol{u}\boldsymbol{u}}_{ab} & \mathbf{K}^{\boldsymbol{u}\phi}_{ab} \\
       \mathbf{K}^{\phi\boldsymbol{u}}_{ab} & K^{\phi\phi}_{ab}
   \end{bmatrix}.

For the standard formulation, the coupled nonlinear system is solved using the Newton-Raphson method.


Penalization scheme
^^^^^^^^^^^^^^^^^^^

To address non-monotonic loading and enforce physical constraints strictly, the monolithic scheme can be augmented with penalty terms. In this approach, the weak forms corresponding to the penalized functional (introduced in :ref:`sec_pff_penalty_model`) are solved simultaneously. The penalty terms incorporate the irreversibility condition ($\phi \ge \phi_{\text{prev}}$) and, if necessary, the boundedness constraints directly into the monolithic system.

The residuals and tangent stiffness matrices for the penalized monolithic scheme can be obtained by adding the corresponding penalty contributions to the standard phase-field equations. Specifically, the elemental phase-field residual $R^\phi_a$ is augmented by adding:

.. math::
   :label: eq_pff_mono_penalty_residual

   R^{\phi,\text{pen}}_a = \rho \int_{\Omega_e} \langle \phi_h - \phi_{\text{prev}} \rangle_- \, N_a \, \mathrm{d}\boldsymbol{x},

where $\rho > 0$ is a sufficiently large penalty parameter, and the phase-field tangent stiffness component $K^{\phi\phi}_{ab}$ is augmented by adding the corresponding derivative term:

.. math::
   :label: eq_pff_mono_penalty_tangent

   K^{\phi\phi,\text{pen}}_{ab} = \rho \int_{\Omega_e} H(\phi_{\text{prev}} - \phi_h) \, N_a N_b \, \mathrm{d}\boldsymbol{x},

where $H(\cdot)$ is the Heaviside step function.


.. _sec_pff_staggered_schemes:

Staggered schemes
~~~~~~~~~~~~~~~~~

Staggered schemes, also referred to as operator-splitting methods, decouple the solution procedure by solving for the displacement and phase-field variables sequentially rather than simultaneously. In a typical algorithmic step, the mechanical equilibrium equation is solved for the displacement field while holding the phase-field variable fixed. Subsequently, the phase-field evolution equation is solved using the updated displacement field. This alternating process is repeated until the coupled system satisfies the staggered convergence criteria.

A key advantage of staggered schemes is their robustness; by decoupling the fields, the resulting sub-problems are typically convex with respect to the variable being solved (assuming the other is fixed). This property simplifies the solution of the systems at each staggered step. Because the fields are solved sequentially rather than simultaneously, the off-diagonal coupling tangent blocks ($\mathbf{K}^{\boldsymbol{u}\phi}$ and $\mathbf{K}^{\phi\boldsymbol{u}}$) are neither computed nor assembled.


Penalization scheme
^^^^^^^^^^^^^^^^^^^

In the penalized staggered scheme, the displacement field is updated based on the phase-field from the previous iteration, and the phase-field is then updated using the new displacements. Within each uncoupled solver step, the residuals and independent tangent components match those presented in the monolithic penalization scheme. The alternating sequence continues until the staggered convergence criteria are satisfied, as discussed in :ref:`sec_pff_staggered_tol`.

.. tikz:: General flowchart of the staggered solution scheme.
   :align: center

   \begin{tikzpicture}[node distance=2.0cm, auto]
        \tikzstyle{process} = [rectangle, minimum width=2.5cm, minimum height=1cm, text centered, text width=4.5cm, align=center, font=\footnotesize, draw=black, fill=gray!10, rounded corners]
        \tikzstyle{decision} = [diamond, minimum width=1.5cm, minimum height=1.0cm, text centered, text width=2.0cm, font=\scriptsize, inner sep=0pt, draw=black, fill=white]
        \tikzstyle{stop} = [rectangle, rounded corners, minimum width=2.6cm, minimum height=1cm, text centered, font=\small, draw=black, fill=gray!30]
        \tikzstyle{arrow} = [thick,->,>=stealth]

        \node (start) [process] {Start Iteration $k$};
        \node (solve_u) [process, below of=start, yshift=-0.3cm] {Solve $\boldsymbol{u}^{(k)}$ such that $\boldsymbol{R}^{\boldsymbol{u}}(\boldsymbol{u}^{(k)}, \phi^{(k-1)}) = \boldsymbol{0}$};
        \node (solve_phi) [process, below of=solve_u, yshift=-0.3cm] {Solve $\phi^{(k)}$ such that $\boldsymbol{R}^{\phi}_{\text{penalty}}(\boldsymbol{u}^{(k)}, \phi^{(k)}) = \boldsymbol{0}$};
        \node (check) [decision, below of=solve_phi, yshift=-0.8cm] {Staggered criterion satisfied?};
        \node (stop) [stop, right of=check, xshift=4.2cm] {Stop};
        \node (next) [process, left of=solve_phi, xshift=-4.2cm] {$k \leftarrow k+1$};

        \draw [arrow] (start) -- (solve_u);
        \draw [arrow] (solve_u) -- (solve_phi);
        \draw [arrow] (solve_phi) -- (check);
        \draw [arrow] (check) -- node[anchor=south] {Yes} (stop);
        \draw [arrow] (check) -| node[anchor=east] {No} (next);
        \draw [arrow] (next) |- (start);
   \end{tikzpicture}


History field method
^^^^^^^^^^^^^^^^^^^^

The history field method, as formulated in :footcite:t:`phase_field_Miehe2010`, is naturally suited for staggered schemes. In this approach, irreversibility is enforced by driving the phase-field evolution with a history variable $\mathcal{H}$, which records the maximum tensile strain energy density over time.

Crucially, implementing this method requires careful handling of the history variable update. Within the staggered loop, the history variable $\mathcal{H}$ must be computed based on the active energy of the current iteration, updated against the maximum value stored from the previous time step.

The numerical residual associated with the displacement field coincides with :eq:`eq_pff_fem_residuals`. For the phase-field residual, the history field replaces the active energy density:

.. math::
   :label: eq_pff_staggered_history_residual

   R^{\phi}_a = \int_{\Omega_e} g'(\phi_h) \mathcal{H} N_a \, \mathrm{d}\boldsymbol{x} + \frac{G_c}{c_0} \int_{\Omega_e} \left( \frac{\alpha'(\phi_h)}{l} N_a + 2l \nabla\phi_h \cdot \nabla N_a \right) \mathrm{d}\boldsymbol{x}.

Since this formulation is solved using a staggered scheme, the tangent block for the displacement field coincides with :eq:`eq_pff_fem_tangents`, while the phase-field tangent block is given by:

.. math::
   :label: eq_pff_staggered_history_tangent

   K^{\phi\phi}_{ab} = \int_{\Omega_e} g''(\phi_h) \mathcal{H} N_a N_b \, \mathrm{d}\boldsymbol{x} + \frac{G_c}{c_0} \int_{\Omega_e} \left( \frac{\alpha''(\phi_h)}{l} N_a N_b + 2l \nabla N_a \cdot \nabla N_b \right) \mathrm{d}\boldsymbol{x}.

The algorithm proceeds as follows:

1. Given the phase-field from the previous iteration, solve the mechanical equilibrium equation using $\mathbf{R}^{\boldsymbol{u}}$ and $\mathbf{K}^{\boldsymbol{u}\boldsymbol{u}}$.
2. Compute the current active energy density, $V_c = \psi_a(\boldsymbol{\epsilon}(\boldsymbol{u}^{(k)}))$.
3. Update the local history field for the current iteration as $\mathcal{H} = \max(V_c, V_n)$, where $V_n$ is the history variable from the previous converged time step.
4. Solve the phase-field evolution equation using $R^{\phi}$ and $K^{\phi\phi}$, where the active energy term $\psi_a(\boldsymbol{\epsilon}(\boldsymbol{u}^{(k)}))$ is replaced by the updated history variable $\mathcal{H}$.

.. tikz:: Flowchart of the staggered scheme with history field.
   :align: center

   \begin{tikzpicture}[node distance=2.0cm, auto]
        \tikzstyle{process} = [rectangle, minimum width=2.5cm, minimum height=1cm, text centered, text width=4.5cm, align=center, font=\footnotesize, draw=black, fill=gray!10, rounded corners]
        \tikzstyle{decision} = [diamond, minimum width=1.5cm, minimum height=1.0cm, text centered, text width=2.0cm, font=\scriptsize, inner sep=0pt, draw=black, fill=white]
        \tikzstyle{stop} = [rectangle, rounded corners, minimum width=2.6cm, minimum height=1cm, text centered, font=\small, draw=black, fill=gray!30]
        \tikzstyle{arrow} = [thick,->,>=stealth]

        \node (start) [process] {Start Iteration $k$};
        \node (disp_step) [process, below of=start, yshift=-0.3cm] {Solve $\boldsymbol{u}^{(k)}$ such that $\boldsymbol{R}^{\boldsymbol{u}}(\boldsymbol{u}^{(k)}, \phi^{(k-1)}) = \boldsymbol{0}$};
        \node (compute_vc) [process, below of=disp_step, yshift=-0.3cm] {Compute $V_c = \psi_a(\boldsymbol{\epsilon}(\boldsymbol{u}^{(k)}))$ \\ and update $H = \max(V_c, V_n)$};
        \node (phi_step) [process, below of=compute_vc, yshift=-0.3cm] {Solve $\phi^{(k)}$ such that $\boldsymbol{R}^{\phi}(\boldsymbol{u}^{(k)}, \phi^{(k)}, H) = \boldsymbol{0}$};
        \node (check) [decision, below of=phi_step, yshift=-0.8cm] {Staggered criterion satisfied?};
        \node (stop) [stop, right of=check, xshift=4.2cm] {Stop};
        \node (next) [process, left of=phi_step, xshift=-4.2cm] {$k \leftarrow k+1$};

        \draw [arrow] (start) -- (disp_step);
        \draw [arrow] (disp_step) -- (compute_vc);
        \draw [arrow] (compute_vc) -- (phi_step);
        \draw [arrow] (phi_step) -- (check);
        \draw [arrow] (check) -- node[anchor=south] {Yes} (stop);
        \draw [arrow] (check) -| node[anchor=east] {No} (next);
        \draw [arrow] (next) |- (start);
   \end{tikzpicture}


.. _sec_pff_staggered_tol:

Staggered tolerance criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Establishing a robust convergence criterion is a critical aspect of any staggered scheme. Since the mechanical and phase-field sub-problems are solved sequentially, satisfying the equilibrium conditions for each field individually does not guarantee that the coupled system has reached a state of global equilibrium. Consequently, iterations are necessary to ensure that the solution properly captures the interaction between the displacement and phase-field variables.

Several approaches are commonly employed in the literature to determine when to terminate this iterative process:

Fixed number of iterations
""""""""""""""""""""""""""

In some implementations, such as the one described in :footcite:t:`phase_field_Miehe2010`, no explicit convergence criterion is defined for the staggered loop. Instead, a fixed number of staggered iterations is performed per time step. To mitigate the resulting splitting error, extremely small time steps are typically required. While this approach ensures that the individual sub-problems (displacement and phase-field) are solved to equilibrium, it does not guarantee that the coupled system reaches global equilibrium simultaneously within the step. Although computationally efficient per step, this method may lead to drift from the true equilibrium path and inaccuracies.


Solution increment norms
""""""""""""""""""""""""

A widely used convergence criterion assesses the relative change in the solution fields between consecutive staggered iterations. Convergence is assumed when the normalized $L^2$ norms of the solution increments fall below a user-defined tolerance:

.. math::
   :label: eq_pff_staggered_norms

   \text{error}_{\boldsymbol{u}} &= \frac{\parallel \boldsymbol{u}^{(k)} - \boldsymbol{u}^{(k-1)} \parallel_{L^2}}{\parallel \boldsymbol{u}^{(k)} \parallel_{L^2}} < \text{tol}, \\
   \text{error}_{\phi} &= \frac{\parallel \phi^{(k)} - \phi^{(k-1)} \parallel_{L^2}}{\parallel \phi^{(k)} \parallel_{L^2}} < \text{tol}.

While this method indicates that the solution has stabilized, it does not strictly enforce the satisfaction of the governing equations' residuals. Furthermore, although the criteria shown above evaluate the increments of both fields, alternative implementations may monitor only one of them, most commonly the phase-field increment. Additionally, while the $L^2$ norm is standard, other metrics are also employed; for instance, :footcite:t:`phase_field_staggered_tol_infinity` considers the unnormalized $L^{\infty}$ norm for the phase-field increment, whereas :footcite:t:`phase_field_Bourdin2000` applies it to the displacement increment.


Residual-based staggered convergence criterion
""""""""""""""""""""""""""""""""""""""""""""""

A more rigorous convergence criterion for the staggered scheme involves directly monitoring the global residuals of the governing equations. The fundamental objective of this scheme is to obtain a specific pair of solution fields, $(\boldsymbol{u}, \phi)$, that simultaneously satisfy both equilibrium conditions at the end of the time step.

Conveniently, when the uncoupled sub-problems are solved using the Newton--Raphson method, this evaluation occurs naturally. The initial residual (evaluated at iteration 0, prior to any nodal updates) explicitly measures whether the fields entering the sub-problem already satisfy the corresponding governing equation. If this initial residual falls below the prescribed tolerance, the current variables are deemed adequate, and no further Newton iterations or staggered steps are required.

Based on this property, the termination of the staggered loop follows an efficient "early exit" logic, as illustrated in the figure below. Because the equations are solved sequentially, the initial residual of the phase-field equation can be evaluated immediately after solving for the displacement field. If this initial residual satisfies the tolerance, the staggered loop terminates, as the displacement field has just been solved to equilibrium and the phase-field inherently satisfies its respective condition.

Symmetric logic applies if the solving sequence is reversed: if the initial residual of the displacement field falls below the prescribed tolerance immediately after solving the phase-field equation, the staggered loop terminates. This robust approach guarantees that the final solution fields strictly satisfy the coupled problem.

Ultimately, this criterion provides a direct global measure of equilibrium. It should be noted that at least one complete staggered iteration must be performed to initialize the cycle. Ideally, the staggered tolerance should match the tolerance imposed on the individual Newton--Raphson schemes, ensuring that the residual equations satisfy the required accuracy bounds both individually and globally. Finally, it is crucial that the variables accepted as the final time-step solution are precisely those satisfying the initial residual condition, ensuring strict enforcement of the coupled governing partial differential equations (PDEs).

.. tikz:: Flowchart of the staggered scheme with "early exit" residual-based convergence checks.
   :align: center

   \begin{tikzpicture}[node distance=2.2cm, auto]
        \tikzstyle{process} = [rectangle, minimum width=2.5cm, minimum height=1cm, text centered, text width=4.5cm, align=center, font=\footnotesize, draw=black, fill=gray!10, rounded corners]
        \tikzstyle{decision} = [diamond, minimum width=3.0cm, minimum height=2.0cm, text centered, text width=3.2cm, font=\scriptsize, inner sep=0pt, thick, draw=black, fill=white]
        \tikzstyle{stop} = [rectangle, rounded corners, minimum width=2.6cm, minimum height=1cm, text centered, font=\small, draw=black, fill=gray!30]
        \tikzstyle{arrow} = [thick,->,>=stealth]

        \node (start) [process] {Start Iteration $k$};
        \node (check_u) [decision, below of=start, yshift=-1cm] {$\begin{array}{c}\|\boldsymbol{R}^{\boldsymbol{u},0}(\boldsymbol{u}^{(k-1)}, \phi^{(k-1)})\|\\<\ \text{tol}_{\boldsymbol{u}}?\end{array}$};
        \node (stop_u) [stop, right of=check_u, xshift=4cm] {Stop (Converged)};
        \node (solve_u) [process, below of=check_u, yshift=-1cm] {Solve $\boldsymbol{u}^{(k)}$ such that $\boldsymbol{R}^{\boldsymbol{u}}(\boldsymbol{u}^{(k)}, \phi^{(k-1)}) = \boldsymbol{0}$};
        \node (check_phi) [decision, below of=solve_u, yshift=-1cm] {$\begin{array}{c}\|R^{\phi,0}(\boldsymbol{u}^{(k)}, \phi^{(k-1)})\|\\<\ \text{tol}_{\phi}?\end{array}$};
        \node (stop_phi) [stop, right of=check_phi, xshift=4cm] {Stop (Converged)};
        \node (solve_phi) [process, below of=check_phi, yshift=-1cm] {Solve $\phi^{(k)}$ such that $R^{\phi}(\boldsymbol{u}^{(k)}, \phi^{(k)}) = 0$};
        \node (next) [process, left of=solve_phi, xshift=-4cm] {$k \leftarrow k+1$};

        \draw [arrow] (start) -- (check_u);
        \draw [arrow] (check_u) -- node[anchor=south] {Yes} (stop_u);
        \draw [arrow] (check_u) -- node[anchor=east] {No} (solve_u);
        \draw [arrow] (solve_u) -- (check_phi);
        \draw [arrow] (check_phi) -- node[anchor=south] {Yes} (stop_phi);
        \draw [arrow] (check_phi) -- node[anchor=east] {No} (solve_phi);
        \draw [arrow] (solve_phi) -- (next);
        \draw [arrow] (next) |- (start);
   \end{tikzpicture}




.. note::
    Please view the examples related to phase-field fracture :ref:`ref_examples_phase_field_fracture`.







.. _sec_pff_modular_framework:

Model summary
~~~~~~~~~~~~~

A complete PFF model is assembled by selecting one formulation for each of the four building
blocks. The previous sections detail each component; the following boxes summarize their
roles and how they connect, following the order in which they contribute to the governing
equations.

.. admonition:: Component 1 — Degradation Function :math:`g(\phi)`
   :class: hint

   Reduces material stiffness as :math:`\phi` evolves from 0 (intact) to 1 (fully broken).
   Standard choice: :math:`g(\phi)=(1-\phi)^2` :footcite:t:`phase_field_Bourdin2000`.
   Cubic and quartic alternatives allow tailoring of the post-peak softening response
   :footcite:t:`phase_field_degradation_functions`. See :ref:`sec_pff_degradation_functions`.

.. admonition:: Component 2 — Energy Split :math:`\psi = \psi_a + \psi_b`
   :class: hint

   Separates the strain energy into crack-driving active (:math:`\psi_a`, degraded by
   :math:`g(\phi)`) and crack-resistant inactive (:math:`\psi_b`, undegraded) contributions.
   The isotropic split is the simplest choice; the spectral
   :footcite:t:`phase_field_Miehe2010` and volumetric-deviatoric
   :footcite:t:`phase_field_Amor2009` splits accurately capture tension–compression
   asymmetry. See :ref:`sec_pff_iso_aniso_models`.

.. admonition:: Component 3 — Crack Surface Density :math:`\gamma(\phi,\nabla\phi)`
   :class: tip

   Regularizes the sharp crack topology via the length-scale parameter :math:`l`. The AT2
   functional naturally satisfies :math:`\phi\geq 0`; AT1
   :footcite:t:`introduction_ambrosio_tortorelli` and Wu :footcite:t:`phase_field_Wu`
   functionals require explicit enforcement of the bounds constraint
   :footcite:t:`phase_field_Gerasimov`. All formulations satisfy :math:`\Gamma`-convergence
   to the Griffith energy. See :ref:`theory_phase_field`.

.. important::

   **Assembled weak form — Components 1, 2, and 3 together.**
   Substituting the choices of :math:`g`, :math:`\psi_a`, :math:`\psi_b`, and :math:`\gamma`
   into the energy functional :eq:`eq_pff_variational_formulation` and enforcing stationarity
   yields the coupled weak form :eq:`eq_pff_weak_forms`. Each ingredient appears explicitly:
   :math:`g(\phi)` degrades the elastic energy; the split stress tensors
   :math:`\boldsymbol{\sigma}_a`, :math:`\boldsymbol{\sigma}_b` enter the mechanical
   equilibrium equation; the CSDF parameters :math:`\alpha'(\phi)` and :math:`c_0` govern
   the crack regularization term in the phase-field equation.

.. admonition:: Component 4 — Irreversibility & Bounds
   :class: warning

   Enforces the physical inequality constraints that are **not** embedded in the functional
   itself and must be imposed separately:

   - **No crack healing:** :math:`\dot{\phi}\geq 0`
   - **Physical admissibility:** :math:`\phi\in[0,1]`

   Common enforcement strategies:

   - **History-field variable** :footcite:t:`phase_field_Miehe2010`: drives crack evolution
     with the maximum historical energy density; simple to implement but breaks strict
     variational consistency.
   - **Penalty approach** :footcite:t:`phase_field_Gerasimov`: adds a penalty term to enforce
     the inequality while maintaining the variational structure.
   - **LVPP algorithm** :footcite:t:`lvpp_Keith`: reformulates the problem as a sequence of
     saddle-point problems involving latent variables; avoids history fields and penalty terms.

   See :ref:`sec_pff_numerical_strategies`.


.. _sec_pff_numerical_strategies:

Numerical solution strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numerical implementation of PFF models presents three primary challenges that must be
addressed to ensure robust and accurate simulations.

First, the interval constraint of the phase-field variable ($\phi \in [0,1]$) is not
naturally guaranteed in all formulations. Specifically, while the AT2 model with a quadratic
degradation function inherently confines $\phi$ to physical values, other models such as AT1
may violate this constraint. To enforce this constraint, techniques such as Lagrange
multipliers or penalty methods are commonly employed :footcite:t:`phase_field_Gerasimov`.

Second, the irreversibility condition ($\dot{\phi} \ge 0$) must be strictly enforced, as
cracks physically cannot heal. This is typically achieved using a history variable approach,
where the maximum historical strain energy is used to drive damage, as presented in
:footcite:t:`phase_field_Miehe2010`, or through penalty terms in the variational formulation,
as done by :footcite:t:`phase_field_Gerasimov`. Each approach involves trade-offs between
implementation simplicity and variational scheme consistency. The novel Latent Variable
Proximal Point (LVPP) algorithm, introduced in :footcite:t:`lvpp_Keith` and applied to
several examples in :footcite:t:`lvpp_Dokken`, handles the PFF inequality-constrained problem
in a robust manner, avoiding the use of history variables or penalty methods by reformulating
it into a sequence of saddle-point problems involving latent variables.

Third, the total energy functional is non-convex with respect to both displacement
$\boldsymbol{u}$ and phase-field $\phi$ simultaneously. Consequently, solving the system in a
monolithic scheme using standard Newton-Raphson procedures often leads to convergence issues.
To overcome this, staggered schemes (or alternate minimization) are widely used
:footcite:t:`phase_field_Miehe2010`, solving for $\boldsymbol{u}$ and $\phi$ sequentially.

Standard displacement-controlled algorithms are generally insufficient for capturing unstable
crack propagation, characterized by *snap-back* phenomena in the load-displacement response.
To trace these equilibrium paths quasi-statically, advanced control schemes are required,
commonly referred to as **equilibrium path tracking algorithms**. While arc-length methods
:footcite:t:`phase_field_snap_Ritukesh` and crack-length control techniques
:footcite:t:`phase_field_snap_pedro` exist, they often introduce significant complexity.


.. _sec_pff_mesh_effects:

Considerations and mesh effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In PFF modeling, the regularization is governed by a length-scale parameter $l$ that controls
the width of the diffuse crack representation. A central challenge in the finite element
implementation lies in the conflicting requirements placed on $l$ and the characteristic mesh
size $h$.

On the one hand, the mesh must be fine enough to resolve the phase-field profile. A common
guideline for the AT2 model :footcite:t:`phase_field_Miehe_lh_relation` is to maintain a
ratio $l/h \ge 2$. On the other hand, to approximate the sharp-crack limit, $l$ must be
sufficiently small. However, this is complicated by a numerical artifact known as
*strain localization* :footcite:t:`phase_field_FrancfortMarigo1998`, where the phase-field
variable artificially saturates at $\phi=1$ across entire finite elements, distorting the
ideal diffuse profile and leading to a significant overestimation of the crack surface area.
This overestimation systematically compromises the accuracy of simulation outputs by
artificially scaling key quantities such as displacements, forces, and energy-related
quantities.

Several strategies have been developed to mitigate this overestimation. One approach scales
the effective energy release rate as $G_c^{\text{eff}} = G_c (1 + h/2l)$. Alternative
post-processing techniques, such as skeletonization algorithms
:footcite:t:`phase_field_skeleton`, estimate the crack area by thresholding the computed
phase-field (e.g., $\phi > 0.9$). These two competing requirements often lead to
computationally prohibitive mesh densities, highlighting the need for more robust and
efficient correction methods.

Another significant artifact is the presence of a non-physical peak force at the onset of
crack initiation, not observed in experimental results :footcite:t:`phase_field_snap_pedro`.
This force overshoot is linked to the nucleation process, where the phase-field variable
evolves from $\phi=0$ to $\phi=1$.


.. _sec_pff_fatigue_background:

Fatigue analysis within PFF framework
---------------------------------------

The extensions of PFF to model fatigue crack propagation are relatively recent
:footcite:t:`phase_field_Carrara2020`. A common ingredient of most proposed models is the
introduction of a fatigue degradation function that reduces fracture toughness according to a
cumulative history variable, where simulation progress is tracked by a pseudotime. This
history variable typically monitors quantities such as accumulated plastic strain or energy
dissipation, capturing the material's fatigue history. In these models, fatigue crack growth
is simulated by incrementally applying cyclic loads and updating the history variable at each
cycle. However, the computational cost is inherently high for high-cycle fatigue scenarios, as
it requires resolving each individual load cycle. Cycle-jumping or cycle-skipping strategies
:footcite:t:`phase_field_fatigue_Heinzmann` have been proposed to accelerate these
simulations.

In classical linear elastic fatigue frameworks, crack propagation is typically modeled using
Paris-type laws, which provide the crack growth rate and direction as a function of the Stress
Intensity Factors (SIF) at the crack tip. To simulate the complete process of crack growth,
numerical techniques such as the Extended Finite Element Method (XFEM)
:footcite:t:`introduction_x_fem` or remeshing strategies :footcite:t:`introduction_remeshing`
are commonly employed. Criteria such as the Maximum Tangential Stress
:footcite:t:`introduction_maximum_tangential_stress_criterion` and the Maximum Energy Release
Rate :footcite:t:`introduction_maximum_energy_release_rate_criterion` are often used to
predict crack propagation direction.







.. footbibliography::
