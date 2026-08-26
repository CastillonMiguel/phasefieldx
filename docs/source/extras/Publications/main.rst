.. _publications:

Publications Using phasefieldx
==============================


.. raw:: html

    <style>
        .pub-title {
            font-size: 28px;
            font-weight: bold;
            margin: 0;
            line-height: 1.2;
            color: #2c3e50; /* Dark blue for light mode */
        }
        .pub-subtitle {
            color: #7f8c8d; /* Muted gray for light mode */
            font-style: italic;
            margin: 5px 0 0 0;
            font-size: 16px;
        }
        .pub-button {
            display: block;
            color: white !important;
            padding: 12px;
            text-decoration: none;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            transition: all 0.3s;
        }
        .pub-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .btn-journal { background: linear-gradient(135deg, #2c5f75 0%, #1e3a5f 100%); }
        .btn-github { background-color: #24292e; }
        .btn-docs { background: linear-gradient(135deg, #2c5f75 0%, #295068 100%); }
        .btn-arxiv { background: linear-gradient(135deg, #B31B1B 0%, #8A1818 100%); }

        [data-theme="dark"] .pub-title {
            color: #ecf0f1; /* Light gray for dark mode */
        }
        [data-theme="dark"] .pub-subtitle {
            color: #95a5a6; /* Lighter gray for dark mode */
        }
        [data-theme="dark"] .card-body {
            color: #bdc3c7;
        }
        .second-pub-card {
            background-color: white;
        }
        [data-theme="dark"] .second-pub-card {
            background-color: var(--sd-color-card-background);
        }
        [data-theme="dark"] .second-pub-card .pub-title,
        [data-theme="dark"] .second-pub-card .pub-subtitle,
        [data-theme="dark"] .second-pub-card,
        [data-theme="dark"] .second-pub-card .card-body,
        [data-theme="dark"] .second-pub-card p,
        [data-theme="dark"] .second-pub-card strong {
            color: #ecf0f1;
        }
    </style>
    
This section collects scientific publications and academic works related to
*PhaseFieldX*. The listed contributions include peer-reviewed journal
articles, doctoral theses, and other scholarly outputs that have employed
or contributed to the development of *PhaseFieldX*.
These works demonstrate the application of *PhaseFieldX* to phase-field
modeling, fracture mechanics, fatigue analysis, and related computational
mechanics problems, and serve as references for users interested in
validated research and published use cases of the framework.

If you have used *phasefieldx* in your research and would like your publication
to be included, please consider contacting the developers or submitting a
request through the project repository.

.. contents::
   :local:
   :depth: 2


Peer-Reviewed Journal Articles
------------------------------

.. grid:: 1
    :gutter: 2

    .. grid-item-card::
        :shadow: lg
        :class-card: first-pub-card

        .. raw:: html

            <div style="text-align: center; margin: 20px 0;">
                <h2 class="pub-title" style="font-size: 26px;">
                    A Correction Method for Crack Area Overestimation in Phase-Field Fracture
                </h2>
            </div>

        ^^^
        .. raw:: html

            <div style="padding: 30px 0; margin: 20px 0;">
                <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
                    <div style="text-align: center;">
                        <img
                        src="../../_static/logo_dgcm.png"
                        width="540px"
                        style="border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);"
                        alt="DGCM Logo"
                        />
                    </div>
                </div>
            </div>
            
        **Authors:** M. Castillón, J. Segurado, I. Romero,   
        **Year:** 2026  
        **Journal:** Computational Mechanics
        **DOI:** `10.1007/s00466-026-02834-2 <https://doi.org/10.1007/s00466-026-02834-2>`_

        Phase-field fracture models are known to overestimate the crack area, a discrepancy that affects the accuracy of fracture predictions. This issue stems from the diffuse crack representation and numerical artifacts, such as strain localization, where the phase-field variable artificially saturates across finite elements. Existing correction strategies, including mesh-dependent factors and skeletonization algorithms, have limitations. Mesh-based corrections are often unreliable for unstructured meshes, while skeletonization can be complex and inaccurate for intricate crack topologies, especially in three dimensions. This paper introduces a correction framework to address this overestimation. Our approach is founded on the principle of energy equipartition, where the energy contributions from the phase-field and its gradient are equal as the length-scale parameter approaches zero. Since numerical artifacts primarily affect the phase-field term while leaving the gradient term largely unperturbed, we propose that the crack area can be approximated as twice the gradient-dependent energy. This method is inherently mesh-independent and readily applicable to the entire domain, including 3D simulations. The proposed methodology is validated against benchmarks with analytical solutions and compared with established methods like skeletonization to demonstrate its accuracy. It is then applied to complex geometries with curvilinear crack paths and evaluated in a three-dimensional simulation.
        +++

        **🔗 Related Resources**

        .. raw:: html

            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 15px 0;">
                <a href="https://doi.org/10.1007/s00466-026-02834-2" target="_blank" class="pub-button btn-journal">
                    📄<br>Journal Paper
                </a>
                <a href="https://github.com/CastillonMiguel/A-Correction-Method-for-Crack-Area-Overestimation-in-Phase-Field-Fracture" target="_blank" class="pub-button btn-github">
                    💻<br>GitHub Repo
                </a>
                <a href="https://doublegradientcorrectionmethod.readthedocs.io" target="_blank" class="pub-button btn-docs">
                    📚<br>Documentation
                </a>
            </div>

    .. grid-item-card:: 
        :shadow: lg
        :class-card: second-pub-card

        .. raw:: html
        
            <div style="text-align: center; margin: 20px 0;">
                <h2 class="pub-title" style="font-size: 26px;">
                    A Phase-Field Approach to Fracture and Fatigue Analysis: Bridging Theory and Simulation
                </h2>
            </div>
        
        ^^^
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

        **Authors:** M. Castillón, I. Romero, J. Segurado  
        **Year:** 2025  
        **Journal:** International Journal of Fatigue  
        **DOI:** `10.1016/j.ijfatigue.2025.109397 <https://doi.org/10.1016/j.ijfatigue.2025.109397>`_
        
        This article presents a novel, robust and efficient framework for fatigue crack-propagation that combines the principles of Linear Elastic Fracture Mechanics (LEFM) with phase-field fracture (PFF). Contrary to cycle-by-cycle PFF approaches, this work relies on a single simulation and uses standard crack propagation models such as Paris' law for the material response, simplifying its parametrization.
        The core of the methodology is the numerical evaluation of the derivative of a specimen's compliance with respect to the crack area. To retrieve this compliance the framework relies on a PFF-FEM simulation, controlled imposing a monotonic crack growth. This control of the loading process is done by a new crack-control scheme which allows to robustly trace the complete equilibrium path of a crack, capturing complex instabilities. The specimen's compliance obtained from the PFF simulation enables the integration of Paris' law to predict fatigue life.
        The proposed methodology is first validated through a series of benchmarks with analytical solutions to demonstrate its accuracy. The framework is then applied to more complex geometries where the crack path is unknown, showing a very good agreement with experimental results of both crack paths and fatigue life.
        +++

        **🔗 Related Resources**

        .. raw:: html

            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 15px 0;">
                <a href="https://doi.org/10.1016/j.ijfatigue.2025.109397" target="_blank" class="pub-button btn-journal">
                    📄<br>Journal Paper
                </a>
                <a href="https://github.com/CastillonMiguel/A-Phase-Field-Approach-to-Fatigue-Analysis-Bridging-Theory-and-Simulation" target="_blank" class="pub-button btn-github">
                    💻<br>GitHub Repo
                </a>
                <a href="https://phasefieldfatigue.readthedocs.io" target="_blank" class="pub-button btn-docs">
                    📚<br>Documentation
                </a>
            </div>

    .. grid-item-card:: 
        :shadow: lg
        :class-card: third-pub-card

        .. raw:: html
        
            <div style="text-align: center; margin: 20px 0;">
                <h2 class="pub-title">
                    PhaseFieldX: An Open-Source Framework for Advanced Phase-Field Simulations
                </h2>
            </div>
        
        ^^^

        .. image:: https://raw.githubusercontent.com/CastillonMiguel/phasefieldx/main/docs/source/_static/logo_name.png
            :width: 400px
            :align: center

        **Authors:** M. Castillón  
        **Year:** 2025  
        **Journal:** Journal of Open Source Software  
        **Volume:** 10(108)  
        **Pages:** 7307  
        **DOI:** `10.21105/joss.07307 <https://doi.org/10.21105/joss.07307>`_

        This publication introduces PhaseFieldX, an open-source framework for advanced phase-field simulations. The project includes comprehensive documentation and is actively maintained on GitHub.

        +++

        **🔗 Related Resources**

        .. raw:: html

            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 15px 0;">
                <a href="https://joss.theoj.org/papers/10.21105/joss.07307" target="_blank" class="pub-button btn-journal">
                    📖<br>Journal Paper
                </a>
                <a href="https://github.com/CastillonMiguel/phasefieldx" target="_blank" class="pub-button btn-github">
                    💻<br>GitHub Repo
                </a>
                <a href="https://phasefieldx.readthedocs.io" target="_blank" class="pub-button btn-docs">
                    📚<br>Documentation
                </a>
            </div>


Doctoral Theses
---------------

.. grid-item-card::
    :shadow: lg
    :class-card: thesis-card

    .. raw:: html

        <div style="text-align: center; margin: 20px 0;">
            <h2 class="pub-title" style="font-size: 26px;">
                Numerical Methods and Algorithms for Phase-Field Fracture Modeling
            </h2>
        </div>

    ^^^

    **Author:** M. Castillón  
    **Year:** 2026  
    **Institution:** Universidad Politécnica de Madrid (UPM), Madrid, Spain  
    **Degree:** PhD Thesis  
    **DOI:** `10.20868/UPM.thesis.96840 <https://doi.org/10.20868/UPM.thesis.96840>`_

    This doctoral thesis presents novel numerical methods and algorithms for
    phase-field fracture modeling, with particular emphasis on fatigue crack
    propagation, crack area measurement, and robust constrained optimization
    techniques.

    The computational challenge of fatigue—where conventional phase-field
    approaches often require cycle-by-cycle simulations over thousands of
    loading cycles—is addressed through a framework that enables fatigue
    analysis using only monotonic loading. A specialized energy-controlled
    solver robustly traces the complete crack equilibrium path, including
    snap-back instabilities, allowing the numerical extraction of the
    compliance rate with respect to crack area. By combining these results
    with Linear Elastic Fracture Mechanics (LEFM) and Paris' law, fatigue
    life can be predicted with dramatically reduced computational cost.

    To improve the accuracy of geometric quantities obtained from
    phase-field simulations, the thesis introduces the Double Gradient
    Correction Method (DGCM). Based on the principle of energy
    equipartition, DGCM provides a mesh-independent estimate of crack area
    by exploiting the gradient contribution of the phase-field energy,
    remaining applicable to complex crack topologies and three-dimensional
    problems.

    The thesis also presents the Latent Variable Proximal Point (LVPP)
    algorithm, a robust framework for solving constrained phase-field
    problems. By reformulating inequality-constrained models through an
    auxiliary latent variable, LVPP transforms the original problem into a
    sequence of unconstrained saddle-point systems suitable for efficient
    Proximal Galerkin discretizations. The approach avoids the parameter
    sensitivity and convergence difficulties commonly associated with
    penalty-based methods.

    All methodologies developed in this thesis are implemented in
    *PhaseFieldX*, ensuring reproducibility and providing an open-source
    platform for future developments in computational fracture mechanics.

    +++

    **🔗 Related Resources**

    .. raw:: html

        <div style="display: grid; grid-template-columns: 1fr; gap: 10px; margin: 15px 0;">
            <a href="https://oa.upm.es/96840/" target="_blank" class="pub-button btn-journal">
                📖<br>PhD Thesis
            </a>
        </div>
