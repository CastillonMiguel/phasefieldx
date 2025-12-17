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
    
This section collects peer-reviewed scientific publications that have used
the *phasefieldx* software in their research. The listed works demonstrate
the application of *phasefieldx* to phase-field modeling, fracture mechanics,
and related computational mechanics problems, and serve as references for
users interested in validated and published use cases of the framework.

If you have used *phasefieldx* in your research and would like your publication
to be included, please consider contacting the developers or submitting a
request through the project repository.


.. grid:: 1
    :gutter: 1

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