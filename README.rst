PhaseFieldX
===========

.. image:: https://raw.githubusercontent.com/CastillonMiguel/phasefieldx/main/docs/source/_static/logo_name.png
   :target: https://phasefieldx.readthedocs.io/en/latest/index.html
   :alt: PhaseFieldX


Welcome to **PhaseFieldX**. `documentation <https://phasefieldx.readthedocs.io/en/latest/index.html>`_

.. image:: https://readthedocs.org/projects/phasefieldx/badge/?version=latest
    :target: https://phasefieldx.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    
.. image:: https://img.shields.io/pypi/v/phasefieldx
    :target: https://pypi.org/project/phasefieldx/
    :alt: PyPI Version

.. image:: https://img.shields.io/pypi/dm/phasefieldx.svg?label=Pypi%20downloads
    :target: https://pypi.org/project/phasefieldx/
    :alt: PyPI Downloads

.. image:: https://img.shields.io/github/license/CastillonMiguel/phasefieldx
    :target: https://github.com/CastillonMiguel/phasefieldx/blob/main/LICENSE
    :alt: License

.. image:: https://joss.theoj.org/papers/10.21105/joss.07307/status.svg
    :target: https://doi.org/10.21105/joss.07307
    :alt: Joss

.. image:: https://github.com/CastillonMiguel/phasefieldx/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/CastillonMiguel/phasefieldx/actions/workflows/testing.yml   
    :alt: Unit Testing
 
.. image:: https://deepwiki.com/badge.svg
   :target: https://deepwiki.com/CastillonMiguel/phasefieldx
   :alt: Ask DeepWiki

Introduction
------------
The **PhaseFieldX** project is designed to simulate and analyze material behavior using phase-field models, which provide a continuous approximation of interfaces, phase boundaries, and discontinuities such as cracks. Leveraging the robust capabilities of *FEniCSx*, a renowned finite element framework for solving partial differential equations, this project facilitates efficient and precise numerical simulations. It supports a wide range of applications, including phase-field fracture, solidification, and other complex material phenomena, making it an invaluable resource for researchers and engineers in materials science.


Purpose
-------
The **PhaseFieldX** project aims to advance phase-field modeling through open-source contributions. By leveraging the powerful *FEniCSx* framework, our goal is to enhance and broaden the application of phase-field simulations across various domains of materials science and engineering. We strive to make these advanced simulation techniques more accessible, enabling researchers and engineers to conduct more accurate and comprehensive scientific investigations. Through collaborative efforts, our mission is to deepen understanding, foster innovation, and contribute to the broader scientific community’s pursuit of knowledge in complex material behaviors.


Key Features
------------
- **Phase-Field Method:** The code employs the phase-field method, a versatile mathematical framework for modeling phenomena such as fracture, phase transitions, and pattern formation as diffuse processes. It enables the simulation of complex behaviors and multiple interacting phenomena within a unified framework.

- **FEniCSx Integration:** Integrated with FEniCSx, a powerful finite element framework, the code provides robust capabilities for solving partial differential equations governing phase-field simulations. This integration ensures efficient computation and adaptive mesh refinement, enhancing simulation accuracy and scalability.
  
- **User-Friendly Interface:** Designed with usability in mind, the code features an intuitive interface for defining material properties, boundary conditions, and simulation parameters. This interface caters to both novice users and experienced researchers, facilitating straightforward setup and execution of simulations.

- **Advanced Visualization:** The code includes advanced visualization tools to depict simulation results effectively. These tools enable comprehensive analysis of crack propagation, stress distributions, and other key quantities, supporting insightful interpretations and comparisons across simulations.


Installation Instructions
--------------------------
To use this repository, you need to have the latest stable release of FEniCSx installed. The latest stable release of FEniCSx is version 0.9. The easiest way to start using FEniCSx on MacOS and other systems is to install it using `conda`.

Follow these steps to set up your environment:

1. Create a new conda environment
   
   .. code-block::
   
      conda create -n phasefieldx-env

2. Activate the new environment
   
   .. code-block::
   
      conda activate phasefieldx-env

3. Install FEniCSx, `mpi4py`, `numpy`, `pandas`, ... from the `conda-forge` channel:
   
   .. code-block:: sh
   
      conda install -c conda-forge fenics-dolfinx=0.9.0 mpi4py numpy pyvista pandas pyvista

4. Install `gmsh`
   
   .. code-block::
   
      pip install --upgrade gmsh
   

5. Finally, install the code from this repository
   
   .. code-block::
   
      pip install phasefieldx


These steps will set up all the necessary dependencies for running the code in this repository. Make sure to activate the `phasefieldx-env` environment whenever you work with this project.

For more detailed installation options and information, please visit the `FEniCSx Project download page <https://fenicsproject.org/download/>`_.

We also provide a pre-built docker image with FEniCSx and phasefieldx installed. You pull this image using the command

.. code-block::

   docker pull ghcr.io/castillonmiguel/phasefieldx:main


Examples
--------
There are numerous examples available to demonstrate the usage of PhaseFieldX for various phase-field simulations. These examples cover different scenarios such as phase-field fracture, phase-field fatigue, and more complex material behavior simulations. Explore the examples in the `documentation <https://phasefieldx.readthedocs.io/en/latest/index.html>`_ to learn more.


API Documentation
-----------------
For detailed API documentation, including class references, function definitions, and usage examples, please refer to the `API documentation <https://phasefieldx.readthedocs.io/en/latest/api/index.html>`_.

You can also explore the project on DeepWiki — an AI-powered, interactive knowledge base built from the code and documentation: `Explore PhaseFieldX on DeepWiki <https://deepwiki.com/CastillonMiguel/phasefieldx>`_

Contributions and Feedback
--------------------------
We welcome contributions and feedback from the community to enhance the code's functionality, reliability, and user experience.To get started, please review our `Contributing Guidelines <https://phasefieldx.readthedocs.io/en/latest/extras/DeveloperNotes/main.html>`_ to share your insights and collaborate with fellow developers.

Thank you for choosing our Phase-Field Fracture simulation code. We trust this tool will prove invaluable in advancing your understanding of fracture mechanics and its practical applications.


Citing PhaseFieldX
------------------
There is a `paper about PhaseFieldX <https://doi.org/10.21105/joss.07307>`_.

If you use **PhaseFieldX** in your scientific research, please consider citing our work to support its development and increase its scientific visibility.

    Castillón, M. (2025). PhaseFieldX: An Open-Source Framework for Advanced Phase-Field Simulations. Journal of Open Source Software, 10(108), 7307, https://doi.org/10.21105/joss.07307


BibTex:

.. code:: latex

    @article{Castillon2025phasefieldx, 
      doi = {10.21105/joss.07307}, 
      url = {https://doi.org/10.21105/joss.07307}, 
      year = {2025}, 
      publisher = {The Open Journal}, 
      volume = {10}, 
      number = {108}, 
      pages = {7307},
      author = {Miguel Castillón}, 
      title = {PhaseFieldX: An Open-Source Framework for Advanced Phase-Field Simulations},
      journal = {Journal of Open Source Software} 
    }
