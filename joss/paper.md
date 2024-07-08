---
title: "PhaseFieldX: An Open-Source Framework for Advanced Phase-Field Simulations"
tags:
  - Fenicsx
  - Phase field
  - Fracture
  - Finite element
authors:
  - name: Miguel Castillón
    orcid: 0009-0008-6383-7103
    affiliation: "1, 2"
affiliations:
  - name: Universidad Politécnica de Madrid, José Gutiérrez Abascal, 2, 28006 Madrid, Spain
    index: 1
  - name: IMDEA Materials Institute, Eric Kandel 2, Getafe, 28906 Madrid, Spain
    index: 2
date: 8 July 2024
bibliography: paper.bib
---

# Summary

The **PhaseFieldX** project is designed to simulate and analyze material behavior using phase-field models, which provide a continuous approximation of interfaces, phase boundaries, and discontinuities such as cracks. Leveraging the robust capabilities of *FEniCSx* [@BarattaEtal2023; @ScroggsEtal2022; @BasixJoss; @AlnaesEtal2014], a renowned finite element framework for solving partial differential equations, this project facilitates efficient and precise numerical simulations. It supports a wide range of applications, including phase-field fracture, solidification, and other complex material phenomena, making it an invaluable resource for researchers and engineers in materials science.

![Logo](./images/logo_name.png){height="80pt"}

# Statement of Need

The **PhaseFieldX** project aims to advance phase-field modeling through open-source contributions. By leveraging the powerful *FEniCSx* framework, our goal is to enhance and broaden the application of phase-field simulations across various domains of materials science and engineering. We strive to make these advanced simulation techniques more accessible, enabling researchers and engineers to conduct more accurate and comprehensive scientific investigations. Through collaborative efforts, our mission is to deepen understanding, foster innovation, and contribute to the broader scientific community’s pursuit of knowledge in complex material behaviors.

# Applications

**PhaseFieldX** offers a wide range of applications in materials science and engineering:

- **Phase-Field Fracture**: Enables phase-field fracture simulations, considering various degradation functions and energy split methods such as volumetric-deviatoric [@Amor2009] or spectral decomposition [@Miehe2010]. It supports different formulations including isotropic, anisotropic, and hybrid [@Ambati2015].

- **Elasticity**: Provides tools for analyzing elasticity problems, foundational to phase-field fracture modeling.

- **Phase-Field Fatigue**: Introduces a phase-field model to study fatigue [@Carrara2020].

- **Other Phase-Field Models**: Facilitates the study of other phase-field models like the Allen-Cahn equation, expanding simulation capabilities to include various phase transformation phenomena.

These capabilities make **PhaseFieldX** a versatile and powerful tool for researchers and engineers aiming to investigate complex material behaviors through advanced phase-field simulations. A complete list of examples is available, demonstrating the breadth of possible applications and scenarios.

# Acknowledgements

This work was supported by the FPI grant PRE2020-092051, funded by MCIN/AEI /10.13039/501100011033.

# References
