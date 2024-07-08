Installation manual
===================

To use this repository, you need to have the latest stable release of FEniCSx installed. The latest stable release of FEniCSx is version 0.8, which was released in April 2024. The easiest way to start using FEniCSx on MacOS and other systems is to install it using `conda`.

Follow these steps to set up your environment:

1. Create a new conda environment:
   
   .. code-block:: sh
   
      conda create -n phasefieldx-env

2. Activate the new environment:
   
   .. code-block:: sh
   
      conda activate phasefieldx-env

3. Install FEniCSx, `mpich`, and `pyvista` from the `conda-forge` channel:
   
   .. code-block:: sh
   
      conda install -c conda-forge fenics-dolfinx mpich pyvista pandas

4. Install gmsh
   
   .. code-block:: sh
   
      pip install --upgrade gmsh
   

5. Finally, install the code from this repository:
   
   .. code-block:: sh
   
      pip install phasefieldx

These steps will set up all the necessary dependencies for running the code in this repository. Make sure to activate the `phasefieldx-env` environment whenever you work with this project.

For more information and additional installation options, please visit the [FEniCSx Project download page](https://fenicsproject.org/download/).
