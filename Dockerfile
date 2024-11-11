# We choose ubuntu 22.04 as our base docker image (linux/amd64)
FROM ghcr.io/fenics/dolfinx/dolfinx:v0.9.0@sha256:61f17eb5de64721e8bbfc06b0e715bd8d0c3b6f8ec37d30f6985016c6ac68472

ENV PYVISTA_JUPYTER_BACKEND="html"

# Requirements for pyvista
RUN apt-get update && apt-get install -y libgl1 libglx-mesa0
RUN apt-get update && apt-get install -y libxrender1 
RUN apt-get update && apt-get install -y xvfb
RUN apt-get update && apt-get install -y nodejs

COPY . /repo
WORKDIR /repo

RUN python3 -m pip install vtk

RUN python3 -m pip install jupyter-book 
RUN python3 -m pip install jupyter
RUN python3 -m pip install pyvista[all]
RUN python3 -m pip install trame-vuetify
RUN python3 -m pip install ipywidgets

RUN python3 -m pip install .

