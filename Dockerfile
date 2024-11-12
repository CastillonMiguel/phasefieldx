# We choose ubuntu 22.04 as our base docker image
FROM ghcr.io/fenics/dolfinx/dolfinx:v0.9.0

COPY . /repo
WORKDIR /repo

# (from fenics: dolfinx/docker/Dockerfile.end-user)
RUN pip install --no-cache-dir jupyter jupyterlab

# pyvista dependencies from apt
RUN apt-get -qq update && \
    apt-get -y install libgl1-mesa-dev xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install pyvista from PyPI. pyvisa depends on (py)vtk), but vtk wheels are not
# available on pypi for linux/arm64, so we use a custom build wheel.
# matplotlib improves plotting quality with better color maps and
# properly rendering colorbars.
# trame is the preferred backend for pyvista.
RUN dpkgArch="$(dpkg --print-architecture)"; \
    pip install matplotlib; \
    case "$dpkgArch" in amd64) \
      pip install --no-cache-dir pyvista[trame]==${PYVISTA_VERSION} ;; \
    esac; \
    case "$dpkgArch" in arm64) \
      pip install --no-cache-dir https://github.com/finsberg/vtk-aarch64/releases/download/vtk-9.3.0-cp312/vtk-9.3.0.dev0-cp312-cp312-linux_aarch64.whl && \
      pip install --no-cache-dir pyvista[trame]==${PYVISTA_VERSION} ;; \
    esac; \
    pip cache purge

EXPOSE 8888/tcp
ENV SHELL /bin/bash
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

RUN python3 -m pip install .
