# This is a Dockerfile intended to build an image containing all necessary dependencies for linux-ppc64le

# -----------------
# Builder container
# -----------------
FROM --platform=linux/ppc64le condaforge/mambaforge:latest as builder

COPY conda-linux-ppc64le.lock /locks/conda-linux-ppc64le.lock

# Install packages from lockfile, cache packages for faster build times when more packages are added
RUN --mount=type=cache,target=/opt/conda/pkgs mamba create -p /opt/env --copy --file /locks/conda-linux-ppc64le.lock 

# -----------------
# Primary container
# -----------------
FROM --platform=linux/ppc64le nvidia/cuda:11.8.0-cudnn8-runtime-ubi8
# copy over the generated environment
COPY --from=builder /opt/env /opt/env
ENV PATH="/opt/env/bin:${PATH}"
RUN echo $(which   python)
RUN ls /opt/env/bin
RUN python -V
