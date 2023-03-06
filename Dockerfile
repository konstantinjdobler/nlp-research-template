# This is a Dockerfile intended to build an image containing all necessary dependencies

# -----------------
# Builder container
# -----------------

# Use fixed cache for layer caching
FROM --platform=$TARGETPLATFORM condaforge/mambaforge:22.11.1-4 as builder

ARG TARGETOS TARGETARCH TARGETPLATFORM

# NOTE: The regular output file of conda-lock for linux/amd64 is conda-linux-64.lock, so we need to rename it to conda-linux-amd64.lock
COPY conda-${TARGETOS}-${TARGETARCH}.lock /locks/conda-${TARGETOS}-${TARGETARCH}.lock

# Install packages from lockfile, cache packages for faster build times when more packages are added
# Need unique cache per platform
RUN --mount=type=cache,target=/opt/conda/pkgs,id=conda-${TARGETPLATFORM} mamba create -p /opt/env --copy --file /locks/conda-${TARGETOS}-${TARGETARCH}.lock 

# -----------------
# Primary container
# -----------------
FROM --platform=$TARGETPLATFORM nvidia/cuda:11.8.0-cudnn8-runtime-ubi8
# copy over the generated environment
COPY --from=builder /opt/env /opt/env
ENV PATH="/opt/env/bin:${PATH}"

# For debugging
# RUN echo $(which python)
# RUN python -V
