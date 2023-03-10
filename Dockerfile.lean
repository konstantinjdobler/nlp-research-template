# This is a Dockerfile intended to build an image containing all necessary dependencies
# The result is a "lean" container, which means that it does not contains a full-fledged conda installation, only the required dependencies

# -----------------
# Builder container
# -----------------

FROM --platform=$TARGETPLATFORM condaforge/mambaforge:22.11.1-4 as builder

RUN . /opt/conda/etc/profile.d/conda.sh && \
    mamba create --name lock && \
    conda activate lock && \
    mamba install --yes pip conda-lock>=1.2.2 setuptools wheel 

ARG TARGETOS TARGETARCH TARGETPLATFORM

COPY conda-lock.yml /locks/conda-lock.yml
# HACK: overwrite generic lockfile if TARGETARCH=ppc64le, otherwise ${TARGETARCH}.conda-lock.yml will not exist
# This way, we have a Dockerfile that works for all architectures
COPY *${TARGETARCH}.conda-lock.yml /locks/conda-lock.yml

# install dependencies from lockfile, cache packages for faster build times when more packages are added
RUN  --mount=type=cache,target=/opt/conda/pkgs,id=conda-${TARGETPLATFORM} \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate lock && \
    conda-lock install \
        --mamba \
        --copy \
        --prefix /opt/env \
        /locks/conda-lock.yml
# optionally you can perfom some more cleanup on your conda install after this
# to get a leaner conda environment


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

