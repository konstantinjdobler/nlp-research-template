# syntax=docker/dockerfile:1.5.2

# This Dockerfile produces a container with all dependencies installed into an environment called "research"
# Additionally, the container has a full-fledged micromamba installation, which is a faster drop-in replacement for conda
# When inside the container, you can install additional dependencies with `conda install <package>`, e.g. `conda install scipy`
# The actual installation is done by micromamba, we have simply provided an alias to link the conda command to micromamba

# The syntax line above is crucial to enable variable expansion for type=cache=mount commands

# We can use the OS_PREFIX build arg to choose between ubi8 and ubuntu as base image (for amd64 processor architecture)
ARG OS_SELECTOR=ubi8

# Load micromamba container to copy from later
FROM --platform=$TARGETPLATFORM mambaorg/micromamba:1.4.2 as micromamba


####################################################
################ BASE IMAGES #######################
####################################################

# -----------------
# base image for amd64
# -----------------
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-runtime-ubi8 as amd64ubi8
# Install compiler for .compile() with PyTorch 2.0 and nano for devcontainers
RUN yum install -y gcc gcc-c++ nano && yum clean all
# Copy lockfile to container
COPY conda-lock.yml /locks/conda-lock.yml

# -----------------
# devcontainer base image for amd64 using Ubuntu
# SLURM + pyxis has a bug on our cluster, where the automatic activation of the conda environment fails if the base image is ubuntu
# But Ubuntu works better for devcontainers than ubi8
# So we use Ubuntu for devcontainers and ubi8 for actual deployment on the cluster
# -----------------
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as amd64ubuntu
# Install compiler for .compile() with PyTorch 2.0 and nano for devcontainers
RUN apt-get update && apt-get install -y gcc g++ nano openssh-client && apt-get clean
# Copy lockfile to container
COPY conda-lock.yml /locks/conda-lock.yml

# -----------------
# base image for ppc64le
# -----------------
FROM --platform=linux/ppc64le nvidia/cuda:11.8.0-cudnn8-runtime-ubi8 as ppc64leubi8
# Install compiler for .compile() with PyTorch 2.0
RUN yum install -y gcc gcc-c++ && yum clean all
# Copy ppc64le specififc lockfile to container
COPY ppc64le.conda-lock.yml /locks/conda-lock.yml



####################################################
################ FINAL IMAGE #######################
####################################################

# -----------------
# Final build image - we choose the correct base image based on the target architecture and OS
# -----------------
ARG TARGETARCH
FROM ${TARGETARCH}${OS_SELECTOR} as final
# From https://github.com/mamba-org/micromamba-docker#adding-micromamba-to-an-existing-docker-image
# The commands below add micromamba to an existing image to give the capability to ad-hoc install new dependencies

####################################################
######### Adding micromamba starts here ############
####################################################
USER root

# if your image defaults to a non-root user, then you may want to make the
# next 3 ARG commands match the values in your image. You can get the values
# by running: docker run --rm -it my/image id -a
ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
# Optional: if you want to customize the ENTRYPOINT and have a conda
# environment activated, then do this:
# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "my_entrypoint_program"]

# You can modify the CMD statement as needed....
CMD ["/bin/bash"]

####################################################
######### Adding micromamba stops here #############
####################################################

# Switch to root user to grant necessary permissions
USER root
# Give user permission to gcc
RUN chown $MAMBA_USER:$MAMBA_USER /usr/bin/gcc
# Necessary to prevent permission error when micromamba tries to install pip dependencies from lockfile
RUN chown $MAMBA_USER:$MAMBA_USER /locks/
RUN chown $MAMBA_USER:$MAMBA_USER /locks/conda-lock.yml
# Provide conda alias for micromamba
RUN echo "alias conda=micromamba" >> /usr/local/bin/_activate_current_env.sh

# Switch back to micromamba user
USER $MAMBA_USER
ARG TARGETPLATFORM
# Use line below to debug if cache is correctly mounted
# RUN --mount=type=cache,target=$MAMBA_ROOT_PREFIX/pkgs,id=conda-$TARGETPLATFORM,uid=$MAMBA_USER_ID,gid=$MAMBA_USER_GID ls -al /opt/conda/pkgs
# Install dependencies from lockfile into environment, cache packages in /opt/conda/pkgs
RUN --mount=type=cache,target=$MAMBA_ROOT_PREFIX/pkgs,id=conda-$TARGETPLATFORM,uid=$MAMBA_USER_ID,gid=$MAMBA_USER_GID \
    micromamba create --name research --yes --file /locks/conda-lock.yml

# Set conda-forge as default channel (otherwise no default channel is set)
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba config prepend channels conda-forge --env
# Disable micromamba banner at every command
RUN micromamba config set show_banner false --env

# Install optional tricky pip dependencies that do not work with conda-lock
# RUN micromamba run -n research pip install example-dependency --no-deps --no-cache-dir

# Use our environment `research` as default
ENV ENV_NAME=research