#!/bin/bash

# Usage: bash ./scripts/run-in-docker.sh [OPTIONS] [COMMAND]
# ---------------------
# Example for opening a console without GPUs: bash ./scripts/run-in-docker.sh bash
# Example for opening a console with GPUs 0,1,5,7: bash ./scripts/run-in-docker.sh -g 0,1,5,7 bash
# Example with GPUs a custom docker image and training script: bash ./scripts/run-in-docker.sh -g 0,1,2,3 -i my-cool/docker-image:latest python train.py ...
# ---------------------
# The current directory is mounted to /workspace in the docker container.
# We automatically detect W&B login credentials in the ~/.netrc file and pass them to the docker container. To store them, do wandb login once on the host machine.

# Default values
image="konstantinjdobler/nlp-research-template:latest"
command="bash"
gpus="none"

set -e

# Check if the current directory is named "scripts"
current_directory=$(basename "$PWD")

if [ "$current_directory" == "scripts" ]; then
    echo "Error: This script should be called from the root of the project."
    echo "Example: bash ./scripts/run-in-docker.sh"
    exit 1
fi

# Function to parse the command line arguments
parse_arguments() {
    local in_command=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
        -g)
            shift
            gpus="$1"
            ;;
        -i)
            shift
            image="$1"
            ;;
        *)
            if [ "$in_command" = false ]; then
                command="$1"
            else
                command="${command} $1"

            fi
            in_command=true
            ;;
        esac
        shift
    done
}

# Call the function to parse arguments
parse_arguments "$@"

# Rest of your script
echo "image: $image"
echo "command: $command"
echo "gpus: $gpus"

# Look for WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' ~/.netrc)
    if [ -z "$WANDB_API_KEY" ]; then
        echo "WANDB_API_KEY not found"
    else
        echo "WANDB_API_KEY found in ~/.netrc"
    fi
else
    echo "WANDB_API_KEY found in environment"
fi

# Tested on chairserver w/ 4x A6000 - doesn't bring speedups
# # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#when-using-ddp-on-a-multi-node-cluster-set-nccl-parameters
# export NCCL_NSOCKS_PERTHREAD=4
# export NCCL_SOCKET_NTHREADS=2
#  --env NCCL_NSOCKS_PERTHREAD --env NCCL_SOCKET_NTHREADS \

# NOTE: --ipc=host for full RAM and CPU access or -m XXXG --cpus XX to control access to RAM and cpus
# You probably want to add addiitonal mounts to your homefolder, e.g. -v /home/username/data:/home/username/data
# IMPORTANT: Use -v /home/username/.cache:/home/mamba/.cache to mount your cache folder to the docker container. The username inside the container is "mamba".
# Other common mounts:  -v /scratch/username/:/scratch/username/ -v /home/username/data/:/home/username/data/
# Add -p 5678:5678 to expose port 5678 for remote debugging. But keep in mind that this will block the port for other docker users on the server, so you might have to choose a different one.
docker run --rm -it --ipc=host \
    -v "$(pwd)":/workspace -w /workspace \
    --user $(id -u):$(id -g) \
    --env XDG_CACHE_HOME --env HF_DATASETS_CACHE --env WANDB_CACHE_DIR --env WANDB_DATA_DIR --env WANDB_API_KEY \
    --gpus=\"device=${gpus}\" $image $command
