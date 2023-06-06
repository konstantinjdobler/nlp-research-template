#!/bin/bash

set e

# Check if the current directory is named "scripts"
current_directory=$(basename "$PWD")

if [ "$current_directory" == "scripts" ]; then
  echo "Error: This script should be called from the root of the project."
  echo "Example: bash ./scripts/console.sh"
  exit 1
fi

docker run -it \
    --gpus='device=CHANGE_ME' \
    --ipc host \
    --env WANDB_API_KEY \
    -v "CHANGE_ME/cache:/home/mamba/.cache" \
    -v "$(pwd)":/workspace \
    -w /workspace \
    DOCKER_TAG_CHANGE_ME \
    bash

# the mounted cache folder has to exist somewhere, before this script can be run