#!/bin/bash

set e

# Check if the current directory is named "scripts"
current_directory=$(basename "$PWD")

if [ "$current_directory" == "scripts" ]; then
  echo "Error: This script should be called from the root of the project."
  echo "Example: bash ./scripts/console.sh"
  exit 1
fi

#Change the following to the actual GPU devices you want to work on (e.g "0,1") or to NONE if you do not plan on using any GPUs
DEVICES="NONE"

#Change the following to your caching directory if you want persistent caching, else set it to NONE
CACHE_DIR="NONE"

docker run -it \
    --user $(id -u):$(id -g) \
    $([[ "$DEVICES" != "NONE" ]] && echo "--gpus=\"device=$DEVICES\"") \
    --ipc host \
    --env WANDB_API_KEY \
    -v "$(pwd)":/workspace \
    -w /workspace \
    $([ "$CACHE_DIR" != "NONE" ] && echo "--mount type=bind,source=$CACHE_DIR,target=/home/mamba/.cache") \
    nlp_template \
    bash

# if you plan on mounting a cache-folder you will have to create one, before you can run this script
