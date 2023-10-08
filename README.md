# An opinionated template for NLP research code

[![Docker Hub](https://img.shields.io/docker/v/konstantinjdobler/nlp-research-template/latest?color=blue&label=docker&logo=docker)](https://hub.docker.com/r/konstantinjdobler/nlp-research-template/tags)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Linter](https://img.shields.io/badge/linter-ruff-blue)
![License: MIT](https://img.shields.io/github/license/konstantinjdobler/nlp-research-template?color=green)

NLP research template for training language models using PyTorch + Lightning + Weights & Biases + HuggingFace. It's built to be customized but provides comprehensive, sensible default functionality.

If you are not doing NLP or want to use your own training code or template, the setup and environment tooling with Docker, `mamba`, and `conda-lock` in this template might still be interesting for you.

## Setup

### Preliminaries

It's recommended to use [`mamba`](https://github.com/mamba-org/mamba) to manage dependencies. `mamba` is a drop-in replacement for `conda` re-written in C++ to speed things up significantly (you can stick with `conda` though). To provide reproducible environments, we use `conda-lock` to generate lockfiles for each platform.

<details><summary>Installing <code>mamba</code></summary>

<p>

On Unix-like platforms, run the snippet below. Otherwise, visit the [mambaforge repo](https://github.com/conda-forge/miniforge#mambaforge). Note this does not use the Anaconda installer, which reduces bloat.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

</details>

<details><summary>Installing <code>conda-lock</code></summary>

<p>

The preferred method is to install `conda-lock` using `pipx install conda-lock`. For other options, visit the [conda-lock repo](https://github.com/conda/conda-lock). For basic usage, have a look at the commands below:

```bash
conda-lock install --name gpt5 conda-lock.yml # create environment with name gpt5 based on lockfile
conda-lock # create new lockfile based on environment.yml
conda-lock --update <package-name> # update specific packages in lockfile
```

</details>

### Environment

Lockfiles are an easy way to **exactly** reproduce an environment.

After having installed `mamba` and `conda-lock`, you can create a `mamba` environment named `gpt5` from a lockfile with all necessary dependencies installed like this:

```bash
conda-lock install --name gpt5 conda-lock.yml
```

You can then activate your environment with

```bash
mamba activate gpt5
```

To generate new lockfiles after updating the `environment.yml` file, simply run `conda-lock -f environment.yml`.

<details><summary>Setup on <code>ppc64le</code></summary>

<p>

**If you're not using a PowerPC machine, do not worry about this.**

Whenever you create an environment for a different processor architecture, some packages (especially `pytorch`) need to be compiled specifically for that architecture. IBM PowerPC machines for example use a processor architecture called <code>ppc64le</code>.
Setting up the environment <code>ppc64le</code> is a bit tricky because the official channels do not provide packages compiled for <code>ppc64le</code>. However, we can use the amazing [Open-CE channel](https://ftp.osuosl.org/pub/open-ce/current/) instead. A lockfile containing the relevant dependencies is already prepared in <code>ppc64le.conda-lock.yml</code> and the environment again can be simply installed with:

```bash
conda-lock install --name gpt5-ppc64le ppc64le.conda-lock.yml
```

Dependencies for <code>ppc64le</code> should go into the separate <code>ppc64le.environment.yml</code> file. Use the following command to generate a new lockfile after updating the dependencies:

```bash
conda-lock --file ppc64le.environment.yml --lockfile ppc64le.conda-lock.yml
```

</p>
</details>

### Docker (recommended)

For fully reproducible environments and running on HPC clusters, we provide pre-built docker images at [konstantinjdobler/nlp-research-template](https://hub.docker.com/r/konstantinjdobler/nlp-research-template/tags). We also provide a `Dockerfile` that allows you to build new docker images with updated dependencies:

```bash
# first update `environment.yml` with your dependencies
# then this command will create a new conda-lock.yml file
conda-lock -f environment.yml
```

```bash
# this automatically uses your latest conda-lock.yml to create a reproducible docker image
docker build --tag <username>/<imagename>:<tag> --platform="linux/amd64" .
```

The specified username should be your personal [`dockerhub`](https://hub.docker.com) username. This will make distribution and usage of your images easier with `docker push/pull <your image>`.

We also provide shell commands and a convenience script to run all your training commands inside docker (recommended).

## Training

After all of this setup you are finally ready for some training. First of all, you need to create your data directory with a `train.txt` and `dev.txt`. Then you can start a training run in your environment with:

```bash
python train.py -n <run-name> -d /path/to/data --model roberta-base --offline
```

To see an overview over all options and their defaults, run `python train.py --help` or have a look inside [`args.py`](./args.py). We have disabled Weights & Biases syncing with the `--offline` flag. If you want to log your results, enable W&B as described [here](#weights--biases) and omit the `--offline` flag.

<details><summary>Using GPUs for hardware acceleration</summary>

<p>

By default, `train.py` tries to use a single CUDA GPU if available. If you want to train on multiple GPUs, increase the `--num_devices` flag (this then uses `DistributedDataParallel` under the hood). **IMPORTANT:** you should always select the GPUs that are visible to the script via the `CUDA_VISIBLE_DEVICES` environment variable (e.g. `CUDA_VISIBLE_DEVICES=0,2 python train.py ...`) or via the docker flags if training inside a container (recommended). To use different hardware accelerators, use the `--accelerator` flag. You can use advanced parallel training strategies with `--distributed_strategy`.

</p>
</details>

### Using the Docker for training **(recommended)**

To conveniently run the training code inside a docker container, you can use the [run-in-docker.sh](./scripts/run-in-docker.sh) script.

```bash
# execute the training inside your container
# -g 2 means only GPU 2 is visible to the script
# -g 0,2 would make the GPUs 0 and 2 visible
bash ./scripts/run-in-docker.sh -g 2 python train.py --num_devices 1 -n <run-name> -d /path/to/data/ --model roberta-base --offline
```

By default (no `-g` flag), no GPUs are available inside the container. You probably want to adjust the `run-in-docker.sh` script to add your own mounts for data and other things you want to load / save.

**Docker + GPUs:** You should **always select specific GPUs** to be visible inside the container. When using the `run-in-docker.sh` script, use the `-g` flag. When using docker natively, use e.g. `--gpus='"device=0,7"'` (for the GPUs `0` and `7`) and adjust the `--num_devices` flag according to your number of selected GPUs. Yes, the weird format of `--gpus='"device=0,7"'` is important, otherwise the shell might not pass the flag correctly to `nvidia-docker` (official Nvidia recommendation).

<details><summary>Single-line docker command</summary>

<p>

You can start a script inside a docker container in a single command:

```bash
docker run -it --user $(id -u):$(id -g) --ipc host -v "$(pwd)":/workspace -w /workspace --gpus='"device=7"' konstantinjdobler/nlp-research-template:latest python train.py --num_devices=1 ...
```

Since we have not mounted any cache directories (only the current working directory with `$(pwd)`), nothing that is written to disk outside `$(pwd)` is persistent in this example. You can add those with `-v` or `--mount`.

</p>
</details>

<details><summary>Using Docker with SLURM / <code>pyxis</code></summary>

<p>

For security reasons, `docker` might be disabled on your HPC cluster. You might be able to use the SLURM plugin `pyxis` instead like this:

```bash
srun ... --container-image konstantinjdobler/nlp-research-template:latest python train.py ...
```

This uses [`enroot`](https://github.com/NVIDIA/enroot) under the hood to import your docker image and run your code inside the container. See the [`pyxis` documentation](https://github.com/NVIDIA/pyxis) for more options, such as `--container-mounts` or `--container-writable`.

It might take a long time to start the container. You can prepare this by doing `enroot import docker://konstantinjdobler/nlp-research-template:latest -o prepared-image.sqsh` and then modify the `srun`:

```bash
srun ... --container-image /path/to/prepared-image.sqsh python train.py ...
```

If you want to run an interactive session with bash don't forget the `--pty` flag.

</p>
</details>

### Weights & Biases

[Weights & Biases](https://wandb.ai/site) allows you to easily log metrics, training results, checkpoints, and hyperparameters. To enable Weights & Biases, enter your `WANDB_ENTITY` and `WANDB_PROJECT` in [train.py](train.py) and omit the `--offline` flag for training.

<details><summary>Weights & Biases + Docker</summary>

<p>

When using docker we also have to get our `WANDB_API_KEY` inside the container. You can find your personal API key at [wandb.ai/authorize](https://app.wandb.ai/authorize). Set `WANDB_API_KEY` on your host machine and use the `docker` flag `--env WANDB_API_KEY` when starting your run. Or just use the `run-in-docker.sh` script, which will try to parse the `WANDB_API_KEY` from your `~/.netrc` file (or get it from the environment).

</p>
</details>

### Configs

To save the exact configurations of experiments and save yourself some time typing out arguments in the command line, you can use `.yml` style config files supplied via the `--config_path` argument. You can also combine multiple configs. The order of importance is default args < config args (multiple configs are resolved in order) < command line args.

```bash
python train.py --config_path ./cfgs/example.yml ./cfgs/llama-from-scratch.yml --devices 8 -n my-training-run ...
```

## Development

If you want to connect to a remote host machine with GPUs for development, we recommend the VS Code [Remote-SSH](https://code.visualstudio.com/docs/remote/ssh) extension.

### Dev Containers **(recommended)**

Ideally, you should also do your development inside the same docker container to reduce a mismatch between training and development. For this, use VS Code `Dev Containers`. They allow you to develop in VS Code inside a docker container with full support for IntelliSense, type hints and more. The template already contains a `.devcontainer` directory, where all the settings for it are stored - you can start right away!

<details><summary>VS Code <code>Dev Container</code> example</summary>

<p>

After having installed the [Remote-SSH-](https://code.visualstudio.com/docs/remote/ssh), and [Dev Containers-Extension](https://code.visualstudio.com/docs/devcontainers/containers), you set up your `Dev Container` in the following way:

1. Establish the SSH-connection with the host by opening your VS Code command pallette and typing <code>Remote-SSH: Connect to Host</code>. Now you can connect to your host machine.
2. Open the folder that contains this template on the host machine.
3. VS Code will automatically detect the `.devcontainer` directory and ask you to reopen the folder in a Dev Container. Alternatively, use the command pallette and type <code>Dev Containers</code>.
4. Press <code>Reopen in Container</code> and wait for VS Code to set everything up. for the first time or when you change `devcontainer.json`, you will need to do <code>Rebuild and reopen in Container</code>.

There is a bit of setup: for a proper dev environment, you will need to configure mounts (cache directories, your datasets, ...) and environment variables like for a regular docker run command, have a look inside [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json).
`conda-lock` is automatically installed for you but you have to add the `--micromamba` flag inside the Dev Container (e.g. `conda-lock --micromamba -f environment.yml`).

If you want to use GPUs for development, you also need to specify the GPU you want to use in [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json). However, this is a bit cumbersome if you are often switching between GPUs. Alternatively, you edit your code in the Dev Container (without a GPU) but start all actual development runs of your script like you would for training with `run-in-docker.sh` and select the GPU ad-hoc. The nice advantage of Dev Containers is that you are still using the exact same docker container for both.

</p>
</details>

### `mamba` and `conda-lock`

Sometimes it's just quicker or unavoidable to create an environment via `conda-lock install --name gpt5 conda-lock.yml` instead of using Docker. In most cases, this is fine since we are using lockfiles but there might be some tricky edge cases depending on the platform and OS. Just be careful to keep any local environments and your docker containers in sync. Docker containers also allow more advanced support for compiled CUDA kernels such as FlashAttention.

### Code style

We use the `ruff` linter and `black` formatter. You should install their VS Code extensions and enable "Format on Save" inside VS Code.
