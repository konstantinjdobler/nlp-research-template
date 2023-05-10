# An opinionated template for NLP research code

[![Docker Hub](https://img.shields.io/docker/v/konstantinjdobler/nlp-research-template/torch2.0.0-cuda11.8?color=blue&label=docker&logo=docker)](https://hub.docker.com/r/konstantinjdobler/nlp-research-template/tags) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![License: MIT](https://img.shields.io/github/license/konstantinjdobler/nlp-research-template?color=green)

NLP research template for training language models from scratch using PyTorch + PyTorch Lightning + Weights & Biases + HuggingFace. It's built to be customized but provides comprehensive, sensible default functionality.

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

<details><summary>using Windows</summary>

<p>

When installing Mamba on Windows you will need to add its Script-folder to your **PATH** environment variable. This folder is by default located at 
```bash
C:\Users\user\mambaforge\Scripts
```
Afterwards, you will be able to run `mamba` commands simply from your command line. Note that you might not be able to activate your created environments within `powershell`. A simple workaround would be to use `cmd` instead.
</details>

</details>

<details><summary>Installing <code>conda-lock</code></summary>

<p>

The preferred method is to install conda-lock into your `mamba` / `conda` `base` environment using `mamba install -c conda-forge -n base conda-lock`. Then, you can access conda-lock via the automatic subcommand discovery (e.g. `mamba lock --version`). Otherwise, visit the [conda-lock repo](https://github.com/conda/conda-lock).

</details>

### Environment

Lockfiles are an easy way to **exactly** reproduce an environment.

After having installed `mamba` and `conda-lock`, you can create lockfiles from a `environment.yml` file, simply by running `mamba lock` in the same working directory.

To create a new `mamba` environment with all necessary dependencies installed from such a lockfile , run:

```bash
mamba lock install --name <environmentName>
```
You can then activate your environment with
```bash
mamba activate <environmentName>
```
For more commands (e.g. updating or removing environments) have a look at the [conda-documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#removing-an-environment).

<details><summary>Setup on <code>ppc64le</code></summary>

<p>

It's slightly more tricky because the official channels do not provide packages compiled for <code>ppc64le</code>. However, we can use the amazing [Open-CE channel](https://ftp.osuosl.org/pub/open-ce/current/) instead. A lockfile containing the relevant dependencies is already prepared in <code>ppc64le.conda-lock.yml</code>.

```bash
mamba lock install --name <gpt4> --file ppc64le.conda-lock.yml
```

Dependencies for <code>ppce64le</code> should go into the seperate <code>ppc64le.environment.yml</code> file. Use the following command to generate a new lockfile after updating the dependencies:

```bash
mamba lock --file ppc64le.environment.yml --lockfile ppc64le.conda-lock.yml
```

</p>
</details>

### Docker

For fully reproducible environments and running on HPC clusters, we provide pre-built docker images at [konstantinjdobler/nlp-research-template](https://hub.docker.com/r/konstantinjdobler/nlp-research-template/tags). We also provide a `Dockerfile` that allows you to build new docker images with updated dependencies:

```bash
docker build --tag <username>/<imagename>:<tag> --platform=linux/<amd64/ppc64le> .
```

## Development
We provide an example setup for a remote development environment on a GPU server using [VS Code](https://code.visualstudio.com/), [Remote - SSH](https://code.visualstudio.com/docs/remote/ssh), and [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers). This allows you to use the same environment for both development and production. For more details, see [here](https://code.visualstudio.com/docs/remote/remote-overview).

Before you can start successfully, you have to adapt `"runArgs": ["--ipc=host", "--gpus", "device=CHANGE_ME"]` and `"mounts": ["source=/CHANGE_ME/.cache,target=/mamba/.cache,type=bind"]` in [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json). Additionally, you can set the `WANDB_API_KEY` in your remote environment; it will then be automatically mapped into the container.

## Training

To start a language model MLM training, run:

```bash
python train.py --data /path/to/data/dir --model roberta-base --gpus 2 --offline
```

By default, `train.txt` and `dev.txt` are expected in the data directory. To see an overview over all options and their defaults, run `python train.py --help`.
We have disabled Weights & Biases syncing with the `--offline` flag. To enable W&B, enter your `WANDB_ENTITY` and `WANDB_PROJECT` in [dlib/frameworks/wandb.py](dlib/frameworks/wandb.py) and simply omit the `--offline` flag.

### Using the Docker environment for training
To run the training code inside the docker environment, use a `docker run` command like this:
```bash
docker run --rm -it --ipc=host --gpus='"device=0,1"' -v "($pwd)":/workspace -w /workspace -v /path/to/data:/data/in/container python train.py --gpus -1 ...
```
The `--gpus='"device=0,1"'` flag (change this to use the GPUs you actually want) selects the GPUs with indices `0` and `1` for the container and `train.py --gpus -1` makes the training script use all available GPUs (which are only the ones selected with the docker flag). 

<details><summary>Weights & Biases + Docker</summary>

<p>

Weights & Biases needs access to your `WANDB_API_KEY` to be able to log results. Either set `WANDB_API_KEY` on your host machine and use the `docker` flag `--env WANDB_API_KEY` or mount your `.netrc` file into the docker container like so: `-v ~/.netrc:~/.netrc`.
</p>
</details>

<details><summary>Using Docker with SLURM / <code>pyxis</code></summary>

<p>

For security reasons, `docker` might be disabled on your HPC cluster. You might be able to use the SLURM plugin `pyxis` instead like this:

```bash
srun ... --container-image konstantinjdobler/nlp-research-template:torch2.0.0-cuda11.8 --container-name torch-cuda python train.py ...
```

This uses [`enroot`](https://github.com/NVIDIA/enroot) under the hood to import your docker image and run your code inside the container. See the [`pyxis` documentation](https://github.com/NVIDIA/pyxis) for more options, such as `--container-mounts` or `--container-writable`.

If you want to run an interactive session with bash don't forget the `--pty` flag, otherwise the environment won't be activated properly.
</p>
</details>





