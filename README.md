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

</details>

<details><summary>Installing <code>conda-lock</code></summary>

<p>

The preferred method is to install conda-lock into your `mamba` / `conda` `base` environment using `mamba install -c conda-forge -n base conda-lock`. Then, you can access conda-lock via the automatic subcommand discovery (e.g. `mamba lock --version`). Otherwise, visit the [conda-lock repo](https://github.com/conda/conda-lock). For basic usage, have a look at the commands below:

```bash
mamba lock install --name gpt5 conda-lock.yml # create environment with name gpt5 based on lockfile
mamba lock # create new lockfile based on environment.yml
mamba lock --update <package-name> # update specific packages in lockfile
```

</details>


### Environment

Lockfiles are an easy way to **exactly** reproduce an environment.

After having installed `mamba` and `conda-lock`, you can create a `mamba` environment named `gpt5` from a lockfile with all necessary dependencies installed like this:

To create a new `mamba` environment with all necessary dependencies installed from such a lockfile , run:

```bash
mamba lock install --name gpt5 conda-lock.yml
```

You can then activate your environment with
```bash
mamba activate gpt5
```

To generate new lockfiles after updating the `environment.yml` file, simply run `mamba lock` in the directory with your `environment.yml` file.

For more advanced usage of environments (e.g. updating or removing environments) have a look at the [conda-documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#removing-an-environment).

<details><summary>Setup on <code>ppc64le</code></summary>

<p>

If you wish to create an environment for a different architecture, you will need to use the packages suited for it. In the case of <code>ppc64le</code> this is a little bit tricky because the official channels do not provide packages compiled for it. However, we can use the amazing [Open-CE channel](https://ftp.osuosl.org/pub/open-ce/current/) instead. We prepared a lockfile containing the relevant dependencies already in <code>ppc64le.conda-lock.yml</code>.

```bash
mamba lock install --name <gpt4> --file ppc64le.conda-lock.yml
# this is still the wrong command
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
The specified username should be your personal [`dockerhub`](https://hub.docker.com) username. This will make distribution and reusage of your images more easy.

## Development
Development for ML can be quite resource intensive. If possible, you can make use of a more powerful host machine to which you connect to with your local PC and start your development on.

This workflow is simplified a lot by using [VS Code](https://code.visualstudio.com/) with the [Remote-SSH-](https://code.visualstudio.com/docs/remote/ssh), and [Dev Containers-Extension](https://code.visualstudio.com/docs/devcontainers/containers). For more details, see [here](https://code.visualstudio.com/docs/remote/remote-overview). Typically, you would want to connect to the host machine via `SSH` and then open your `DEV Container` afterwards. The template already contains a `.devcontainer` directory, where all the settings for it are stored in JSON-format.

<details><summary>VS Code example</summary>

<p>

After having installed both extensions, you set up your `DEV Container` in the following way.

1. Establish the SSH-connection with the host by opening your VS Code command pallet and typing <code>Remote-SSH: Connect to Host</code>. Now you can connect to your host machine.
2. Open the folder that contains this template on the host machine.
3. VS Code will automatically detect the `.devcontainer` directory and ask you to reopen the folder in a DEV-Container.
4. Press `Reopen in Container' and wait for VS Code to set everything up.

When using this workflow you will have to adapt `"runArgs": ["--ipc=host", "--gpus", "device=CHANGE_ME"]` in [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) and specify the GPU-devices you are actually going to use on the host-machine for your development.

Additionally, you can set the `WANDB_API_KEY` in your remote environment; it will then be automatically mapped into the container.

</p>
</details>



## Training

After all of this setup you are finally ready for some training. First of all, you need to create your data directory with a `train.txt` and your `dev.txt`. Then you can start a basic training with the following command and make sure everything works properly.   

```bash
python train.py -n <runName> -d /path/to/data/dir --model roberta-base --device=1 --offline 
# only use this to test your setup
```
This should create and start a training-run with the specified name in your current environment. Be aware that this is only useful to test your setup, because it uses no GPU's for training. When you are convinced everything is working as it should you can terminate the process early. 

To create a training run with hardware-acceleration use the following command, to use all available GPUs.

```bash
python train.py -n <runName> -d /path/to/data/dir --model roberta-base --gpus=-1 --offline
# prefix with CUDA_VISIBLE_DEVICES=...
```

To see an overview over all options and their defaults, run `python train.py --help`.
We have disabled Weights & Biases syncing with the `--offline` flag. If you want to log your results, enable W&B as described [here](#weights--biases) and omit the `--offline` flag. We also set --gpus=-1 to use all GPU's available.

### Using the Docker environment for training
To run the training code inside the docker environment, use a `docker run` command like this:
```bash
docker run -it --user $(id -u):$(id -g) --gpus='device=0' --ipc=host -v "($pwd)":/workspace -w /workspace imagename bash
```
The `--gpus='device=0'` flag (change this to use the GPUs you actually want) selects the GPU with indice `0` for the container. Inside the container you can now execute your training-script as before.

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

### Weights & Biases
Weights & Biases is a platform that helps ml-researches to log their metrics for a training-run in an easy way. It lets you create checkpoints of your best models, can save the hyperparameters of your model and even supports Sweeps for parameter-optimization. For more information you can visit the [wandb](https://wandb.ai/site)-Website.
To enable Weights & Biases, enter your `WANDB_ENTITY` and `WANDB_PROJECT` in [dlib/frameworks/wandb.py](dlib/frameworks/wandb.py).
<details><summary>Weights & Biases + Docker</summary>

<p>

 When using docker you also have to provide your `WANDB_API_KEY`. You can find your personal key at [wandb.ai/authorize](https://app.wandb.ai/authorize). Either set `WANDB_API_KEY` on your host machine and use the `docker` flag `--env WANDB_API_KEY` when starting your run or mount your `.netrc` file into the docker container like so: `-v ~/.netrc:~/.netrc`.

</p>
</details>







