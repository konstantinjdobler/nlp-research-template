# An opinionated template for NLP research code

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

NLP research template using PyTorch + PyTorch Lightning + Weights & Biases + HuggingFace. It's built to be customized but provides comprehensive, sensible default functionality.

It also uses [`dlib - the Deep Learning utility library`](https://github.com/konstantinjdobler/dlib). `dlib` is integrated as a `git-subtree` -- for you this just means that all files live inside the `dlib` folder.

## Setup

It's recommended to use [`mamba`](https://github.com/mamba-org/mamba) to manage dependencies. `mamba` is a drop-in replacement for `conda` re-written in C++ to speed things up significantly (you can stick with conda though). To provide reproducible environments, we use `conda-lock` to generate lockfiles for each platform. You can create a `conda` environment from a lockfile like this:

```bash
mamba create --name <gpt4> --file cpu-linux-64.lock
```
That's it -- this is the power of lockfiles.

To generate new lockfiles after updating the `environment.yml` file, run:

```bash
conda-lock -k explicit --conda mamba
```

This will create `conda-<platform>.lock` files for all platforms specified in `environment.yml`. To create lockfiles for CPU, comment out `nvidia::pytorch-cuda=<version.number>` in `environment.yml`.

## Training
To start a language model MLM training, run:
```bash
python train.py --data /path/to/data/dir --model roberta-base --gpus 2 --offline
```
By default, `train.txt` and `dev.txt` are expected in the data directory. To see an overview over all options and their defaults, run `python train.py --help`. 
We have disabled Weights & Biases syncing with the `--offline` flag. To enable W&B, enter your `WANDB_ENTITY` and `WANDB_PROJECT` in [dlib/frameworks/wandb.py](dlib/frameworks/wandb.py) and simply omit the `--offline` flag.

## Component Overview
Coming soon.

