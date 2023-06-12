import os
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from train import TrainingArgs


def infer_batch_size_per_device(
    num_devices: int, effective_batch_size: int, batch_size_per_device: int
):
    needed_batch_size_per_device = int(effective_batch_size / num_devices)
    assert needed_batch_size_per_device == effective_batch_size / num_devices

    needed_gradient_accum_steps = 1
    while (needed_gradient_accum_steps * batch_size_per_device < needed_batch_size_per_device) or (
        effective_batch_size / (needed_gradient_accum_steps * num_devices)
        != int(effective_batch_size / (needed_gradient_accum_steps * num_devices))
    ):
        needed_gradient_accum_steps += 1

    resulting_batch_size_per_device = effective_batch_size / (
        needed_gradient_accum_steps * num_devices
    )
    assert resulting_batch_size_per_device <= batch_size_per_device
    assert resulting_batch_size_per_device == int(resulting_batch_size_per_device)

    batch_size_per_device = int(resulting_batch_size_per_device)
    effective_batch_size_per_step = num_devices * batch_size_per_device

    return batch_size_per_device, needed_gradient_accum_steps, effective_batch_size_per_step


def choose_auto_accelerator():
    """Choose hardware accelerator depending on availability. TODO: TPU"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_auto_devices(accelerator: str):
    """Choose number of devices depending on availability or sane defaults."""
    match accelerator:
        case "cuda":
            if not os.environ.get("CUDA_VISIBLE_DEVICES"):
                logger.warning(
                    "CUDA_VISIBLE_DEVICES not set and `devices=-1`, using all available GPUs."
                )
            return torch.cuda.device_count()
        case "mps":
            return 1  # There's only a single GPU supported on Apple Silicon for now
        case "tpu":
            return 8  # default from lightning
        case "cpu":
            return 1  # default from lightning
        case _:
            raise ValueError(f"Cannot use auto number of devices with accelerator {accelerator}.")


def handle_batch_size_logic_(args: "TrainingArgs"):
    """Calculates and sets effective batch size / gradient accumulation steps."""
    ACCELERATOR = args.accelerator.upper() if args.accelerator != "cuda" else "GPU"
    if args.effective_batch_size:
        logger.info(
            f"Trying to auto-infer settings for effective batch size {args.effective_batch_size}..."
        )
        (
            args.batch_size_per_device,
            args.gradient_accumulation_steps,
            effective_batch_size_per_step,
        ) = infer_batch_size_per_device(
            args.num_devices, args.effective_batch_size, args.batch_size_per_device
        )

        logger.info(
            f"Using effective batch size {args.effective_batch_size} "
            f"with {args.num_devices} {ACCELERATOR}s, "
            f"{args.batch_size_per_device} batch size per {ACCELERATOR} and "
            f"{args.gradient_accumulation_steps} gradient accumulation steps."
        )
    else:
        effective_batch_size_per_step = args.num_devices * args.batch_size_per_device
        args.effective_batch_size = effective_batch_size_per_step * args.gradient_accumulation_steps
        logger.info(
            f"Effective batch size {args.effective_batch_size} based on specified args "
            f"{args.num_devices} {ACCELERATOR}s, "
            f"{args.batch_size_per_device} batch size per {ACCELERATOR} and "
            f"{args.gradient_accumulation_steps} gradient accumulation steps."
        )
    return effective_batch_size_per_step
