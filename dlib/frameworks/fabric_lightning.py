import contextlib
import os
import subprocess
import threading
import time
from pathlib import Path
from time import sleep
from typing import Any

import lightning as L
import torch
import wandb
from lightning import Callback, Fabric, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger as LightningWandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from print_on_steroids import logger as cli_logger


@contextlib.contextmanager
def fabric_main_process_first(
    fabric: Fabric,
    description="",
    active=True,
    time_buffer_after_main: bool | int = False,
):
    """
    Context manager that executes the wrapped code on the main process first and then on all other processes. Useful for e.g. dataset preprocessing.
    Inspiration from Huggingface: https://github.com/huggingface/transformers/pull/12351/files
    """
    if torch.distributed.is_available() and active:
        local_rank = fabric.local_rank
        try:
            if local_rank > 0:
                print(f"Process {local_rank} | {description} | Waiting for main process...")
                fabric.barrier()
            yield
        finally:
            if local_rank == 0:
                print(f"Main process | {description} | Done. Executing on parallel processes now...")
                fabric.barrier()
                if time_buffer_after_main:
                    time_buffer_after_main = time_buffer_after_main if isinstance(time_buffer_after_main, int) else 5
                    sleep(time_buffer_after_main)  # Give other processes time to catch up
    else:
        yield


def lora_filter(key: str, value: Any) -> bool:
    return "lora_" in key


def save_checkpoint(
    fabric: L.Fabric,
    state,
    out_dir: Path,
    checkpoint_parameter_keys: set[str],
    tags: list[str] = [],
):
    wandb_logger: LightningWandbLogger = fabric.logger

    wandb_run_id = str(wandb_logger.experiment.id)
    file_path = out_dir / wandb_run_id / f"iter-{state['iter_num']:06d}-ckpt.pth"
    os.makedirs(file_path.parent, exist_ok=True)

    cli_logger.info(f"Saving checkpoint weights to {str(file_path)!r}")

    previous_ckpts = sorted((out_dir / wandb_run_id).glob("iter-*-ckpt.pth"))
    if len(previous_ckpts) > 0:
        cli_logger.warning(f"Deleting previous checkpoints: {previous_ckpts}")
        for p in previous_ckpts:
            p.unlink()

    def filter_func(key: str, value: torch.Tensor):
        return key in checkpoint_parameter_keys

    # Always save LoRA s.t. we can reconstruct as diff to Llama2 base checkpoint
    filter_funcs = (filter_func, lora_filter)

    def merged_weight_filter_func(k, v):
        return any([f(k, v) for f in filter_funcs])

    fabric.save(file_path, state, filter={"model": merged_weight_filter_func})

    cli_logger.info("Logging to wandb...")
    if fabric.is_global_zero:
        # NOTE: Important to add `wandb_logger.experiment.id` (or some other unique string) s.t. different runs get different artifacts
        artifact = wandb.Artifact(
            name=f"model-checkpoint-{wandb_logger.experiment.id}",
            type="model",
            metadata=dict(iter=state["iter_num"], optimizer_step=state["step_count"]),
        )
        artifact.add_file(str(file_path), name="checkpoint.pth")
        tags.append("latest")
        wandb_logger.experiment.log_artifact(artifact, aliases=tags)


class WandbCleanupDiskAndCloudSpaceCallback(Callback):
    def __init__(self, cleanup_local=True, cleanup_online=True, size_limit=0, backoff=10) -> None:
        super().__init__()
        self.cleanup_local = cleanup_local
        self.cleanup_online = cleanup_online
        self.size_limit = size_limit
        self.backoff = backoff
        self.counter = 0

    @rank_zero_only
    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.counter < self.backoff:
            self.counter += 1
            return
        else:
            self.counter = 0
        # run = wandb.run  # type: ignore
        run = trainer.logger.experiment  # type: ignore

        # Delete outdated online artifacts
        if self.cleanup_online:  # BUG: this doesn't work....
            if getattr(run, "logged_artifacts", None) is not None:
                for artifact in run.logged_artifacts():
                    aliases = [x["alias"] for x in artifact._attrs["aliases"]]
                    if "best" not in aliases and "keep" not in aliases:
                        cli_logger.info(f"Deleting outdated artifact with aliases {aliases}")
                        artifact.delete()
            else:
                cli_logger.error("wandb run has no logged artifacts")

        # Free up local wandb cache (This is often A LOT of memory)
        if self.cleanup_local:
            cli_logger.info("Starting wandb artifact cache cleanup timeout")
            cache_cleanup_callback = lambda: subprocess.run(  # noqa: E731
                ["wandb", "artifact", "cache", "cleanup", f"{self.size_limit}GB"]
            )
            timer = threading.Timer(
                120.0, cache_cleanup_callback
            )  # Delay cleanupcall to avoid cleaning a temp file from the ModelCheckpoint hook that is needed to upload current checkpoint
            timer.start()


class CUDAMetricsCallback(Callback):
    """
    Log CUDA stats. Adapted from https://github.com/Lightning-AI/lightning-GPT/blob/main/lightning_gpt/callbacks.py
    """

    def on_validation_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(self.root_gpu(trainer))
        torch.cuda.synchronize(self.root_gpu(trainer))
        self.start_time = time.time()

    def on_validation_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        torch.cuda.synchronize(self.root_gpu(trainer))
        max_memory = torch.cuda.max_memory_allocated(self.root_gpu(trainer)) / 2**20
        start_time = getattr(self, "start_time", None)
        if start_time:
            time_since_last_validation = time.time() - self.start_time

            max_memory = trainer.strategy.reduce(max_memory, reduce_op=torch.max)

            time_since_last_validation = trainer.strategy.reduce(time_since_last_validation)

            rank_zero_info(f"Average time: {time_since_last_validation:.2f} seconds")
            rank_zero_info(f"Max Peak memory {max_memory:.2f}MiB")

            trainer.logger.log_metrics(
                {
                    "System/Max. Peak CUDA memory": max_memory,
                    "System/Avg. Training Time": time_since_last_validation,
                },
                step=trainer.fit_loop.epoch_loop._batches_that_stepped,
            )

    def root_gpu(self, trainer: "Trainer") -> int:
        return trainer.strategy.root_device.index
