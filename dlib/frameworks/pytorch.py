import contextlib
import os
from time import sleep

import torch
from torch import nn


def set_torch_file_sharing_strategy_to_system(worker_id: int = 0) -> None:
    """
    When having many workers for dataloaders / many tensors per batch, torch uses file descriptors to share data between processes.
    UNIX systems have upper limits for the number of open file descriptors allowed, given enough workers / tensors this limit will be reached and the process will be killed.
    https://github.com/pytorch/pytorch/issues/11201#issuecomment-895047235
    """
    torch.multiprocessing.set_sharing_strategy("file_system")


@contextlib.contextmanager
def main_process_first(
    local_rank: int,
    description="",
    active=True,
    time_buffer_after_main: bool | int = False,
):
    """
    Context manager that executes the wrapped code on the main process first and then on all other processes. Useful for e.g. dataset preprocessing.
    Inspiration from Huggingface: https://github.com/huggingface/transformers/pull/12351/files
    """
    if torch.distributed.is_available() and active:
        try:
            if local_rank > 0:
                print(f"Process {local_rank} | {description} | Waiting for main process...")
                torch.distributed.barrier()
            yield
        finally:
            if local_rank == 0:
                print(f"Main process | {description} | Done. Executing on parallel processes now...")
                torch.distributed.barrier()
                if time_buffer_after_main:
                    time_buffer_after_main = time_buffer_after_main if isinstance(time_buffer_after_main, int) else 5
                    sleep(time_buffer_after_main)  # Give other processes time to catch up
    else:
        yield


def num_parameters(module: nn.Module, requires_grad: bool | None = None) -> int:
    """From lit-gpt."""
    return sum(p.numel() for p in module.parameters() if requires_grad is None or p.requires_grad == requires_grad)


def get_rank() -> int:
    if not torch.distributed.is_available():
        return 0  # Training on CPU
    if not torch.distributed.is_initialized():
        # LOCAL_RANK from pytorch-lightning
        rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
        if rank is not None:
            return int(rank)
        else:
            return 0
    else:
        return torch.distributed.get_rank()
