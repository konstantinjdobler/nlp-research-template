import time

import lightning as L
import torch
import wandb
from lightning.fabric.strategies.strategy import _Sharded
from tqdm.asyncio import tqdm


def clean_name(n):
    """Remove common wrapper prefixes from module names."""
    return (
        n.replace("_forward_module.", "")
        .replace("_original_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_fsdp_wrapped_module.", "")
    )


def histogram(xs, bins):
    """
    Like torch.histogram, but works with cuda

    from: https://github.com/pytorch/pytorch/issues/69519#issuecomment-1183866843
    """
    min, max = xs.min().item(), xs.max().item()
    counts = torch.histc(xs, bins, min=min, max=max)
    # counts = counts / counts.sum() # convert to pdf
    boundaries = torch.linspace(min, max, bins + 1)
    return counts, boundaries


def work_then_gather_then_process_on_rank0(tensor, work_func, rank0_func, fabric: L.Fabric):
    tensor_is_sharded_elsewhere = tensor is None or tensor.numel() == 0
    result = None
    if not tensor_is_sharded_elsewhere:
        result = work_func(tensor)

    if isinstance(fabric.strategy, _Sharded):
        object_gather_list = None
        if fabric.local_rank == 0:
            object_gather_list = [None for _ in range(fabric.world_size)]
        torch.distributed.gather_object(result, object_gather_list, dst=0)
        if fabric.local_rank == 0:
            result = [r for r in object_gather_list if r is not None][0]
            rank0_func(result)
    else:
        rank0_func(result)


def create_check_bf16_numerics_funcs(name: str, lr: float):
    def work_func(param: torch.nn.Parameter):
        weight = param.data
        grad = param.grad
        updates = grad.reshape(-1) * lr
        weights = weight.reshape(-1)

        # fp32_updates = updates.to(torch.float32)
        epsilons = weights.to(torch.float32) / (2**7)
        n_underflow_math = torch.lt(updates.to(torch.float32), epsilons).sum()
        # ratios = updates.to(torch.float32) / weights.to(torch.float32)
        # n_underflow_math = torch.lt(ratios, 1 / (2**7)).sum()
        # print("mathemtical", n_underflow_math, weights.numel(), n_underflow_math / weights.numel())

        weights_after_update = weights - updates
        # high_precision_update = weights.to(torch.float32) - updates.to(torch.float32)
        # reconstructed_update = simulated_update.to(torch.float32) - weights.to(torch.float32)
        no_change = torch.eq(weights_after_update, weights)

        non_zero_update = torch.ne(updates, 0.0)

        n_underflow_empirical = (no_change & non_zero_update).sum()
        # print("empirical", n_underflow_empirical, weights.numel(), n_underflow_empirical / weights.numel())
        return n_underflow_empirical, n_underflow_math, weights.numel()

    def rank0_func(r):
        n_underflow_empirical, n_underflow_math, n = r
        wandb.run._log(
            {
                f"bf16-e/{name}": n_underflow_empirical / n,
                f"bf16-m/{name}": n_underflow_math / n,
            },
            commit=False,
        )

    return work_func, rank0_func


def create_log_tensor_histogram_funcs(name: str, prefix: str):
    def work_func(tensor: torch.Tensor):
        return histogram(tensor.reshape(-1).to(torch.float32), bins=64)

    def rank0_func(r):
        counts, bins = r
        wandb.run._log(
            {f"{prefix}/{name}": wandb.Histogram(np_histogram=(counts.tolist(), bins.tolist()))},
            commit=False,
        )

    return work_func, rank0_func


def log_param_stats(
    fabric: L.Fabric,
    param: torch.nn.Parameter,
    name: str,
    log_weights=True,
    log_grads=True,
    check_bf16_underflow=True,
):
    weight = param.data
    grad = param.grad

    if log_weights:
        work_func, rank0_func = create_log_tensor_histogram_funcs(name, prefix="parameters")
        work_then_gather_then_process_on_rank0(weight, work_func, rank0_func, fabric)

    if log_grads:
        work_func, rank0_func = create_log_tensor_histogram_funcs(name, prefix="gradients")
        work_then_gather_then_process_on_rank0(grad, work_func, rank0_func, fabric)

    if check_bf16_underflow:
        work_func, rank0_func = create_check_bf16_numerics_funcs(name, lr=0.1)
        work_then_gather_then_process_on_rank0(param, work_func, rank0_func, fabric)


def log_model_stats_to_wandb(
    fabric: L.Fabric,
    model: torch.nn.Module,
    log_weights=True,
    log_grads=True,
    check_bf16_underflow=False,
):
    """
    Log paramters and gradients of a model to wandb.
    Handles sharded parameters in case of FSDP (we calculate tensor histograms in the local rank and then gather to rank0).

    Experimental feature: `check_bf16_underflow`.
    If training in true bf16 precision, we try to check if any weight updates are so small that they are rounded to zero.
    Probably still buggy / not accurate.

    Returns: time spent in this function (seconds).
    """

    grad_tracking_t0 = time.perf_counter()
    for n, p in tqdm(model.named_parameters(), desc="Logging model states", leave=False):
        if p.requires_grad:
            log_param_stats(
                fabric,
                p,
                clean_name(n),
                log_grads=log_grads,
                log_weights=log_weights,
                check_bf16_underflow=check_bf16_underflow,
            )
    grad_tracking_t1 = time.perf_counter()
    return grad_tracking_t1 - grad_tracking_t0
