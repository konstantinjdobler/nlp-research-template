"""
Adapted from lit-gpt by Konstantin Dobler.
"""


from collections import deque
from contextlib import nullcontext
from typing import Any, Callable, Deque, Dict, Optional

import torch
from lightning import Fabric
from lightning.fabric.utilities.rank_zero import rank_zero_only as fabric_rank_zero_only
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from .pytorch import num_parameters

GPU_AVAILABLE_FLOPS = {
    # source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    # nvidia publishes spec sheet with a 2x sparsity factor
    "h100-sxm": {
        "64-true": 67e12,
        "32-true": 67e12,
        "16-true": 1.979e15 / 2,
        "16-mixed": 1.979e15 / 2,
        "bf16-true": 1.979e15 / 2,
        "bf16-mixed": 1.979e15 / 2,
        "8-true": 3.958e15 / 2,
        "8-mixed": 3.958e15 / 2,
    },
    "h100-pcie": {
        "64-true": 51e12,
        "32-true": 51e12,
        "16-true": 1.513e15 / 2,
        "16-mixed": 1.513e15 / 2,
        "bf16-true": 1.513e15 / 2,
        "bf16-mixed": 1.513e15 / 2,
        "8-true": 3.026e15 / 2,
        "8-mixed": 3.026e15 / 2,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    "a100": {
        "64-true": 19.5e12,
        "32-true": 19.5e12,
        "16-true": 312e12,
        "16-mixed": 312e12,
        "bf16-true": 312e12,
        "bf16-mixed": 312e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
    "a10g": {
        "32-true": 31.2e12,
        "16-true": 125e12,
        "16-mixed": 125e12,
        "bf16-true": 125e12,
        "bf16-mixed": 125e12,
    },
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    "v100-sxm": {
        "64-true": 7.8e12,
        "32-true": 15.7e12,
        "16-true": 125e12,
        "16-mixed": 125e12,
    },
    "v100-pcie": {
        "64-true": 7e12,
        "32-true": 14e12,
        "16-true": 112e12,
        "16-mixed": 112e12,
    },
    "v100s-pcie": {
        "64-true": 8.2e12,
        "32-true": 16.4e12,
        "16-true": 130e12,
        "16-mixed": 130e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    # sxm and pcie have same flop counts
    "t4": {
        "32-true": 8.1e12,
        "16-true": 65e12,
        "16-mixed": 65e12,
        "8-true": 130e12,
        "int4": 260e12,
    },
    # https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-5000-data-sheet-us-nvidia-704120-r4-web.pdf
    "quadro rtx 5000": {"32-true": 11.2e12, "16-true": 89.2e12, "16-mixed": 89.2e12},
    # NOTE: A6000 added by Konstantin Dobler
    # https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf
    # (1) https://www.topcpu.net/en/gpu-c/NVIDIA-RTX-A6000-vs-NVIDIA-A100-PCIe-80-GB
    # (2) https://www.techpowerup.com/gpu-specs/rtx-a6000.c3686
    # nvidia says more on their spec sheet but with "sparsity". A100 also is the number without sparisty, sources (1) and (2) say the same
    # => divide by 2 to get the true number (sparsity is 2x factor)
    "rtx a6000": {
        "32-true": 38.7e12,
        "16-true": 309.7e12 / 2,
        "bf16-true": 309.7e12 / 2,
        "16-mixed": 309.7e12 / 2,
        "bf16-mixed": 309.7e12 / 2,
    },
}

TPU_AVAILABLE_FLOPS = {
    # flop count for each TPU generation is the same for all precisions
    # since bfloat16 precision is always used for performing matrix operations
    # for more info: https://cloud.google.com/tpu/docs/bfloat16#choosing_bfloat16
    # source: https://arxiv.org/pdf/1907.10701.pdf
    "v2": 45e12,
    # source: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v3
    "v3": 123e12,
    # source: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v4
    "v4": 275e12,
    # source: https://cloud.google.com/tpu/docs/v5e-training
    "v5litepod": 197e12,
}


def get_flops_available(device: torch.device, precision: str) -> Optional[float]:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device).lower()
        if "h100" in device_name and "hbm3" in device_name:
            device_name = "h100-sxm"
        elif "h100" in device_name and ("pcie" in device_name or "hbm2e" in device_name):
            device_name = "h100-pcie"
        elif "a100" in device_name:
            device_name = "a100"
        elif "a10g" in device_name:
            device_name = "a10g"
        elif "v100-sxm" in device_name:
            device_name = "v100-sxm"
        elif "v100-pcie" in device_name:
            device_name = "v100-pcie"
        elif "t4" in device_name:
            device_name = "t4"
        elif "quadro rtx 5000" in device_name:
            device_name = "quadro rtx 5000"
        elif "rtx a6000" in device_name:
            device_name = "rtx a6000"
        else:
            device_name = None

        if device_name is not None:
            try:
                if precision == "transformer-engine":
                    precision = "8-mixed"
                return int(GPU_AVAILABLE_FLOPS[device_name][precision])
            except KeyError:
                raise KeyError(
                    f"flop count not found for {device_name} with precision: {precision}; "
                    "MFU cannot be calculated and reported."
                )
    elif device.type == "xla":
        from lightning.fabric.accelerators.xla import _XLA_GREATER_EQUAL_2_1

        if _XLA_GREATER_EQUAL_2_1:
            from torch_xla._internal import tpu
        else:
            from torch_xla.experimental import tpu

        device_name = tpu.get_tpu_env()["TYPE"].lower()
        try:
            return int(TPU_AVAILABLE_FLOPS[device_name])
        except KeyError:
            raise KeyError(
                f"flop count not found for {device_name} with precision: {precision}; " "MFU cannot be calculated and reported."
            )

    return None


# Adapted from https://github.com/mosaicml/composer/blob/f2a2dc820cb75023b9eb7c46fdfd25273712abd0/composer/callbacks/speed_monitor.py
# Adapted again from lit-gpt by Konstantin Dobler


class SpeedMonitor:
    def __init__(
        self,
        flops_available: float,
        world_size: int,
        model_flops_fwd_bwd: float | None,
        log_dict: Callable[[Dict, int], None],
        window_size: int = 100,
    ):
        self.hardware_flops_per_sec_promised = flops_available
        self.model_flops_fwd_bwd = model_flops_fwd_bwd
        self.log_dict = log_dict
        self.world_size = world_size

        # Track stats over a window of batches
        self.history_samples: Deque[int] = deque(maxlen=window_size)
        self.history_times: Deque[float] = deque(maxlen=window_size)
        self.history_e2e_times: Deque[float] = deque(maxlen=window_size)
        self.history_tokens: Deque[int] = deque(maxlen=window_size)
        self.history_flops: Deque[int] = deque(maxlen=window_size)

        # Keep track of toal times
        self.total_eval_time_elapsed = 0.0
        self.total_train_time_elapsed = 0.0
        self.step = -1

    def on_train_batch_end(
        self,
        samples: int,  # samples seen (per device)
        train_time: float,  # elapsed training time (seconds), just the actual iter
        train_time_e2e: float | None = None,  # elapsed training time (seconds) including everything
        model_flops_fwd_bwd: Optional[int] = None,  # (per device)
        tokens: Optional[int] = None,  # tokens seen this batch (per device)
        compute: bool = True,  # whether to compute & log metrics or only update the running counts
        step_kwargs: dict = {},  # additional kwargs to pass to log_dict, e.g. optimizer_step
    ):
        self.step += 1
        model_flops_fwd_bwd = model_flops_fwd_bwd or self.model_flops_fwd_bwd
        metrics = {}
        self.total_train_time_elapsed += train_time
        self.history_samples.append(samples)
        self.history_times.append(train_time)
        if tokens is not None:
            self.history_tokens.append(tokens)
            # if lengths are passed, there should be as many values as samples
            assert len(self.history_samples) == len(self.history_tokens)
        if model_flops_fwd_bwd is not None:
            self.history_flops.append(model_flops_fwd_bwd)
            assert len(self.history_samples) == len(self.history_flops)
        if train_time_e2e is not None:
            self.history_e2e_times.append(train_time_e2e)
            assert len(self.history_samples) == len(self.history_e2e_times)

        if not compute:
            return

        if len(self.history_times) == self.history_times.maxlen:
            elapsed_batches = len(self.history_samples)
            elapsed_samples = sum(self.history_samples)
            elapsed_time = sum(self.history_times)
            samples_per_sec = elapsed_samples / elapsed_time
            batches_per_sec = elapsed_batches / elapsed_time
            metrics.update(
                {
                    "throughput/batches_per_sec": batches_per_sec * self.world_size,
                    "throughput/samples_per_sec": samples_per_sec * self.world_size,
                    "throughput/device/batches_per_sec": batches_per_sec,
                    "throughput/device/samples_per_sec": samples_per_sec,
                }
            )
            if train_time_e2e is not None:
                elapsed_e2e_time = sum(self.history_e2e_times)
                samples_per_sec_e2e = elapsed_samples / elapsed_e2e_time
                batches_per_sec_e2e = elapsed_batches / elapsed_e2e_time
                metrics.update(
                    {
                        "throughput-e2e/batches_per_sec": batches_per_sec_e2e * self.world_size,
                        "throughput-e2e/samples_per_sec": samples_per_sec_e2e * self.world_size,
                        "throughput-e2e/device/batches_per_sec": batches_per_sec_e2e,
                        "throughput-e2e/device/samples_per_sec": samples_per_sec_e2e,
                    }
                )
            if tokens is not None:
                elapsed_lengths = sum(self.history_tokens)
                tokens_per_sec = elapsed_lengths / elapsed_time
                metrics.update(
                    {
                        "throughput/tokens_per_sec": tokens_per_sec * self.world_size,
                        "throughput/device/tokens_per_sec": tokens_per_sec,
                    }
                )
                if train_time_e2e is not None:
                    tokens_per_sec_e2e = elapsed_lengths / elapsed_e2e_time
                    metrics.update(
                        {
                            "throughput-e2e/tokens_per_sec": tokens_per_sec_e2e * self.world_size,
                            "throughput-e2e/device/tokens_per_sec": tokens_per_sec_e2e,
                        }
                    )

            if model_flops_fwd_bwd is not None:
                elapsed_flops = sum(self.history_flops)
                achieved_flops_per_sec = elapsed_flops / elapsed_time
                metrics.update(
                    {
                        "throughput/flops_per_sec": achieved_flops_per_sec * self.world_size,
                        "throughput/device/flops_per_sec": achieved_flops_per_sec,
                        "throughput/tflops_per_sec": achieved_flops_per_sec * self.world_size / 1e12,
                        "throughput/device/tflops_per_sec": achieved_flops_per_sec / 1e12,
                    }
                )
                if self.hardware_flops_per_sec_promised:
                    metrics["throughput/device/mfu"] = achieved_flops_per_sec / self.hardware_flops_per_sec_promised
                if train_time_e2e is not None:
                    achieved_flops_per_sec_e2e = elapsed_flops / elapsed_e2e_time
                    metrics.update(
                        {
                            "throughput-e2e/flops_per_sec": achieved_flops_per_sec_e2e * self.world_size,
                            "throughput-e2e/device/flops_per_sec": achieved_flops_per_sec_e2e,
                            "throughput-e2e/tflops_per_sec": achieved_flops_per_sec_e2e * self.world_size / 1e12,
                            "throughput-e2e/device/tflops_per_sec": achieved_flops_per_sec_e2e / 1e12,
                        }
                    )
                    if self.hardware_flops_per_sec_promised:
                        metrics["throughput-e2e/device/mfu"] = achieved_flops_per_sec_e2e / self.hardware_flops_per_sec_promised

        metrics.update(
            {
                "time/train": self.total_train_time_elapsed,
                "time/val": self.total_eval_time_elapsed,
                "time/total": self.total_train_time_elapsed + self.total_eval_time_elapsed,
                "samples": samples * self.world_size,
            }
        )

        self.log_dict(metrics | step_kwargs, self.step)
        return metrics

    def eval_end(self, eval_elapsed: float):
        self.total_eval_time_elapsed += eval_elapsed  # seconds


class SpeedMonitorFabric(SpeedMonitor):
    def __init__(self, fabric: Fabric, *args: Any, **kwargs: Any) -> None:
        # TODO: this will not work properly if a precision plugin is passed to Fabric
        flops_available = get_flops_available(fabric.device, fabric._connector._precision_input)
        super().__init__(flops_available, *args, log_dict=fabric.log_dict, **kwargs)

    @fabric_rank_zero_only
    def on_train_batch_end(self, *args: Any, **kwargs: Any):
        super().on_train_batch_end(*args, **kwargs)


######################################
#### FLOPS estimation / measuring ####
######################################


def estimate_flops(model: nn.Module, block_size: int, num_layers: int, hidden_size: int) -> int:
    """Measures estimated FLOPs for MFU, for Transformer models.

    Refs:
        * https://ar5iv.labs.arxiv.org/html/2205.05198#A1
        * https://ar5iv.labs.arxiv.org/html/2204.02311#A2
    """
    # using all parameters for this is a naive over estimation because not all model parameters actually contribute to
    # this FLOP computation (e.g. embedding, norm). For this reason, the result will be higher by a fixed percentage
    # (~10%) compared to the measured FLOPs, making those lower but more realistic.
    # For a proper estimate, this needs a more fine-grained calculation as in Appendix A of the paper.
    n_trainable_params = num_parameters(model, requires_grad=True)
    trainable_flops = flops_per_param(block_size, num_layers, hidden_size, n_trainable_params)
    # forward + backward + gradients (assumes no gradient accumulation)
    ops_per_step = 3 if model.training else 1
    n_frozen_params = num_parameters(model, requires_grad=False)
    frozen_flops = flops_per_param(block_size, num_layers, hidden_size, n_frozen_params)
    # forward + backward
    frozen_ops_per_step = 2 if model.training else 1
    return ops_per_step * trainable_flops + frozen_ops_per_step * frozen_flops


def flops_per_param(max_seq_length: int, num_layers: int, hidden_size: int, num_params: int) -> int:
    flops_per_token = 2 * num_params  # each parameter is used for a MAC (2 FLOPS) per network operation
    # this assumes that all samples have a fixed length equal to the block size
    # which is most likely false during finetuning
    flops_per_seq = flops_per_token * max_seq_length
    attn_flops_per_seq = num_layers * 2 * 2 * (hidden_size * (max_seq_length**2))
    return flops_per_seq + attn_flops_per_seq


def measure_flops(model: nn.Module, x: torch.Tensor) -> int:
    """Measures real FLOPs for HFU"""
    flop_counter = FlopCounterMode(model, display=False)
    ctx = nullcontext() if model.training else torch.no_grad()
    with ctx, flop_counter:
        y = model(x)
        if model.training:
            y.sum().backward()
    return flop_counter.get_total_flops()


def measure_model_flops(
    fabric: Fabric,
    batch_size: int,
    block_size: int,
    new_model_func: Callable,
    model: nn.Module | None = None,
    parameter_lookup: dict[str, tuple[tuple[int], bool]] | None = None,
    num_layers: int | None = None,
    hidden_size: int | None = None,
) -> tuple[int, int, int]:
    """
    We use either the original model or a parameter_lookup dict to exactly reconstruct the model (shapes, requires_grad).
    Using the parameter lookup instead of the original model can be helpful when the original model has been modified a lot since it's instantiation (e.g. FSDP, activation checkpointing, manual edits).

    Adapted from lit-gpt.
    """

    if parameter_lookup is None and model is None:
        raise ValueError("Either parameter_lookup or model must be passed.")

    if parameter_lookup is not None and model is not None:
        raise ValueError("Only one of parameter_lookup or model must be passed.")

    from print_on_steroids import logger as printer

    with torch.device("meta"), fabric.strategy.precision.init_context():
        new_model = new_model_func()

        for n, p in new_model.named_parameters():
            if model:
                shape, grad = (model.get_parameter(n).shape, model.get_parameter(n).requires_grad)
            else:
                shape, grad = parameter_lookup[n]
            p.data = torch.empty(shape, requires_grad=grad, device=torch.device("meta"))
        if num_layers is not None and hidden_size is not None:
            estimated_flops = estimate_flops(new_model, block_size, num_layers, hidden_size) * batch_size
            printer.info(f"Estimated TFLOPs: {estimated_flops / 1e12:.2f}")
        x = torch.randint(10, 42, (batch_size, block_size))
        new_model.train()
        forward_backward_flops = measure_flops(new_model, x)
        new_model.eval()
        forward_flops = measure_flops(new_model, x)
        printer.info(f"Measured TFLOPs: {forward_backward_flops  / 1e12:.2f}")
        printer.info(f"Measured TFLOPs (just forward): {forward_flops  / 1e12:.2f}")

    del new_model, x
    return forward_backward_flops, forward_flops, estimated_flops
