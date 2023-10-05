from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from simple_parsing import field, list_field


@dataclass(kw_only=True)
class TrainingArgs:
    """
    Argument class for use with simple_parsing that handles the basics of most LLM training scripts. Subclass this to add more arguments.
    """

    data_dir: Path = field(alias="-d")
    run_name: str = field(aliase="-n")  # Run name for logging

    base_unit: Literal["samples", "tokens", "optimizer-steps", "iters"] = field(default="iters")
    "Unit of all training constants. They will be converted to optimizer_steps and iters in __post_init__."

    training_goal: int = field(default=100_000)
    eval_interval: float = field(default=0.1)
    "Interval between evaluations. If < 1, use as percentage of training_goal."

    eval_samples: int = field(default=100)
    "Number of iters on the val dataset during evaluation. If -1, use full val dataset."

    save_interval: float = field(default=0.1)
    "Interval between model checkpoints. If < 1, use as percentage of training_goal."

    log_interval: float = field(default=-1)
    "Interval between log prints. If < 1, use as percentage of training_goal. If -1, print log after every batch."

    warmup_period: float = field(default=0.005)
    "Length of lr warmup. If < 1, use as percentage of training_goal."

    lr_decay_period: int = field(default=-1)
    "If -1, decay until end of training."

    seed: int = field(default=42)
    only_val: bool = field(default=False)

    pretrained_checkpoint_dir: Path = field(default="checkpoints/meta-llama/Llama-2-7b-hf")
    resume: bool = False
    out_dir: Path = field(default="out/")

    learning_rate: float = field(default=3e-4)
    batch_size: int = field(default=128, alias="-b")
    block_size: int | None = field(default=None)
    "If None, load from model config."

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = field(default=1.0)
    "If -1, disable."

    decay_lr: bool = True
    min_lr: float = 6e-5

    wandb_tags: list[str] = list_field(default=[], alias="-t")
    "Tags for wandb."

    offline: bool = field(default=False)
    "If True, don't log to wandb."

    debug: bool = field(default=False)
    "If true, wait for debugger to attach at the start of the script."

    devices: int = field(default=1)
    tpu: bool = False
    precision: Literal["32-true", "16-true", "bf16-true", "16-mixed", "bf16-mixed"] = "bf16-true"
    use_anyprecision_adamw: bool = field(default=False)
    activation_checkpointing: bool = field(default=True)
    micro_batch_size: int = field(default=4, alias="--mb")
    eval_micro_batch_size: int = field(default=None)
    gradient_accumulation_steps: int = field(default=-1)
    "If -1, set automatically based on batch_size and micro_batch_size."

    fast_model_loading: bool = field(default=True, alias="--fml")
    "Instantiate model with empty weights. Use if you are loading weights from a checkpoint. Saves a lot of time.",

    compile: bool = field(default=False)
    "torch.compile model for faster training."

    fsdp_sharding_strategy: Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"] = field(default="SHARD_GRAD_OP")
    fsdp_limit_all_gathers: bool = field(default=True)
    fsdp_cpu_offload: bool = field(default=False)
    smart_cuda_alloc: bool = field(default=False)

    workers: int = field(default=4, alias="-w")
    preprocessing_workers: int = field(default=4, alias="--pw")
    data_preprocessing_only: bool = field(default=False)
    conserve_disk_space: bool = field(default=False)
    overwrite_data_cache: bool = field(default=False)

    def __post_init__(self):
        self.iter_batch_size = self.micro_batch_size * self.devices

        if self.eval_interval < 1:
            self.eval_interval = int(self.eval_interval * self.training_goal)
        if self.save_interval < 1:
            self.save_interval = int(self.save_interval * self.training_goal)
        if self.log_interval < 1 and self.log_interval != -1:
            self.log_interval = int(self.log_interval * self.training_goal)
        if self.warmup_period < 1:
            self.warmup_period = int(self.warmup_period * self.training_goal)
        if self.lr_decay_period == -1:
            self.lr_decay_period = self.training_goal
        elif self.lr_decay_period < 1:
            self.lr_decay_period = int(self.lr_decay_period * self.training_goal)

        assert self.batch_size % self.micro_batch_size == 0
        if self.gradient_accumulation_steps == -1:
            self.gradient_accumulation_steps = self.batch_size // self.iter_batch_size
        assert self.gradient_accumulation_steps > 0
        assert self.batch_size == self.micro_batch_size * self.devices * self.gradient_accumulation_steps

        if self.tokenizer is None:
            self.tokenizer = self.pretrained_checkpoint_dir
            assert self.pretrained_checkpoint_dir is not None

        if self.eval_micro_batch_size is None:
            self.eval_micro_batch_size = self.micro_batch_size

        # Calculate training constants
        if self.base_unit == "samples":
            UNITS_PER_ITER = self.iter_batch_size
            UNITS_PER_STEP = self.batch_size
        elif self.base_unit == "tokens":
            UNITS_PER_ITER = self.iter_batch_size * self.block_size
            UNITS_PER_STEP = self.batch_size * self.block_size
        elif self.base_unit == "optimizer-steps":
            UNITS_PER_ITER = 1 / self.gradient_accumulation_steps
            UNITS_PER_STEP = 1
        elif self.base_unit == "iters":
            UNITS_PER_ITER = 1
            UNITS_PER_STEP = self.gradient_accumulation_steps
        else:
            raise ValueError(f"Unknown training goal unit: {self.base_unit}")

        self.training_goal = int(self.training_goal / UNITS_PER_ITER)
        self.eval_interval = int(self.eval_interval / UNITS_PER_STEP)
        self.save_interval = int(self.save_interval / UNITS_PER_STEP)
        if self.log_interval == -1:
            self.log_interval = 1
        else:
            self.log_interval = int(self.log_interval / UNITS_PER_STEP)
        self.warmup_period = int(self.warmup_period / UNITS_PER_ITER)
        self.lr_decay_period = int(self.lr_decay_period / UNITS_PER_ITER)
