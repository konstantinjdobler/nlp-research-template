import multiprocessing
import os
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

    model_path: str = field(alias="--model")
    "Model identifier. This is used to construct the model architecture and load pretrained weights if not specified otherwise."

    from_scratch: bool = field(default=False)
    "Do not use pre-trained weights to intialize the model."

    saved_checkpoint_path: str | None = field(default=None, alias="--checkpoint")
    "Path to a checkpoint saved during training. Use the wandb:<wandb-run-id> syntax to load a checkpoint from W&B."

    resume: bool = False

    train_file: str = field(default="train.txt")
    "Name of the training file."

    val_file: str = field(default="val.txt")
    "Name of the validation file."

    tokenizer_path: str | None = field(default=None)
    "Path to a saved tokenizer to switch the vocabulary. If None, use the model_path."

    ###############################
    ##### Training constants ######
    ###############################

    base_unit: Literal["samples", "tokens", "optimizer-steps", "iters"] = field(default="optimizer-steps")
    "Unit of all training constants. They will be converted to optimizer_steps in __post_init__."

    training_goal: int = field(default=100_000)
    eval_interval: float = field(default=0.1)
    "Interval between evaluations. If < 1, use as percentage of training_goal."

    eval_samples: int = field(default=-1)
    "Number of samples on the val dataset during evaluation. If -1, use full val dataset."

    save_interval: int | float = field(default=0.1)
    "Interval between model checkpoints. If < 1, use as percentage of training_goal."

    log_interval: float = field(default=-1)
    "Interval between log prints. If < 1, use as percentage of training_goal. If -1, print log after every batch."

    warmup_period: float = field(default=0.005)
    "Length of lr warmup. If < 1, use as percentage of training_goal."

    lr_decay_period: int = field(default=-1)
    "If -1, decay until end of training."

    ###########################
    ##### Hyperparameters #####
    ###########################
    block_size: int = field(default=512)
    "The sequence length of samples."

    learning_rate: float = field(default=3e-4)
    batch_size: int = field(default=128, alias="-b")
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = field(default=1.0)
    "If -1, disable."

    decay_lr: bool = True
    min_lr: float = 6e-5

    #######################################
    ## Hardware acceleration & precision ##
    #######################################

    accelerator: Literal["cuda", "cpu", "tpu", "mps"] = field(default="cuda")
    "Hardware accelerator to use."
    num_devices: int = field(default=1)
    activation_checkpointing: bool = field(default=True)

    distributed_strategy: Literal["ddp", "fsdp", "auto"] = field(
        default="auto",
        help="Distributed training strategy to use. If `auto`, will select automatically (no distributed strategy is used when using a single device).",
        aliases="--ds",
    )

    micro_batch_size: int = field(default=None, alias="--mb")
    """If None, use batch_size // num_devices. This is the batch size per device, not the total batch size.
    You should tune this so that you do not get GPU RAM OOM errors. We automatically calculate the gradient accumulation steps to achieve your desired `batch_size`."""

    eval_micro_batch_size: int = field(default=None)
    "If None, use micro_batch_size."

    gradient_accumulation_steps: int = field(default=-1)
    "If -1, set automatically based on batch_size and micro_batch_size."

    precision: Literal["32-true", "16-mixed", "bf16-mixed", "bf16-true", "16-true"] = "bf16-true"
    compile: bool = field(default=False)
    "torch.compile model for faster training."

    workers: int = field(default=4, alias="-w")
    preprocessing_workers: int = field(default=4, aliases="--pw")
    "Number of workers for preprocessing the datasets. If -1, use all available CPUs."

    data_preprocessing_only: bool = field(default=False)
    conserve_disk_space: bool = field(default=False)

    ############################
    ###### Logging & Misc ######
    ############################

    run_name: str = field(default=None, alias="-n")
    "Run name for logging."

    seed: int | None = field(default=None)

    only_val: bool = field(default=False)
    "Only run validation."

    val_before_training: bool = field(default=True)
    "Run one validation epoch before training."

    out_dir: Path = field(default="out/")

    wandb_tags: list[str] = list_field(default=[], alias="-t")
    "Tags for wandb."

    offline: bool = field(default=False)
    "If True, don't log to wandb."

    debug: bool = field(default=False)
    "If true, wait for debugger to attach at the start of the script."

    force_deterministic: bool = field(default=False)
    "Force PyTorch operations to be deterministic. Could be slower."

    fast_dev_run: bool = field(default=False)
    "Do fast run through training and validation with reduced sizes."

    def __post_init__(self):
        if self.micro_batch_size is None:
            # NOTE: you need to make sure that micro_batch_size can fit into the GPU memory
            self.micro_batch_size = self.batch_size // self.num_devices
            assert self.batch_size % self.num_devices == 0

        self.iter_batch_size = self.micro_batch_size * self.num_devices

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
        assert self.batch_size == self.micro_batch_size * self.num_devices * self.gradient_accumulation_steps

        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
            assert self.model_path is not None

        if self.eval_micro_batch_size is None:
            self.eval_micro_batch_size = self.micro_batch_size

        # Calculate training constants
        if self.base_unit == "samples":
            UNITS_PER_STEP = self.batch_size
        elif self.base_unit == "tokens":
            assert self.block_size is not None, "block_size must be set if base_unit is tokens"
            UNITS_PER_STEP = self.batch_size * self.block_size
        elif self.base_unit == "optimizer-steps":
            UNITS_PER_STEP = 1
        elif self.base_unit == "iters":
            UNITS_PER_STEP = self.gradient_accumulation_steps
        else:
            raise ValueError(f"Unknown training goal unit: {self.base_unit}")

        self.training_goal = int(self.training_goal / UNITS_PER_STEP)
        self.eval_interval = int(self.eval_interval / UNITS_PER_STEP)
        self.save_interval = int(self.save_interval / UNITS_PER_STEP)
        if self.log_interval == -1:
            self.log_interval = 1
        else:
            self.log_interval = int(self.log_interval / UNITS_PER_STEP)
        self.warmup_period = int(self.warmup_period / UNITS_PER_STEP)
        self.lr_decay_period = int(self.lr_decay_period / UNITS_PER_STEP)

        if self.preprocessing_workers == -1:
            # Set to all available CPUs, handle SLURM case when only some CPUs are available to the job
            self.preprocessing_workers = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", multiprocessing.cpu_count()))
