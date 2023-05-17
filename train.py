import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import wandb
from dargparser import dArg, dargparse
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import (
    LightningEnvironment,
    SLURMEnvironment,
)
from lightning.pytorch.strategies import DDPStrategy
from loguru import logger
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from dlib.frameworks.lightning import CUDAMetricsCallback
from dlib.frameworks.pytorch import (
    get_effective_batch_size_per_step,
    get_num_devices,
    get_rank,
    set_torch_file_sharing_strategy_to_system,
)
from dlib.frameworks.wandb import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    WandbCleanupDiskAndCloudSpaceCallback,
    check_checkpoint_path_for_wandb,
    check_for_wandb_checkpoint_and_download_if_necessary,
)
from src.data_loading import LMDataModule
from src.helpers import infer_batch_size_per_device
from src.model import BasicLM


@dataclass
class TrainingArgs:
    model_name_or_path: str = dArg(
        default="roberta-base",
        help="HuggingFace model identifier. This is used to construct the model architecture and load pretrained weights if not specified otherwise.",  # noqa: E501
        aliases="--model",
    )
    language_modeling_strategy: Literal["mlm", "clm"] = dArg(
        default="mlm",
        help="Whether to train a masked language model or a causal language model.",
    )
    resume_training: bool = dArg(
        default=False,
        help="Whether to resume training form checkpoint or only load the weights.",
        aliases="--resume",
    )
    checkpoint_path: str | None = dArg(
        default=None,
        help="Path to a saved pytorch-lightning checkpoint. Use the wandb:<wandb-run-id> syntax to load a checkpoint from W&B.",  # noqa: E501
        aliases="--checkpoint",
    )
    tokenizer_path: str | None = dArg(
        default=None,
        help="Path to a directory containing a saved Huggingface PreTrainedTokenizer.",
        aliases="--tokenizer",
    )
    data_dir: str = dArg(
        default="./data",
        help="Path to the data directory. By default, expects a train.txt and dev.txt file inside the directory.",  # noqa: E501
        aliases="-d",
    )
    train_file: str = dArg(default="train.txt")
    dev_file: str = dArg(default="dev.txt")
    line_by_line: bool = dArg(
        default=False, help="Process dataset line by line instead of chunking."
    )
    language: str = dArg(
        default=None,
        help="If specified, the data is expected to lie inside a subdirectory with this name.",
        aliases=["--lang", "--lg", "-l"],
    )
    max_sequence_length: int = dArg(
        default=512,
        help="Sequence length for dataset tokenization.",
        aliases=["--max_seq_length", "--block_size"],
    )
    overwrite_data_cache: bool = dArg(
        default=False, help="Overwrite the cached preprocessed datasets or not.", aliases="--odc"
    )
    conserve_disk_space: bool = dArg(
        default=False, help="Cleanup cache files whenever possible to save disk space."
    )
    data_preprocessing_only: bool = dArg(
        default=False, help="Exit the script after data preprocessing. Do not start training."
    )

    ####### Hardware ###########
    accelerator: Literal["gpu", "cpu", "tpu", "mps", "auto"] = dArg(
        default="auto",
        help='Hardware accelerator to use. Can be gpu, cpu, tpu, mps, etc. If "auto", will auto-detect available hardware accelerator.',  # noqa: E501
    )
    distributed_strategy: Literal["ddp", "ddp_smart", "ddp_spawn", "ddp_fork", "dp", "auto"] = dArg(
        default="auto",
        help="Distributed training strategy to use.",
        aliases=["--dist_strategy", "--ds"],
    )
    devices: int | None = dArg(
        default=None,
        aliases=["--gpus", "--cpus", "--tpus"],
        help="Number of devices to use for distributed training. If -1, will use all available devices.",  # noqa: E501
    )
    workers: int = dArg(
        default=4,
        help="Number of workers for dataloaders. *Every device* weill use that many workers.",
        aliases="-w",
    )
    preprocessing_workers: int = dArg(
        default=4,
        help="Number of workers for preprocessing the datasets. Cached datasets are only valid for the same number of preprocessing workers.",  # noqa: E501
        aliases="--pw",
    )
    precision: Literal["16-mixed", "bf16-mixed", 32] = dArg(
        default=32,
        help="Floating point precision to use during training. Might require specific hardware.",
    )
    compile: bool = dArg(
        default=False,
        help="Whether to compile the model with using `torch.compile`. Requires torch>=2.0",
    )

    ####### General training ###########
    training_goal: int = dArg(default=10_000, help="Number K samples to train for.", aliases="--tg")
    val_frequency: float = dArg(
        default=0.05,
        help="Do validation every K samples. If <1, compute as fraction of training_goal",
        aliases="--vfq",
    )
    model_log_frequency: float = dArg(
        default=0.1,
        help="Log a model checkpoint every K samples. If <1, compute as fraction of training_goal",
        aliases="--mfq",
    )
    val_before_training: bool = dArg(default=True, help="Run one validation epoch before training.")
    batch_size_per_device: int = dArg(
        default=8,
        help="Batch size per device. If --effective_batch_size is specified, this is the maximum batch size per device (you should test when you cannot get larger without CUDA OOM errors.).",  # noqa: E501
        aliases=["--batch_size_per_gpu", "-b"],
    )
    effective_batch_size: int | None = dArg(
        default=None,
        help="If set, try to auto-infer batch_size_per_device and gradient_accumulation_steps based on number of devices given by --devices.",  # noqa: E501
        aliases=["--eb"],
    )
    learning_rate: float = dArg(default=5e-5, aliases="--lr")
    lr_warmup: float = dArg(
        default=0.1,
        help="Number of K samples to do a learning rate warmup. If <1, compute as fraction of training_goal.",  # noqa: E501
    )
    lr_schedule: Literal[
        "cosine", "linear", "reduce_on_plateau", "constant", "cosine_with_restarts", "polynomial"
    ] = dArg(default="cosine", help="Learning rate schedule.")
    weight_decay: float = dArg(default=0.0, aliases="--wd")
    gradient_clipping: float | None = dArg(default=None, aliases="--gc")
    gradient_accumulation_steps: int = dArg(default=1, aliases=["--gas", "--accum"])
    train_only_embeddings: bool = dArg(
        default=False,
        help="Train only the embedding layer of the model and keep the other transformer layers frozen.",  # noqa: E501
        aliases="--only_embeddings",
    )
    from_scratch: bool = dArg(
        default=False, help="Do not use pre-trained weights to intialize the model."
    )
    from_scratch_embeddings: bool = dArg(
        default=False, help="Do not use pre-trained weights to intialize the token embeddings."
    )

    def __post_init__(self):
        if self.val_frequency < 1:
            self.val_frequency = int(self.training_goal * self.val_frequency)
        if self.model_log_frequency < 1:
            self.model_log_frequency = int(self.training_goal * self.model_log_frequency)
        if self.lr_warmup < 1:
            self.lr_warmup = int(self.training_goal * self.lr_warmup)


@dataclass
class MiscArgs:
    seed: int | None = None
    force_deterministic: bool = dArg(
        default=False, help="Force PyTorch operations to be deterministic."
    )
    offline: bool = dArg(default=False, help="Disable W&B online syncing.")
    fast_dev_run: bool = dArg(
        default=False, help="Do fast run through training and validation with reduced sizes."
    )
    wandb_run_name: str | None = dArg(
        default=None, help="Run name for the W&B online UI.", aliases="-n"
    )
    wandb_tags: list[str] = dArg(default=[])
    wandb_project: str = dArg(default=None)
    too_many_open_files_fix: bool = dArg(
        default=False,
        help='Apply fix to circumvent "Too many open files" error caused by the PyTorch Dataloader when using many workers or large batches.',  # noqa: E501
        aliases="--open_files_fix",
    )


@logger.catch(reraise=True)
def main(parsed_arg_groups: tuple[TrainingArgs, MiscArgs]):
    args, misc_args = parsed_arg_groups

    ################ Apply fixes ##############
    if misc_args.too_many_open_files_fix:
        logger.info("Setting torch sharing strategy to 'file_system'")
        set_torch_file_sharing_strategy_to_system()

    ############# Seed & print args ##############
    misc_args.seed = seed_everything(workers=True, seed=misc_args.seed)
    current_process_rank = get_rank()
    if current_process_rank == 0:
        for arg_group in parsed_arg_groups:
            logger.info(arg_group)

    ############# Construct W&B Logger ##############
    if misc_args.offline or misc_args.fast_dev_run or args.data_preprocessing_only:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb_extra_args = dict(
        name=misc_args.wandb_run_name,
    )
    if (
        args.checkpoint_path
        and args.resume_training
        and check_checkpoint_path_for_wandb(args.checkpoint_path)
    ):
        logger.info("Resuming training from W&B")
        wandb_extra_args = dict(
            id=check_checkpoint_path_for_wandb(args.checkpoint_path), resume="must"
        )  # resume W&B run
    else:
        args.resume_training = False

    wandb_logger = WandbLogger(
        project=misc_args.wandb_project or WANDB_PROJECT,
        entity=WANDB_ENTITY,
        log_model="all",
        tags=misc_args.wandb_tags,
        **wandb_extra_args,
    )

    #### Calculate effective batch size / gradient accumulation steps ####
    ACCELERATOR = args.accelerator.upper()
    num_devices = get_num_devices(args.devices)
    if args.effective_batch_size:
        logger.info(
            f"Trying to auto-infer settings for effective batch size {args.effective_batch_size}..."
        )
        (
            args.batch_size_per_device,
            args.gradient_accumulation_steps,
            effective_batch_size_per_step,
        ) = infer_batch_size_per_device(
            num_devices, args.effective_batch_size, args.batch_size_per_device
        )

        logger.info(
            f"Using effective batch size {args.effective_batch_size}"
            f"with {num_devices} {ACCELERATOR}s, "
            f"{args.batch_size_per_device} batch size per {ACCELERATOR} and "
            f"{args.gradient_accumulation_steps} gradient accumulation steps."
        )
    else:
        effective_batch_size_per_step = get_effective_batch_size_per_step(
            args.devices, args.batch_size_per_device
        )  # does not take accumulation into account
        args.effective_batch_size = effective_batch_size_per_step * args.gradient_accumulation_steps
        logger.info(
            f"Effective batch size {args.effective_batch_size} based on specified args"
            f"{num_devices} {ACCELERATOR}s, "
            f"{args.batch_size_per_device} batch size per {ACCELERATOR} and"
            f"{args.gradient_accumulation_steps} gradient accumulation steps."
        )

    for arg_group in parsed_arg_groups:
        wandb_logger.log_hyperparams(dataclasses.asdict(arg_group))

    if current_process_rank == 0 and not args.resume_training and not misc_args.offline:
        if misc_args.wandb_run_name is None:
            logger.warning("No run name specified with `--wandb_run_name`. Using W&B default (randomly generated name).")
        else:
            wandb_logger.experiment.name = (
                misc_args.wandb_run_name + "-" + wandb_logger.version
            )  # Append id to name for easier recognition in W&B UI

    ########### Calulate training constants ###########
    KSAMPLES = 1000
    args.training_goal = int(
        args.training_goal * KSAMPLES / args.effective_batch_size
    )  # Lightning does `gradient_accumulation_steps` many forward passes per step (step := optimization step aka backward pass)
    val_frequency_in_optimization_steps = int(
        args.val_frequency * KSAMPLES / args.effective_batch_size
    )
    args.val_frequency = int(
        args.val_frequency * KSAMPLES / effective_batch_size_per_step
    )  # val_frequency in lightning is every forward pass NOT optimization step
    args.model_log_frequency = int(args.model_log_frequency * KSAMPLES / args.effective_batch_size)
    args.lr_warmup = int(args.lr_warmup * KSAMPLES / args.effective_batch_size)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_name_or_path, use_fast=True
    )

    vocab_size = len(tokenizer)  # NOTE: tokenizer.vocab_size returns size without added vocab
    checkpoint_callback = ModelCheckpoint(
        filename="snap-{step}-ksamples-{progress/ksamples:.2f}-loss-{val/loss:.2f}",
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
        every_n_train_steps=args.model_log_frequency,
    )
    wandb_disk_cleanup_callback = WandbCleanupDiskAndCloudSpaceCallback(
        cleanup_local=True, cleanup_online=False, size_limit=20
    )

    ################# Construct model ##############
    model_extra_args = dict(
        effective_batch_size_per_step=effective_batch_size_per_step, vocab_size=vocab_size
    )

    # Resume from checkpoint if specified
    if args.checkpoint_path:
        args.checkpoint_path = check_for_wandb_checkpoint_and_download_if_necessary(
            args.checkpoint_path, wandb_logger.experiment
        )

        if args.resume_training:  # load weights, optimizer states, scheduler state, ...\
            if current_process_rank == 0:
                resume_ksamples = wandb_logger.experiment.summary[
                    "progress/ksamples"
                ]  # TODO: use ksamples from checkpoint instead of run summary
                os.environ["DLIB_PROGRESS_KSAMPLES"] = str(resume_ksamples)
            else:
                # need to pass via env var because wandb_logger is only available on main process
                resume_ksamples = float(os.environ["DLIB_PROGRESS_KSAMPLES"])

            print("Resuming from", resume_ksamples)
            model = BasicLM.load_from_checkpoint(
                args.checkpoint_path,
                ksamples_processed=resume_ksamples,
                effective_batch_size_per_step=effective_batch_size_per_step,
            )
            print(model.hparams.ksamples_processed)
        else:  # load only weights
            model = BasicLM(training_args=args, **model_extra_args)
            torch_load = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(torch_load["state_dict"], strict=False)
    else:
        model = BasicLM(training_args=args, **model_extra_args)

    if args.train_only_embeddings:
        if get_rank() == 0:
            logger.info("Training only embedding layer")
        for param in model.model.parameters():
            param.requires_grad = False
        model.model.get_input_embeddings().weight.requires_grad = True

    if args.from_scratch_embeddings:
        torch.nn.init.xavier_uniform_(model.model.get_input_embeddings().weight)
        # torch.nn.init.normal_(self.model.get_input_embeddings().weight) # alternative

    if current_process_rank == 0:
        model.on_train_start = lambda: logger.info(
            f"Total training steps: {args.training_goal} | "
            f"LR warmup steps: {args.lr_warmup} | "
            f"Validation Frequency: {val_frequency_in_optimization_steps} | "
            f"Model Log Frequencey: {args.model_log_frequency} | "
            f"Effective batch size: {args.effective_batch_size}"
        )
    wandb_logger.watch(model, log="gradients", log_freq=500, log_graph=False)

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("high")
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"The current torch version ({torch.__version__}) does not have support for compile."  # noqa: E501
                "Please install torch >= 2.0 or disable compile."
            )
        model = torch.compile(model)

    #################### Construct dataloaders & trainer #################
    dm = LMDataModule(training_args=args, misc_args=misc_args)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, wandb_disk_cleanup_callback, lr_monitor]
    if args.accelerator == "gpu":
        callbacks.append(CUDAMetricsCallback())

    # "smart" DDP skipping the find_unused_parameters step - slightly faster
    distributed_strategy = (
        DDPStrategy(find_unused_parameters=False)
        if args.accelerator == "gpu" and args.distributed_strategy == "ddp_smart"
        else args.distributed_strategy
    )

    plugins = None
    if SLURMEnvironment.detect():
        logger.info("Disabling SLURMEnvironment (we use lightning's native DDP launcher)")
        plugins = [LightningEnvironment()]

    # Initialize trainer
    trainer = Trainer(
        max_steps=args.training_goal,
        val_check_interval=args.val_frequency,
        check_val_every_n_epoch=None,  # validation based on steps instead of epochs
        devices=args.devices,
        accelerator=args.accelerator,
        strategy=distributed_strategy,
        logger=wandb_logger,
        deterministic=misc_args.force_deterministic,
        callbacks=callbacks,
        plugins=plugins,
        enable_checkpointing=True,
        precision=args.precision,
        gradient_clip_val=args.gradient_clipping,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        fast_dev_run=misc_args.fast_dev_run,
        inference_mode=not args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other  # noqa: E501
    )

    if args.val_before_training:
        # TODO: we could use a new trainer with Trainer(devices=1, num_nodes=1) to prevent samples from possibly getting replicated with DistributedSampler here.  # noqa: E501
        logger.info(f"Rank {current_process_rank} | Validation before training...")
        val_result = trainer.validate(model, dm)
        print(val_result)

    logger.info(f"Rank {current_process_rank} | Starting training...")
    try:
        trainer.fit(model, dm, ckpt_path=args.checkpoint_path if args.resume_training else None)

        logger.success("Fit complete, starting validation...")
        # Validate after training has finished
        trainer.validate(model, dm)
    except Exception as e:
        logger.exception(e)
    finally:
        if current_process_rank == 0:
            logger.info("Trying to save checkpoint....")

            save_path = str(Path(checkpoint_callback.dirpath) / "last_model_ckpt.ckpt")
            trainer.save_checkpoint(save_path)

            logger.info("Collecting PL checkpoint for wandb...")
            artifact = wandb.Artifact(name=f"model-{wandb_logger.experiment.id}", type="model")
            artifact.add_file(save_path, name="model.ckpt")

            logger.info('Collecting "raw" HF checkpoint for wandb...')
            # Also save raw huggingface checkpoint, so that we don't need lightning and the current code structure to load the weights  # noqa: E501
            raw_huggingface_model: PreTrainedModel = trainer.lightning_module.model
            save_path = str(Path(checkpoint_callback.dirpath) / "raw_huggingface")
            raw_huggingface_model.save_pretrained(save_path)
            artifact.add_dir(save_path, name="raw_huggingface")

            logger.info("Pushing to wandb...")
            aliases = ["train_end", "latest"]
            wandb_logger.experiment.log_artifact(artifact, aliases=aliases)

            logger.success("Saving finished!")


if __name__ == "__main__":
    parsed_arg_groups = dargparse(dataclasses=(TrainingArgs, MiscArgs))
    main(parsed_arg_groups)
