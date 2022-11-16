import dataclasses
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import wandb
from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.data_loading import LMDataModule
from src.model import BasicLM
from dlib.argparsing.dArgumentParser import dArg, parse_args_into_dataclasses
from dlib.frameworks.pytorch import get_effective_batch_size_per_step, get_rank, set_torch_file_sharing_strategy_to_system
from dlib.frameworks.wandb import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    WandbCleanupDiskAndCloudSpaceCallback,
    check_checkpoint_path_for_wandb,
    check_for_wandb_checkpoint_and_download_if_necessary,
)


class dSchedulerType(Enum):
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


@dataclass
class TrainingArgs:
    model_name_or_path: str = dArg(
        default="roberta-base",
        help="HuggingFace model identifier. This is used to construct the model architecture and load pretrained weights if not specified otherwise",
        aliases="--model",
    )
    resume_training: bool = dArg(
        default=False, help="Whether to resume training form checkpoint or only load the weights.", aliases="--resume"
    )
    checkpoint_path: Optional[str] = dArg(
        default=None,
        help="Path to a saved PyTorch Lightning checkpoint. You can use wandb:<wandb-run-id> syntax to load a checkpoint from W&B.",
        aliases="--checkpoint",
    )
    tokenizer_path: Optional[str] = dArg(
        default=None,
        help="Path to a directory containing a saved Huggingface PreTrainedTokenizer.",
        aliases="--tokenizer",
    )
    data_dir: str = dArg(
        default="./data",
        help="Path to the data directory. By default, expects a train.txt and dev.txt file inside the directory.",
        aliases="-d",
    )
    language: str = dArg(
        default=None,
        help="If specified, the data is expected to lie inside a subdirectory with this name.",
        aliases=["--lang", "--lg", "-l"],
    )
    max_sequence_length: int = dArg(default=256, help="Sequence length for dataset tokenization.", aliases="--max_seq_length")
    overwrite_data_cache: bool = dArg(default=False, help="Overwrite the cached preprocessed datasets or not.", aliases="--odc")
    gpus: Optional[int] = dArg(default=None)
    workers: int = dArg(default=4, help="Number of workers for dataloader.", aliases="-w")
    preprocessing_workers: int = dArg(
        default=1,
        help="Number of workers for preprocessing the datasets. Cached datasets are only valid for the same number of preprocessing workers.",
        aliases="--pw",
    )
    precision: str = dArg(
        default="32",
        help="Floating point precision to use during training. Might require specific hardware. Options: [32, 16, bf16].",
    )

    ####### General training ###########
    training_goal: int = dArg(default=50_000, help="Number K samples to train for.", aliases="--tg")
    val_frequency: int = dArg(default=1000, help="Do validation every K samples.", aliases="--vfq")
    val_before_training: bool = dArg(default=False, help="Run one validation epoch before training.")
    batch_size_per_device: int = dArg(default=8, aliases=["--batch_size_per_gpu", "-b"])
    learning_rate: float = dArg(default=5e-5, aliases="--lr")
    lr_warmup: int = dArg(default=4000, help="Number of K samples to do a learning rate warmup.")
    lr_schedule: dSchedulerType = dArg(default=dSchedulerType.COSINE.value, help="Learning rate schedule.")
    weight_decay: float = dArg(default=0.0, aliases="--wd")
    gradient_clipping: Optional[float] = dArg(default=None, aliases="--gc")
    gradient_accumulation_steps: int = dArg(default=1, aliases="--accum")
    train_only_embeddings: bool = dArg(
        default=False,
        help="Train only the embedding layer of the model and keep the other transformer layers frozen.",
        aliases="--only_embeddings",
    )
    from_scratch: bool = dArg(default=False, help="Do not use pre-trained weights to intialize the model.")
    from_scratch_embeddings: bool = dArg(
        default=False, help="Do not use pre-trained weights to intialize the token embeddings."
    )


@dataclass
class MiscArgs:
    seed: Optional[int] = None
    force_deterministic: bool = dArg(default=False, help="Force PyTorch operations to be deterministic.")
    offline: bool = dArg(default=False, help="Disable W&B online syncing.")
    fast_dev_run: bool = dArg(default=False, help="Do fast run through training and validation with reduced sizes.")
    wandb_run_name: Optional[str] = dArg(default=None, help="Run name for the W&B online UI.", aliases="-n")
    wandb_tags: list[str] = dArg(default_factory=list)
    too_many_open_files_fix: bool = dArg(
        default=False,
        help='Apply fix to circumvent "Too many open files" error caused by the PyTorch Dataloader when using many workers or large batches.',
        aliases="--open_files_fix",
    )
    wandb_project: str = dArg(default=None)


def main():
    args = parse_args_into_dataclasses(
        dataclasses=(
            TrainingArgs,
            MiscArgs,
        )
    )

    training_args, misc_args = args
    try:
        training_args.precision = int(training_args.precision)
    except ValueError:
        pass

    misc_args.seed = seed_everything(workers=True, seed=misc_args.seed)

    current_process_rank = get_rank()

    if current_process_rank == 0:
        for arg_group in args:
            logger.info(arg_group)

    ################ Apply fixes ##############
    if misc_args.too_many_open_files_fix:
        logger.info("Setting torch sharing strategy to 'file_system'")
        set_torch_file_sharing_strategy_to_system()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    ############# Construct W&B Logger ##############
    if misc_args.offline or misc_args.fast_dev_run:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb_extra_args = dict(
        name=misc_args.wandb_run_name,
    )

    RESUMING_TRAINING = False
    if (
        training_args.checkpoint_path
        and training_args.resume_training
        and check_checkpoint_path_for_wandb(training_args.checkpoint_path)
    ):
        RESUMING_TRAINING = True
        logger.info("Resuming training from W&B")
        wandb_extra_args = dict(
            id=check_checkpoint_path_for_wandb(training_args.checkpoint_path), resume="must"
        )  # resume W&B run

    wandb_logger = WandbLogger(
        project=misc_args.wandb_project or WANDB_PROJECT,
        entity=WANDB_ENTITY,
        log_model="all",
        tags=misc_args.wandb_tags,
        **wandb_extra_args,
    )

    effective_batch_size_per_step = get_effective_batch_size_per_step(
        training_args.gpus, training_args.batch_size_per_device
    )  # does not take accumulation into account

    for arg_group in args:
        wandb_logger.log_hyperparams(dataclasses.asdict(arg_group))
    wandb_logger.log_hyperparams(
        {"effective_batch_size": effective_batch_size_per_step * training_args.gradient_accumulation_steps}
    )
    if current_process_rank == 0 and not RESUMING_TRAINING:
        wandb_logger.experiment.name = (
            misc_args.wandb_run_name + "-" + wandb_logger.version
        )  # Append id to name for easier recognition in W&B UI

    ########### Calulate training constants ###########
    KSAMPLES = 1000
    total_effective_batch_size = effective_batch_size_per_step * training_args.gradient_accumulation_steps
    training_args.training_goal = int(
        training_args.training_goal * KSAMPLES / total_effective_batch_size
    )  # Lightning does grad_accum forward passes per step
    val_frequency_in_optimization_steps = int(training_args.val_frequency * KSAMPLES / total_effective_batch_size)
    training_args.val_frequency = int(
        training_args.val_frequency * KSAMPLES / effective_batch_size_per_step
    )  # val_frequency in lightning is every batch NOT optmization step
    training_args.lr_warmup = int(training_args.lr_warmup * KSAMPLES / total_effective_batch_size)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        training_args.tokenizer_path or training_args.model_name_or_path, use_fast=True
    )

    vocab_size = len(tokenizer)  # NOTE: tokenizer.voab_size returns size without added vocab
    checkpoint_callback = ModelCheckpoint(
        filename="snap-{step}-ksamples-{progress/ksamples:.2f}-loss-{val/loss:.2f}",
        monitor="val/loss",
        mode="min",
        # save_last=True,
        auto_insert_metric_name=False,
        every_n_train_steps=5 * val_frequency_in_optimization_steps,  # don't save on every val frqcy
    )
    wandb_disk_cleanup_callback = WandbCleanupDiskAndCloudSpaceCallback(cleanup_local=True, cleanup_online=False, size_limit=20)

    ################# Construct model ##############
    model_extra_args = dict(effective_batch_size=effective_batch_size_per_step, vocab_size=vocab_size)

    # Resume from checkpoint if specified
    if training_args.checkpoint_path:
        training_args.checkpoint_path = check_for_wandb_checkpoint_and_download_if_necessary(
            training_args.checkpoint_path, wandb_logger.experiment
        )

        if training_args.resume_training:  # load weights, optimizer states, scheduler state, ...\
            if current_process_rank == 0:
                resume_ksamples = wandb_logger.experiment.summary["progress/ksamples"]
                os.environ["DLIB_PROGRESS_KSAMPLES"] = str(resume_ksamples)
            else:
                resume_ksamples = float(os.environ["DLIB_PROGRESS_KSAMPLES"])

            print("resuming from", resume_ksamples)
            model = BasicLM.load_from_checkpoint(
                training_args.checkpoint_path,
                ksamples_processed=resume_ksamples,
                effective_batch_size=effective_batch_size_per_step,
            )
            print(model.hparams.ksamples_processed)
        else:  # load only weights
            model = BasicLM(training_args=training_args, **model_extra_args)
            torch_load = torch.load(training_args.checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(torch_load["state_dict"], strict=False)
    else:
        model = BasicLM(training_args=training_args, **model_extra_args)

    if current_process_rank == 0:
        model.on_train_start = lambda: logger.info(
            f"Total training steps: {training_args.training_goal} | LR warmup steps: {training_args.lr_warmup} | Validation Frequency: {val_frequency_in_optimization_steps} | Effective batch size: {total_effective_batch_size}"
        )

    wandb_logger.watch(model, log="gradients", log_freq=500, log_graph=False)

    #################### Construct dataloaders & trainer #################
    dm = LMDataModule(training_args=training_args, misc_args=misc_args)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize trainer
    trainer = Trainer(
        max_steps=training_args.training_goal,
        val_check_interval=training_args.val_frequency,
        gpus=training_args.gpus,
        strategy=DDPPlugin(find_unused_parameters=False) if training_args.gpus is not None else None,
        logger=wandb_logger,
        deterministic=misc_args.force_deterministic,
        callbacks=[checkpoint_callback, wandb_disk_cleanup_callback, lr_monitor],
        enable_checkpointing=True,
        precision=training_args.precision,
        gradient_clip_val=training_args.gradient_clipping,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        fast_dev_run=misc_args.fast_dev_run,
    )

    if training_args.val_before_training:
        logger.info(f"Rank {current_process_rank} | Validation before training...")
        val_result = trainer.validate(model, dm)
        print(val_result)
        exit(0)

    # if RESUMING_TRAINING:
    #     trainer.fit_loop.epoch_loop.global_step = wandb_logger.experiment.summary["trainer/global_step"]

    logger.info(f"Rank {current_process_rank} | Starting training...")
    trainer.fit(model, dm, ckpt_path=training_args.resume_training and training_args.checkpoint_path)

    if current_process_rank == 0:
        logger.info("Trying to save checkpoint....")

        save_path = str(Path(checkpoint_callback.dirpath) / "last_model_ckpt.ckpt")
        trainer.save_checkpoint(save_path)
        artifact = wandb.Artifact(name=f"model-{wandb_logger.experiment.id}", type="model")
        artifact.add_file(save_path, name="model.ckpt")
        aliases = ["train_end", "latest"]
        wandb_logger.experiment.log_artifact(artifact, aliases=aliases)


if __name__ == "__main__":
    main()
