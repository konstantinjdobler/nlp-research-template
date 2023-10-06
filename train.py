import dataclasses
import os
from pathlib import Path

import torch
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import LightningEnvironment, SLURMEnvironment
from print_on_steroids import graceful_exceptions, logger
from simple_parsing import parse
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from args import TrainingArgs
from dlib import CUDAMetricsCallback, WandbCleanupDiskAndCloudSpaceCallback, get_rank, log_slurm_info, wait_for_debugger
from src.data_loading import LMDataModule
from src.helpers import (
    ProgressMetricCallback,
    check_checkpoint_path_for_wandb,
    check_for_wandb_checkpoint_and_download_if_necessary,
)
from src.model import BasicLM

WANDB_PROJECT = "nlp-research-template"
WANDB_ENTITY = "konstantinjdobler"


def main(args: TrainingArgs):
    ########### CUDA checks ###########
    current_process_rank = get_rank()
    logger.config(rank=current_process_rank, print_rank0_only=True)
    if args.accelerator == "cuda":
        num_available_gpus = torch.cuda.device_count()
        if num_available_gpus > args.num_devices:
            logger.warning(
                f"Requested {args.num_devices} GPUs but {num_available_gpus} are available.",
                f"Using first {args.num_devices} GPUs. You should set CUDA_VISIBLE_DEVICES or the docker --gpus flag to the desired GPU ids.",
            )
        if not torch.cuda.is_available():
            logger.error("CUDA is not available, you should change the accelerator with --accelerator cpu|tpu|mps.")
            exit(1)
    if current_process_rank == 0 and args.debug:
        wait_for_debugger()
    args.seed = seed_everything(workers=True, seed=args.seed)

    ############# Construct W&B Logger ##############
    if args.offline or args.fast_dev_run or args.data_preprocessing_only:
        os.environ["WANDB_MODE"] = "dryrun"
    wandb_extra_args = dict(name=args.run_name)
    if args.saved_checkpoint_path and args.resume and check_checkpoint_path_for_wandb(args.saved_checkpoint_path):
        logger.info("Resuming training from W&B")
        wandb_extra_args = dict(id=check_checkpoint_path_for_wandb(args.saved_checkpoint_path), resume="must")  # resume W&B run
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        log_model="all",
        tags=args.wandb_tags,
        save_dir="logs/",
        **wandb_extra_args,
    )
    wandb_logger.log_hyperparams(dataclasses.asdict(args))
    wandb_logger.experiment.log_code(".")  # log code to wandb to be able to reproduce the run
    if current_process_rank == 0:
        logger.info(args)
    if current_process_rank == 0 and not args.resume and not args.offline:
        if args.run_name is None:
            logger.warning("No run name specified with `--run_name`. Using W&B default (randomly generated name).")
        else:
            assert wandb_logger.version is not None
            wandb_logger.experiment.name = (
                args.run_name + "-" + wandb_logger.version
            )  # Append id to name for easier recognition in W&B UI
    IS_ON_SLURM = SLURMEnvironment.detect()
    if IS_ON_SLURM and current_process_rank == 0:
        log_slurm_info()

    ################# Construct model ##############

    # Resume from checkpoint if specified
    model_args = dict(
        model_name_or_path=args.hf_model_name,
        lm_objective=args.language_modeling_objective,
        from_scratch=args.from_scratch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        lr_schedule=args.lr_schedule,
        warmup_period=args.warmup_period,
        eval_interval=args.eval_interval,
    )
    if args.saved_checkpoint_path:
        args.saved_checkpoint_path = check_for_wandb_checkpoint_and_download_if_necessary(
            args.saved_checkpoint_path, wandb_logger.experiment
        )

        if args.resume:  # load weights, optimizer states, scheduler state, ...\
            model = BasicLM.load_from_checkpoint(args.saved_checkpoint_path, save_hyperparameters=False)
            # we will resume via trainer.fit(ckpt_path=...)
        else:  # load only weights
            model = BasicLM(**model_args)
            torch_load = torch.load(args.saved_checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(torch_load["state_dict"], strict=False)
    else:
        model = BasicLM(**model_args)

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.tokenizer_path or args.hf_model_name, use_fast=True)
    if not args.resume:
        pretrained_vocab_size = model.model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) != pretrained_vocab_size:
            logger.warning(f"Resizing embedding size from {pretrained_vocab_size} to match tokenizer ({len(tokenizer)}).")
            model.model.resize_token_embeddings(len(tokenizer))

    wandb_logger.watch(model, log="all", log_freq=500, log_graph=False)

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
    dm = LMDataModule(training_args=args, tokenizer=tokenizer)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    wandb_disk_cleanup_callback = WandbCleanupDiskAndCloudSpaceCallback(cleanup_local=True, cleanup_online=False, size_limit=20)
    checkpoint_callback = ModelCheckpoint(
        filename="snap-{step}-samples-{progress/samples}-{progress/tokens}-loss-{val/loss:.2f}",
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
        every_n_train_steps=int(args.save_interval),
    )
    callbacks = [checkpoint_callback, wandb_disk_cleanup_callback, lr_monitor, ProgressMetricCallback()]
    if args.accelerator == "cuda":
        callbacks.append(CUDAMetricsCallback())

    plugins = None
    if IS_ON_SLURM:
        logger.info("Disabling SLURMEnvironment (we use lightning's native DDP launcher)")
        plugins = [LightningEnvironment()]

    # lightning wants val_check_interval in num forward passes (iters) not num optimization steps
    val_frequency_in_iters = args.eval_interval * args.gradient_accumulation_steps

    # Initialize trainer
    trainer = Trainer(
        max_steps=args.training_goal,
        val_check_interval=val_frequency_in_iters,
        check_val_every_n_epoch=None,  # validation based on steps instead of epochs
        devices=args.num_devices,
        accelerator=args.accelerator,
        strategy=args.distributed_strategy,
        logger=wandb_logger,
        deterministic=args.force_deterministic,
        callbacks=callbacks,
        plugins=plugins,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        fast_dev_run=args.fast_dev_run,
        limit_val_batches=None if args.eval_samples == -1 else (args.eval_samples // args.eval_micro_batch_size),
        inference_mode=not args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other
    )

    if current_process_rank == 0:
        logger.info(
            f"Total optimizer steps: {args.training_goal} | "
            f"LR warmup steps: {args.warmup_period} | "
            f"Validation Frequency: {args.eval_interval} | "
            f"Model Log Frequency: {args.save_interval} | "
            f"Effective batch size: {args.batch_size} | "
            f"Micro batch size (per device and forward pass): {args.eval_micro_batch_size} | "
            f"Gradient accumulation steps: {args.gradient_accumulation_steps} | "
        )

    ########### Start val & train loop ###########
    if args.val_before_training and not args.resume:
        # TODO: we could use a new trainer with Trainer(devices=1, num_nodes=1) to prevent samples from possibly getting replicated with DistributedSampler here.
        logger.info(f"Rank {current_process_rank} | Validation before training...")
        val_result = trainer.validate(model, dm)
        print(val_result)
        if args.only_val:
            exit(0)

    logger.info(f"Rank {current_process_rank} | Starting training...")
    trainer.fit(model, dm, ckpt_path=args.saved_checkpoint_path if args.resume else None)
    if trainer.interrupted and IS_ON_SLURM:
        logger.error(
            "Detected keyboard interrupt, not trying to save latest checkpoint right now because we detected SLURM and do not want to drain the node..."
        )
    else:
        if trainer.interrupted:
            logger.warning("Detected keyboard interrupt, trying to save latest checkpoint...")
        else:
            logger.success("Fit complete, starting validation...")
            trainer.validate(model, dm)

        if current_process_rank == 0:
            logger.info("Trying to save checkpoint....")

            save_path = str(Path(checkpoint_callback.dirpath) / "last_model_ckpt.ckpt")
            trainer.save_checkpoint(save_path)

            logger.info("Collecting PL checkpoint for wandb...")
            artifact = wandb.Artifact(name=f"model-{wandb_logger.experiment.id}", type="model")
            artifact.add_file(save_path, name="model.ckpt")

            logger.info("Pushing to wandb...")
            aliases = ["train_end", "latest"]
            wandb_logger.experiment.log_artifact(artifact, aliases=aliases)

            logger.success("Saving finished!")


if __name__ == "__main__":
    parsed_arg_groups = parse(TrainingArgs, add_config_path_arg=True)
    current_process_rank = get_rank()
    with graceful_exceptions(extra_message=f"Rank: {current_process_rank}"):
        main(parsed_arg_groups)
