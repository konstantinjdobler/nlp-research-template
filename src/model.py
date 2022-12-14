# need this to avoid cyclical import
from __future__ import annotations  # fmt: skip
from typing import TYPE_CHECKING  # fmt: skip
if TYPE_CHECKING:  # fmt: skip
    from train import TrainingArgs  # fmt: skip

from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from loguru import logger
from torch import nn
from torch.optim import AdamW
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
from transformers.optimization import get_scheduler
from warmup_scheduler import GradualWarmupScheduler

from dlib.frameworks.pytorch import get_rank


@dataclass
class ModelArgs:
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


class BasicLM(pl.LightningModule):
    def __init__(
        self,
        training_args: TrainingArgs,
        adhoc_args: ModelArgs = ModelArgs(),
        effective_batch_size_per_step=-100000,
        vocab_size=None,
        ksamples_processed=0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["from_scratch", "from_scratch_embeddings", "train_only_embeddings", "effective_batch_size"]
        )
        self.args = training_args
        self.adhoc_args = adhoc_args
        config = AutoConfig.from_pretrained(self.args.model_name_or_path, return_dict=True)

        self.model: PreTrainedModel = (
            AutoModelForMaskedLM.from_pretrained(self.args.model_name_or_path, config=config)
            if not self.args.from_scratch
            else AutoModelForMaskedLM.from_config(config=config)
        )

        self.model.resize_token_embeddings(vocab_size)
        if self.args.from_scratch and get_rank() == 0:
            logger.info("Training from scratch without pretrained weights")

        if self.args.train_only_embeddings:
            if get_rank() == 0:
                logger.info("Training only embedding layer")
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.get_input_embeddings().weight.requires_grad = True

        if self.args.from_scratch_embeddings:
            nn.init.xavier_uniform_(self.model.get_input_embeddings().weight)

        self.effective_batch_size_per_step = effective_batch_size_per_step

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("train/loss", loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0) -> None:
        self.hparams.ksamples_processed += self.effective_batch_size_per_step / 1000
        self.log("progress/ksamples", self.hparams.ksamples_processed, rank_zero_only=True)

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log_dict({"val/loss": loss, "progress/ksamples": self.hparams.ksamples_processed}, sync_dist=True)

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.args.learning_rate}, weight decay: {self.args.weight_decay} and warmup steps: {self.args.lr_warmup}"
            )

        named_parameters = list(self.model.named_parameters())

        ### Filter out parameters that are not optmized (requires_grad == False)
        named_parameters = list(filter(lambda named_param: named_param[1].requires_grad, named_parameters))

        ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_parameters,
            self.args.learning_rate,
            betas=(self.adhoc_args.adam_beta1, self.adhoc_args.adam_beta2),
            eps=self.adhoc_args.adam_epsilon,
        )

        if self.args.lr_schedule == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
            if self.args.lr_warmup > 0:  # Wrap ReduceLROnPlatesu to enable LR warmup
                scheduler = GradualWarmupScheduler(
                    optimizer, multiplier=1, total_epoch=self.args.lr_warmup, after_scheduler=scheduler
                )
            scheduler_config = {"frequency": self.args.val_frequency, "monitor": "train/loss"}
        else:
            scheduler_name = self.args.lr_schedule
            if scheduler_name == "constant" and self.args.lr_warmup > 0:
                scheduler_name += "_with_warmup"
            scheduler = get_scheduler(
                scheduler_name,
                optimizer,
                num_warmup_steps=self.args.lr_warmup,
                num_training_steps=self.trainer.max_steps,
            )
            scheduler_config = {"frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", **scheduler_config}}
