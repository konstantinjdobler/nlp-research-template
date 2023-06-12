from dataclasses import dataclass
from typing import TYPE_CHECKING

import lightning as L
import torch
from loguru import logger
from torch.optim import AdamW
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
from transformers.optimization import get_scheduler
from warmup_scheduler import GradualWarmupScheduler

from dlib.frameworks.pytorch import get_rank

if TYPE_CHECKING:
    from train import TrainingArgs


@dataclass
class ModelArgs:
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


class BasicLM(L.LightningModule):
    def __init__(
        self,
        training_args: "TrainingArgs",  # do in string to remove dependency when loading.
        adhoc_args: ModelArgs = ModelArgs(),
        effective_batch_size_per_step=-100000,
        vocab_size=None,
        samples_processed=0.0,
        tokens_processed=0.0,
    ) -> None:
        super().__init__()
        if not training_args.resume_training:
            self.save_hyperparameters(
                ignore=["effective_batch_size_per_step", "samples_processed", "tokens_processed"]
            )
        self.args = training_args
        self.adhoc_args = adhoc_args
        config = AutoConfig.from_pretrained(self.args.model_name_or_path, return_dict=True)

        if self.args.language_modeling_strategy == "mlm":
            self.model: PreTrainedModel = (
                AutoModelForMaskedLM.from_pretrained(self.args.model_name_or_path, config=config)
                if not self.args.from_scratch
                else AutoModelForMaskedLM.from_config(config=config)
            )
        elif self.args.language_modeling_strategy == "clm":
            self.model: PreTrainedModel = (
                AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path, config=config)
                if not self.args.from_scratch
                else AutoModelForCausalLM.from_config(config=config)
            )

        self.model.resize_token_embeddings(vocab_size)

        if self.args.from_scratch and get_rank() == 0:
            logger.info("Training from scratch without pretrained weights")

        self.effective_batch_size_per_step = effective_batch_size_per_step
        self.register_buffer("samples_processed", torch.tensor(samples_processed))
        self.register_buffer("tokens_processed", torch.tensor(tokens_processed))

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0) -> None:
        self.samples_processed += self.effective_batch_size_per_step
        self.tokens_processed += self.effective_batch_size_per_step * self.args.max_sequence_length
        self.log_dict(
            {
                "progress/samples": self.samples_processed,
                "progress/tokens": self.tokens_processed,
            },
            rank_zero_only=True,
            on_step=True,
            on_epoch=False,
        )

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss

        self.log_dict(
            {
                "val/loss": loss,
                "progress/samples": self.samples_processed,
                "progress/tokens": self.tokens_processed,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.args.learning_rate}, weight decay: {self.args.weight_decay} and warmup steps: {self.args.lr_warmup}"
            )

        named_parameters = list(self.model.named_parameters())

        ### Filter out parameters that are not optimized (requires_grad == False)
        named_parameters = list(
            filter(lambda named_param: named_param[1].requires_grad, named_parameters)
        )

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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, verbose=True
            )
            if self.args.lr_warmup > 0:  # Wrap ReduceLROnPlateau to enable LR warmup
                scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=1,
                    total_epoch=self.args.lr_warmup,
                    after_scheduler=scheduler,
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

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", **scheduler_config},
        }
