from typing import Literal

import lightning as L
import torch
from print_on_steroids import logger
from torch.optim import AdamW
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModelForMaskedLM
from transformers.optimization import get_scheduler
from warmup_scheduler import GradualWarmupScheduler


class BasicLM(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        lm_objective: Literal["mlm", "clm"],
        from_scratch: bool,
        learning_rate: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        lr_schedule: str,
        warmup_period: int,
        eval_interval: int,
        epsilon: float = 1e-8,
        save_hyperparameters: bool = True,
    ) -> None:
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])
        config = AutoConfig.from_pretrained(model_name_or_path, return_dict=True)

        if lm_objective == "mlm":
            self.model: PreTrainedModel = (
                AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
                if not from_scratch
                else AutoModelForMaskedLM.from_config(config=config)
            )
        elif lm_objective == "clm":
            self.model: PreTrainedModel = (
                AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
                if not from_scratch
                else AutoModelForCausalLM.from_config(config=config)
            )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_schedule = lr_schedule
        self.warmup_period = warmup_period
        self.eval_interval = eval_interval
        self.epsilon = epsilon

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.learning_rate}, weight decay: {self.weight_decay} and warmup steps: {self.warmup_period}"
            )

        named_parameters = list(self.model.named_parameters())

        ### Filter out parameters that are not optimized (requires_grad == False)
        optimized_named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]

        ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in optimized_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in optimized_named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_parameters,
            self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,  # You can also tune this
        )

        if self.lr_schedule == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
            if self.warmup_period > 0:  # Wrap ReduceLROnPlateau to enable LR warmup
                scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=1,
                    total_epoch=self.warmup_period,
                    after_scheduler=scheduler,
                )
            scheduler_config = {"frequency": self.eval_interval, "monitor": "train/loss"}
        else:
            scheduler_name = self.lr_schedule
            if scheduler_name == "constant" and self.warmup_period > 0:
                scheduler_name += "_with_warmup"
            scheduler = get_scheduler(
                scheduler_name,
                optimizer,
                num_warmup_steps=int(self.warmup_period),
                num_training_steps=self.trainer.max_steps,
            )
            scheduler_config = {"frequency": 1}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", **scheduler_config},
        }
