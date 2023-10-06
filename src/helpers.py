import os
import re
from typing import Any

from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
from print_on_steroids import logger

from dlib.frameworks.pytorch import get_rank


def check_for_wandb_checkpoint_and_download_if_necessary(
    checkpoint_path: str,
    wandb_run_instance,
    wandb_entity=None,
    wandb_project=None,
    suffix="/model.ckpt",
) -> str:
    from train import WANDB_ENTITY, WANDB_PROJECT

    """
    Checks the provided checkpoint_path for the wandb regex r\"wandb:.*\".
    If matched, download the W&B artifact indicated by the id in the provided string and return its path.
    If not, just returns provided string.
    """
    wandb_model_id_regex = r"wandb:.*"
    if re.search(wandb_model_id_regex, checkpoint_path):
        if get_rank() == 0:
            logger.info("Downloading W&B checkpoint...")
        wandb_model_id = checkpoint_path.split(":")[1]
        model_tag = checkpoint_path.split(":")[2] if len(checkpoint_path.split(":")) == 3 else "latest"

        """
        Only the main process should download the artifact in DDP. We add this environment variable as a guard. 
        This works only if this function is called first on the main process.
        """
        if os.environ.get(f"MY_WANDB_ARTIFACT_PATH_{wandb_model_id}_{model_tag}"):
            checkpoint_path = os.environ[f"MY_WANDB_ARTIFACT_PATH_{wandb_model_id}_{model_tag}"]
        else:
            artifact = wandb_run_instance.use_artifact(
                f"{wandb_entity or WANDB_ENTITY}/{wandb_project or WANDB_PROJECT}/model-{wandb_model_id}:{model_tag}"
            )
            checkpoint_path = artifact.download() + suffix
            logger.info(f"Path of downloaded W&B artifact: {checkpoint_path}")
            os.environ[f"MY_WANDB_ARTIFACT_PATH_{wandb_model_id}_{model_tag}"] = checkpoint_path
    return checkpoint_path


def check_checkpoint_path_for_wandb(checkpoint_path: str):
    wandb_model_id_regex = r"wandb:.*"
    if re.search(wandb_model_id_regex, checkpoint_path):
        wandb_model_id = checkpoint_path.split(":")[1]
        return wandb_model_id
    return None


class ProgressMetricCallback(Callback):
    """
    # BUG: Tying the validation progress to the number of samples and tokens is not working yet.
    """

    def __init__(
        self,
        samples_processed=0.0,
        tokens_processed=0.0,
    ):
        super().__init__()
        self.samples_processed = samples_processed
        self.tokens_processed = tokens_processed

    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, pl_module: Any, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        B, T = batch["input_ids"].shape
        self.samples_processed += B * trainer.num_devices
        self.tokens_processed += B * T * trainer.num_devices

        self.log_dict(
            {
                "progress/samples": self.samples_processed,
                "progress/tokens": self.tokens_processed,
                "trainer/global_step": float(trainer.global_step),
            },
            rank_zero_only=True,
            on_step=True,
            on_epoch=False,
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: Any,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        self.log_dict(
            {
                "progress/samples": self.samples_processed,
                "progress/tokens": self.tokens_processed,
                "trainer/global_step": float(trainer.global_step),
            },
            rank_zero_only=True,
            on_step=True,
            on_epoch=False,
        )
