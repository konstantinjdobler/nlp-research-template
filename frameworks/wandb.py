import os
import re
import subprocess
import threading
from pathlib import Path

import pytorch_lightning as pl
import wandb
from loguru import logger
from pytorch_lightning import Callback
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

from ..frameworks.pytorch import get_rank

WANDB_PROJECT = "<your project>"
WANDB_ENTITY = "<your entity>"

if WANDB_ENTITY == "<your entity>" or WANDB_PROJECT == "<your project>":
    logger.warning(
        "dLib error: You need to specify WANDB_ENTITY and WANDB_PROJECT in dlib/frameworks/wandb.py when using the wandb module."
    )


class MyWandbLogger(WandbLogger):
    def _scan_and_log_checkpoints(self, checkpoint_callback) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = {
            checkpoint_callback.last_model_path: checkpoint_callback.current_score,
            checkpoint_callback.best_model_path: checkpoint_callback.best_model_score,
            **checkpoint_callback.best_k_models,
        }
        checkpoints = sorted(
            (Path(p).stat().st_mtime, p, s)
            for p, s in checkpoints.items()
            if Path(p).is_file()
        )
        checkpoints = [
            c
            for c in checkpoints
            if c[1] not in self._logged_model_time.keys()
            or self._logged_model_time[c[1]] < c[0]
        ]

        # log iteratively all new checkpoints
        for t, p, s in checkpoints:
            metadata = {
                "score": s,
                "original_filename": Path(p).name,
                "ModelCheckpoint": {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                        "_every_n_val_epochs",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            artifact = wandb.Artifact(
                name=f"model-{self.experiment.id}", type="model", metadata=metadata
            )
            artifact.add_file(p, name="model.ckpt")
            aliases = (
                ["latest", "best"]
                if p == checkpoint_callback.best_model_path
                else ["latest"]
            )
            self.experiment.log_artifact(artifact, aliases=aliases)

            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t


class WandbCleanupDiskAndCloudSpaceCallback(Callback):
    def __init__(
        self, cleanup_local=True, cleanup_online=True, size_limit=0, backoff=10
    ) -> None:
        super().__init__()
        self.cleanup_local = cleanup_local
        self.cleanup_online = cleanup_online
        self.size_limit = size_limit
        self.backoff = backoff
        self.counter = 0

    @rank_zero_only
    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.counter < self.backoff:
            self.counter += 1
            return
        else:
            self.counter = 0
        # run = wandb.run  # type: ignore
        run = trainer.logger.experiment  # type: ignore

        # Delete outdated online artifacts
        if self.cleanup_online:  # BUG: this doesn't work....
            if getattr(run, "logged_artifacts", None) is not None:
                for artifact in run.logged_artifacts():
                    aliases = [x["alias"] for x in artifact._attrs["aliases"]]
                    if "best" not in aliases and "keep" not in aliases:
                        logger.info(
                            f"Deleting outdated artifact with aliases {aliases}"
                        )
                        artifact.delete()
            else:
                logger.error("wandb run has no logged artifacts")

        # Free up local wandb cache (This is often A LOT of memory)
        if self.cleanup_local:
            logger.info("Starting wandb artifact cache cleanup timeout")
            cache_cleanup_callback = lambda: subprocess.run(
                ["wandb", "artifact", "cache", "cleanup", f"{self.size_limit}GB"]
            )
            timer = threading.Timer(
                120.0, cache_cleanup_callback
            )  # Delay cleanupcall to avoid cleaning a temp file from the ModelCheckpoint hook that is needed to upload current checkpoint
            timer.start()


def check_for_wandb_checkpoint_and_download_if_necessary(
    checkpoint_path: str, wandb_run_instance
) -> str:
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
        model_tag = (
            checkpoint_path.split(":")[2]
            if len(checkpoint_path.split(":")) == 3
            else "latest"
        )

        """
        Only the main process should download the artifact in DDP. We add this environment variable as a guard. 
        This works only if this function is called first on the main process.
        """
        if os.environ.get(f"DLIB_ARTIFACT_PATH_{wandb_model_id}_{model_tag}"):
            checkpoint_path = os.environ[
                f"DLIB_ARTIFACT_PATH_{wandb_model_id}_{model_tag}"
            ]
        else:
            artifact = wandb_run_instance.use_artifact(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}/model-{wandb_model_id}:{model_tag}"
            )
            checkpoint_path = artifact.download() + "/model.ckpt"
            logger.info(f"Path of downloaded W&B artifact: {checkpoint_path}")
            os.environ[
                f"DLIB_ARTIFACT_PATH_{wandb_model_id}_{model_tag}"
            ] = checkpoint_path
    return checkpoint_path


def check_checkpoint_path_for_wandb(checkpoint_path: str):
    wandb_model_id_regex = r"wandb:.*"
    if re.search(wandb_model_id_regex, checkpoint_path):
        wandb_model_id = checkpoint_path.split(":")[1]
        return wandb_model_id
    return None
