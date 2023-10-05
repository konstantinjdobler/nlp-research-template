import os

from print_on_steroids import logger


def log_slurm_info():
    # The info doesn't always seem to be in the same environment variable, so we just check all of them
    gpu_identifiers = (
        os.environ.get("SLURM_GPUS")
        or os.environ.get("SLURM_GPUS_PER_TASK")
        or os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_STEP_GPUS")
        or len(os.environ.get("CUDA_VISIBLE_DEVICES", []))
    )
    logger.info(
        f"Detected SLURM environment. SLURM Job ID: {os.environ.get('SLURM_JOB_ID')}, "
        f"SLURM Host Name: {os.environ.get('SLURM_JOB_NODELIST')}, "
        f"SLURM Job Name: {os.environ.get('SLURM_JOB_NAME')}, "
        f"SLURM GPUS: {gpu_identifiers}"
    )
