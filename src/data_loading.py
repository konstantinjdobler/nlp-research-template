import errno
import glob
import os
import shutil
import tempfile
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import datasets
import lightning as L
from print_on_steroids import logger
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

from dlib.frameworks.pytorch import get_rank

if TYPE_CHECKING:
    from train import TrainingArgs


class LMDataModule(L.LightningDataModule):
    def __init__(
        self,
        training_args: "TrainingArgs",
        tokenizer: PreTrainedTokenizerFast,
    ):
        super().__init__()
        self.args = training_args
        self.data_dir = training_args.data_dir
        train_file, val_file = (
            self.data_dir / self.args.train_file,
            self.data_dir / self.args.val_file,
        )

        logger.debug(f"Train file path: {train_file} val file path: {val_file}")

        self.train_file = str(train_file)
        self.val_file = str(val_file)
        self.tokenizer_path = self.args.tokenizer_path or self.args.hf_model_name
        self.local_rank = get_rank()

        self.tokenizer = tokenizer

    def prepare_data(self) -> None:
        cache_exists, cache_path = self._get_dataset_cache_path(self.tokenizer_path)
        if not cache_exists:
            logger.info(f"Could not find cached processed dataset: {cache_path}, creating it now...")
            processed_datasets = self.load_and_process_dataset(self.tokenizer, str(self.data_dir / "tokenized"))
            logger.info(f"Saving dataset to {cache_path}...")
            processed_datasets.save_to_disk(cache_path, num_proc=self.args.preprocessing_workers)
        else:
            logger.success(f"Found cached processed dataset: {cache_path}.")
        if self.args.data_preprocessing_only:
            exit(0)

    def setup(self, stage):
        cache_exists, cache_path = self._get_dataset_cache_path(self.tokenizer_path)
        assert (
            cache_exists
        ), f"Could not find cached processed dataset: {cache_path}, should have been created in prepare_data()"

        logger.info(f"Loading cached processed dataset from {cache_path}...", rank0_only=False)
        processed_datasets = datasets.load_from_disk(cache_path)

        pad_to_multiple_of = 8 if self.args.precision in ["16-mixed", "bf16-mixed"] else None
        if self.args.language_modeling_objective == "clm":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of
            )
        elif self.args.language_modeling_objective == "mlm":
            DataCollatorClass = DataCollatorForLanguageModeling
            data_collator = DataCollatorClass(
                tokenizer=self.tokenizer,
                mlm=True,
                pad_to_multiple_of=pad_to_multiple_of,
            )

        self.train_dataset = processed_datasets["train"]
        self.val_dataset = processed_datasets["val"]
        self.data_collator = data_collator

    def load_and_process_dataset(self, tokenizer, tokenized_data_dir):
        extension = self.train_file.split(".")[-1]
        if extension in ("txt", "raw"):
            extension = "text"

        data_files = {"train": self.train_file, "val": self.val_file}

        logger.info("Loading raw dataset...")
        tmp_load_dataset_cache_dir = tempfile.mkdtemp(dir=tokenized_data_dir) if self.args.conserve_disk_space else None
        train_val_datasets = datasets.load_dataset(
            extension,
            data_files=data_files,
            name=str(self.data_dir).replace("/", "_"),
            num_proc=self.args.preprocessing_workers,
            cache_dir=tmp_load_dataset_cache_dir,
        )

        if self.local_rank == 0:
            logger.debug((train_val_datasets, train_val_datasets["train"][:2]))

        if self.args.conserve_disk_space:
            datasets.fingerprint.disable_caching()

        processed_datasets = self.process_dataset_in_chunks(tokenizer=tokenizer, train_val_datasets=train_val_datasets)

        # processed_datasets["train"] = processed_datasets["train"].shuffle(seed=self.args.seed) # <-- this is bad, triggers super expensive .flatten_indices op when .save_to_disk
        logger.success(
            f"Rank {self.local_rank} | Finished processing datasets: {processed_datasets} | First sample len: {len(processed_datasets['train'][0]['input_ids'])}"
        )

        if self.args.conserve_disk_space:
            logger.info("Cleaning dataset loading cache...")
            try:
                shutil.rmtree(tmp_load_dataset_cache_dir)
            except OSError as e:
                # Reraise unless ENOENT: No such file or directory
                # (ok if directory has already been deleted)
                if e.errno != errno.ENOENT:
                    raise

            datasets.fingerprint.enable_caching()

        return processed_datasets

    def process_dataset_in_chunks(self, tokenizer, train_val_datasets):
        """Expects input data to be one document per line. Tokenizes the documents and splits into chunks of max_sequence_legth."""
        tokenized_datasets = train_val_datasets.map(
            make_tokenize_function(tokenizer, max_seq_length=None, truncate=False),
            batched=True,
            num_proc=1,  # Should use only one process to leverage tokenizers parallelism
            remove_columns=["text"],
            load_from_cache_file=not self.args.overwrite_data_cache,
            desc="Running tokenizer on every text in dataset",
        )

        processed_datasets = tokenized_datasets.map(
            make_group_text_function(self.args.block_size),
            batched=True,
            batch_size=16_000,
            num_proc=self.args.preprocessing_workers,
            load_from_cache_file=not self.args.overwrite_data_cache,
            desc=f"Grouping texts in chunks of {self.args.block_size}",
        )

        return processed_datasets

    def train_dataloader(self):
        common_args = dict(
            batch_size=self.args.micro_batch_size,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            shuffle=True,
        )
        return DataLoader(self.train_dataset, collate_fn=self.data_collator, **common_args)

    def val_dataloader(self):
        common_args = dict(
            batch_size=self.args.eval_micro_batch_size,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
        )
        return DataLoader(self.val_dataset, collate_fn=self.data_collator, **common_args)

    def _get_dataset_cache_path(self, tokenizer_name: str):
        tokenizer_name = Path(self.tokenizer_path).as_posix().replace("/", "_")
        tokenize_fn = make_tokenize_function(self.tokenizer, self.args.block_size)
        tokenize_fn_hash = datasets.fingerprint.Hasher.hash(tokenize_fn)
        tokenized_data_dir = str(self.data_dir / "tokenized")
        cache_path = os.path.join(
            tokenized_data_dir,
            f"{self.args.train_file}.{self.args.val_file}.seq_len_{self.args.block_size}.tokenizer_{tokenizer_name}.tokenize_fn_hash_{tokenize_fn_hash}.arrow",
        )
        maybe_cache_path = os.path.join(
            tokenized_data_dir,
            f"{self.args.train_file}.{self.args.val_file}.seq_len_{self.args.block_size}.tokenizer_{tokenizer_name}.tokenize_fn_hash_.*.arrow",
        )
        maybe_cache_path_match_list = glob.glob(maybe_cache_path)

        if os.path.exists(cache_path):
            return True, cache_path
        elif len(maybe_cache_path_match_list) > 0 and os.path.exists(maybe_cache_path_match_list[0]):
            logger.warning(
                f"Rank {self.local_rank} | Did not find cached processed dataset: {cache_path} but {maybe_cache_path_match_list[0]}.",
                "The tokenize function hash can change with small, functionally meaningless code changes in the tokenizers library.",
                "Proceeding with existing found cache.",
            )
            return True, maybe_cache_path_match_list[0]
        else:
            return False, cache_path


def make_tokenize_function(tokenizer, max_seq_length=None, truncate=True):
    """Needs to be outside of DataModule because of hashing error in dataset.map"""

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=truncate,
            max_length=max_seq_length,
            # We use return_special_tokens_mask=True because DataCollatorForLanguageModeling is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    return tokenize_function


def make_group_text_function(max_seq_length):
    """Needs to be outside of DataModule because of hashing error in dataset.map"""

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    return group_texts
