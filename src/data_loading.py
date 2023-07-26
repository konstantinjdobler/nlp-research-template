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
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollatorForWholeWordMask
from transformers.models.auto.tokenization_auto import AutoTokenizer

from dlib.frameworks.pytorch import (
    get_rank,
    main_process_first,
    set_torch_file_sharing_strategy_to_system,
)

if TYPE_CHECKING:
    from train import MiscArgs, TrainingArgs


class LMDataModule(L.LightningDataModule):
    def __init__(
        self,
        training_args: "TrainingArgs",
        misc_args: "MiscArgs",
        mlm_probability=0.15,
        whole_word_masking=False,
    ):
        super().__init__()
        self.args = training_args
        self.misc_args = misc_args
        self.data_dir = (
            Path(training_args.data_dir) / training_args.language
            if training_args.language
            else Path(training_args.data_dir)
        )
        train_file, dev_file = (
            self.data_dir / self.args.train_file,
            self.data_dir / self.args.dev_file,
        )

        logger.debug(f"Train file path: {train_file} Dev file path: {dev_file}")

        self.train_file = str(train_file)
        self.dev_file = str(dev_file)
        self.mlm_probability = mlm_probability
        self.whole_word_masking = whole_word_masking
        self.tokenizer_path = self.args.tokenizer_path or self.args.model_name_or_path

    def setup(self, stage):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)

        if get_rank() == 0:
            logger.debug(f"Loaded tokenizer: {tokenizer}")

        tokenizer_name = self.tokenizer_path.rstrip("/").replace("/", "_")
        tokenize_fn = make_tokenize_function(tokenizer, self.args.max_sequence_length)
        tokenize_fn_hash = datasets.fingerprint.Hasher.hash(tokenize_fn)

        tokenized_data_dir = str(self.data_dir / "tokenized")

        cache_path = os.path.join(
            tokenized_data_dir,
            f"{self.args.train_file}.{self.args.dev_file}.seq_len_{self.args.max_sequence_length}.tokenizer_{tokenizer_name}.tokenize_fn_hash_{tokenize_fn_hash}.arrow",
        )
        maybe_cache_path = os.path.join(
            tokenized_data_dir,
            f"{self.args.train_file}.{self.args.dev_file}.seq_len_{self.args.max_sequence_length}.tokenizer_{tokenizer_name}.tokenize_fn_hash_.*.arrow",
        )
        maybe_cache_path_match_list = glob.glob(maybe_cache_path)
        logger.info(f"Rank {get_rank()} | Cache path: {cache_path}")

        with main_process_first(description="Loading dataset", active=(self.args.num_devices > 1)):
            if os.path.exists(cache_path):
                logger.success(f"Rank {get_rank()} | Found cached processed dataset: {cache_path}")
                processed_datasets = datasets.load_from_disk(cache_path)
                logger.success(
                    f"Rank {get_rank()} | Loaded cached processed dataset: {processed_datasets}"
                )
            elif len(maybe_cache_path_match_list) > 0 and os.path.exists(
                maybe_cache_path_match_list[0]
            ):
                logger.warning(
                    f"Rank {get_rank()} | Did not find cached processed dataset: {cache_path} but {maybe_cache_path_match_list[0]}. The tokenize function hash can change with small, functionally meaningless code changes in the tokenizers library. Proceeding with existing found cache."
                )
                processed_datasets = datasets.load_from_disk(maybe_cache_path_match_list[0])
                logger.success(
                    f"Rank {get_rank()} | Loaded cached processed dataset: {processed_datasets}"
                )
            else:
                processed_datasets = self.load_and_process_dataset(tokenizer, tokenized_data_dir)
                logger.info(f"Saving dataset to {cache_path}...")
                processed_datasets.save_to_disk(
                    cache_path, num_proc=self.args.preprocessing_workers
                )
        pad_to_multiple_of = 8 if self.args.precision in ["16-mixed", "bf16-mixed"] else None
        if self.args.language_modeling_strategy == "clm":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of
            )
        elif self.args.language_modeling_strategy == "mlm":
            DataCollatorClass = (
                DataCollatorForWholeWordMask
                if self.whole_word_masking
                else DataCollatorForLanguageModeling
            )
            data_collator = DataCollatorClass(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=self.mlm_probability,
                pad_to_multiple_of=pad_to_multiple_of,
            )

        self.train_dataset = processed_datasets["train"]
        self.dev_dataset = processed_datasets["dev"]
        self.data_collator = data_collator

        if self.args.data_preprocessing_only:
            exit(0)

    def load_and_process_dataset(self, tokenizer, tokenized_data_dir):
        extension = self.train_file.split(".")[-1]
        if extension in ("txt", "raw"):
            extension = "text"

        data_files = {"train": self.train_file, "dev": self.dev_file}

        logger.info("Loading raw dataset...")
        tmp_load_dataset_cache_dir = (
            tempfile.mkdtemp(dir=tokenized_data_dir) if self.args.conserve_disk_space else None
        )
        train_dev_datasets = datasets.load_dataset(
            extension,
            data_files=data_files,
            name=str(self.data_dir).replace("/", "_"),
            num_proc=self.args.preprocessing_workers,
            cache_dir=tmp_load_dataset_cache_dir,
        )

        if get_rank() == 0:
            logger.debug((train_dev_datasets, train_dev_datasets["train"][:2]))

        if self.args.conserve_disk_space:
            datasets.fingerprint.disable_caching()

        if self.args.line_by_line:
            processed_datasets = self.process_dataset_line_by_line(
                tokenizer=tokenizer,
                tokenizer_path=self.tokenizer_path,
                train_dev_datasets=train_dev_datasets,
            )
        else:
            processed_datasets = self.process_dataset_in_chunks(
                tokenizer=tokenizer, train_dev_datasets=train_dev_datasets
            )

        # processed_datasets["train"] = processed_datasets["train"].shuffle(seed=self.misc_args.seed) # <-- this is bad, triggers super expensive .flatten_indices op when .save_to_disk
        logger.success(
            f"Rank {get_rank()} | Finished processing datasets: {processed_datasets} | First sample len: {len(processed_datasets['train'][0]['input_ids'])}"
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

    def process_dataset_in_chunks(self, tokenizer, train_dev_datasets):
        """Expects input data to be one document per line. Tokenizes the documents and splits into chunks of max_sequence_legth."""
        tokenized_datasets = train_dev_datasets.map(
            make_tokenize_function(tokenizer, max_seq_length=None, truncate=False),
            batched=True,
            num_proc=1,  # Should use only one process to leverage tokenizers parallelism
            remove_columns=["text"],
            load_from_cache_file=not self.args.overwrite_data_cache,
            desc="Running tokenizer on every text in dataset",
        )

        processed_datasets = tokenized_datasets.map(
            make_group_text_function(self.args.max_sequence_length),
            batched=True,
            batch_size=16_000,
            num_proc=self.args.preprocessing_workers,
            load_from_cache_file=not self.args.overwrite_data_cache,
            desc=f"Grouping texts in chunks of {self.args.max_sequence_length}",
        )

        return processed_datasets

    def process_dataset_line_by_line(self, tokenizer, tokenizer_path, train_dev_datasets):
        tokenized_data_dir = self.data_dir / "tokenized" / tokenizer_path
        os.makedirs(tokenized_data_dir, exist_ok=True)

        tokenize_fn = make_tokenize_function(tokenizer, self.args.max_sequence_length)
        tokenize_fn_hash = datasets.fingerprint.Hasher.hash(tokenize_fn)
        final_tokenized_filenames = {
            "train": os.path.join(
                tokenized_data_dir,
                f"seq_len_{self.args.max_sequence_length}.tokenize_fn_hash_{tokenize_fn_hash}.{self.args.train_file}",
            ),
            "dev": os.path.join(
                tokenized_data_dir,
                f"seq_len_{self.args.max_sequence_length}.tokenize_fn_hash_{tokenize_fn_hash}.{self.args.dev_file}",
            ),
        }
        cache_exists = os.path.exists(final_tokenized_filenames["train"]) and os.path.exists(
            final_tokenized_filenames["dev"]
        )
        logger.debug(
            f"Rank {get_rank()} | {tokenizer_path} | Cache exists: {cache_exists} | {'Loading cache...' if cache_exists else 'Starting dataset tokenization...'}"
        )

        # Always load from cache when not main process, dataset was already processed in main process
        load_from_cache = get_rank() != 0 or not self.args.overwrite_data_cache
        processed_datasets = train_dev_datasets.map(
            make_tokenize_function(tokenizer, self.args.max_sequence_length),
            batched=True,
            num_proc=1,  # Should use only one process to leverage tokenizers parallelism
            remove_columns=["text"],
            load_from_cache_file=load_from_cache,
            cache_file_names=final_tokenized_filenames,
            desc="Tokenizing dataset...",
        )

        return processed_datasets

    def train_dataloader(self):
        common_args = dict(
            batch_size=self.args.batch_size_per_device,
            num_workers=self.args.workers,
            persistent_workers=(True if self.args.workers > 0 else False),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system
            if self.misc_args.too_many_open_files_fix
            else None,
            shuffle=True,
        )
        return DataLoader(self.train_dataset, collate_fn=self.data_collator, **common_args)

    def val_dataloader(self):
        common_args = dict(
            batch_size=self.args.batch_size_per_device,
            num_workers=self.args.workers,
            persistent_workers=(True if self.args.workers > 0 else False),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system
            if self.misc_args.too_many_open_files_fix
            else None,
        )
        return DataLoader(self.dev_dataset, collate_fn=self.data_collator, **common_args)


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
