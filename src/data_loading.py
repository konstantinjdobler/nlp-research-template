from itertools import chain  # fmt: skip
from typing import TYPE_CHECKING  # fmt: skip

# need this to avoid cyclical import
if TYPE_CHECKING:
    from train import TrainingArgs, MiscArgs

import os
from pathlib import Path

import pytorch_lightning as pl
from datasets.load import load_dataset
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollatorForWholeWordMask
from transformers.models.auto.tokenization_auto import AutoTokenizer

from dlib.frameworks.pytorch import get_rank, main_process_first, set_torch_file_sharing_strategy_to_system


class LMDataModule(pl.LightningDataModule):
    """ """

    def __init__(
        self,
        training_args: "TrainingArgs",
        misc_args: "MiscArgs",
        mlm_probability=0.15,
        whole_word_masking=False,
        train_file_name="train.txt",
        dev_file_name="dev.txt",
    ):
        super().__init__()
        self.args = training_args
        self.misc_args = misc_args
        self.data_dir = (
            Path(training_args.data_dir) / training_args.language if training_args.language else Path(training_args.data_dir)
        )
        train_file, dev_file = (
            self.data_dir / train_file_name,
            self.data_dir / dev_file_name,
        )

        logger.debug(f"Train file path: {train_file} Dev file path: {dev_file}")

        self.train_file = str(train_file)
        self.dev_file = str(dev_file)
        self.max_seq_length = training_args.max_sequence_length
        self.mlm_probability = mlm_probability
        self.whole_word_masking = whole_word_masking
        self.tokenizer_path = self.args.tokenizer_path or self.args.model_name_or_path
        self.train_file_name = train_file_name
        self.dev_file_name = dev_file_name

    def setup(self, stage):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)

        if get_rank() == 0:
            logger.debug(f"Loaded tokenizer: {tokenizer}")

        extension = self.train_file.split(".")[-1]
        if extension in ("txt", "raw"):
            extension = "text"

        data_files = {"train": self.train_file, "dev": self.dev_file}
        train_dev_datasets = load_dataset(extension, data_files=data_files)
        if get_rank() == 0:
            logger.debug((train_dev_datasets, train_dev_datasets["train"][:2]))

        with main_process_first(description="Tokenizing datasets", active=self.args.gpus is not None):
            processed_datasets, data_collator = self.map_and_tokenize_datasets(
                tokenizer=tokenizer, tokenizer_path=self.tokenizer_path, datasets=train_dev_datasets
            )

        self.train_dataset = processed_datasets["train"]
        self.dev_dataset = processed_datasets["dev"]
        self.data_collator = data_collator

    def map_and_tokenize_datasets(self, tokenizer, tokenizer_path, datasets, mlm=True, truncation_buffer=0):
        tokenized_data_dir = self.data_dir / "tokenized" / tokenizer_path
        os.makedirs(tokenized_data_dir, exist_ok=True)

        actual_max_seq_len = self.max_seq_length + truncation_buffer
        final_tokenized_filenames = {
            "train": os.path.join(tokenized_data_dir, f"{actual_max_seq_len}.{self.args.language}.{self.train_file_name}"),
            "dev": os.path.join(tokenized_data_dir, f"{actual_max_seq_len}.{self.args.language}.{self.dev_file_name}"),
        }
        cache_exists = os.path.exists(final_tokenized_filenames["train"])
        logger.debug(
            f"Rank {get_rank()} | {tokenizer_path} | Cache exists: {cache_exists} | {'Loading cache...' if cache_exists else 'Starting dataset tokenization...'}"
        )

        # Always load from cache when not main process, dataset was already processed in main process
        load_from_cache = get_rank() != 0 or not self.args.overwrite_data_cache
        processed_datasets = datasets.map(
            make_tokenize_function(tokenizer, actual_max_seq_len),
            batched=True,
            num_proc=self.args.preprocessing_workers,  # Should use only one process to leverage tokenizers parallelism
            remove_columns=["text"],
            load_from_cache_file=load_from_cache,
            cache_file_names=final_tokenized_filenames,
            desc="Tokenizing dataset...",
        )

        processed_datasets["train"] = processed_datasets["train"].shuffle(seed=self.misc_args.seed)
        logger.success(
            f"Rank {get_rank()} | Loaded datasets: {processed_datasets} | {len(processed_datasets['train'][0]['input_ids'])}"
        )
        pad_to_multiple_of = 8 if self.args.precision in [16, "bf16"] else None
        DataCollatorClass = DataCollatorForWholeWordMask if self.whole_word_masking else DataCollatorForLanguageModeling
        data_collator = DataCollatorClass(
            tokenizer=tokenizer, mlm=mlm, mlm_probability=self.mlm_probability, pad_to_multiple_of=pad_to_multiple_of
        )

        return processed_datasets, data_collator

    def train_dataloader(self):
        common_args = dict(
            batch_size=self.args.batch_size_per_device,
            num_workers=self.args.workers,
            persistent_workers=True,
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system if self.misc_args.too_many_open_files_fix else None
            # shuffle=True, # DO NOT shuffle here, we have already pre-shuffled the data
        )
        return DataLoader(self.train_dataset, collate_fn=self.data_collator, **common_args)

    def val_dataloader(self):
        common_args = dict(
            batch_size=self.args.batch_size_per_device,
            num_workers=self.args.workers,
            persistent_workers=True,
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system if self.misc_args.too_many_open_files_fix else None,
        )
        return DataLoader(self.dev_dataset, collate_fn=self.data_collator, **common_args)


def make_tokenize_function(tokenizer, max_seq_length):
    """Needs to be outside of DataModule because of hashing error in dataset.map"""

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
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
