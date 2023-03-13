"""
This script allows you to download and preprocess datasets for language modeling, specifically mc4 and cc100. You can customize it to your own needs.
"""

import errno
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

import datasets
from dargparser import dArg, dargparse
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@dataclass
class Args:
    language: str = dArg(help="Language identifier from mC4", aliases="--lg")
    dataset: Literal["mc4", "cc100"] = dArg(default="mc4")
    reference_tokenizer: str = dArg(
        default=None,
        help="Tokenizer used to split text into chunks based on number of tokens. Only necessary for chunking=True",
    )
    max_train_size: int = dArg(
        default=50_000_000,
        help="Maximum number of lines to write to disk. Use to limit very large datasets that you will not exhaust during training anyway. Use -1 to disable.",
    )
    dev_size: int = dArg(default=25_000)
    test_size: int = dArg(default=25_000)
    out_dir: str = dArg(default="./data/mc4/")
    processes: int = dArg(default=4)
    chunking: bool = dArg(
        default=False, help="Whether to split the text into chunks of a fixed number of tokens or return one document per line."
    )
    chunking_max_len: int = dArg(default=256)
    sep_token: str = dArg(default="</s><s>")
    split: str = dArg(default="train", help="Select percentage of dataset like so: --split=train[:50%]")
    conserve_disk_space: bool = dArg(default=False, aliases="--disk_space")
    pre_discard_factor: float = dArg(default=None)
    stream: bool = dArg(
        default=False,
        help="Couple with max_train_size to avoid having to load the entire dataset. Use streaming mode to load dataset and process dataset.",
    )


def main(args: Args):
    logger.info(args)
    if args.chunking:
        logger.info(f"Using sep token (in between ||): |{args.sep_token}|")
    if args.max_train_size == -1:
        args.max_train_size = None
    if args.conserve_disk_space:
        # Disable caching because we write the end result to disk anyways. Intermediary caches just clutter the disk!
        logger.info("Disabling caching to conserve disk space.")
        datasets.fingerprint.disable_caching()

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("Downloading dataset. This can take some time, so sit back and relax...")

    tmp_cache_dir = None
    if args.conserve_disk_space:
        tmp_cache_dir = os.path.join(args.out_dir, args.language, "tmp_download_cache")
        os.makedirs(tmp_cache_dir, exist_ok=True)

    if args.dataset == "mc4":
        dataset = load_dataset("mc4", args.language, split=args.split, cache_dir=tmp_cache_dir, streaming=args.stream)
    elif args.dataset == "cc100":
        if args.stream:
            logger.warning("Streaming mode for cc100 might lose some sample documents.")
            # Streaming mode is not trivial for cc100, since we need to group samples into documents.
            # To loose no samples, we need to set batch_size=len(dataset) but this si not possible for IteratableDataset.
            # We can accept loosing some samples by setting batch_size to a large number.
        dataset = load_dataset("cc100", lang=args.language, split=args.split, cache_dir=tmp_cache_dir, streaming=args.stream)

    if args.dataset == "cc100":

        def document_grouping_f(examples: Dict[str, list[str]]):
            documents = []
            current_doc = ""
            for example in examples["text"]:
                if example == "\n":
                    documents.append(current_doc)
                    current_doc = ""
                else:
                    current_doc += example
            return {"docs": documents}

        batch_size = 16_000 if args.stream else len(dataset)
        map_args = dict(batched=True, batch_size=batch_size, remove_columns=["text", "id"])
        if not args.stream:
            map_args["num_proc"] = args.processes  # stream does not support multiprocessing
        dataset = dataset.map(document_grouping_f, **map_args)
        dataset = dataset.rename_column("docs", "text")

    if not args.stream:
        dataset_len = len(dataset)
        logger.info("Dataset len:", dataset_len)

        if args.pre_discard_factor:
            assert args.max_train_size is not None
            if dataset_len > args.pre_discard_factor * args.max_train_size:
                dataset = dataset.shuffle(seed=42).select(range(int(args.pre_discard_factor * args.max_train_size)))

    logger.info(dataset)

    if args.chunking:
        assert args.reference_tokenizer is not None
        reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_tokenizer)

    def chunking_f(examples: Dict[str, list[str]]):
        chunked_lines = []
        discard = 0
        left_over_lines = []
        RE_COMBINE_WHITESPACE = re.compile(r"\s+")

        batch_tokens = reference_tokenizer(examples["text"], add_special_tokens=False, return_offsets_mapping=True)
        for token_offsets, example in zip(batch_tokens["offset_mapping"], examples["text"]):
            if example == "\n":
                continue

            for token_chunk in chunks(token_offsets, args.chunking_max_len):
                # print(len(token_chunk))
                start = token_chunk[0][0]
                end = token_chunk[-1][1]
                # print(start, end)
                chunk_str = example[start : end + 1]
                if len(token_chunk) < args.chunking_max_len:
                    left_over_lines.append((chunk_str, len(token_chunk)))
                    discard += 1
                else:
                    clean_str = RE_COMBINE_WHITESPACE.sub(" ", chunk_str).strip()
                    chunked_lines.append(clean_str + "\n")

        # Chunk leftovers
        current_str = ""
        current_len = 0
        for line in left_over_lines:
            text = line[0]
            text = RE_COMBINE_WHITESPACE.sub(" ", text).strip()
            length = line[1]
            if current_len + length >= args.chunking_max_len:
                chunked_lines.append(current_str + args.sep_token + text + "\n")
                current_str = ""
                current_len = 0
            else:
                current_str = current_str + args.sep_token + text
                current_len += length

        return {"processed_text": chunked_lines}

    def clean_f(examples: Dict[str, list[str]]):
        RE_COMBINE_WHITESPACE = re.compile(r"\s+")

        return {
            "processed_text": [
                RE_COMBINE_WHITESPACE.sub(" ", example).strip() + "\n" for example in examples["text"] if example != "\n"
            ]
        }

    f = chunking_f if args.chunking else clean_f

    if args.stream:
        # dataset.columns_names is not available for streaming datasets
        cols = ["text"]
        if args.dataset == "mc4":
            cols.extend(["timestamp", "url"])
        dataset = dataset.map(f, batch_size=16_000, batched=True, remove_columns=cols)
    else:
        logger.info("Starting mapping & chunking")
        dataset = dataset.map(
            f,
            num_proc=args.processes,
            batch_size=16_000,
            batched=True,
            remove_columns=dataset.column_names,
        )
        logger.success("Processing finished!")

    logger.info("Shuffling and splitting into sets...")
    if args.stream:
        # careful here, this would not truly shuffle the data, only samples within a buffer
        # Take care of true shuffling in the Dataloader
        # dataset = dataset.shuffle(seed=42, buffer_size=50_000)

        dev_paragraphs = dataset.take(args.dev_size)
        dataset = dataset.skip(args.dev_size)

        test_paragraphs = dataset.take(args.test_size)
        dataset = dataset.skip(args.test_size)

        train_paragraphs = dataset.take(args.max_train_size)

        logger.info(f"Example train split data: {list(train_paragraphs.take(4))}")
    else:
        total_len = len(dataset)
        logger.info(f"Dataset len after processing: {total_len}")

        dataset = dataset.shuffle(seed=42)

        dev_test_size = args.dev_size + args.test_size
        train_end_idx = total_len - dev_test_size
        train_paragraphs = dataset.select(range(train_end_idx))
        if args.max_train_size and len(train_paragraphs) > args.max_train_size:
            train_paragraphs = train_paragraphs.select(range(args.max_train_size))

        dev_paragraphs = dataset.select(range(train_end_idx, train_end_idx + args.dev_size))
        test_paragraphs = dataset.select(range(train_end_idx + args.dev_size, total_len))
        logger.info(f"Example train split data: {train_paragraphs[:4]}")
        logger.info(f"len: {len(train_paragraphs)}")

    if args.conserve_disk_space:
        logger.info("Cleaning download cache")
        try:
            shutil.rmtree(tmp_cache_dir)
        except OSError as e:
            # Reraise unless ENOENT: No such file or directory
            # (ok if directory has already been deleted)
            if e.errno != errno.ENOENT:
                raise

    logger.info("Writing data...")
    output_dir = Path(args.out_dir) / args.language
    chunking_prefix = f"chunked{args.chunking_max_len}." if args.chunking else ""
    os.makedirs(str(output_dir), exist_ok=True)
    with open(str(output_dir / f"{chunking_prefix}train.txt"), "w+") as file:
        file.writelines((t["processed_text"] for t in tqdm(train_paragraphs, desc="Writing train data...")))

    with open(str(output_dir / f"{chunking_prefix}dev.txt"), "w+") as file:
        file.writelines((t["processed_text"] for t in tqdm(dev_paragraphs, desc="Writing dev data...")))

    with open(str(output_dir / f"{chunking_prefix}test.txt"), "w+") as file:
        file.writelines((t["processed_text"] for t in tqdm(test_paragraphs, desc="Writing test data...")))


if __name__ == "__main__":
    args = dargparse(Args)
    main(args)
