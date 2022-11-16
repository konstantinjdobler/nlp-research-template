import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import tempfile
import shutil
import errno
from typing import Union
import datasets
from datasets.load import load_dataset
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer

from dlib.argparsing.dArgumentParser import dArg, parse_args_into_dataclasses


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


@dataclass
class Args:
    language: str = dArg(default="en", help="Language identifier from CC100", aliases="--lg")
    dev_size: int = dArg(default=50_000)
    test_size: int = dArg(default=10_000)
    lower_case: bool = dArg(default=False)
    out_dir: str = dArg(default="./data/cc100/")
    processes: int = dArg(default=4)
    chunking: bool = dArg(default=True)
    chunking_chars_per_token: float = dArg(default=5.0)
    reference_tokenizer: Optional[str] = dArg(default=None)
    chunking_max_len: int = dArg(default=256)
    aggressive_cleaning: bool = dArg(default=False)
    sep_token: str = dArg(default=" ")  # NOTE: default is whitespace as seperator between lines in CC100 dataset
    split: str = dArg(default="train", help="Select percentage of dataset like so: --split=train[:50%]")
    temp_cache_dir: Union[str, None] = dArg(default=None)
    conserve_disk_space: bool = dArg(default=False, aliases="--disk_space")


def main():
    (args,) = parse_args_into_dataclasses(dataclasses=(Args,))
    print(f"using sep token: ##{args.sep_token}##")
    if args.conserve_disk_space:
        # Disable caching because we write the end result to disk anyways. Intermediary caches just clutter the disk!
        datasets.fingerprint.disable_caching()

    print("Downloading dataset. This can take some time, so sit back and relax...")
    tmp_cache_dir = args.temp_cache_dir or (tempfile.mkdtemp(dir=args.out_dir) if args.conserve_disk_space else None)
    dataset = load_dataset(
        "cc100",
        lang=args.language,
        split=args.split,
        cache_dir=tmp_cache_dir,
    )

    print(dataset)
    print("Dataset len:", len(dataset))

    reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_tokenizer) if args.reference_tokenizer else None

    def estimated_tokenized_len(s: str) -> int:
        if reference_tokenizer:
            return len(reference_tokenizer(s).input_ids)
        else:
            return len(s) / args.chunking_chars_per_token

    def chunking_f(examples: Dict[str, list[str]]):
        chunked_lines = []
        current_line = ""
        current_len = 0

        for example in examples["text"]:
            if example == "\n":
                continue

            example = remove_control_characters(" ".join(example.split())) + "\n"
            if args.lower_case:
                example = example.lower()
            new_len = estimated_tokenized_len(example)

            if new_len > args.chunking_max_len:
                chunked_lines.append(example.rstrip() + "\n")
                continue
            if current_len + new_len > args.chunking_max_len or example == "\n":
                prepped_line = current_line.rstrip(" \n").replace("\n", args.sep_token) + "\n"
                if current_len > 20:
                    chunked_lines.append(prepped_line)
                else:
                    pass

                current_line = ""
                current_len = 0

            current_line = current_line + example
            current_len += new_len
        if current_line != "":
            chunked_lines.append(current_line.rstrip(" \n").replace("\n", args.sep_token) + "\n")

        return {"processed_text": chunked_lines}

    def filter_f(examples: Dict[str, list[str]]):
        return {
            "processed_text": [
                example.lower() if args.lower_case else example for example in examples["text"] if example != "\n"
            ]
        }

    f = chunking_f if args.chunking else filter_f
    dataset = dataset.map(f, num_proc=args.processes, batch_size=16_000, batched=True, remove_columns=["text", "id"])

    print("Processing finished!")
    print("Shuffling and splitting into sets...")

    dataset = dataset.shuffle(seed=42)
    total_len = len(dataset)
    dev_test_size = args.dev_size + args.test_size

    print("len", total_len)
    train_end_idx = total_len - dev_test_size
    train_paragraphs = dataset.select(range(train_end_idx))
    dev_paragraphs = dataset.select(range(train_end_idx, train_end_idx + args.dev_size))
    test_paragraphs = dataset.select(range(train_end_idx + args.dev_size, total_len))
    print(train_paragraphs[:4], "len:", len(train_paragraphs))

    if args.conserve_disk_space:
        print("Cleaning download cache")
        try:
            shutil.rmtree(tmp_cache_dir)
        except OSError as e:
            # Reraise unless ENOENT: No such file or directory
            # (ok if directory has already been deleted)
            if e.errno != errno.ENOENT:
                raise

    print("Writing train, dev and test data...")
    cc100_lg_dir = Path(args.out_dir) / (args.language + ("-uncased" if args.lower_case else ""))
    chunking_prefix = f"chunked{args.chunking_max_len}." if args.chunking else ""
    os.makedirs(str(cc100_lg_dir), exist_ok=True)
    with open(str(cc100_lg_dir / f"{chunking_prefix}{args.language}.train.txt"), "w+") as file:
        file.writelines((t["processed_text"] for t in tqdm(train_paragraphs, desc="Writing train data...")))

    with open(str(cc100_lg_dir / f"{chunking_prefix}{args.language}.devid.txt"), "w+") as file:
        file.writelines((t["processed_text"] for t in tqdm(dev_paragraphs, desc="Writing dev data...")))

    with open(str(cc100_lg_dir / f"{chunking_prefix}{args.language}.test.txt"), "w+") as file:
        file.writelines((t["processed_text"] for t in tqdm(test_paragraphs, desc="Writing test data...")))


if __name__ == "__main__":
    main()
