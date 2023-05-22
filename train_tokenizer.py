"""
This script allows you train and compare tokenizers using the HuggingFace transformers and tokenizers librarires.

Example usage:
python train_tokenizer.py --language de --train_file ./data/de/train.txt --vocab_size 32_000 --custom_tokenizers xlm-roberta-base bpe --off_the_shelf_tokenizers xlm-roberta-base

This would train a XLM-RoBERTa tokenizer and a BPE tokenizer with a vocab size of 32k on the provided data and compare it to the pretrained XLM-RoBERTa tokenizer.
"""


import os
from dataclasses import dataclass

from dargparser import dArg, dargparse
from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
from tqdm import tqdm
from transformers import AutoTokenizer, XLMRobertaTokenizerFast


@dataclass
class Args:
    train_file: str
    training_cutoff: int = dArg(
        default=50_000_000, help="Cutoff tokenizer training after that many samples."
    )
    language: str = dArg(
        default="de", help="Language identifier. Just used for naming.", aliases="--lg"
    )
    prefix: str = dArg(default="", help="Prefix to prepend to the tokenizer name.")
    vocab_size: int = dArg(
        default=50_048,
        help="Vocab size for the tokenizer. Consider setting to a multiple of 64 for best usage of hardware accelerators.",
        aliases="--vs",
    )
    lower_case: bool = dArg(default=False)
    overwrite_cache: bool = dArg(default=False)
    custom_tokenizers: list[str] = dArg(
        default=[],
        help="A list of tokenizers to train on the provided data. Use tokenizer spec form, i.e. <tokenizer_name>"
        "or <tokenizer_name>:overwrite to overwrite any existing trained tokenizer with the same name."
        "The tokenizer_name must be a valid HuggingFace tokenizer name or bpe to train a Byte-Level BPE tokenizer.",
        aliases="--ct",
    )
    off_the_shelf_tokenizers: list[str] = dArg(
        default=[],
        help="A list of off-the-shelf HuggingFace tokenizers to compare against.",
        aliases="--otst",
    )


def get_training_corpus(datasets, batch_size=1000, cutoff=None):
    ABSURDLY_LARGE_LEN = 1_000_000_000_000
    return (
        [t.replace("</s>", " ") for t in datasets["train"][i : i + batch_size]["text"]]
        for i in range(0, min(cutoff or ABSURDLY_LARGE_LEN, len(datasets["train"])), batch_size)
    )


def eval_tokenizer(datasets, tokenizer, encode_batch=False):
    tokens = 0
    num_samples = 0
    total_chars = 0
    for batch in tqdm(get_training_corpus(datasets, batch_size=10_000, cutoff=500_000)):
        # batch = batch["text"]
        num_samples += len(batch)
        total_chars += sum((len(line) for line in batch))
        if encode_batch:
            encoding = (enc.tokens for enc in tokenizer.encode_batch(batch))
        else:
            encoding = tokenizer(batch).input_ids
        for enc in encoding:
            tokens += len(enc)
    return tokens, num_samples, total_chars


def transformers_tokenizer(
    name: str,
    datasets,
    tokenizer_lg_tag: str,
    vocab_size: int,
    lower_case: bool,
    training_cutoff: int = 50_000_000,
    adapt=False,
    overwrite_cache=False,
):
    cache_path = f"./tokenizers/{tokenizer_lg_tag}/{name}-{int(vocab_size/1000)}k"

    print("\n", f"############# {name} {'ADAPTED' if adapt else ''} ##############")
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, add_prefix_space=True)
    if adapt:
        if os.path.exists(cache_path) and not overwrite_cache:
            tokenizer = AutoTokenizer.from_pretrained(cache_path)
        else:
            tokenizer = tokenizer.train_new_from_iterator(
                get_training_corpus(datasets, cutoff=training_cutoff),
                vocab_size,
                lowercase=lower_case,
            )
            tokenizer.save_pretrained(cache_path)
    total_tokens, num_samples, total_chars = eval_tokenizer(datasets, tokenizer)
    print_eval_results(name, total_tokens, num_samples, total_chars)

    print(tokenizer.tokenize(EXAMPLE_TEXT, add_special_tokens=True), "\n")


def bpe_huggingface_tokenizer(
    datasets,
    tokenizer_lg_tag: str,
    vocab_size: int,
    lower_case: bool,
    training_cutoff: int = 50_000_000,
    overwrite_cache=False,
):
    cache_path = f"./tokenizers/{tokenizer_lg_tag}/bpe-{int(vocab_size/1000)}k"
    print(overwrite_cache)
    if not os.path.exists(cache_path) or overwrite_cache:
        print("\n", tokenizer_lg_tag, vocab_size)
        print("\n", "############# BPE ##############")

        # TODO: make normalization configurable
        bpe_tokenizer = ByteLevelBPETokenizer(
            add_prefix_space=True, lowercase=lower_case
        )  # x unicode_normalizer="nfkc")
        # bpe_tokenizer.normalizer = normalizers.NFKC()

        print(
            getattr(bpe_tokenizer, "normalizer", None),
            bpe_tokenizer.pre_tokenizer,
            bpe_tokenizer.post_processor,
        )
        bpe_tokenizer.train_from_iterator(
            iterator=get_training_corpus(datasets, cutoff=training_cutoff),
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
            ],
        )
        bpe_tokenizer.post_processor = RobertaProcessing(
            ("</s>", bpe_tokenizer.token_to_id("</s>")),
            ("<s>", bpe_tokenizer.token_to_id("<s>")),
        )
        huggingface_bpe_tokenizer = XLMRobertaTokenizerFast(tokenizer_object=bpe_tokenizer)
        huggingface_bpe_tokenizer.save_pretrained(cache_path)

    bpe_tokenizer = AutoTokenizer.from_pretrained(cache_path)
    total_tokens, num_samples, total_chars = eval_tokenizer(datasets, bpe_tokenizer)
    print_eval_results("Byte-Level BPE", total_tokens, num_samples, total_chars)
    bpe_result = bpe_tokenizer.tokenize(EXAMPLE_TEXT, add_special_tokens=True)
    print(bpe_result, "\n")


def print_eval_results(name: str, total_tokens, num_samples, total_chars):
    print(
        f"Tokenizer: {name}\n",
        f"Total tokens in corpus: {total_tokens}\n",
        f"Number of samples in corpus: {num_samples}\n",
        f"Number of chars in corpus: {total_chars}\n",
        f"Tokens per sample: {total_tokens / num_samples}\n",
        f"Tokens per char: {total_tokens / total_chars}\n",
        f"Chars per token: { total_chars / total_tokens}\n",
    )


EXAMPLE_TEXT = "lorm ipsum."


def decompose_tokenizer_spec(s: str):
    split_s = s.split(":")
    tokenizer_name = split_s[0]
    overwrite_cache = False

    if len(split_s) == 2:
        overwrite_cache = split_s[1] == "overwrite"
    return tokenizer_name, overwrite_cache


def main(args: Args):
    if not os.path.exists(args.train_file):
        raise "Train file does not exist"

    data_files = {"train": args.train_file}
    datasets = load_dataset("text", data_files=data_files)

    tokenizer_lg_tag = f"{args.prefix}{args.language}"

    global EXAMPLE_TEXT
    EXAMPLE_TEXT = get_training_corpus(datasets, batch_size=1, cutoff=10).__next__()[0]

    for tokenizer_spec in args.off_the_shelf_tokenizers:
        tokenizer_name, _ = decompose_tokenizer_spec(tokenizer_spec)
        transformers_tokenizer(
            tokenizer_name,
            datasets,
            tokenizer_lg_tag,
            vocab_size=args.vocab_size,
            lower_case=args.lower_case,
        )

    for tokenizer_spec in args.custom_tokenizers:
        tokenizer_name, overwrite_cache = decompose_tokenizer_spec(tokenizer_spec)
        if tokenizer_name == "bpe":
            bpe_huggingface_tokenizer(
                datasets,
                tokenizer_lg_tag=tokenizer_lg_tag,
                vocab_size=args.vocab_size,
                lower_case=args.lower_case,
                overwrite_cache=overwrite_cache,
                training_cutoff=args.training_cutoff,
            )
        else:
            transformers_tokenizer(
                tokenizer_name,
                datasets,
                tokenizer_lg_tag,
                vocab_size=args.vocab_size,
                lower_case=args.lower_case,
                adapt=True,
                overwrite_cache=overwrite_cache,
                training_cutoff=args.training_cutoff,
            )


if __name__ == "__main__":
    args = dargparse(Args)
    print(args)
    main(args)
