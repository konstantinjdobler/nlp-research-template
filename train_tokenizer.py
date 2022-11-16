import os
from dataclasses import dataclass

from datasets.load import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
from tqdm import tqdm
from transformers import XLMRobertaTokenizerFast
from transformers.models.auto.tokenization_auto import AutoTokenizer

from dlib.argparsing.dArgumentParser import dArg, parse_args_into_dataclasses


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
    training_cutoff: int = 100_000_000,
    adapt=False,
    overwrite_cache=False,
):
    cache_path = f"./tokenizers/{tokenizer_lg_tag}/{name}-{int(vocab_size/1000)}k"

    print("\n", f"############# {name} {'ADAPTED' if adapt else ''} ##############")
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, add_prefix_space=True, do_lower_case=lower_case)
    if adapt:
        if os.path.exists(cache_path) and not overwrite_cache:
            tokenizer = AutoTokenizer.from_pretrained(cache_path)
        else:
            tokenizer = tokenizer.train_new_from_iterator(
                get_training_corpus(datasets, cutoff=training_cutoff), vocab_size, lowercase=lower_case
            )
            tokenizer.save_pretrained(cache_path)
    total_tokens, num_samples, total_chars = eval_tokenizer(datasets, tokenizer)
    print_eval_results(name, total_tokens, num_samples, total_chars)

    print(
        tokenizer.tokenize(TEXT, add_special_tokens=True),
        "\n",
        tokenizer.decode(tokenizer("LOLwasfür eins äq wiüiünicer boi").input_ids),
        "\n",
    )


def bpe_huggingface_tokenizer(
    datasets,
    tokenizer_lg_tag: str,
    vocab_size: int,
    lower_case: bool,
    training_cutoff: int = 100_000_000,
    overwrite_cache=False,
):
    cache_path = f"./tokenizers/{tokenizer_lg_tag}/trans-bpe-{int(vocab_size/1000)}k"
    print(overwrite_cache)
    if not os.path.exists(cache_path) or overwrite_cache:
        print("\n", tokenizer_lg_tag, vocab_size)
        print("\n", "############# BPE ##############")
        bpe_tokenizer = ByteLevelBPETokenizer(add_prefix_space=True, lowercase=lower_case)  # x unicode_normalizer="nfkc")
        # bpe_tokenizer.normalizer = normalizers.NFKC()

        print(getattr(bpe_tokenizer, "normalizer", None), bpe_tokenizer.pre_tokenizer, bpe_tokenizer.post_processor)
        bpe_tokenizer.train_from_iterator(
            iterator=get_training_corpus(datasets, cutoff=training_cutoff),
            vocab_size=vocab_size,
            min_frequency=2,
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
        trans_bpe_tokenizer = XLMRobertaTokenizerFast(tokenizer_object=bpe_tokenizer)
        trans_bpe_tokenizer.save_pretrained(cache_path)

    bpe_tokenizer = AutoTokenizer.from_pretrained(cache_path)
    total_tokens, num_samples, total_chars = eval_tokenizer(datasets, bpe_tokenizer)
    print_eval_results("Byte-Level BPE", total_tokens, num_samples, total_chars)
    bpe_result = bpe_tokenizer.tokenize(TEXT, add_special_tokens=True)
    print(bpe_result, "\n", bpe_tokenizer.decode(bpe_tokenizer("LOLwasfür eins äq wiüiünicer boi").input_ids))


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


# NEW_VOCAB_SIZE = 50_000
# LG = "de"
# TOKENIZER_LG = "cc100-de"
TEXT = "BREAKING: JPM ATEUA MWANASHERIA MKUU MPYA WA SERIKALI, MASAJU AHAMISHWA</s>Taarifa kuhusu kifo cha Radio wa Uganda</s>Hospitali ya Apollo yatoa “zawadi ya maisha” kwa kufanikisha upandikizaji wa...</s>Matokeo ya awali ya uchaguzi Tanzania.</s>Kwa habari za kitaifa na kimataifa, biashara, burudani na michezo. Breaking News zote utazipata hapa. Tafadhali endelea kutembelea tovuti hii."
# TEXT = "bürgerrecht hanseaten hanseatischer handel modalität wird in dieser auffassung bürgermeisterwahl hanseatische marine donaudampfschifffahrtskapitänunterhosenknopfloch als einstellung des sprechers zum. gesprochenen definiert, wobei mit einstellung die subjektive bewertung gemeint ist.  »linke woche der zukunft« \"Peter\" 'peter' „besser als im kino und auf cd“"
# CC100 = "cc100/"


@dataclass
class Args:
    language: str = dArg(default="de", help="Language identifier from CC100", aliases="--lg")
    # seed: int = dArg(default=42, help="Seed to use for preprocessing.")
    vocab_size: int = dArg(default=50_000)
    train_file: str = dArg(default="")
    lower_case: bool = dArg(default=False)
    # tokenizer_name: str = dArg(required=True)
    cc100: bool = dArg(default=True)
    overwrite_cache: bool = dArg(default=False)
    custom_tokenizers: list[str] = dArg(default_factory=list, aliases="--ct")
    off_the_shelf_tokenizers: list[str] = dArg(default_factory=list, aliases="--otst")


def decompose_tokenizer_spec(s: str):
    split_s = s.split(":")
    tokenizer_name = split_s[0]
    overwrite_cache = False

    if len(split_s) == 2:
        overwrite_cache = split_s[1] == "overwrite"
    return tokenizer_name, overwrite_cache


def main():
    (args,) = parse_args_into_dataclasses(dataclasses=(Args,))
    print(args)

    CC100_prefix = "cc100/" if args.cc100 else ""
    # tokenizer = AutoTokenizer.from_pretrained("xlm-mlm-17-1280", use_fast=True)
    # print(len(tokenizer.get_vocab()))
    # train_file = f"./data/{CC100_prefix}{LG}/chunked256.{LG.split('-')[0]}.train.txt"
    if not os.path.exists(args.train_file):
        raise "Train file does not exist"
    data_files = {"train": args.train_file}
    datasets = load_dataset("text", data_files=data_files)

    tokenizer_lg_tag = f"{'cc100-' if args.cc100 else ''}{args.language}"
    
    global TEXT
    TEXT = get_training_corpus(datasets, batch_size=1, cutoff=10).__next__()[0]
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
            )


if __name__ == "__main__":
    main()
