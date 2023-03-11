from dataclasses import dataclass

from dargparser import dargparse

from src.model import BasicLM


@dataclass
class Args:
    pl_checkpoint: str
    out_file: str


def main(args: Args):
    lit_model = BasicLM.load_from_checkpoint(args.pl_checkpoint)
    hugging_face_model = lit_model.model
    hugging_face_model.save_pretrained(args.out_file)


if __name__ == "__main__":
    args = dargparse(Args)
    main(args)
