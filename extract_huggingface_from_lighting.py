import click

from src.model import BasicLM


@click.command()
######## Data and Training Setup ########
@click.option("--pl-checkpoint", type=click.Path(dir_okay=False))
@click.option("--out-file", "-o", type=click.Path(file_okay=False))
def main(pl_checkpoint, out_file):
    lit_model = BasicLM.load_from_checkpoint(pl_checkpoint)
    hugging_face_model = lit_model.model
    hugging_face_model.save_pretrained(out_file)


if __name__ == "__main__":
    main()
