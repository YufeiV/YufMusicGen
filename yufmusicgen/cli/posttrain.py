from __future__ import annotations

import argparse

from .train_args import add_training_args, run_training


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="YufMusicGen supervised caption-to-music post-training")
    add_training_args(parser, default_lr=5e-5)
    run_training(parser.parse_args(argv), phase="posttrain", supervised=True)


if __name__ == "__main__":
    main()
