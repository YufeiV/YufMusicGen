from __future__ import annotations

import argparse

from .train_args import add_training_args, run_training


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="YufMusicGen autoregressive pretraining")
    add_training_args(parser, default_lr=3e-4)
    run_training(parser.parse_args(argv), phase="pretrain", supervised=False)


if __name__ == "__main__":
    main()
