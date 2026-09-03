from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _restart_windows_in_utf8_mode() -> None:
    """Make torch.compile subprocesses use UTF-8 instead of the local codepage."""

    if (
        os.name == "nt"
        and sys.flags.utf8_mode == 0
        and os.environ.get("YUFMUSICGEN_UTF8_REEXEC") != "1"
        and os.environ.get("YUFMUSICGEN_DISABLE_UTF8_REEXEC") != "1"
    ):
        os.environ["PYTHONUTF8"] = "1"
        os.environ["YUFMUSICGEN_UTF8_REEXEC"] = "1"
        entrypoint = Path(sys.argv[0])
        if entrypoint.suffix.lower() in {".py", ".pyw"}:
            result = subprocess.run([sys.executable, "-X", "utf8", *sys.argv])
            raise SystemExit(result.returncode)


_restart_windows_in_utf8_mode()

from ..config import ModelConfig, TrainConfig
from ..training import load_model_config_from_checkpoint, model_config_from_dataset, train


def add_training_args(parser: argparse.ArgumentParser, default_lr: float) -> None:
    parser.add_argument("--dataset", required=True, help="processed dataset directory")
    parser.add_argument("--manifest", default=None, help="JSONL manifest; defaults to dataset/manifest.jsonl")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable CUDA automatic mixed precision (BF16 by default)",
    )
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker processes; use 0 to disable multiprocessing",
    )
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=default_lr)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--d-model", type=int, default=640)
    parser.add_argument("--n-layers", type=int, default=14)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--head-size", type=int, default=40)
    parser.add_argument("--rosa-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--disable-cuda-kernel",
        action="store_true",
        help="force the PyTorch RWKV-7 recurrence instead of the fused CUDA operator",
    )
    parser.add_argument(
        "--disable-rosa-scan",
        action="store_true",
        help="force the reference ROSA recurrence instead of CUDA scan",
    )


def run_training(args: argparse.Namespace, phase: str, supervised: bool) -> None:
    dataset = Path(args.dataset)
    manifest = args.manifest or str(dataset / "manifest.jsonl")
    if args.init_checkpoint:
        model_config = load_model_config_from_checkpoint(args.init_checkpoint)
    else:
        model_config = model_config_from_dataset(
            dataset,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            head_size=args.head_size,
            rosa_size=args.rosa_size,
            dropout=args.dropout,
            use_cuda_kernel=not args.disable_cuda_kernel,
            use_rosa_scan=not args.disable_rosa_scan,
        )
    train_config = TrainConfig(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        grad_clip=args.grad_clip,
        seed=args.seed,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    train(
        manifest=manifest,
        dataset_dir=dataset,
        output_dir=args.output,
        model_config=model_config,
        train_config=train_config,
        phase=phase,
        device_name=args.device,
        init_checkpoint=args.init_checkpoint,
        supervised=supervised,
    )
