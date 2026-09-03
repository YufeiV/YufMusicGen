"""Shared training loop for pretraining and supervised post-training."""

from __future__ import annotations

import contextlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig, dataclass_from_dict, config_to_dict
from .data import ResumableShuffleSampler, load_json, make_dataset
from .model import YufMusicGen


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def model_config_from_dataset(
    dataset_dir: str | Path,
    d_model: int = 640,
    n_layers: int = 14,
    n_heads: int = 16,
    head_size: int = 40,
    rosa_size: int = 128,
    dropout: float = 0.0,
    use_cuda_kernel: bool = True,
    use_rosa_scan: bool = True,
) -> ModelConfig:
    tokenizer_info = load_json(Path(dataset_dir) / "tokenizer.json")
    midi_offset = int(tokenizer_info.get("midi_offset", 260))
    midi_vocab_size = int(tokenizer_info["midi_vocab_size"])
    vocab_size = midi_offset + midi_vocab_size
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        head_size=head_size,
        rosa_size=rosa_size,
        dropout=dropout,
        use_cuda_kernel=use_cuda_kernel,
        use_rosa_scan=use_rosa_scan,
    )


def _load_checkpoint(
    checkpoint_path: str | Path,
    model: YufMusicGen,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, Any]:
    """Restore model/optimizer state and return resume metadata.

    ``samples_seen`` is the number of samples already consumed by completed
    training steps.  Older checkpoints (which did not store it) fall back to
    ``step * batch_size * grad_accumulation`` from their saved TrainConfig.
    """

    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    step = int(payload.get("step", 0))
    fallback_samples = 0
    saved_train_config = payload.get("train_config")
    if saved_train_config:
        try:
            saved_cfg = dataclass_from_dict(TrainConfig, saved_train_config)
            fallback_samples = step * saved_cfg.batch_size * saved_cfg.grad_accumulation
        except Exception:
            fallback_samples = 0
    samples_seen = payload.get("samples_seen")
    if samples_seen is None:
        samples_seen = fallback_samples
    return {
        "step": step,
        "phase": payload.get("phase"),
        "samples_seen": samples_seen,
        "dataset_size": payload.get("dataset_size"),
    }


def save_checkpoint(
    path: Path,
    model: YufMusicGen,
    optimizer: torch.optim.Optimizer,
    step: int,
    phase: str,
    train_config: TrainConfig,
    codec_config: dict[str, Any] | None,
    samples_seen: int | None = None,
    dataset_size: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "phase": phase,
            "samples_seen": samples_seen,
            "dataset_size": dataset_size,
            "model_config": config_to_dict(model.config),
            "train_config": config_to_dict(train_config),
            "codec_config": codec_config,
        },
        path,
    )


def load_model_config_from_checkpoint(checkpoint_path: str | Path) -> ModelConfig:
    payload = torch.load(checkpoint_path, map_location="cpu")
    return dataclass_from_dict(ModelConfig, payload["model_config"])


def train(
    manifest: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    phase: str,
    device_name: str = "auto",
    init_checkpoint: str | Path | None = None,
    supervised: bool = False,
) -> Path:
    model_config.validate()
    set_seed(train_config.seed)
    device = resolve_device(device_name)
    if train_config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if train_config.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if train_config.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = YufMusicGen(model_config).to(device)
    optimizer_kwargs = {
        "lr": train_config.learning_rate,
        "weight_decay": train_config.weight_decay,
    }
    optimizer_name = "AdamW"
    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **optimizer_kwargs)
            optimizer_name = "AdamW(fused)"
        except (TypeError, RuntimeError):
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    amp_enabled = device.type == "cuda" and train_config.amp
    amp_dtype = None
    if amp_enabled:
        if train_config.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        elif train_config.amp_dtype in {"bfloat16", "float16"}:
            amp_dtype = torch.float16
            if train_config.amp_dtype == "bfloat16":
                print("BF16 is unavailable on this GPU; falling back to FP16")
        else:
            raise ValueError(f"unsupported amp_dtype: {train_config.amp_dtype}")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16
    )

    print(
        f"device={device} optimizer={optimizer_name} amp="
        f"{amp_dtype or 'off'} workers={train_config.num_workers} "
        f"batch={train_config.batch_size} seq={train_config.sequence_length}"
    )
    start_meta: dict[str, Any] = {
        "step": 0,
        "phase": phase,
        "samples_seen": 0,
        "dataset_size": None,
    }
    if init_checkpoint:
        start_meta = _load_checkpoint(init_checkpoint, model, optimizer, device)
        print(
            f"loaded checkpoint {init_checkpoint} at step {start_meta['step']}"
        )
    start_step = int(start_meta["step"])
    # Only continue the data stream when resuming the same training phase; a
    # pretrain checkpoint handed to post-training starts a fresh data stream.
    continue_stream = start_meta["phase"] == phase
    samples_seen = int(start_meta["samples_seen"]) if continue_stream else 0

    dataset = make_dataset(
        manifest,
        sequence_length=train_config.sequence_length,
        supervised=supervised,
        random_crop=not supervised,
        seed=train_config.seed,
    )
    saved_dataset_size = start_meta["dataset_size"]
    if saved_dataset_size is not None and saved_dataset_size != len(dataset):
        print(
            f"warning: checkpoint was trained on {saved_dataset_size} records "
            f"but the manifest now has {len(dataset)}; the resumed data stream "
            "will not match the original run"
        )
    loader_kwargs: dict[str, Any] = {
        "batch_size": train_config.batch_size,
        "sampler": ResumableShuffleSampler(
            dataset,
            batch_size=train_config.batch_size,
            seed=train_config.seed,
            samples_seen=samples_seen,
        ),
        "shuffle": False,
        # A short final batch causes an avoidable utilization dip and makes
        # throughput noisy, so keep every optimization step the same shape.
        "drop_last": True,
        "num_workers": train_config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if samples_seen:
        print(f"continuing data stream at sample {samples_seen}")
    if train_config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = train_config.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    if len(loader) == 0:
        raise ValueError(
            f"dataset has {len(dataset)} records, which is smaller than batch_size="
            f"{train_config.batch_size}"
        )
    batches = iter(loader)
    model.train()
    output_dir = Path(output_dir)
    codec_config = None
    codec_path = Path(dataset_dir) / "codec.json"
    tokenizer_json_path = Path(dataset_dir) / "miditok" / "tokenizer.json"
    if codec_path.exists() and tokenizer_json_path.exists():
        codec_metadata = json.loads(codec_path.read_text(encoding="utf-8"))
        codec_config = {
            "type": "miditok",
            "tokenization": codec_metadata.get("tokenization", "REMI"),
            "midi_offset": codec_metadata.get("midi_offset", 260),
            "tokenizer_json": tokenizer_json_path.read_text(encoding="utf-8"),
        }

    final_step = start_step + train_config.max_steps
    loss_ema: float | None = None
    samples_per_step = train_config.batch_size * train_config.grad_accumulation
    for step in range(start_step + 1, final_step + 1):
        local_step = step - start_step
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        running_loss = torch.zeros((), device=device)
        running_tokens = torch.zeros((), device=device)
        for _ in range(train_config.grad_accumulation):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(loader)
                batch = next(batches)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_enabled
                else contextlib.nullcontext()
            )
            with amp_context:
                logits, _ = model(input_ids)
                token_loss = F.cross_entropy(
                    logits.float().reshape(-1, model_config.vocab_size),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape_as(labels)
                loss = (token_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
            scaled_loss = loss / train_config.grad_accumulation
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            running_loss += loss.detach()
            running_tokens += loss_mask.sum().detach()

        # Schedule off the global step so a resumed run continues the warmup /
        # cosine decay where it left off instead of restarting it.
        if train_config.warmup_steps > 0 and step <= train_config.warmup_steps:
            scale = step / train_config.warmup_steps
        else:
            progress = (step - train_config.warmup_steps) / max(
                1, final_step - train_config.warmup_steps
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        for group in optimizer.param_groups:
            group["lr"] = train_config.learning_rate * max(0.05, scale)
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"non-finite gradient norm at {phase} step {local_step}: {grad_norm.item()}"
            )
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        samples_seen += samples_per_step

        if local_step % train_config.log_every == 0 or local_step == 1:
            lr = optimizer.param_groups[0]["lr"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - step_started, 1e-6)
            token_count = int(running_tokens.item())
            throughput = token_count / elapsed
            memory = (
                torch.cuda.max_memory_allocated(device) / (1024**3)
                if device.type == "cuda"
                else 0.0
            )
            loss_value = running_loss.item() / train_config.grad_accumulation
            loss_ema = loss_value if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_value
            print(
                f"{phase} step={local_step}/{train_config.max_steps} "
                f"loss={loss_value:.4f} ema={loss_ema:.4f} "
                f"grad_norm={grad_norm.item():.3e} "
                f"lr={lr:.3e} tokens/s={throughput:.0f} "
                f"max_mem={memory:.2f}GiB"
            )
        if local_step % train_config.save_every == 0 or local_step == train_config.max_steps:
            checkpoint = output_dir / f"{phase}-step-{step:08d}.pt"
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                step,
                phase,
                train_config,
                codec_config,
                samples_seen=samples_seen,
                dataset_size=len(dataset),
            )
            print(f"saved {checkpoint}")

    return output_dir / f"{phase}-step-{final_step:08d}.pt"
