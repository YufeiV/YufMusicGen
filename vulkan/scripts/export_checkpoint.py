"""Export a PyTorch YufMusicGen checkpoint to the ``.yuf`` container.

The Vulkan client reads ``.yuf`` files, not raw ``.pt`` checkpoints.  This
script converts a training checkpoint (which also contains optimizer state and
Python pickles) into a flat, self-contained binary format:

    magic "YUFM" | version u32 | header_len u64 | JSON header | f32 tensor data

The header carries the model config, the MidiTok REMI vocabulary and the
offset/size of every tensor, so the Rust side needs no Python or PyTorch at
runtime.

Usage:
    python scripts/export_checkpoint.py --checkpoint ../checkpoints/...pt \
        --output model.yuf
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from yufmusicgen.codec import MidiCodec
from yufmusicgen.config import ModelConfig, dataclass_from_dict


MAGIC = b"YUFM"
VERSION = 1


def _ordered_vocab(codec: MidiCodec) -> list[str]:
    """Return the token names indexed by raw MidiTok id."""

    vocab: list[str] = [""] * len(codec.tokenizer)
    for name, token_id in codec.tokenizer.vocab.items():
        vocab[int(token_id)] = name
    if any(not name for name in vocab):
        raise RuntimeError("MidiTok vocabulary has gaps; cannot export")
    return vocab


def export_checkpoint(checkpoint_path: str, output_path: str) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = dataclass_from_dict(ModelConfig, payload["model_config"])
    model_config.validate()
    state = payload["model"]

    codec_values = payload.get("codec_config") or {}
    if codec_values.get("tokenizer_json"):
        codec = MidiCodec.from_config_dict(codec_values)
    else:
        raise RuntimeError("checkpoint has no codec_config.tokenizer_json; cannot export")

    if codec.vocab_size + 260 != model_config.vocab_size:
        raise RuntimeError(
            "codec vocabulary does not match model config "
            f"({codec.vocab_size} + 260 != {model_config.vocab_size})"
        )

    # Drop the tied lm_head: the client reuses token_embedding.weight.
    lm_head_alias = None
    if model_config.tie_embeddings and "lm_head.weight" in state:
        lm_head_alias = "token_embedding.weight"
        state = {k: v for k, v in state.items() if k != "lm_head.weight"}

    # Canonical tensor order: embeddings first, then per-layer blocks, final norm.
    def sort_key(name: str) -> tuple[int, int]:
        if name == "token_embedding.weight":
            return (0, 0)
        if name == "final_norm.weight":
            return (2, 0)
        if name == "final_norm.bias":
            return (2, 1)
        parts = name.split(".")
        layer = int(parts[1]) if parts[0] == "blocks" else -1
        return (1, layer)

    names = sorted(state.keys(), key=sort_key)
    tensors: list[dict] = []
    blob = bytearray()
    for name in names:
        tensor = state[name].detach().to(dtype=torch.float32).contiguous().cpu()
        values = tensor.numpy().reshape(-1)
        tensors.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "offset": len(blob) // 4,
                "count": int(values.size),
            }
        )
        blob.extend(values.astype(np.float32, copy=False).tobytes())

    header = {
        "format": "yufmusicgen-checkpoint",
        "version": VERSION,
        "model_config": {
            "vocab_size": model_config.vocab_size,
            "d_model": model_config.d_model,
            "n_layers": model_config.n_layers,
            "n_heads": model_config.n_heads,
            "head_size": model_config.head_size,
            "rosa_size": model_config.rosa_size,
            "dropout": model_config.dropout,
            "tie_embeddings": model_config.tie_embeddings,
        },
        "midi": {
            "tokenization": codec.tokenization,
            "midi_offset": 260,
            "midi_vocab_size": codec.vocab_size,
            "vocab": _ordered_vocab(codec),
            "tokenizer_json": codec_values["tokenizer_json"],
        },
        "source": {
            "checkpoint": checkpoint_path.name,
            "step": int(payload.get("step", -1)),
            "phase": str(payload.get("phase", "")),
        },
        "lm_head_alias": lm_head_alias,
        "tensors": tensors,
    }
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", VERSION))
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(blob)

    print(
        f"exported {len(names)} tensors, "
        f"{len(blob) / 1e6:.1f} MB -> {output_path} "
        f"(step {header['source']['step']}, phase {header['source']['phase']!r})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--output", required=True, help="output .yuf path")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
