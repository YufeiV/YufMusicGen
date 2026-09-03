"""Dump post-conditioning logits (reference math path) for comparison."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch

from yufmusicgen.codec import MidiCodec
from yufmusicgen.config import ModelConfig, dataclass_from_dict
from yufmusicgen.generation import build_condition
from yufmusicgen.model import YufMusicGen
from yufmusicgen.tokenizer import MusicTokenizer, TokenSpec
from yufmusicgen.training import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2348975934)
    args = parser.parse_args()

    set_seed(args.seed)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model_config = dataclass_from_dict(ModelConfig, payload["model_config"])
    model = YufMusicGen(model_config).to("cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    codec = MidiCodec.from_config_dict(payload["codec_config"])
    tokenizer = MusicTokenizer(TokenSpec(codec.vocab_size, codec.midi_offset))
    condition_tokens, _ = build_condition(
        tokenizer, codec, text=args.prompt, instrument=None, prompt_midi_ids=[]
    )
    condition = torch.tensor([condition_tokens], dtype=torch.long, device="cpu")
    with torch.no_grad():
        logits, _state = model(condition)
        logits = logits[:, -1]
    values = logits.reshape(-1).tolist()
    with open(args.out, "w", encoding="utf-8") as fh:
        for v in values:
            fh.write(f"{struct.unpack('<I', struct.pack('<f', v))[0]:08x}\n")
    top = torch.topk(logits, 10)
    print(
        "top10:",
        top.indices.reshape(-1).tolist(),
        "logits:",
        [round(float(v), 4) for v in top.values.reshape(-1).tolist()],
    )


if __name__ == "__main__":
    main()
