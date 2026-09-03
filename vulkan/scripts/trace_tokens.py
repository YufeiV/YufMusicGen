"""Trace generated token ids using the reference (non-fused) math path.

Mirrors yufmusicgen.cli.generate exactly (conditioning, allowed mask, EOS
warm-up, sample_token with the same seed) but prints the token ids instead of
writing a MIDI file, so the sequence can be compared with the Vulkan client.

Set YUFMUSICGEN_DISABLE_CUDA=1 to force the reference recurrence.
"""

from __future__ import annotations

import argparse
import sys

import torch

from yufmusicgen.codec import MidiCodec
from yufmusicgen.config import ModelConfig, dataclass_from_dict
from yufmusicgen.generation import build_condition
from yufmusicgen.instruments import resolve_program
from yufmusicgen.model import YufMusicGen
from yufmusicgen.tokenizer import EOS, MIDI_OFFSET, MusicTokenizer, TokenSpec
from yufmusicgen.training import resolve_device, set_seed


def sample_token(logits, temperature, top_p):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    remove = cumulative - probabilities > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return sorted_indices.gather(
        -1, torch.multinomial(torch.softmax(sorted_logits, -1), 1)
    ).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2348975934)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dump-logits", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    model_config = dataclass_from_dict(ModelConfig, payload["model_config"])
    model = YufMusicGen(model_config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    codec = MidiCodec.from_config_dict(payload["codec_config"])
    tokenizer = MusicTokenizer(TokenSpec(codec.vocab_size, codec.midi_offset))

    condition_tokens, _ = build_condition(
        tokenizer, codec, text=args.prompt, instrument=None, prompt_midi_ids=[]
    )
    condition = torch.tensor([condition_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, state = model(condition)
        logits = logits[:, -1]
        if args.dump_logits:
            with open(args.dump_logits, "w", encoding="utf-8") as fh:
                for v in logits.reshape(-1).tolist():
                    fh.write(f"{v:.6f}\n")
        generated: list[int] = []
        for _ in range(args.steps):
            allowed = torch.full_like(logits, float("-inf"))
            allowed[:, MIDI_OFFSET:] = logits[:, MIDI_OFFSET:]
            if len(generated) >= 16:
                allowed[:, EOS] = logits[:, EOS]
            token = sample_token(allowed, args.temperature, args.top_p)
            token_id = int(token.item())
            if args.verbose:
                top = torch.topk(allowed, 5)
                print(
                    f"step {len(generated)}: sampled={token_id} "
                    f"top5={top.indices.reshape(-1).tolist()} "
                    f"vals={[round(float(v), 4) for v in top.values.reshape(-1).tolist()]}",
                    file=sys.stderr,
                )
            if token_id == EOS:
                break
            generated.append(token_id)
            logits, state = model.step(token, state)

    print(" ".join(map(str, generated)))


if __name__ == "__main__":
    main()
