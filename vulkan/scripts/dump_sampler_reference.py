"""Dump fixed logits + CPU-sampled token sequence for Rust sampler cross-check.

Mirrors yufmusicgen.cli.generate.sample_token exactly:
    logits / temperature -> sort desc -> softmax -> top-p mask
    -> torch.multinomial (CPU, default generator seeded by torch.manual_seed).

Writes one text file per step (one f32 per line, token order) into a scratch
directory and prints the reference token ids on the last line.
"""

import argparse
import os
from pathlib import Path

import torch


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=520)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2348975934)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Fixed synthetic logits: deterministic per step, independent of the RNG
    # stream used by multinomial (so the two streams stay separable).
    rng = torch.Generator().manual_seed(args.seed ^ 0x5DEECE66D)
    tokens = []
    for step in range(args.steps):
        logits = torch.randn(args.vocab, generator=rng) * 1.5
        token = int(sample_token(logits, args.temperature, args.top_p))
        tokens.append(token)
        # Write exact f32 bit patterns (hex) so the Rust side parses the
        # identical float values without decimal-rounding error.
        with open(args.out / f"logits_{step:04d}.txt", "w", encoding="utf-8") as fh:
            for v in logits.view(torch.int32).tolist():
                fh.write(f"{v & 0xFFFFFFFF:08x}\n")

    print(" ".join(map(str, tokens)))


if __name__ == "__main__":
    main()
