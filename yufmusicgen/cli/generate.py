from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

import torch

from ..codec import MidiCodec
from ..config import MidiCodecConfig, ModelConfig, dataclass_from_dict
from ..generation import build_condition, program_token_ids
from ..instruments import GM_PROGRAMS, instrument_name, resolve_program
from ..midi_io import midi_duration_seconds, read_midi, write_midi
from ..model import YufMusicGen
from ..tokenizer import EOS, MIDI_OFFSET, MusicTokenizer, TokenSpec
from ..training import resolve_device, set_seed


# Rough REMI tokens-per-second estimate used when --seconds is given instead
# of an explicit --steps budget.
TOKENS_PER_SECOND = 20.0
MIN_MIDI_TOKENS_BEFORE_EOS = 16


@dataclass
class GenerationArgs:
    checkpoint: str
    prompt: str = ""
    instrument: str | None = None
    instrument_only: bool = False
    prompt_midi: str | None = None
    prompt_max_tokens: int = 512
    output: str = "outputs/generated.mid"
    steps: int | None = None
    seconds: float | None = None
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int = 1337
    device: str = "auto"


def sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
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


def run_generation(
    params: GenerationArgs,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Load a checkpoint, sample MIDI tokens and write a ``.mid`` file.

    ``progress`` receives ``(fraction, status_text)`` updates; when omitted a
    ``tqdm`` bar is shown.
    """

    if progress:
        progress(0.02, "loading checkpoint")
    set_seed(params.seed)
    device = resolve_device(params.device)
    payload = torch.load(params.checkpoint, map_location=device)
    model_config = dataclass_from_dict(ModelConfig, payload["model_config"])
    model = YufMusicGen(model_config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    codec_values = payload.get("codec_config")
    if codec_values:
        codec = MidiCodec.from_config_dict(codec_values)
    else:
        codec = MidiCodec(MidiCodecConfig())
    tokenizer = MusicTokenizer(TokenSpec(codec.vocab_size, codec.midi_offset))
    if tokenizer.spec.vocab_size != model_config.vocab_size:
        raise ValueError("codec/tokenizer vocabulary does not match checkpoint model")

    if progress:
        progress(0.08, "encoding prompt MIDI")
    prompt_raw_ids: list[int] = []
    if params.prompt_midi:
        prompt_score = read_midi(params.prompt_midi)
        prompt_raw_ids = codec.encode(prompt_score)
    if params.instrument is not None:
        requested_program = resolve_program(params.instrument)
    else:
        requested_program = None
    if params.instrument_only and requested_program is None:
        raise ValueError("--instrument-only requires --instrument")

    blocked_programs: set[int] = set()
    if params.instrument_only:
        program_ids = program_token_ids(codec)
        blocked_programs = {
            raw_id
            for program, raw_id in program_ids.items()
            if program != requested_program
        }

    if progress:
        progress(0.12, "building condition")
    condition_tokens, prompt_raw_ids = build_condition(
        tokenizer,
        codec,
        text=params.prompt,
        instrument=params.instrument,
        prompt_midi_ids=prompt_raw_ids,
        prompt_max_tokens=params.prompt_max_tokens,
    )
    condition = torch.tensor([condition_tokens], dtype=torch.long, device=device)

    if params.steps:
        target_steps = params.steps
    elif params.seconds:
        target_steps = max(1, int(params.seconds * TOKENS_PER_SECOND))
    else:
        target_steps = 512
    target_steps = max(1, target_steps)

    with torch.no_grad():
        if progress:
            progress(0.15, "conditioning")
        logits, state = model(condition)
        logits = logits[:, -1]
        generated: list[int] = []
        iterator = tqdm(
            range(target_steps), desc="Generating MIDI", disable=progress is not None
        )
        for index in iterator:
            if progress:
                progress(0.15 + 0.8 * (index / target_steps), f"sampling {index + 1}/{target_steps}")
            allowed = torch.full_like(logits, float("-inf"))
            # Only MIDI tokens (and EOS after a short warm-up) are selectable.
            allowed[:, MIDI_OFFSET:] = logits[:, MIDI_OFFSET:]
            for raw_id in blocked_programs:
                allowed[:, MIDI_OFFSET + raw_id] = float("-inf")
            if index >= MIN_MIDI_TOKENS_BEFORE_EOS:
                allowed[:, EOS] = logits[:, EOS]
            token = sample_token(allowed, params.temperature, params.top_p)
            token_id = int(token.item())
            if token_id == EOS:
                break
            generated.append(token_id)
            logits, state = model.step(token, state)

    if progress:
        progress(0.96, "decoding MIDI")
    midi_ids = tokenizer.decode_midi(generated)
    combined_ids = [*prompt_raw_ids, *midi_ids]
    score = codec.decode(combined_ids)
    output = Path(params.output)
    write_midi(output, score)

    info: dict[str, Any] = {
        "output": output,
        "midi_tokens": len(midi_ids),
        "prompt_tokens": len(prompt_raw_ids),
        "tracks": len(score.tracks),
        "notes": sum(len(track.notes) for track in score.tracks),
        "duration_seconds": midi_duration_seconds(score),
        "instrument": (
            f"{instrument_name(requested_program)} (program {requested_program})"
            if requested_program is not None
            else None
        ),
    }
    if progress:
        progress(1.0, "done")
    return output, info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MIDI music from a YufMusicGen checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument(
        "--instrument",
        default=None,
        help="instrument name or GM program number (e.g. piano, violin, 40); "
        "puts the corresponding Program token at the generation point",
    )
    parser.add_argument(
        "--instrument-only",
        action="store_true",
        help="with --instrument, mask every other Program token so the output "
        "uses only the requested instrument (approximate under BPE)",
    )
    parser.add_argument(
        "--prompt-midi",
        default=None,
        help="condition on an existing MIDI file and continue from it",
    )
    parser.add_argument(
        "--prompt-max-tokens",
        type=int,
        default=512,
        help="max MidiTok tokens kept from --prompt-midi (keeps the tail)",
    )
    parser.add_argument(
        "--list-instruments",
        action="store_true",
        help="print the GM instrument table and exit",
    )
    parser.add_argument("--output", default="outputs/generated.mid")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="number of MIDI tokens to generate; overrides --seconds",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="approximate target length in seconds (maps to steps at ~20 tokens/s)",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.list_instruments:
        for program, name in enumerate(GM_PROGRAMS):
            print(f"{program:3d}  {name}")
        print(" -1  Drums")
        return
    params = GenerationArgs(
        **{field.name: getattr(args, field.name) for field in fields(GenerationArgs)}
    )
    output, info = run_generation(params)
    print(
        f"generated {info['midi_tokens']} midi tokens "
        f"(prompt {info['prompt_tokens']} + continuation) -> {output}"
    )


if __name__ == "__main__":
    main()
