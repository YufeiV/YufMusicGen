from __future__ import annotations

import argparse
import os

from ..config import MidiCodecConfig
from ..preprocess import preprocess_dataset


def _parse_pitch_range(value: str) -> tuple[int, int]:
    low, high = (int(part) for part in value.split(","))
    return (low, high)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MIDI files to YufMusicGen tokens with MidiTok"
    )
    parser.add_argument("--input", required=True, help="MIDI directory, file, or JSONL manifest")
    parser.add_argument("--output", required=True, help="processed dataset directory")
    parser.add_argument(
        "--tokenization",
        default="REMI",
        choices=("REMI", "TSD", "MIDILike", "CPWord", "Structured", "Octuple"),
        help="MidiTok tokenization scheme (default: REMI)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=0,
        help="BPE vocabulary size; 0 keeps the plain MidiTok vocabulary",
    )
    parser.add_argument(
        "--pitch-range",
        default="21,109",
        type=_parse_pitch_range,
        help="piano pitch range as 'low,high' (default: 21,109)",
    )
    parser.add_argument("--num-velocities", type=int, default=32)
    parser.add_argument(
        "--use-chords",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="add chord tokens (default: off)",
    )
    parser.add_argument(
        "--use-tempos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="add tempo tokens (default: off)",
    )
    parser.add_argument(
        "--use-time-signatures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="add time signature tokens (default: off)",
    )
    parser.add_argument(
        "--disable-velocities",
        action="store_true",
        help="drop velocity tokens from the vocabulary",
    )
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="parallel encoding processes (default: min(8, cpu_count); 1 disables)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = MidiCodecConfig(
        tokenization=args.tokenization,
        vocab_size=args.vocab_size,
        pitch_range=args.pitch_range,
        num_velocities=args.num_velocities,
        use_velocities=not args.disable_velocities,
        use_chords=args.use_chords,
        use_tempos=args.use_tempos,
        use_time_signatures=args.use_time_signatures,
    )
    manifest = preprocess_dataset(
        args.input,
        args.output,
        codec_config=config,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
