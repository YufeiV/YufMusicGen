"""Helpers for building generation conditioning prefixes."""

from __future__ import annotations

from typing import Any

from .instruments import instrument_name, resolve_program
from .tokenizer import MusicTokenizer


def program_token_ids(codec: Any) -> dict[int, int]:
    """Map MIDI program number -> raw token id for every ``Program_*`` token."""

    program_ids: dict[int, int] = {}
    for name in codec.tokenizer.vocab:
        if name.startswith("Program_"):
            program = int(name.split("_", 1)[1])
            program_ids[program] = int(codec.tokenizer[name])
    return program_ids


def build_condition(
    tokenizer: MusicTokenizer,
    codec: Any,
    text: str = "",
    instrument: int | str | None = None,
    prompt_midi_ids: list[int] | None = None,
    prompt_max_tokens: int = 512,
) -> tuple[list[int], list[int]]:
    """Build the model condition for generation.

    Layout: ``BOS + UTF-8 text bytes + SEP`` followed by the raw MidiTok
    prefix the model should continue from: the (possibly truncated) MIDI
    prompt first, then an optional ``Program_<instrument>`` token right at the
    generation point.  The returned ``output_prefix_raw_ids`` mirrors that
    tail so the decoded output starts with the same context the model saw.
    """

    condition = tokenizer.condition_tokens(text)
    prompt_raw: list[int] = []
    if prompt_midi_ids:
        prompt_raw = list(prompt_midi_ids)
        if len(prompt_raw) > max(1, prompt_max_tokens):
            # Keep the tail: generation continues from the end of the prompt.
            prompt_raw = prompt_raw[-prompt_max_tokens:]
    output_prefix = list(prompt_raw)
    if instrument is not None:
        program = resolve_program(instrument)
        raw_program_id = int(codec.tokenizer[f"Program_{program}"])
        output_prefix.append(raw_program_id)
    if output_prefix:
        condition.extend(tokenizer.encode_midi(output_prefix))
    return condition, output_prefix
