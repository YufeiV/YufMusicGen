"""Token layout for text-conditioned MIDI generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PAD = 0
BOS = 1
EOS = 2
SEP = 3
TEXT_OFFSET = 4
TEXT_SIZE = 256
MIDI_OFFSET = TEXT_OFFSET + TEXT_SIZE


@dataclass(frozen=True)
class TokenSpec:
    """Vocabulary layout: special tokens + UTF-8 bytes + shifted MidiTok ids."""

    midi_vocab_size: int
    midi_offset: int = MIDI_OFFSET

    @property
    def vocab_size(self) -> int:
        return self.midi_offset + self.midi_vocab_size

    def midi_id(self, raw: int) -> int:
        if not 0 <= raw < self.midi_vocab_size:
            raise ValueError(f"midi token must be in [0, {self.midi_vocab_size})")
        return self.midi_offset + raw

    def raw_id(self, token_id: int) -> int:
        if not self.is_midi(token_id):
            raise ValueError(f"token {token_id} is not a midi token")
        return token_id - self.midi_offset

    def is_midi(self, token_id: int) -> bool:
        return self.midi_offset <= token_id < self.vocab_size


class MusicTokenizer:
    """Byte-level text prefix plus shifted MidiTok token ids."""

    def __init__(self, spec: TokenSpec | None = None) -> None:
        self.spec = spec or TokenSpec(midi_vocab_size=413)

    def encode_text(self, text: str) -> list[int]:
        return [TEXT_OFFSET + byte for byte in text.encode("utf-8")]

    def decode_text(self, tokens: list[int]) -> str:
        raw = bytes(
            token - TEXT_OFFSET
            for token in tokens
            if TEXT_OFFSET <= token < MIDI_OFFSET
        )
        return raw.decode("utf-8", errors="replace")

    def encode_midi(self, ids: list[int] | np.ndarray) -> list[int]:
        """Shift raw MidiTok ids into the model's vocabulary space."""

        return [self.spec.midi_id(int(raw)) for raw in np.asarray(ids).reshape(-1)]

    def decode_midi(self, tokens: list[int] | np.ndarray) -> list[int]:
        """Un-shift flat model tokens back to raw MidiTok ids.

        Leading condition tokens (BOS/text/SEP) are skipped; once the first
        MIDI token appears, any non-MIDI token (or out-of-range id) ends the
        stream.  Everything after the first invalid token is dropped so that
        sampled sequences stay decodable.
        """

        raw: list[int] = []
        started = False
        for token in np.asarray(tokens).reshape(-1):
            token = int(token)
            if not self.spec.is_midi(token):
                if started:
                    break
                continue
            started = True
            raw.append(self.spec.raw_id(token))
        return raw

    def build_sequence(self, text: str, midi_ids: list[int]) -> np.ndarray:
        sequence = [
            BOS,
            *self.encode_text(text),
            SEP,
            *self.encode_midi(midi_ids),
            EOS,
        ]
        return np.asarray(sequence, dtype=np.int64)

    def condition_tokens(self, text: str) -> list[int]:
        return [BOS, *self.encode_text(text), SEP]
