"""MidiTok-backed MIDI <-> token codec.

The codec wraps a :class:`miditok.MusicTokenizer` so that the rest of the
project only deals with plain integer token ids.  ``MidiCodec.encode`` turns a
MIDI file (or ``symusic.Score``) into raw MidiTok ids; ``MidiCodec.decode``
turns ids back into a ``symusic.Score``.  An optional BPE vocabulary can be
trained on the dataset before preprocessing, and the resulting tokenizer is
persisted next to the dataset (``miditok/tokenizer.json``) and embedded inside
checkpoints so inference is self-contained.
"""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

from miditok import (
    CPWord,
    MIDILike,
    Octuple,
    REMI,
    Structured,
    TSD,
    TokSequence,
    TokenizerConfig,
)
from symusic import Score

from .config import MidiCodecConfig, config_to_dict
from .tokenizer import MIDI_OFFSET


_TOKENIZER_CLASSES = {
    "REMI": REMI,
    "TSD": TSD,
    "MIDILike": MIDILike,
    "CPWord": CPWord,
    "Structured": Structured,
    "Octuple": Octuple,
}


def _build_tokenizer(config: MidiCodecConfig):
    tokenizer_cls = _TOKENIZER_CLASSES[config.tokenization]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Attribute controls are not compatible.*",
        )
        return tokenizer_cls(TokenizerConfig(**config.tokenizer_kwargs()))


class MidiCodec:
    """Deterministic MIDI token codec built on top of MidiTok."""

    def __init__(
        self,
        config: MidiCodecConfig | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config or MidiCodecConfig()
        self.config.validate()
        self.tokenizer = tokenizer or _build_tokenizer(self.config)

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    @property
    def midi_offset(self) -> int:
        return MIDI_OFFSET

    @property
    def tokenization(self) -> str:
        return self.config.tokenization

    def encode(self, midi: Score | str | Path) -> list[int]:
        """Tokenize a MIDI into raw MidiTok ids (one single-token stream)."""

        sequence = self.tokenizer(midi)
        if isinstance(sequence, TokSequence):
            return list(sequence.ids)
        if isinstance(sequence, list) and len(sequence) == 1:
            return list(sequence[0].ids)
        raise ValueError(
            "the MidiTok configuration produced multiple token streams; "
            "use one_token_stream_for_programs=True for a single LM stream"
        )

    def decode(self, ids: list[int] | tuple[int, ...]) -> Score:
        """Detokenize raw MidiTok ids back into a ``symusic.Score``."""

        return self.tokenizer.decode(list(ids))

    def train_vocab(
        self, files_paths: list[str | Path], vocab_size: int | None = None
    ) -> None:
        """Train a BPE vocabulary on the dataset (no-op when too small)."""

        vocab_size = self.config.vocab_size if vocab_size is None else vocab_size
        if vocab_size <= 0 or vocab_size <= self.vocab_size:
            return
        self.tokenizer.train(vocab_size=vocab_size, files_paths=list(files_paths))

    def save(self, dataset_dir: str | Path) -> None:
        """Persist the codec next to a processed dataset."""

        dataset_dir = Path(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_dir = dataset_dir / "miditok"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(tokenizer_dir)
        (dataset_dir / "codec.json").write_text(
            json.dumps(config_to_dict(self.config), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (dataset_dir / "tokenizer.json").write_text(
            json.dumps(
                {
                    "type": "miditok",
                    "tokenization": self.config.tokenization,
                    "midi_offset": self.midi_offset,
                    "midi_vocab_size": self.vocab_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def tokenizer_json(self) -> str:
        """Serialize the MidiTok tokenizer to a JSON string."""

        with tempfile.TemporaryDirectory(prefix="yufmusicgen-miditok-") as temporary:
            tokenizer_dir = Path(temporary) / "tokenizer"
            self.tokenizer.save_pretrained(tokenizer_dir)
            return (tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8")

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Build the codec payload stored inside training checkpoints."""

        return {
            "type": "miditok",
            "tokenization": self.config.tokenization,
            "midi_offset": self.midi_offset,
            "tokenizer_json": self.tokenizer_json(),
        }

    @classmethod
    def from_dataset(cls, dataset_dir: str | Path) -> "MidiCodec":
        """Rebuild the codec from a processed dataset directory."""

        dataset_dir = Path(dataset_dir)
        codec_path = dataset_dir / "codec.json"
        if not codec_path.exists():
            raise FileNotFoundError(f"missing codec metadata: {codec_path}")
        config = MidiCodecConfig(**json.loads(codec_path.read_text(encoding="utf-8")))
        tokenizer_json_path = dataset_dir / "miditok" / "tokenizer.json"
        if not tokenizer_json_path.exists():
            raise FileNotFoundError(f"missing MidiTok tokenizer: {tokenizer_json_path}")
        return cls.from_tokenizer_json(
            config, tokenizer_json_path.read_text(encoding="utf-8")
        )

    @classmethod
    def from_tokenizer_json(
        cls, config: MidiCodecConfig, tokenizer_json: str
    ) -> "MidiCodec":
        """Rebuild a codec from its config plus a serialized tokenizer."""

        config.validate()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", encoding="utf-8", delete=False
        ) as handle:
            handle.write(tokenizer_json)
            path = Path(handle.name)
        try:
            tokenizer_cls = _TOKENIZER_CLASSES[config.tokenization]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Attribute controls are not compatible.*",
                )
                tokenizer = tokenizer_cls(params=path)
        finally:
            path.unlink(missing_ok=True)
        return cls(config, tokenizer)

    @classmethod
    def from_config_dict(cls, values: dict[str, Any]) -> "MidiCodec":
        """Rebuild a codec from a checkpoint's ``codec_config`` payload."""

        config = MidiCodecConfig(tokenization=values.get("tokenization", "REMI"))
        return cls.from_tokenizer_json(config, values["tokenizer_json"])
