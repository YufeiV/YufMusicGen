"""Configuration objects shared by preprocessing, training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MidiCodecConfig:
    """MidiTok-backed MIDI<->token codec configuration.

    ``tokenization`` selects the MidiTok scheme (REMI, TSD, MIDILike, CPWord,
    Structured or Octuple).  ``vocab_size`` > 0 trains a BPE vocabulary on the
    dataset; 0 keeps the plain (non-trained) MidiTok vocabulary.  The
    remaining fields are forwarded to :class:`miditok.TokenizerConfig`.
    """

    tokenization: str = "REMI"
    vocab_size: int = 0
    pitch_range: tuple[int, int] = (21, 109)
    num_velocities: int = 32
    use_velocities: bool = True
    use_chords: bool = False
    use_rests: bool = False
    use_tempos: bool = False
    use_time_signatures: bool = False
    use_sustain_pedals: bool = False
    use_pitch_bends: bool = False
    use_programs: bool = True
    one_token_stream_for_programs: bool = True

    def validate(self) -> None:
        if not self.tokenization or self.tokenization not in {
            "REMI",
            "TSD",
            "MIDILike",
            "CPWord",
            "Structured",
            "Octuple",
        }:
            raise ValueError(
                "tokenization must be one of REMI/TSD/MIDILike/CPWord/"
                "Structured/Octuple"
            )
        if self.vocab_size < 0:
            raise ValueError("vocab_size must be >= 0 (0 disables BPE)")
        if self.pitch_range[0] < 0 or self.pitch_range[1] > 127:
            raise ValueError("pitch_range must stay inside [0, 127]")
        if self.num_velocities < 1:
            raise ValueError("num_velocities must be positive")
        if self.one_token_stream_for_programs and not self.use_programs:
            raise ValueError("one_token_stream_for_programs requires use_programs")

    def tokenizer_kwargs(self) -> dict[str, Any]:
        """Forward the MidiTok-facing fields to ``miditok.TokenizerConfig``."""

        return {
            "pitch_range": tuple(self.pitch_range),
            "num_velocities": self.num_velocities,
            "use_velocities": self.use_velocities,
            "use_chords": self.use_chords,
            "use_rests": self.use_rests,
            "use_tempos": self.use_tempos,
            "use_time_signatures": self.use_time_signatures,
            "use_sustain_pedals": self.use_sustain_pedals,
            "use_pitch_bends": self.use_pitch_bends,
            "use_programs": self.use_programs,
            "one_token_stream_for_programs": self.one_token_stream_for_programs,
        }


@dataclass
class ModelConfig:
    vocab_size: int = 673
    # ~0.1B parameter default (≈100M): 640 * 14 layers + RWKV-7/ROSA blocks.
    d_model: int = 640
    n_layers: int = 14
    n_heads: int = 16
    head_size: int = 40
    rosa_size: int = 128
    dropout: float = 0.0
    tie_embeddings: bool = True
    use_cuda_kernel: bool = True
    use_rosa_scan: bool = True

    def validate(self) -> None:
        if self.d_model != self.n_heads * self.head_size:
            raise ValueError("d_model must equal n_heads * head_size")
        if self.d_model < 32 or self.n_layers < 1:
            raise ValueError("model is too small to be useful")
        if self.head_size > 1024:
            raise ValueError("head_size must fit within one CUDA thread block")


@dataclass
class TrainConfig:
    sequence_length: int = 2048
    batch_size: int = 2
    grad_accumulation: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 10000
    log_every: int = 10
    save_every: int = 1000
    grad_clip: float = 1.0
    seed: int = 1337
    amp: bool = True
    amp_dtype: str = "bfloat16"
    num_workers: int = 2
    prefetch_factor: int = 4


def dataclass_from_dict(cls: type[Any], values: dict[str, Any]) -> Any:
    """Create a config while ignoring unknown keys from older checkpoints."""

    fields = cls.__dataclass_fields__
    return cls(**{key: value for key, value in values.items() if key in fields})


def config_to_dict(config: Any) -> dict[str, Any]:
    return asdict(config)
