"""YufMusicGen: compact recurrent MIDI music generation toolkit."""

from .config import ModelConfig

__all__ = ["ModelConfig", "YufMusicGen"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "YufMusicGen":
        from .model import YufMusicGen

        return YufMusicGen
    raise AttributeError(name)
