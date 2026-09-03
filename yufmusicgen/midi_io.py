"""MIDI file I/O helpers backed by symusic."""

from __future__ import annotations

from pathlib import Path

from symusic import Score


DEFAULT_TEMPO_QPM = 120.0


def read_midi(path: str | Path) -> Score:
    """Load a MIDI file as a ``symusic.Score``."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {path}")
    return Score(str(path))


def write_midi(path: str | Path, score: Score) -> None:
    """Dump a ``symusic.Score`` to a MIDI file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(str(path))


def midi_duration_seconds(score: Score) -> float:
    """Estimate the musical duration of a score in seconds.

    The end tick is converted to seconds using the first tempo change (or
    120 BPM when the file carries no tempo).
    """

    end_tick = int(score.end())
    ticks_per_quarter = int(score.ticks_per_quarter) or 480
    qpm = float(score.tempos[0].qpm) if score.tempos else DEFAULT_TEMPO_QPM
    qpm = qpm if qpm > 0 else DEFAULT_TEMPO_QPM
    return end_tick / ticks_per_quarter / (qpm / 60.0)


def truncate_midi(score: Score, max_seconds: float) -> Score:
    """Drop events after ``max_seconds`` and clamp notes crossing the cut.

    Operates in place on a copy so callers keep the original object.
    """

    import copy

    truncated = copy.deepcopy(score)
    ticks_per_quarter = int(truncated.ticks_per_quarter) or 480
    qpm = float(truncated.tempos[0].qpm) if truncated.tempos else DEFAULT_TEMPO_QPM
    qpm = qpm if qpm > 0 else DEFAULT_TEMPO_QPM
    cutoff = int(max_seconds * (qpm / 60.0) * ticks_per_quarter)
    if cutoff <= 0:
        raise ValueError("max_seconds is too small to truncate MIDI events")

    for track in truncated.tracks:
        track.notes = [note for note in track.notes if note.time < cutoff]
        for note in track.notes:
            if note.end > cutoff:
                note.end = cutoff
        track.controls = [control for control in track.controls if control.time < cutoff]
        track.pitch_bends = [
            bend for bend in track.pitch_bends if bend.time < cutoff
        ]
    truncated.tempos = [tempo for tempo in truncated.tempos if tempo.time < cutoff]
    truncated.time_signatures = [
        time_signature
        for time_signature in truncated.time_signatures
        if time_signature.time < cutoff
    ]
    return truncated
