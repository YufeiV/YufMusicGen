import pytest

gr = pytest.importorskip("gradio")
pytest.importorskip("matplotlib")

from symusic import Score, Tempo, TimeSignature, Track, Note

from yufmusicgen.webui import _instrument_choices, build_app, render_piano_roll


def test_build_app_and_instrument_choices():
    demo = build_app()
    assert isinstance(demo, gr.Blocks)
    choices = _instrument_choices()
    assert choices[0] == "auto"
    assert "0: Acoustic Grand Piano" in choices
    assert "-1: Drums" in choices


def test_render_piano_roll(tmp_path):
    score = Score(480)
    track = Track(program=0, is_drum=False, name="Piano")
    track.notes.append(Note(0, 480, 60, 90))
    track.notes.append(Note(480, 960, 64, 90))
    score.tracks.append(track)
    score.tempos.append(Tempo(0, 120))
    score.time_signatures.append(TimeSignature(4, 4, 0))

    image = render_piano_roll(score, tmp_path / "preview.png")
    assert image.is_file()
    assert image.stat().st_size > 1000
