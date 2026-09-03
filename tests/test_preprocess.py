import json

from symusic import Score, Tempo, TimeSignature, Track, Note

from yufmusicgen.config import MidiCodecConfig
from yufmusicgen.preprocess import _iter_midi_paths, preprocess_dataset


def _write_corpus(tmp_path, count=3):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index in range(count):
        score = Score(480)
        track = Track(program=index % 4, is_drum=False, name=f"t{index}")
        for j in range(16):
            track.notes.append(Note(j * 240, 240, 60 + (j % 12), 70 + (j % 20)))
        score.tracks.append(track)
        score.tempos.append(Tempo(0, 120))
        score.time_signatures.append(TimeSignature(4, 4, 0))
        midi_path = corpus / f"song{index:02d}.mid"
        score.dump_midi(str(midi_path))
        (corpus / f"song{index:02d}.txt").write_text(f"caption {index}", encoding="utf-8")
    return corpus


def test_iter_midi_paths_only_yields_midi(tmp_path):
    corpus = _write_corpus(tmp_path, count=2)
    (corpus / "notes.txt").write_text("not a midi", encoding="utf-8")
    paths = sorted(_iter_midi_paths(corpus))
    assert len(paths) == 2
    assert all(path.suffix.lower() == ".mid" for path in paths)


def test_preprocess_dataset_builds_manifest(tmp_path):
    corpus = _write_corpus(tmp_path, count=3)
    dataset = tmp_path / "dataset"
    manifest = preprocess_dataset(
        corpus,
        dataset,
        codec_config=MidiCodecConfig(tokenization="REMI"),
        workers=1,
    )
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert records[0]["text"] == "caption 0"
    assert (dataset / "tokens").is_dir()
    assert (dataset / "miditok" / "tokenizer.json").is_file()
    assert (dataset / "codec.json").is_file()
