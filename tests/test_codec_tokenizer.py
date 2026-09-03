import pytest

from symusic import Score, Tempo, TimeSignature, Track, Note

from yufmusicgen.codec import MidiCodec
from yufmusicgen.config import MidiCodecConfig
from yufmusicgen.midi_io import midi_duration_seconds, truncate_midi
from yufmusicgen.tokenizer import BOS, EOS, MIDI_OFFSET, MusicTokenizer, SEP, TokenSpec


def make_score() -> Score:
    score = Score(480)
    track = Track(program=0, is_drum=False, name="Piano")
    track.notes.append(Note(0, 480, 60, 100))
    track.notes.append(Note(480, 480, 64, 90))
    track.notes.append(Note(960, 960, 67, 80))
    score.tracks.append(track)
    score.tempos.append(Tempo(0, 120))
    score.time_signatures.append(TimeSignature(4, 4, 0))
    return score


def test_tokenizer_round_trip_midi_tokens():
    tokenizer = MusicTokenizer(TokenSpec(midi_vocab_size=413))
    raw = [4, 190, 44, 118, 133]
    sequence = tokenizer.build_sequence("piano", raw)
    assert sequence[0] == BOS
    assert sequence[1] == 4 + ord("p")
    assert SEP in sequence
    assert sequence[-1] == EOS
    assert tokenizer.decode_midi(sequence.tolist()) == raw
    assert all(MIDI_OFFSET <= token < MIDI_OFFSET + 413 for token in sequence[7:-1])


def test_tokenizer_stops_at_first_non_midi():
    tokenizer = MusicTokenizer(TokenSpec(midi_vocab_size=413))
    raw = [4, 190, 44, 118]
    sequence = tokenizer.build_sequence("piano", raw)
    decoded = tokenizer.decode_midi(sequence.tolist())
    assert decoded == raw
    # Once MIDI tokens start, any non-MIDI token ends the stream.
    assert tokenizer.decode_midi([MIDI_OFFSET + 4, BOS, MIDI_OFFSET + 190]) == [4]


def test_midi_codec_round_trip():
    codec = MidiCodec(MidiCodecConfig(tokenization="REMI"))
    score = make_score()
    ids = codec.encode(score)
    assert ids
    decoded = codec.decode(ids)
    assert len(decoded.tracks[0].notes) == len(score.tracks[0].notes)
    assert codec.vocab_size == len(codec.tokenizer)


def test_midi_codec_persistence(tmp_path):
    config = MidiCodecConfig(tokenization="TSD")
    codec = MidiCodec(config)
    score = make_score()
    ids = codec.encode(score)
    codec.save(tmp_path)

    reloaded = MidiCodec.from_dataset(tmp_path)
    assert reloaded.encode(score) == ids

    checkpoint_dict = reloaded.to_checkpoint_dict()
    from_checkpoint = MidiCodec.from_config_dict(checkpoint_dict)
    assert from_checkpoint.encode(score) == ids
    assert from_checkpoint.midi_offset == MIDI_OFFSET


def test_midi_codec_bpe_vocab(tmp_path):
    config = MidiCodecConfig(tokenization="REMI", vocab_size=600)
    codec = MidiCodec(config)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index in range(3):
        score = make_score()
        for extra in range(12):
            score.tracks[0].notes.append(
                Note(extra * 240, 480, 60 + (extra % 12), 70 + (extra % 20))
            )
        score.dump_midi(str(corpus / f"song{index}.mid"))
    base_vocab = codec.vocab_size
    codec.train_vocab(list(corpus.glob("*.mid")), config.vocab_size)
    assert codec.vocab_size > base_vocab
    score = make_score()
    ids = codec.encode(score)
    assert codec.decode(ids).tracks[0].notes


def test_midi_duration_and_truncation():
    score = make_score()
    assert midi_duration_seconds(score) == pytest.approx(2.0, rel=0.05)
    truncated = truncate_midi(score, 1.5)
    assert midi_duration_seconds(truncated) <= 1.5
    assert len(truncated.tracks[0].notes) == 3
    assert truncated.tracks[0].notes[-1].end == 1440
