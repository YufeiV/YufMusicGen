from yufmusicgen.codec import MidiCodec
from yufmusicgen.config import MidiCodecConfig
from yufmusicgen.generation import build_condition, program_token_ids
from yufmusicgen.tokenizer import BOS, MIDI_OFFSET, MusicTokenizer, SEP, TokenSpec


def _fixture():
    codec = MidiCodec(MidiCodecConfig(tokenization="REMI"))
    tokenizer = MusicTokenizer(TokenSpec(codec.vocab_size, codec.midi_offset))
    return codec, tokenizer


def test_condition_with_text_only():
    codec, tokenizer = _fixture()
    tokens, prompt = build_condition(tokenizer, codec, text="piano")
    assert tokens[:2] == [BOS, 4 + ord("p")]
    assert SEP in tokens
    assert prompt == []


def test_condition_with_instrument():
    codec, tokenizer = _fixture()
    tokens, prefix = build_condition(tokenizer, codec, instrument="violin")
    program_id = codec.tokenizer["Program_40"]
    assert prefix == [program_id]
    assert tokens == [BOS, SEP, MIDI_OFFSET + program_id]


def test_condition_with_midi_prompt():
    codec, tokenizer = _fixture()
    prompt_ids = [284, 190, 44, 118]
    tokens, prompt = build_condition(
        tokenizer, codec, text="", prompt_midi_ids=prompt_ids
    )
    assert prompt == prompt_ids
    assert tokens == [BOS, SEP, *(MIDI_OFFSET + id_ for id_ in prompt_ids)]


def test_condition_prompt_truncation_keeps_tail():
    codec, tokenizer = _fixture()
    prompt_ids = list(range(10))
    tokens, prompt = build_condition(
        tokenizer, codec, prompt_midi_ids=prompt_ids, prompt_max_tokens=4
    )
    assert prompt == [6, 7, 8, 9]
    assert tokens[-4:] == [MIDI_OFFSET + 6, MIDI_OFFSET + 7, MIDI_OFFSET + 8, MIDI_OFFSET + 9]


def test_condition_with_instrument_and_prompt():
    codec, tokenizer = _fixture()
    prompt_ids = [284, 190, 44]
    tokens, prompt = build_condition(
        tokenizer, codec, instrument="piano", prompt_midi_ids=prompt_ids
    )
    program_id = codec.tokenizer["Program_0"]
    assert prompt == [*prompt_ids, program_id]
    assert tokens == [
        BOS,
        SEP,
        MIDI_OFFSET + 284,
        MIDI_OFFSET + 190,
        MIDI_OFFSET + 44,
        MIDI_OFFSET + program_id,
    ]


def test_program_token_ids_covers_gm_and_drums():
    codec, _ = _fixture()
    program_ids = program_token_ids(codec)
    assert program_ids[0] == codec.tokenizer["Program_0"]
    assert program_ids[40] == codec.tokenizer["Program_40"]
    assert program_ids[-1] == codec.tokenizer["Program_-1"]
    assert len(program_ids) == 129
