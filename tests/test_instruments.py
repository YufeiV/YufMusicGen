import pytest

from yufmusicgen.instruments import instrument_name, resolve_program


def test_resolve_program_numbers():
    assert resolve_program(0) == 0
    assert resolve_program("40") == 40
    assert resolve_program("127") == 127
    assert resolve_program(-1) == -1


def test_resolve_program_names():
    assert resolve_program("piano") == 0
    assert resolve_program("violin") == 40
    assert resolve_program("Acoustic Guitar (nylon)") == 24
    assert resolve_program("drums") == -1
    assert resolve_program("DRUM") == -1


def test_resolve_program_unknown():
    with pytest.raises(ValueError, match="unknown instrument"):
        resolve_program("theremin")


def test_instrument_name():
    assert instrument_name(0) == "Acoustic Grand Piano"
    assert instrument_name(-1) == "Drums"
    with pytest.raises(ValueError):
        instrument_name(128)
