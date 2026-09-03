"""General MIDI instrument name <-> program number helpers."""

from __future__ import annotations


# General MIDI Level 1 program names, index = MIDI program number.
GM_PROGRAMS = [
    "Acoustic Grand Piano",
    "Bright Acoustic Piano",
    "Electric Grand Piano",
    "Honky-tonk Piano",
    "Electric Piano 1",
    "Electric Piano 2",
    "Harpsichord",
    "Clavi",
    "Celesta",
    "Glockenspiel",
    "Music Box",
    "Vibraphone",
    "Marimba",
    "Xylophone",
    "Tubular Bells",
    "Dulcimer",
    "Drawbar Organ",
    "Percussive Organ",
    "Rock Organ",
    "Church Organ",
    "Reed Organ",
    "Accordion",
    "Harmonica",
    "Tango Accordion",
    "Acoustic Guitar (nylon)",
    "Acoustic Guitar (steel)",
    "Electric Guitar (jazz)",
    "Electric Guitar (clean)",
    "Electric Guitar (muted)",
    "Overdriven Guitar",
    "Distortion Guitar",
    "Guitar Harmonics",
    "Acoustic Bass",
    "Electric Bass (finger)",
    "Electric Bass (pick)",
    "Fretless Bass",
    "Slap Bass 1",
    "Slap Bass 2",
    "Synth Bass 1",
    "Synth Bass 2",
    "Violin",
    "Viola",
    "Cello",
    "Contrabass",
    "Tremolo Strings",
    "Pizzicato Strings",
    "Orchestral Harp",
    "Timpani",
    "String Ensemble 1",
    "String Ensemble 2",
    "Synth Strings 1",
    "Synth Strings 2",
    "Choir Aahs",
    "Voice Oohs",
    "Synth Voice",
    "Orchestra Hit",
    "Trumpet",
    "Trombone",
    "Tuba",
    "Muted Trumpet",
    "French Horn",
    "Brass Section",
    "Synth Brass 1",
    "Synth Brass 2",
    "Soprano Sax",
    "Alto Sax",
    "Tenor Sax",
    "Baritone Sax",
    "Oboe",
    "English Horn",
    "Bassoon",
    "Clarinet",
    "Piccolo",
    "Flute",
    "Recorder",
    "Pan Flute",
    "Blown Bottle",
    "Shakuhachi",
    "Whistle",
    "Ocarina",
    "Lead 1 (square)",
    "Lead 2 (sawtooth)",
    "Lead 3 (calliope)",
    "Lead 4 (chiff)",
    "Lead 5 (charang)",
    "Lead 6 (voice)",
    "Lead 7 (fifths)",
    "Lead 8 (bass + lead)",
    "Pad 1 (new age)",
    "Pad 2 (warm)",
    "Pad 3 (polysynth)",
    "Pad 4 (choir)",
    "Pad 5 (bowed)",
    "Pad 6 (metallic)",
    "Pad 7 (halo)",
    "Pad 8 (sweep)",
    "FX 1 (rain)",
    "FX 2 (soundtrack)",
    "FX 3 (crystal)",
    "FX 4 (atmosphere)",
    "FX 5 (brightness)",
    "FX 6 (goblins)",
    "FX 7 (echoes)",
    "FX 8 (sci-fi)",
    "Sitar",
    "Banjo",
    "Shamisen",
    "Koto",
    "Kalimba",
    "Bag Pipe",
    "Fiddle",
    "Shanai",
    "Tinkle Bell",
    "Agogo",
    "Steel Drums",
    "Woodblock",
    "Taiko Drum",
    "Melodic Tom",
    "Synth Drum",
    "Reverse Cymbal",
    "Guitar Fret Noise",
    "Breath Noise",
    "Seashore",
    "Bird Tweet",
    "Telephone Ring",
    "Helicopter",
    "Applause",
    "Gunshot",
]

DRUM_PROGRAM = -1

_DRUM_ALIASES = {"drums", "drum", "drumkit", "drum kit", "percussion"}


def instrument_name(program: int) -> str:
    """Return the GM name for a program number (``-1`` for drums)."""

    if program == DRUM_PROGRAM:
        return "Drums"
    if not 0 <= program <= 127:
        raise ValueError(f"program must be in [0, 127] or -1 for drums, got {program}")
    return GM_PROGRAMS[program]


def resolve_program(value: int | str) -> int:
    """Resolve an instrument name or program number to a MIDI program.

    Accepts integers/``"40"`` for program numbers, ``"drums"`` for the drum
    track (program -1), an exact GM instrument name, or a substring such as
    ``"piano"`` / ``"violin"``.
    """

    if isinstance(value, int):
        program = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("empty instrument value")
        if text.lstrip("-").isdigit():
            program = int(text)
        elif text.lower() in _DRUM_ALIASES:
            return DRUM_PROGRAM
        else:
            return _match_name(text)
    if not -1 <= program <= 127:
        raise ValueError(
            f"program must be in [0, 127] or -1 for drums, got {program}"
        )
    return program


def _match_name(text: str) -> int:
    normalized = _normalize(text)
    exact = [index for index, name in enumerate(GM_PROGRAMS) if _normalize(name) == normalized]
    if exact:
        return exact[0]
    matches = [index for index, name in enumerate(GM_PROGRAMS) if normalized in _normalize(name)]
    if matches:
        return matches[0]
    suggestions = ", ".join(GM_PROGRAMS[:12])
    raise ValueError(
        f"unknown instrument {text!r}; use a GM program number (0-127), "
        f"'drums', or a name such as {suggestions}..."
    )


def _normalize(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())
