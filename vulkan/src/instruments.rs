//! General MIDI instrument names (index = program number).

pub const GM_PROGRAMS: [&str; 128] = [
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
];

pub const DRUM_PROGRAM: i32 = -1;

pub fn gm_name(program: i32) -> &'static str {
    if program == DRUM_PROGRAM {
        return "Drums";
    }
    if (0..128).contains(&program) {
        GM_PROGRAMS[program as usize]
    } else {
        "Unknown"
    }
}

/// Resolve an instrument name or program number, mirroring
/// `yufmusicgen.instruments.resolve_program`.
pub fn resolve_program(value: &str) -> Result<i32, String> {
    let text = value.trim();
    if text.is_empty() {
        return Err("empty instrument value".into());
    }
    if let Ok(program) = text.parse::<i32>() {
        if (-1..=127).contains(&program) {
            return Ok(program);
        }
        return Err(format!("program must be in [0, 127] or -1 for drums, got {program}"));
    }
    let lowered = text.to_lowercase();
    const DRUM_ALIASES: [&str; 5] = ["drums", "drum", "drumkit", "drum kit", "percussion"];
    if DRUM_ALIASES.contains(&lowered.as_str()) {
        return Ok(DRUM_PROGRAM);
    }
    let normalized = normalize(text);
    if let Some(index) = GM_PROGRAMS
        .iter()
        .position(|name| normalize(name) == normalized)
    {
        return Ok(index as i32);
    }
    if let Some(index) = GM_PROGRAMS
        .iter()
        .position(|name| normalize(name).contains(&normalized))
    {
        return Ok(index as i32);
    }
    Err(format!(
        "unknown instrument {text:?}; use a GM program number (0-127), 'drums', \
         or a name such as 'piano', 'violin', 'flute'..."
    ))
}

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

