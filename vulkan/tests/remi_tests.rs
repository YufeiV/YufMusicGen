//! CPU-only parity tests for the REMI codec and MIDI I/O against the MidiTok
//! reference behaviour captured from `yufmusicgen` + Python.

use std::path::Path;

use yufmusicgen_vulkan::midi::{Note, Score, Track, parse_midi, render_midi};
use yufmusicgen_vulkan::remi::RemiCodec;

fn codec() -> RemiCodec {
    let vocab: Vec<String> = serde_json::from_str(include_str!("../testdata/vocab.json"))
        .expect("vocab fixture");
    RemiCodec::new(vocab).expect("codec")
}

fn ids(names: &[&str]) -> Vec<u32> {
    let codec = codec();
    names
        .iter()
        .map(|name| codec.token_id(name).unwrap_or_else(|| panic!("missing token {name}")))
        .collect()
}

fn two_track_score() -> Score {
    Score {
        ticks_per_quarter: 480,
        tracks: vec![
            Track {
                program: 0,
                is_drum: false,
                name: "Piano".into(),
                notes: vec![
                    Note { start: 0, duration: 120, pitch: 60, velocity: 96 },
                    Note { start: 240, duration: 240, pitch: 64, velocity: 100 },
                    Note { start: 480, duration: 60, pitch: 67, velocity: 80 },
                ],
            },
            Track {
                program: 40,
                is_drum: false,
                name: "Violin".into(),
                notes: vec![Note { start: 120, duration: 240, pitch: 72, velocity: 88 }],
            },
        ],
    }
}

#[test]
fn encode_matches_miditok_reference() {
    let expected = [
        4, 190, 284, 44, 117, 127, 192, 324, 56, 115, 129, 194, 284, 48, 118, 129, 198, 284,
        51, 113, 126,
    ];
    let actual = codec().encode(&two_track_score()).expect("encode");
    assert_eq!(actual, expected);
}

#[test]
fn decode_matches_miditok_reference() {
    let tokens = [
        4, 190, 284, 44, 117, 127, 192, 324, 56, 115, 129, 194, 284, 48, 118, 129, 198, 284,
        51, 113, 126,
    ];
    let score = codec().decode(&tokens);
    assert_eq!(score.ticks_per_quarter, 8);
    assert_eq!(score.tracks.len(), 2);
    assert_eq!(score.tracks[0].program, 0);
    assert_eq!(
        score.tracks[0].notes,
        vec![
            Note { start: 0, duration: 2, pitch: 60, velocity: 95 },
            Note { start: 4, duration: 4, pitch: 64, velocity: 99 },
            Note { start: 8, duration: 1, pitch: 67, velocity: 79 },
        ]
    );
    assert_eq!(score.tracks[1].program, 40);
    assert_eq!(
        score.tracks[1].notes,
        vec![Note { start: 2, duration: 4, pitch: 72, velocity: 87 }]
    );
}

#[test]
fn duration_quantization_matches_miditok() {
    let names = [
        "Bar_None",
        "Position_0",
        "Program_0",
        "Pitch_60",
        "Velocity_99",
        "Duration_0.2.8",
        "Position_12",
        "Program_0",
        "Pitch_61",
        "Velocity_99",
        "Duration_0.3.8",
        "Position_23",
        "Program_0",
        "Pitch_62",
        "Velocity_99",
        "Duration_0.5.8",
        "Bar_None",
        "Position_3",
        "Program_0",
        "Pitch_63",
        "Velocity_99",
        "Duration_1.2.8",
        "Position_15",
        "Program_0",
        "Pitch_64",
        "Velocity_99",
        "Duration_1.7.8",
        "Position_26",
        "Program_0",
        "Pitch_65",
        "Velocity_99",
        "Duration_4.1.4",
    ];
    let expected = ids(&names);
    let score = Score {
        ticks_per_quarter: 480,
        tracks: vec![Track {
            program: 0,
            is_drum: false,
            name: String::new(),
            notes: vec![
                Note { start: 0, duration: 90, pitch: 60, velocity: 100 },
                Note { start: 700, duration: 150, pitch: 61, velocity: 100 },
                Note { start: 1400, duration: 300, pitch: 62, velocity: 100 },
                Note { start: 2100, duration: 620, pitch: 63, velocity: 100 },
                Note { start: 2800, duration: 900, pitch: 64, velocity: 100 },
                Note { start: 3500, duration: 2000, pitch: 65, velocity: 100 },
            ],
        }],
    };
    let actual = codec().encode(&score).expect("encode");
    assert_eq!(actual, expected);
}

#[test]
fn decode_encode_roundtrip_is_stable() {
    let score = two_track_score();
    let tokens = codec().encode(&score).expect("encode");
    let decoded = codec().decode(&tokens);
    let reencoded = codec().encode(&decoded).expect("re-encode");
    assert_eq!(reencoded, tokens);
}

#[test]
fn midi_file_roundtrip() {
    let score = two_track_score();
    let bytes = render_midi(&score).expect("render");
    let parsed = parse_midi(&bytes).expect("parse");
    assert_eq!(parsed.ticks_per_quarter, 480);
    assert_eq!(parsed.tracks.len(), 2);
    let mut sorted: Vec<(i32, i32, u8, u8)> = parsed.tracks[0]
        .notes
        .iter()
        .map(|n| (n.start, n.duration, n.pitch, n.velocity))
        .collect();
    sorted.sort();
    assert_eq!(
        sorted,
        vec![(0, 120, 60, 96), (240, 240, 64, 100), (480, 60, 67, 80)]
    );
}

#[test]
fn decode_midi_render_parse_roundtrip() {
    let tokens = [
        4, 190, 284, 44, 117, 127, 192, 324, 56, 115, 129, 194, 284, 48, 118, 129, 198, 284,
        51, 113, 126,
    ];
    let score = codec().decode(&tokens);
    let bytes = render_midi(&score).expect("render");
    let parsed = parse_midi(&bytes).expect("parse");
    let mut notes: Vec<(i32, i32, u8, u8)> = parsed
        .tracks
        .iter()
        .flat_map(|track| track.notes.iter().map(|n| (n.start, n.duration, n.pitch, n.velocity)))
        .collect();
    notes.sort();
    assert_eq!(
        notes,
        vec![(0, 2, 60, 95), (2, 4, 72, 87), (4, 4, 64, 99), (8, 1, 67, 79)]
    );
}

#[test]
fn write_read_midi_file() {
    let dir = std::env::temp_dir().join("yufmusicgen_vulkan_test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = Path::new(&dir).join("roundtrip.mid");
    let score = two_track_score();
    yufmusicgen_vulkan::midi::write_midi(&path, &score).expect("write");
    let loaded = yufmusicgen_vulkan::midi::read_midi(&path).expect("read");
    assert_eq!(loaded.tracks.len(), 2);
    assert_eq!(loaded.tracks[0].notes.len(), 3);
    std::fs::remove_file(&path).ok();
}

