//! REMI (MidiTok v3) token <-> MIDI conversion for the YufMusicGen codec.
//! REMI (MidiTok v3) token <-> MIDI conversion for the YufMusicGen codec.
//!
//! Mirrors the deterministic vocabulary and encode/decode behaviour of MidiTok
//! 3.0.x for the project's codec configuration (REMI, no BPE, no tempos /
//! time signatures / rests / chords, one token stream for programs).  The
//! decoded score uses the same 8 ticks-per-quarter grid that MidiTok writes.

use std::collections::HashMap;

use anyhow::{Result, ensure};

use crate::midi::{Note, Score, Track};

pub const TICKS_PER_QUARTER: u16 = 8;
pub const TICKS_PER_BAR: i32 = 32;
pub const TICKS_PER_POSITION: i32 = 1;
const PITCH_MIN: i32 = 21;
const PITCH_MAX: i32 = 109;
const DRUM_PITCH_MIN: i32 = 27;
const DRUM_PITCH_MAX: i32 = 88;

fn velocities() -> Vec<i32> {
    (3..=127).step_by(4).collect()
}

pub struct RemiCodec {
    /// Raw MidiTok id -> token name ("Pitch_60", "Duration_0.2.8", ...).
    pub vocab: Vec<String>,
    name_to_id: HashMap<String, u32>,
    /// Sorted (ticks, value) for every Duration token at TPQ 8.
    duration_values: Vec<(i32, String)>,
}

impl RemiCodec {
    pub fn new(vocab: Vec<String>) -> Result<Self> {
        let mut name_to_id = HashMap::new();
        for (id, name) in vocab.iter().enumerate() {
            name_to_id.insert(name.clone(), id as u32);
        }
        let mut duration_values: Vec<(i32, String)> = Vec::new();
        for name in &vocab {
            if let Some((kind, value)) = name.split_once('_') {
                if kind == "Duration" {
                    let parts: Vec<i32> = value
                        .split('.')
                        .filter_map(|part| part.parse().ok())
                        .collect();
                    ensure!(parts.len() == 3, "malformed Duration token {name}");
                    let (beat, position, resolution) = (parts[0], parts[1], parts[2]);
                    let ticks =
                        (beat * resolution + position) * TICKS_PER_QUARTER as i32 / resolution;
                    duration_values.push((ticks, value.to_string()));
                }
            }
        }
        duration_values.sort_by_key(|(ticks, _)| *ticks);
        duration_values.dedup_by_key(|(ticks, _)| *ticks);
        Ok(Self {
            vocab,
            name_to_id,
            duration_values,
        })
    }

    pub fn token_name(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(String::as_str)
    }

    pub fn token_id(&self, name: &str) -> Option<u32> {
        self.name_to_id.get(name).copied()
    }

    // -----------------------------------------------------------------------
    // Decoding: raw MidiTok ids -> Score
    // -----------------------------------------------------------------------

    pub fn decode(&self, ids: &[u32]) -> Score {
        let mut tracks: Vec<Track> = Vec::new();
        let mut track_index: HashMap<i32, usize> = HashMap::new();
        let mut current_bar: i32 = -1;
        let mut tick_at_current_bar: i32 = 0;
        let mut current_tick: i32 = 0;
        let mut current_program: i32 = 0;

        let ensure_track = |tracks: &mut Vec<Track>,
                            track_index: &mut HashMap<i32, usize>,
                            program: i32| {
            if let Some(&index) = track_index.get(&program) {
                return index;
            }
            let is_drum = program == -1;
            let index = tracks.len();
            tracks.push(Track {
                program,
                is_drum,
                name: crate::instruments::gm_name(program).to_string(),
                notes: Vec::new(),
            });
            track_index.insert(program, index);
            index
        };

        let mut index = 0usize;
        while index < ids.len() {
            let Some(name) = self.token_name(ids[index]) else {
                index += 1;
                continue;
            };
            let Some((kind, value)) = name.split_once('_') else {
                index += 1;
                continue;
            };
            match kind {
                "Bar" => {
                    current_bar += 1;
                    tick_at_current_bar = current_bar * TICKS_PER_BAR;
                    current_tick = tick_at_current_bar;
                }
                "Position" => {
                    if current_bar == -1 {
                        current_bar = 0;
                    }
                    let position: i32 = value.parse().unwrap_or(0);
                    current_tick = tick_at_current_bar + position * TICKS_PER_POSITION;
                }
                "Program" => {
                    current_program = value.parse().unwrap_or(0);
                }
                "Pitch" | "PitchDrum" => {
                    let Ok(pitch) = value.parse::<i32>() else {
                        index += 1;
                        continue;
                    };
                    if !(PITCH_MIN..=PITCH_MAX).contains(&pitch) {
                        index += 1;
                        continue;
                    }
                    // Expect Velocity_* then Duration_* immediately after.
                    let velocity = if index + 1 < ids.len() {
                        self.token_name(ids[index + 1])
                            .and_then(|next| next.split_once('_'))
                            .filter(|(kind, _)| *kind == "Velocity")
                            .and_then(|(_, value)| value.parse::<u8>().ok())
                    } else {
                        None
                    };
                    let duration = if index + 2 < ids.len() {
                        self.token_name(ids[index + 2])
                            .and_then(|next| next.split_once('_'))
                            .filter(|(kind, _)| *kind == "Duration")
                            .and_then(|(_, value)| self.duration_ticks(value))
                    } else {
                        None
                    };
                    if let (Some(velocity), Some(duration)) = (velocity, duration) {
                        let track_index_id =
                            ensure_track(&mut tracks, &mut track_index, current_program);
                        tracks[track_index_id].notes.push(Note {
                            start: current_tick,
                            duration,
                            pitch: pitch as u8,
                            velocity,
                        });
                    }
                }
                _ => {}
            }
            index += 1;
        }

        Score {
            ticks_per_quarter: TICKS_PER_QUARTER,
            tracks,
        }
    }

    fn duration_ticks(&self, value: &str) -> Option<i32> {
        let parts: Vec<i32> = value
            .split('.')
            .filter_map(|part| part.parse().ok())
            .collect();
        if parts.len() != 3 {
            return None;
        }
        let (beat, position, resolution) = (parts[0], parts[1], parts[2]);
        if resolution <= 0 {
            return None;
        }
        Some(
            (beat * resolution + position) * TICKS_PER_QUARTER as i32 / resolution,
        )
    }

    // -----------------------------------------------------------------------
    // Encoding: Score -> raw MidiTok ids
    // -----------------------------------------------------------------------

    pub fn encode(&self, score: &Score) -> Result<Vec<u32>> {
        ensure!(
            score.ticks_per_quarter > 0,
            "score has no ticks-per-quarter division"
        );
        let factor = TICKS_PER_QUARTER as f64 / score.ticks_per_quarter as f64;

        // Merge tracks that share a program, mirroring MidiTok's
        // `merge_same_program_tracks` for one-token-stream mode.
        let mut merged: Vec<(i32, bool, Vec<Note>)> = Vec::new();
        for track in &score.tracks {
            let key = (track.program, track.is_drum);
            match merged.iter_mut().find(|(p, d, _)| (*p, *d) == key) {
                Some((_, _, notes)) => notes.extend(track.notes.iter().cloned()),
                None => merged.push((track.program, track.is_drum, track.notes.clone())),
            }
        }

        let mut events: Vec<Event> = Vec::new();
        for (program, is_drum, notes) in merged {
            let mut notes: Vec<Note> = notes
                .into_iter()
                .filter(|note| {
                    let (min, max) = if is_drum {
                        (DRUM_PITCH_MIN, DRUM_PITCH_MAX)
                    } else {
                        (PITCH_MIN, PITCH_MAX)
                    };
                    (min..=max).contains(&(note.pitch as i32))
                })
                .map(|note| Note {
                    start: round_half_up(note.start as f64 * factor),
                    duration: round_half_up(note.duration as f64 * factor).max(1),
                    pitch: note.pitch,
                    velocity: velocities()[closest_tie_up(&velocities(), note.velocity as i32)]
                        as u8,
                })
                .collect();
            // symusic keeps notes sorted by (time, duration, pitch).
            notes.sort_by_key(|n| (n.start, n.duration, n.pitch));

            for note in notes {
                let duration_value = self.closest_duration_value(note.duration);
                let pitch_kind = if is_drum { "PitchDrum" } else { "Pitch" };
                // This codec runs with `program_changes=False`: every note
                // carries its own Program event (mirrors MidiTok's
                // `_create_track_events` for one-token-stream mode).
                events.push(Event {
                    time: note.start,
                    kind: "Program".into(),
                    value: program.to_string(),
                });
                events.push(Event {
                    time: note.start,
                    kind: pitch_kind.to_string(),
                    value: note.pitch.to_string(),
                });
                events.push(Event {
                    time: note.start,
                    kind: "Velocity".into(),
                    value: note.velocity.to_string(),
                });
                events.push(Event {
                    time: note.start,
                    kind: "Duration".into(),
                    value: duration_value,
                });
            }
        }
        // Stable sort by time.
        events.sort_by_key(|event| event.time);
        if std::env::var("YUF_DEBUG_ENCODE").is_ok() {
            for event in &events {
                eprintln!("[encode] {} {} @{}", event.kind, event.value, event.time);
            }
        }
        self.add_time_events(&events)
    }

    fn closest_duration_value(&self, ticks: i32) -> String {
        let values: Vec<i32> = self.duration_values.iter().map(|(t, _)| *t).collect();
        let index = closest_tie_up(&values, ticks);
        self.duration_values[index].1.clone()
    }

    fn add_time_events(&self, events: &[Event]) -> Result<Vec<u32>> {
        let mut output: Vec<String> = Vec::new();
        let mut current_bar: i32 = -1;
        let mut tick_at_current_bar: i32 = 0;
        let mut previous_tick: i32 = -1;

        for event in events {
            if event.time != previous_tick {
                let new_bars = event.time / TICKS_PER_BAR - current_bar;
                for _ in 0..new_bars {
                    current_bar += 1;
                    tick_at_current_bar = current_bar * TICKS_PER_BAR;
                    output.push("Bar_None".to_string());
                }
                let position = (event.time - tick_at_current_bar) / TICKS_PER_POSITION;
                output.push(format!("Position_{position}"));
                previous_tick = event.time;
            }
            let name = format!("{}_{}", event.kind, event.value);
            output.push(name);
        }

        let mut ids = Vec::with_capacity(output.len());
        for name in output {
            let id = self
                .token_id(&name)
                .ok_or_else(|| anyhow::anyhow!("token {name} is not in the vocabulary"))?;
            ids.push(id);
        }
        Ok(ids)
    }
}

#[derive(Debug, Clone)]
struct Event {
    time: i32,
    kind: String,
    value: String,
}

fn round_half_up(value: f64) -> i32 {
    (value + 0.5).floor() as i32
}

/// Closest value in a sorted array; ties resolve to the larger value,
/// matching `np_get_closest` / `np.searchsorted(side="left")` in MidiTok.
fn closest_tie_up(sorted: &[i32], value: i32) -> usize {
    let mut low = 0usize;
    let mut high = sorted.len();
    while low < high {
        let mid = (low + high) / 2;
        if sorted[mid] < value {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if low == sorted.len() {
        return low - 1;
    }
    if low > 0 {
        let distance_prev = (value - sorted[low - 1]).abs();
        let distance_next = (sorted[low] - value).abs();
        if distance_prev < distance_next {
            return low - 1;
        }
    }
    low
}
