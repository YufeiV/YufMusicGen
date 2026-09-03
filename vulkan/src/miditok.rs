//! Deterministic MidiTok REMI vocabulary reconstruction from a
//! `tokenizer.json` configuration.
//!
//! MidiTok v3 does not store a learned vocabulary: the REMI vocabulary is a
//! pure function of `TokenizerConfig`. This module parses the subset of the
//! config used by the project codec (REMI, no BPE, no tempos / rests / chords
//! / time signatures / sustain / pitch bends) and regenerates the exact
//! `PAD_None … Program_-1` id order that `miditok.REMI` produces, so a
//! `tokenizer.json` can be loaded at runtime instead of trusting an embedded
//! vocabulary.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use serde_json::Value;

/// MidiTok tokenizer config subset used to rebuild the REMI vocabulary.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub pitch_range: (i32, i32),
    pub drums_pitch_range: (i32, i32),
    pub num_velocities: i32,
    pub use_velocities: bool,
    pub use_pitchdrum_tokens: bool,
    pub use_programs: bool,
    pub use_note_duration_programs: Vec<i32>,
    /// Max number of beats from the supported time signatures (default 4/4).
    pub max_num_beats: i32,
    /// Ordered `(beat_start, beat_end, resolution)` ranges, preserving JSON
    /// key order (`"0_4": 8, "4_12": 4`).
    pub beat_res: Vec<(i32, i32, i32)>,
    pub programs: Vec<i32>,
}

/// Parse the MidiTok configuration stored in a `tokenizer.json` (`config`
/// object, plus a top-level `tokenization` selector).
pub fn parse_tokenizer_json(json: &str) -> Result<TokenizerConfig> {
    let root: Value = serde_json::from_str(json).context("tokenizer.json is not valid JSON")?;
    let config = root
        .get("config")
        .context("tokenizer.json has no \"config\" object")?;

    let tokenization = root
        .get("tokenization")
        .and_then(Value::as_str)
        .unwrap_or("REMI");
    ensure!(
        tokenization == "REMI",
        "only the REMI tokenization is supported, got {tokenization}"
    );

    let pitch_range = parse_range(config, "pitch_range")?;
    let drums_pitch_range = parse_range(config, "drums_pitch_range")?;
    let num_velocities = config
        .get("num_velocities")
        .and_then(Value::as_i64)
        .unwrap_or(32) as i32;
    let use_velocities = config
        .get("use_velocities")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let use_pitchdrum_tokens = config
        .get("use_pitchdrum_tokens")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let use_programs = config
        .get("use_programs")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let use_note_duration_programs = config
        .get("use_note_duration_programs")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_i64)
                .map(|v| v as i32)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let programs = config
        .get("programs")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_i64)
                .map(|v| v as i32)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let max_num_beats = if config
        .get("use_time_signatures")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        config
            .get("time_signatures")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|ts| ts.get(0).and_then(Value::as_i64))
                    .map(|v| v as i32)
                    .max()
                    .unwrap_or(4)
            })
            .unwrap_or(4)
    } else {
        4
    };

    let mut beat_res: Vec<(i32, i32, i32)> = Vec::new();
    if let Some(map) = config.get("beat_res").and_then(Value::as_object) {
        // serde_json::Map preserves insertion order for the "preserve_order"
        // feature; otherwise keys arrive sorted. MidiTok iterates the dict in
        // insertion order ("0_4" before "4_12").
        let mut entries: Vec<(String, i32)> = map
            .iter()
            .filter_map(|(key, value)| {
                value.as_i64().map(|v| (key.clone(), v as i32))
            })
            .collect();
        entries.sort_by_key(|(key, _)| parse_beat_key(key));
        for (key, resolution) in entries {
            let (start, end) = parse_beat_range(&key)?;
            beat_res.push((start, end, resolution));
        }
    }
    ensure!(!beat_res.is_empty(), "tokenizer.json has no beat_res");

    Ok(TokenizerConfig {
        pitch_range,
        drums_pitch_range,
        num_velocities,
        use_velocities,
        use_pitchdrum_tokens,
        use_programs,
        use_note_duration_programs,
        max_num_beats,
        beat_res,
        programs,
    })
}

fn parse_range(config: &Value, key: &str) -> Result<(i32, i32)> {
    let array = config
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("tokenizer.json config.{key} is not an array"))?;
    ensure!(array.len() == 2, "config.{key} must have two elements");
    let a = array[0].as_i64().context("range start")? as i32;
    let b = array[1].as_i64().context("range end")? as i32;
    Ok((a, b))
}

fn parse_beat_key(key: &str) -> i32 {
    key.split('_').next().and_then(|v| v.parse().ok()).unwrap_or(i32::MAX)
}

fn parse_beat_range(key: &str) -> Result<(i32, i32)> {
    let parts: Vec<i32> = key
        .split('_')
        .filter_map(|part| part.parse().ok())
        .collect();
    ensure!(parts.len() == 2, "malformed beat_res key {key}");
    Ok((parts[0], parts[1]))
}

impl TokenizerConfig {
    /// Rebuild the exact MidiTok REMI vocabulary (raw id -> token name),
    /// following `midi_tokenizer._create_base_vocabulary` ordering:
    /// specials, Bar, Pitch, Velocity, Duration, PitchDrum, Program.
    pub fn build_vocab(&self) -> Vec<String> {
        let mut vocab: Vec<String> = vec![
            "PAD_None".into(),
            "BOS_None".into(),
            "EOS_None".into(),
            "MASK_None".into(),
            "Bar_None".into(),
        ];
        for pitch in self.pitch_range.0..=self.pitch_range.1 {
            vocab.push(format!("Pitch_{pitch}"));
        }
        if self.use_velocities {
            for velocity in velocities(self.num_velocities) {
                vocab.push(format!("Velocity_{velocity}"));
            }
        }
        if !self.use_note_duration_programs.is_empty() {
            for (beat, pos, resolution) in durations(&self.beat_res) {
                vocab.push(format!("Duration_{beat}.{pos}.{resolution}"));
            }
        }
        let max_num_pos_per_beat = self
            .beat_res
            .iter()
            .map(|(_, _, res)| *res)
            .max()
            .unwrap_or(1);
        for position in 0..(max_num_pos_per_beat * self.max_num_beats) {
            vocab.push(format!("Position_{position}"));
        }
        if self.use_pitchdrum_tokens {
            for pitch in self.drums_pitch_range.0..=self.drums_pitch_range.1 {
                vocab.push(format!("PitchDrum_{pitch}"));
            }
        }
        if self.use_programs {
            for program in &self.programs {
                vocab.push(format!("Program_{program}"));
            }
        }
        vocab
    }

    /// Index vocab names for O(1) lookup.
    pub fn name_to_id(&self, vocab: &[String]) -> HashMap<String, u32> {
        let mut map = HashMap::with_capacity(vocab.len());
        for (id, name) in vocab.iter().enumerate() {
            map.insert(name.clone(), id as u32);
        }
        map
    }
}

/// `np.linspace(0, 127, num_velocities + 1, dtype=np.intc)[1:]` — truncating
/// toward zero, so e.g. 32 velocities yield 3, 7, 11, …, 127.
pub fn velocities(num_velocities: i32) -> Vec<i32> {
    (1..=num_velocities)
        .map(|i| (127.0 * i as f64 / num_velocities as f64) as i32)
        .collect()
}

/// `(beat, pos, res)` tuples from `_create_durations_tuples`:
/// each beat_res range contributes `(beat, pos, res)` for every beat/pos
/// pair, then the final sentinel `(max_beat_end, 0, max_res)`, minus the
/// zero-duration `(0, 0, _)` entry.
pub fn durations(beat_res: &[(i32, i32, i32)]) -> Vec<(i32, i32, i32)> {
    let mut out = Vec::new();
    for &(start, end, resolution) in beat_res {
        for beat in start..end {
            for pos in 0..resolution {
                out.push((beat, pos, resolution));
            }
        }
    }
    let (max_end, max_res) = beat_res
        .iter()
        .map(|(_, end, res)| (*end, *res))
        .max_by_key(|(end, res)| (*end, *res))
        .unwrap_or((0, 1));
    out.push((max_end, 0, max_res));
    if out.first().map(|(beat, pos, _)| *beat == 0 && *pos == 0) == Some(true) {
        out.remove(0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER_JSON: &str = r#"{
        "tokenization": "REMI",
        "config": {
            "pitch_range": [21, 109],
            "drums_pitch_range": [27, 88],
            "num_velocities": 32,
            "use_velocities": true,
            "use_pitchdrum_tokens": true,
            "use_programs": true,
            "use_note_duration_programs": [0, 1],
            "beat_res": {"0_4": 8, "4_12": 4},
            "programs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                         30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                         44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                         58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                         72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
                         86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                         100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                         111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                         122, 123, 124, 125, 126, 127, -1]
        }
    }"#;

    #[test]
    fn rebuilds_reference_remi_vocab() {
        let config = parse_tokenizer_json(TOKENIZER_JSON).expect("parse");
        let vocab = config.build_vocab();
        assert_eq!(vocab.len(), 413);
        assert_eq!(&vocab[..5], &["PAD_None", "BOS_None", "EOS_None", "MASK_None", "Bar_None"]);
        assert_eq!(vocab[5], "Pitch_21");
        assert_eq!(vocab[93], "Pitch_109");
        assert_eq!(vocab[94], "Velocity_3");
        assert_eq!(vocab[125], "Velocity_127");
        assert_eq!(vocab[126], "Duration_0.1.8");
        assert_eq!(vocab[189], "Duration_12.0.4");
        assert_eq!(vocab[190], "Position_0");
        assert_eq!(vocab[221], "Position_31");
        assert_eq!(vocab[222], "PitchDrum_27");
        assert_eq!(vocab[283], "PitchDrum_88");
        assert_eq!(vocab[284], "Program_0");
        assert_eq!(vocab[411], "Program_127");
        assert_eq!(vocab[412], "Program_-1");
    }

    #[test]
    fn velocities_match_numpy_linspace() {
        assert_eq!(velocities(32), vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55,
            59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]);
    }

    #[test]
    fn duration_tuples_match_miditok() {
        let config = parse_tokenizer_json(TOKENIZER_JSON).expect("parse");
        let durations = durations(&config.beat_res);
        assert_eq!(durations.len(), 64);
        assert_eq!(durations[0], (0, 1, 8));
        assert_eq!(durations[30], (3, 7, 8));
        assert_eq!(durations[31], (4, 0, 4));
        assert_eq!(durations[63], (12, 0, 4));
    }
}
