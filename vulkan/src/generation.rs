//! Text-conditioned MIDI generation, mirroring `yufmusicgen.cli.generate`.

use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use crate::checkpoint::Checkpoint;
use crate::compute::model::Model;
use crate::instruments;
use crate::midi::{self, Score};
use crate::miditok::parse_tokenizer_json;
use crate::remi::RemiCodec;
use crate::sampler::{self, Mt19937};

pub const BOS: u32 = 1;
pub const EOS: u32 = 2;
pub const SEP: u32 = 3;
pub const TEXT_OFFSET: u32 = 4;
pub const TEXT_SIZE: u32 = 256;
pub const MIDI_OFFSET: u32 = TEXT_OFFSET + TEXT_SIZE;
pub const TOKENS_PER_SECOND: f64 = 20.0;
pub const MIN_MIDI_TOKENS_BEFORE_EOS: usize = 16;

#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub checkpoint: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub prompt: String,
    pub instrument: Option<String>,
    pub instrument_only: bool,
    pub prompt_midi: Option<PathBuf>,
    pub prompt_max_tokens: usize,
    pub output: PathBuf,
    pub steps: Option<usize>,
    pub seconds: Option<f64>,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

/// Build the REMI codec, preferring a `tokenizer.json` supplied at runtime,
/// then the one embedded in the checkpoint header, then the embedded
/// vocabulary. When a tokenizer JSON is used its derived vocabulary is
/// cross-checked against the embedded vocabulary.
pub fn build_codec(checkpoint: &Checkpoint, tokenizer: Option<&PathBuf>) -> Result<RemiCodec> {
    let embedded_vocab = &checkpoint.header.midi.vocab;
    let json_source = if let Some(path) = tokenizer {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read tokenizer {}", path.display()))?;
        Some(text)
    } else {
        checkpoint.header.midi.tokenizer_json.clone()
    };

    let vocab = match json_source {
        Some(json) => {
            let config = parse_tokenizer_json(&json)?;
            let derived = config.build_vocab();
            if derived.len() != embedded_vocab.len() {
                eprintln!(
                    "[warn] tokenizer-derived vocab ({} entries) differs in size from the \
                     checkpoint vocab ({}); using the tokenizer-derived one",
                    derived.len(),
                    embedded_vocab.len()
                );
            } else {
                let mismatches = derived
                    .iter()
                    .zip(embedded_vocab.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                if mismatches > 0 {
                    eprintln!(
                        "[warn] tokenizer-derived vocab differs from the checkpoint vocab in \
                         {mismatches} entries; using the tokenizer-derived one"
                    );
                }
            }
            derived
        }
        None => embedded_vocab.clone(),
    };
    RemiCodec::new(vocab).context("cannot build REMI codec")
}

#[derive(Debug, Clone)]
pub struct GenerationInfo {
    pub output: PathBuf,
    pub midi_tokens: usize,
    pub prompt_tokens: usize,
    pub tracks: usize,
    pub notes: usize,
    pub duration_seconds: f64,
    pub instrument: Option<String>,
    pub steps_done: usize,
}

pub fn run_generation(
    params: &GenerateParams,
    mut progress: impl FnMut(f32, &str),
) -> Result<GenerationInfo> {
    progress(0.02, "loading checkpoint");
    let checkpoint = Checkpoint::load(&params.checkpoint)
        .with_context(|| format!("cannot load {}", params.checkpoint.display()))?;
    let config = checkpoint.header.model_config;
    let midi_offset = checkpoint.header.midi.midi_offset;
    ensure!(
        midi_offset == MIDI_OFFSET as usize,
        "checkpoint uses midi_offset {midi_offset}, expected {}",
        MIDI_OFFSET
    );
    let codec = build_codec(&checkpoint, params.tokenizer.as_ref())?;
    ensure!(
        checkpoint.header.midi.midi_vocab_size + MIDI_OFFSET as usize == config.vocab_size,
        "codec vocabulary ({}) does not match model vocab size ({})",
        checkpoint.header.midi.midi_vocab_size + MIDI_OFFSET as usize,
        config.vocab_size
    );

    progress(0.08, "encoding prompt MIDI");
    let mut prompt_raw: Vec<u32> = Vec::new();
    if let Some(prompt_path) = &params.prompt_midi {
        let score = midi::read_midi(prompt_path)
            .with_context(|| format!("cannot read prompt MIDI {}", prompt_path.display()))?;
        prompt_raw = codec.encode(&score)?;
        if prompt_raw.len() > params.prompt_max_tokens.max(1) {
            prompt_raw = prompt_raw[prompt_raw.len() - params.prompt_max_tokens.max(1)..].to_vec();
        }
    }

    let requested_program = match &params.instrument {
        Some(value) => Some(instruments::resolve_program(value).map_err(anyhow::Error::msg)?),
        None => None,
    };
    if params.instrument_only && requested_program.is_none() {
        bail!("--instrument-only requires --instrument");
    }
    let blocked_programs: Vec<u32> = if params.instrument_only {
        let program = requested_program.unwrap();
        codec_program_ids(&codec)?
            .into_iter()
            .filter(|(id, _)| *id != program)
            .map(|(_, raw_id)| raw_id)
            .collect()
    } else {
        Vec::new()
    };

    progress(0.12, "building condition");
    let mut condition: Vec<u32> = vec![BOS];
    for byte in params.prompt.as_bytes() {
        condition.push(TEXT_OFFSET + *byte as u32);
    }
    condition.push(SEP);
    let mut output_prefix: Vec<u32> = prompt_raw.clone();
    if let Some(program) = requested_program {
        let raw_id = program_token_id(&codec, program)
            .with_context(|| format!("no Program_{program} token in the vocabulary"))?;
        output_prefix.push(raw_id);
    }
    for raw in &output_prefix {
        condition.push(MIDI_OFFSET + raw);
    }

    let target_steps = params.steps.unwrap_or_else(|| {
        params
            .seconds
            .map(|seconds| (seconds * TOKENS_PER_SECOND) as usize)
            .unwrap_or(512)
    })
    .max(1);

    progress(0.15, "initializing Vulkan compute");
    let mut model = Model::new(&checkpoint).context("cannot initialize Vulkan model")?;
    let mut rng = Mt19937::new(params.seed);

    let mut logits = Vec::new();
    for (index, token) in condition.iter().enumerate() {
        logits = model.step(*token)?;
        if index % 32 == 0 || index + 1 == condition.len() {
            progress(
                0.15,
                &format!("conditioning {}/{}", index + 1, condition.len()),
            );
        }
    }

    let mut generated: Vec<u32> = Vec::new();
    for index in 0..target_steps {
        if index % 16 == 0 {
            progress(
                0.15 + 0.8 * (index as f32 / target_steps as f32),
                &format!("sampling {}/{}", index + 1, target_steps),
            );
        }
        let mut allowed = vec![f32::NEG_INFINITY; config.vocab_size];
        allowed[MIDI_OFFSET as usize..]
            .copy_from_slice(&logits[MIDI_OFFSET as usize..]);
        for raw_id in &blocked_programs {
            allowed[MIDI_OFFSET as usize + *raw_id as usize] = f32::NEG_INFINITY;
        }
        if index >= MIN_MIDI_TOKENS_BEFORE_EOS {
            allowed[EOS as usize] = logits[EOS as usize];
        }
        let token = sampler::sample_token(&allowed, params.temperature, params.top_p, &mut rng)
            as u32;
        if token == EOS {
            break;
        }
        generated.push(token);
        if std::env::var("YUF_PRINT_TOKENS").is_ok() {
            println!("{token}");
        }
        logits = model.step(token)?;
    }

    progress(0.96, "decoding MIDI");
    let mut midi_ids: Vec<u32> = Vec::new();
    let mut started = false;
    for token in &generated {
        if (MIDI_OFFSET..config.vocab_size as u32).contains(&token) {
            started = true;
            midi_ids.push(token - MIDI_OFFSET);
        } else if started {
            break;
        }
    }
    let combined: Vec<u32> = output_prefix
        .iter()
        .chain(midi_ids.iter())
        .copied()
        .collect();
    let score: Score = codec.decode(&combined);
    midi::write_midi(&params.output, &score)
        .with_context(|| format!("cannot write {}", params.output.display()))?;

    let duration_seconds = score_duration_seconds(&score);
    let info = GenerationInfo {
        output: params.output.clone(),
        midi_tokens: midi_ids.len(),
        prompt_tokens: output_prefix.len(),
        tracks: score.tracks.len(),
        notes: score.tracks.iter().map(|t| t.notes.len()).sum(),
        duration_seconds,
        instrument: requested_program.map(|program| {
            format!("{} (program {program})", instruments::gm_name(program))
        }),
        steps_done: generated.len(),
    };
    progress(1.0, "done");
    Ok(info)
}

fn score_duration_seconds(score: &Score) -> f64 {
    // 120 BPM default, matching `midi_duration_seconds` in the Python client.
    let ticks_per_quarter = score.ticks_per_quarter.max(1) as f64;
    let qpm = 120.0;
    score.end_tick() as f64 / ticks_per_quarter / (qpm / 60.0)
}

/// Map GM program -> raw MidiTok id for every `Program_*` token.
pub fn codec_program_ids(codec: &RemiCodec) -> Result<Vec<(i32, u32)>> {
    let mut ids = Vec::new();
    for (id, name) in codec.vocab.iter().enumerate() {
        if let Some((kind, value)) = name.split_once('_') {
            if kind == "Program" {
                let program: i32 = value
                    .parse()
                    .with_context(|| format!("invalid Program token {name}"))?;
                ids.push((program, id as u32));
            }
        }
    }
    Ok(ids)
}

pub fn program_token_id(codec: &RemiCodec, program: i32) -> Option<u32> {
    codec
        .token_id(&format!("Program_{program}"))
}

pub fn list_instruments() {
    println!("program  name");
    for (program, name) in instruments::GM_PROGRAMS.iter().enumerate() {
        println!("{program:3}  {name}");
    }
    println!(" -1  Drums");
}
