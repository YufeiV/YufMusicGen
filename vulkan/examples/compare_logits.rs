//! Prints top-k logits and a greedy token sequence for parity checks against
//! the PyTorch reference implementation.

use std::path::PathBuf;

use anyhow::{Context, Result};
use yufmusicgen_vulkan::checkpoint::Checkpoint;
use yufmusicgen_vulkan::compute::model::Model;
use yufmusicgen_vulkan::generation::{BOS, MIDI_OFFSET, SEP, TEXT_OFFSET};

fn main() -> Result<()> {
    let checkpoint_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("testdata/tiny.yuf"));
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "test".into());
    let greedy_steps: usize = std::env::args()
        .nth(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(13);

    let checkpoint = Checkpoint::load(&checkpoint_path)?;
    let config = checkpoint.header.model_config;
    let mut model = Model::new(&checkpoint)?;
    if std::env::var("YUF_PROBE_NOOP").is_ok() {
        model.probe_noop()?;
        println!("noop probe ok");
        return Ok(());
    }
    if std::env::var("YUF_PROBE_EMBED").is_ok() {
        model.probe_embed()?;
        println!("embed probe ok");
        return Ok(());
    }
    if std::env::var("YUF_PROBE_BARE").is_ok() {
        model.probe_bare_noop()?;
        println!("bare noop probe ok");
        return Ok(());
    }

    let mut condition = vec![BOS];
    for byte in prompt.as_bytes() {
        condition.push(TEXT_OFFSET + *byte as u32);
    }
    condition.push(SEP);

    let mut logits = Vec::new();
    for (index, token) in condition.iter().enumerate() {
        logits = model
            .step(*token)
            .with_context(|| format!("model.step({token}) failed (condition {index})"))?;
    }
    if std::env::var("YUF_DUMP").is_ok() {
        let dump_value = std::env::var("YUF_DUMP").unwrap();
        let names: Vec<&str> = dump_value.split(',').collect();
        model.debug_dump(&names);
        return Ok(());
    }
    if std::env::var("YUF_RAW_LOGITS").is_ok() {
        for (index, value) in logits.iter().enumerate() {
            println!("{index} {value:.4}");
        }
        return Ok(());
    }
    print_top("after conditioning", &logits, &checkpoint, 8);

    let mut sequence = Vec::new();
    for _ in 0..greedy_steps {
        let mut allowed = vec![f32::NEG_INFINITY; config.vocab_size];
        allowed[MIDI_OFFSET as usize..].copy_from_slice(&logits[MIDI_OFFSET as usize..]);
        let token = argmax(&allowed);
        sequence.push(token);
        logits = model.step(token)?;
    }
    println!("greedy seq: {sequence:?}");
    let names: Vec<String> = sequence
        .iter()
        .map(|token| {
            if *token >= MIDI_OFFSET {
                let raw = token - MIDI_OFFSET;
                checkpoint
                    .header
                    .midi
                    .vocab
                    .get(raw as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("raw{raw}"))
            } else {
                format!("{token}")
            }
        })
        .collect();
    println!("names: {names:?}");
    Ok(())
}

fn print_top(label: &str, logits: &[f32], checkpoint: &Checkpoint, k: usize) {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    print!("{label} top{k}: [");
    for &index in &indices[..k] {
        print!("{index} ");
    }
    print!("] logits [");
    for &index in &indices[..k] {
        print!("{:.4} ", logits[index]);
    }
    println!("]");
    let names: Vec<String> = indices[..k]
        .iter()
        .map(|&index| {
            if index >= MIDI_OFFSET as usize {
                checkpoint
                    .header
                    .midi
                    .vocab
                    .get((index - MIDI_OFFSET as usize) as usize)
                    .cloned()
                    .unwrap_or_default()
            } else {
                String::new()
            }
        })
        .collect();
    println!("{label} names: {names:?}");
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index as u32)
        .unwrap_or(0)
}
