//! Verify that the Rust `tokenizer.json` loader reproduces the vocabulary
//! embedded in a `.yuf` checkpoint (and therefore matches `miditok.REMI`).

use std::path::PathBuf;

use anyhow::{Context, Result};
use yufmusicgen_vulkan::checkpoint::Checkpoint;
use yufmusicgen_vulkan::miditok::parse_tokenizer_json;

fn main() -> Result<()> {
    let checkpoint_path = std::env::args().nth(1).map(PathBuf::from).unwrap();
    let tokenizer_path = std::env::args().nth(2).map(PathBuf::from).unwrap();

    let checkpoint = Checkpoint::load(&checkpoint_path)
        .with_context(|| format!("cannot load {}", checkpoint_path.display()))?;
    let json = std::fs::read_to_string(&tokenizer_path)
        .with_context(|| format!("cannot read {}", tokenizer_path.display()))?;
    let config = parse_tokenizer_json(&json)?;
    let rebuilt = config.build_vocab();
    let embedded = &checkpoint.header.midi.vocab;

    println!("rebuilt vocab: {} entries", rebuilt.len());
    println!("embedded vocab: {} entries", embedded.len());
    let mismatches: Vec<usize> = (0..rebuilt.len().min(embedded.len()))
        .filter(|&i| rebuilt[i] != embedded[i])
        .collect();
    println!("vocab mismatches: {}", mismatches.len());
    if !mismatches.is_empty() {
        for i in mismatches.iter().take(5) {
            println!(
                "  [{i}] rebuilt={:?} embedded={:?}",
                rebuilt[*i], embedded[*i]
            );
        }
    }
    Ok(())
}
