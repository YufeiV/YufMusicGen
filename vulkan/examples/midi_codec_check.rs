//! Cross-check the Rust REMI codec against MidiTok.
//!
//! Loads a `.yuf` checkpoint (for its embedded MidiTok vocabulary), reads a
//! MIDI file, encodes it to raw MidiTok ids and prints them, then decodes
//! them back to a MIDI file. Compare the token ids with
//! `yufmusicgen.codec.MidiCodec.encode` on the same input.

use std::path::PathBuf;

use anyhow::{Context, Result};
use yufmusicgen_vulkan::checkpoint::Checkpoint;
use yufmusicgen_vulkan::midi::{read_midi, render_midi};
use yufmusicgen_vulkan::remi::RemiCodec;

fn main() -> Result<()> {
    let checkpoint_path = std::env::args().nth(1).map(PathBuf::from).unwrap();
    let midi_in = std::env::args().nth(2).map(PathBuf::from).unwrap();
    let midi_out = std::env::args().nth(3).map(PathBuf::from);

    let checkpoint = Checkpoint::load(&checkpoint_path)
        .with_context(|| format!("cannot load {}", checkpoint_path.display()))?;
    let codec = RemiCodec::new(checkpoint.header.midi.vocab.clone())
        .context("cannot build REMI codec")?;
    let score = read_midi(&midi_in).with_context(|| format!("cannot parse {}", midi_in.display()))?;
    let ids = codec.encode(&score).context("encode failed")?;
    println!("tokens: {}", ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" "));

    if let Some(out) = midi_out {
        let decoded = codec.decode(&ids);
        let bytes = render_midi(&decoded)?;
        std::fs::write(&out, bytes)?;
        println!("wrote {}", out.display());
    }
    Ok(())
}
