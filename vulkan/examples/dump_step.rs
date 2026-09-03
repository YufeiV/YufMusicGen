//! Run a single model step and dump named intermediate work/state regions.
//!
//! Usage: dump_step <checkpoint.yuf> [token[,token...]]
//!
//! Set `YUF_DUMP=region1,region2,...` to select regions (defaults to all);
//! set `YUF_DEBUG_MAX_DISPATCH=N` to stop the recorded step after N
//! dispatches so the work buffers hold early-layer intermediates.

use std::path::PathBuf;

use anyhow::Result;
use yufmusicgen_vulkan::checkpoint::Checkpoint;
use yufmusicgen_vulkan::compute::model::Model;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("model.yuf"));
    let tokens: Vec<u32> = args
        .get(2)
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.parse().ok())
                .collect()
        })
        .filter(|tokens: &Vec<u32>| !tokens.is_empty())
        .unwrap_or_else(|| vec![1]);

    let checkpoint = Checkpoint::load(&checkpoint_path)?;
    let mut model = Model::new(&checkpoint)?;
    for token in tokens {
        model.step(token)?;
    }

    let dump_value = std::env::var("YUF_DUMP").unwrap_or_default();
    let regions: Vec<&str> = dump_value
        .split(',')
        .filter(|region| !region.is_empty())
        .collect();
    if regions.is_empty() {
        model.debug_dump(&[
            "ln0", "r", "k", "v", "w", "a", "g", "o", "ln_o", "h1", "ln1", "cand", "write",
            "read_gate", "read", "ln2", "fin", "fgate", "final", "logits", "tm_mem", "rosa_mem",
            "prevs",
        ]);
    } else {
        model.debug_dump(&regions);
    }
    Ok(())
}
