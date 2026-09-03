//! Cross-check the PyTorch-compatible sampler against Python's CPU reference.
//!
//! Reads logits files (one f32 per line, token order; hexadecimal bit
//! patterns accepted) and prints the sampled token sequence using the same
//! MT19937 + Gumbel-softmax path as `yufmusicgen.cli.generate.sample_token`.
//! Generate reference logits with `scripts/dump_sampler_reference.py`.

use std::path::PathBuf;

use yufmusicgen_vulkan::sampler::{Mt19937, sample_token};

fn parse_logits(text: &str) -> Vec<f32> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let line = line.trim();
            if line.len() == 8 && line.chars().all(|c| c.is_ascii_hexdigit()) {
                f32::from_bits(u32::from_str_radix(line, 16).unwrap())
            } else {
                line.parse().unwrap()
            }
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let temperature: f32 = args.next().unwrap().parse().unwrap();
    let top_p: f32 = args.next().unwrap().parse().unwrap();
    let seed: u64 = args.next().unwrap().parse().unwrap();
    let logits_files: Vec<PathBuf> = args.map(PathBuf::from).collect();

    let mut rng = Mt19937::new(seed);
    let mut tokens = Vec::new();
    for path in &logits_files {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let logits = parse_logits(&text);
        tokens.push(sample_token(&logits, temperature, top_p, &mut rng));
    }
    println!("{}", tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
}
