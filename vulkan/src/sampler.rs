//! Temperature / top-p sampling, bit-compatible with PyTorch's CPU
//! `torch.multinomial(probs, 1)` driven by `torch.manual_seed`.
//!
//! PyTorch's CPU generator is `at::mt19937` (the reference MT19937 with the
//! 2002 initializer). For a single sample (`n_sample == 1`, the default in
//! `yufmusicgen.cli.generate.sample_token`) PyTorch uses a Gumbel-softmax
//! fast path instead of the cumulative-search kernel:
//!
//! ```text
//! u   = ((engine() << 32 | engine()) & ((1 << 53) - 1)) / 2^53   // f64
//! q   = -log1p(-u)   // Exp(1), f64, then cast to f32
//! idx = argmax(p / q)
//! ```
//!
//! This module replicates that stream and arithmetic exactly, so a fixed seed
//! yields the same token sequence as `yufmusicgen.cli.generate` running on
//! CPU.

/// Reference MT19937 engine, matching `at::mt19937` in
/// `ATen/core/MT19937RNGEngine.h`.
#[derive(Debug, Clone)]
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

impl Mt19937 {
    /// Seed exactly like `at::mt19937(seed)`: the 32-bit truncation is stored
    /// as `state[0]` and the remaining words are filled by the 2002 init
    /// generator (no twist happens until the first `next_u32`).
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; N];
        state[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..N {
            let prev = state[j - 1];
            state[j] = 1812433253u64
                .wrapping_mul((prev ^ (prev >> 30)) as u64)
                .wrapping_add(j as u64) as u32;
        }
        Mt19937 { state, index: N }
    }

    fn twist(&mut self) {
        for j in 0..N {
            let u = (self.state[j] & UPPER_MASK) | (self.state[(j + 1) % N] & LOWER_MASK);
            let mut next = u >> 1;
            if u & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[j] = self.state[(j + M) % N] ^ next;
        }
        self.index = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// One uniform draw exactly as `at::uniform_real_distribution<double>(0, 1)`
    /// does for the multinomial kernel: consume two 32-bit words, keep the top
    /// 53 bits of the 64-bit composite and scale by 2^-53.
    pub fn next_uniform(&mut self) -> f64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        let composite = (hi << 32) | lo;
        (composite & ((1u64 << 53) - 1)) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

impl rand::RngCore for Mt19937 {
    fn next_u32(&mut self) -> u32 {
        Mt19937::next_u32(self)
    }

    fn next_u64(&mut self) -> u64 {
        (self.next_u32() as u64) << 32 | self.next_u32() as u64
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(4);
        for chunk in &mut chunks {
            chunk.copy_from_slice(&self.next_u32().to_le_bytes());
        }
        let rest = chunks.into_remainder();
        if !rest.is_empty() {
            let bytes = self.next_u32().to_le_bytes();
            rest.copy_from_slice(&bytes[..rest.len()]);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl rand::SeedableRng for Mt19937 {
    type Seed = [u8; 8];

    fn from_seed(seed: [u8; 8]) -> Self {
        let mut value = 0u64;
        for byte in seed {
            value = (value << 8) | byte as u64;
        }
        Mt19937::new(value)
    }
}

/// Sample a token id from logits, mirroring `yufmusicgen.cli.generate.sample_token`.
///
/// `temperature <= 0` performs greedy argmax. Otherwise logits are scaled,
/// sorted descending, softmaxed, top-p filtered (`cumulative - probability >
/// top_p` removed, exactly like PyTorch), renormalized and sampled with
/// PyTorch's single-sample Gumbel-softmax path (`argmax(p / q)` with
/// `q ~ Exp(1)` drawn from the MT19937 stream).
pub fn sample_token(logits: &[f32], temperature: f32, top_p: f32, rng: &mut Mt19937) -> usize {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let scaled: Vec<f32> = logits.iter().map(|logit| logit / temperature).collect();

    // Order indices by descending logit.
    let mut order: Vec<usize> = (0..scaled.len()).collect();
    order.sort_by(|&a, &b| {
        scaled[b]
            .partial_cmp(&scaled[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Softmax over the sorted logits, replicating torch's CPU softmax:
    // exponentials are accumulated in 8 lanes (AVX2 vector width), then
    // reduced pairwise, and the final scaling uses `1/sum` * exp.
    let max_logit = scaled[order[0]];
    let weights: Vec<f32> = scaled
        .iter()
        .map(|logit| (logit - max_logit).exp())
        .collect();
    let mut lanes = [0.0f32; 8];
    let mut i = 0usize;
    while i + 8 <= weights.len() {
        for l in 0..8 {
            lanes[l] += weights[i + l];
        }
        i += 8;
    }
    let mut tail = 0.0f32;
    for &w in &weights[i..] {
        tail += w;
    }
    let l2 = [
        lanes[0] + lanes[1],
        lanes[2] + lanes[3],
        lanes[4] + lanes[5],
        lanes[6] + lanes[7],
    ];
    let l3 = [l2[0] + l2[1], l2[2] + l2[3]];
    let sum = l3[0] + l3[1] + tail;
    let inv_sum = 1.0 / sum;
    let weights: Vec<f32> = weights.iter().map(|w| w * inv_sum).collect();
    let mut masked = vec![f32::NEG_INFINITY; scaled.len()];
    let mut cumulative = 0.0f32;
    let mut kept_count = 0usize;
    for &index in &order {
        let probability = weights[index];
        if cumulative - probability > top_p {
            break;
        }
        masked[index] = probability;
        cumulative += probability;
        kept_count += 1;
    }
    if kept_count == 0 {
        return argmax(logits);
    }

    // Renormalize exactly like torch's second softmax over the masked vector:
    // the removed entries contribute exp(-inf) = 0, and the sum is computed
    // with the same 8-lane tree reduction, then scaled by 1/sum.
    let mut kept_weights: Vec<f32> = Vec::with_capacity(kept_count);
    for &index in &order {
        if masked[index].is_finite() {
            kept_weights.push(masked[index]);
        }
    }
    let mut lanes2 = [0.0f32; 8];
    let mut j = 0usize;
    while j + 8 <= masked.len() {
        for l in 0..8 {
            let v = masked[j + l];
            lanes2[l] += if v.is_finite() { v } else { 0.0 };
        }
        j += 8;
    }
    let mut tail2 = 0.0f32;
    for &v in &masked[j..] {
        tail2 += if v.is_finite() { v } else { 0.0 };
    }
    let m2 = [
        lanes2[0] + lanes2[1],
        lanes2[2] + lanes2[3],
        lanes2[4] + lanes2[5],
        lanes2[6] + lanes2[7],
    ];
    let m3 = [m2[0] + m2[1], m2[2] + m2[3]];
    let kept_sum = m3[0] + m3[1] + tail2;
    let inv_kept = 1.0 / kept_sum;
    for weight in &mut kept_weights {
        *weight *= inv_kept;
    }

    // PyTorch fast path for single-sample multinomial (used whenever
    // n_sample == 1, even with replacement): Gumbel-softmax sampling over the
    // *full* probability vector (masked entries have probability 0 and can
    // never win). `q ~ Exp(1)` draws a uniform via
    // `uniform_real_distribution<double>` (two 32-bit engine words -> 53-bit
    // mantissa) and applies `-log1p(-u)` in f64, then is cast to f32; the
    // score is `p / q` in f32 and the argmax is taken (first index wins ties).
    let mut best: Option<(f32, usize)> = None;
    for (position, &token) in order.iter().enumerate() {
        let p = if position < kept_count {
            masked[token]
        } else {
            0.0
        };
        let u = rng.next_uniform();
        let q = -log1p_f64(-u) as f32;
        let score = if q == 0.0 { f32::INFINITY } else { p / q };
        if score.is_nan() {
            continue;
        }
        if best
            .as_ref()
            .map(|(best_score, _)| score > *best_score)
            .unwrap_or(true)
        {
            best = Some((score, token));
        }
    }
    best.map(|(_, token)| token)
        .unwrap_or_else(|| order[0])
}

/// f64 `log1p`, matching libm so the Exp(1) transform agrees with PyTorch's
/// CPU `at::log1p`.
fn log1p_f64(x: f64) -> f64 {
    x.ln_1p()
}

pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}
