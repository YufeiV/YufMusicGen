//! YufMusicGen model inference on top of the Vulkan compute context.

use anyhow::Result;
use ash::vk;

use crate::checkpoint::{Checkpoint, ModelConfig};

use super::{
    ComputeContext, FLAG_ADD_RESIDUAL, FLAG_HAS_BIAS, FLAG_SIGMOID, FLAG_TANH,
    FLAG_TRANSPOSE_W, FLAG_USE_GATE, PushConsts,
};

pub struct Model {
    pub context: ComputeContext,
    pub config: ModelConfig,
    work: WorkLayout,
    state: StateLayout,
    weights: WeightRefs,
    dispatch_count: usize,
    max_dispatch: Option<usize>,
}

impl Model {
    pub fn new(checkpoint: &Checkpoint) -> Result<Self> {
        let config = checkpoint.header.model_config;
        config.validate()?;
        let work = WorkLayout::build(&config);
        let state = StateLayout::build(&config);
        let weights = WeightRefs::resolve(checkpoint, &config)?;
        let max_dispatch = std::env::var("YUF_DEBUG_MAX_DISPATCH")
            .ok()
            .and_then(|value| value.parse().ok());
        if max_dispatch.is_some() {
            eprintln!("[model] YUF_DEBUG_MAX_DISPATCH={}", max_dispatch.unwrap());
        }
        let mut model = Self {
            context: ComputeContext::new(
                &checkpoint.data,
                work.size as usize,
                state.size as usize,
            )?,
            config,
            work,
            state,
            weights,
            dispatch_count: 0,
            max_dispatch,
        };
        model.record_step()?;
        Ok(model)
    }

    /// Run one forward step for `token` and return the vocab-sized logits.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        self.context.write_work(self.work.token as usize, &[token as f32]);
        self.context.execute_step()?;
        Ok(self
            .context
            .read_work(self.work.logits as usize, self.config.vocab_size))
    }

    /// Reset all recurrent state (previous token, TimeMix memory, ROSA memory).
    pub fn reset_state(&mut self) -> Result<()> {
        self.context.reset_state()
    }

    /// Debug helper: run a single no-op compute dispatch on the queue.
    pub fn probe_noop(&self) -> Result<()> {
        let cmd = self.context.begin_step_record()?;
        let pc = PushConsts::new();
        self.dispatch_debug(cmd, "noop", &pc);
        self.context.end_step_record(cmd)?;
        self.context.execute_step()
    }

    /// Debug helper: run a single embedding lookup dispatch.
    pub fn probe_embed(&self) -> Result<()> {
        let cmd = self.context.begin_step_record()?;
        let mut pc = PushConsts::new();
        pc.token_off = self.work.token;
        pc.weight_off = self.weights.token_embedding;
        pc.out_off = self.work.hidden;
        pc.cols = self.config.d_model as u32;
        self.dispatch_debug(cmd, "embed", &pc);
        self.context.end_step_record(cmd)?;
        self.context.execute_step()
    }

    /// Debug helper: bare dispatch of the no-op pipeline (no descriptors, no
    /// push constants).
    pub fn probe_bare_noop(&self) -> Result<()> {
        let cmd = self.context.begin_step_record()?;
        self.context.record_bare_dispatch(cmd, "noop", 1);
        self.context.end_step_record(cmd)?;
        self.context.execute_step()
    }

    /// Debug helper: dump named work-buffer regions for parity checks.
    pub fn debug_dump(&self, names: &[&str]) {
        let full = std::env::var("YUF_DUMP_FULL").is_ok();
        let dump = |label: &str, offset: u32, count: u32| {
            let values = self.context.read_work(offset as usize, count as usize);
            if full {
                let all: Vec<String> = values.iter().map(|v| format!("{v:.9}")).collect();
                eprintln!("{label} {} {}", values.len(), all.join(" "));
            } else {
                let head: Vec<String> = values.iter().take(8).map(|v| format!("{v:.4}")).collect();
                eprintln!("{label} {} {}", values.len(), head.join(" "));
            }
        };
        for name in names {
            match *name {
                "ln0" => dump("ln0", self.work.ln0, self.config.d_model as u32),
                "mix" => dump(
                    "mix",
                    self.work.mix,
                    6 * self.config.d_model as u32,
                ),
                "prev" => dump("prev", self.work.prev_base, self.config.d_model as u32),
                "r" => dump("r", self.work.r, self.config.d_model as u32),
                "k" => dump("k", self.work.k, self.config.d_model as u32),
                "v" => dump("v", self.work.v, self.config.d_model as u32),
                "w" => dump("w", self.work.w, self.config.d_model as u32),
                "a" => dump("a", self.work.a, self.config.d_model as u32),
                "g" => dump("g", self.work.g, self.config.d_model as u32),
                "o" => dump("o", self.work.o, self.config.d_model as u32),
                "ln_o" => dump("ln_o", self.work.ln_o, self.config.d_model as u32),
                "h1" | "h2" | "h3" => dump(
                    name,
                    self.work.hidden,
                    self.config.d_model as u32,
                ),
                "ln1" => dump("ln1", self.work.ln1, self.config.d_model as u32),
                "ln2" => dump("ln2", self.work.ln2, self.config.d_model as u32),
                "cand" => dump("cand", self.work.cand, self.config.rosa_size as u32),
                "write" => dump("write", self.work.write, self.config.rosa_size as u32),
                "read_gate" => dump("read_gate", self.work.read_gate, self.config.rosa_size as u32),
                "read" => dump("read", self.work.read, self.config.rosa_size as u32),
                "fin" => dump("fin", self.work.fin, 4 * self.config.d_model as u32),
                "fgate" => dump("fgate", self.work.fgate, 4 * self.config.d_model as u32),
                "final" => dump("final", self.work.ln_final, self.config.d_model as u32),
                "logits" => dump("logits", self.work.logits, self.config.vocab_size as u32),
                "reduce" => {
                    let values = self.context.read_work(self.work.reduce as usize, 2);
                    eprintln!("reduce {} {:.6} {:.6}", values.len(), values[0], values[1]);
                }
                "tm_mem" => {
                    let values = self
                        .context
                        .read_state(
                            self.state.layer_base(0).timemix_memory as usize,
                            (self.config.n_heads
                                * self.config.head_size
                                * self.config.head_size) as usize,
                        )
                        .unwrap();
                    if full {
                        let all: Vec<String> =
                            values.iter().map(|v| format!("{v:.9}")).collect();
                        eprintln!("tm_mem {} {}", values.len(), all.join(" "));
                    } else {
                        let head: Vec<String> =
                            values.iter().take(8).map(|v| format!("{v:.4}")).collect();
                        eprintln!("tm_mem {} {}", values.len(), head.join(" "));
                    }
                }
                name if name.starts_with("tm_mem_L") => {
                    let layer: usize = name[8..].parse().unwrap_or(0);
                    let values = self
                        .context
                        .read_state(
                            self.state.layer_base(layer).timemix_memory as usize,
                            (self.config.n_heads
                                * self.config.head_size
                                * self.config.head_size) as usize,
                        )
                        .unwrap();
                    if full {
                        let all: Vec<String> =
                            values.iter().map(|v| format!("{v:.9}")).collect();
                        eprintln!("tm_mem_L{layer} {} {}", values.len(), all.join(" "));
                    } else {
                        let head: Vec<String> =
                            values.iter().take(8).map(|v| format!("{v:.4}")).collect();
                        eprintln!("tm_mem_L{layer} {} {}", values.len(), head.join(" "));
                    }
                }
                "rosa_mem" => {
                    let values = self
                        .context
                        .read_state(
                            self.state.layer_base(0).rosa_memory as usize,
                            self.config.rosa_size as usize,
                        )
                        .unwrap();
                    if full {
                        let all: Vec<String> =
                            values.iter().map(|v| format!("{v:.9}")).collect();
                        eprintln!("rosa_mem {} {}", values.len(), all.join(" "));
                    } else {
                        let head: Vec<String> =
                            values.iter().take(8).map(|v| format!("{v:.4}")).collect();
                        eprintln!("rosa_mem {} {}", values.len(), head.join(" "));
                    }
                }
                name if name.starts_with("rosa_mem_L") => {
                    let layer: usize = name[11..].parse().unwrap_or(0);
                    let values = self
                        .context
                        .read_state(
                            self.state.layer_base(layer).rosa_memory as usize,
                            self.config.rosa_size as usize,
                        )
                        .unwrap();
                    if full {
                        let all: Vec<String> =
                            values.iter().map(|v| format!("{v:.9}")).collect();
                        eprintln!("rosa_mem_L{layer} {} {}", values.len(), all.join(" "));
                    } else {
                        let head: Vec<String> =
                            values.iter().take(8).map(|v| format!("{v:.4}")).collect();
                        eprintln!("rosa_mem_L{layer} {} {}", values.len(), head.join(" "));
                    }
                }
                "prevs" => {
                    let d = self.config.d_model as u32;
                    for layer in 0..self.config.n_layers {
                        let values = self.context.read_work(
                            (self.work.prev_base + (layer as u32) * d) as usize,
                            d as usize,
                        );
                        if full {
                            let all: Vec<String> =
                                values.iter().map(|v| format!("{v:.9}")).collect();
                            eprintln!("prev[{layer}] {} {}", values.len(), all.join(" "));
                        } else {
                            let head: Vec<String> =
                                values.iter().take(4).map(|v| format!("{v:.4}")).collect();
                            eprintln!("prev[{layer}] {}", head.join(" "));
                        }
                    }
                }
                other => eprintln!("unknown region {other}"),
            }
        }
    }

    fn dispatch_debug(&self, cmd: vk::CommandBuffer, kernel: &str, pc: &PushConsts) {
        self.context.record_dispatch(cmd, kernel, pc, 1);
    }

    fn dispatch(&mut self, cmd: vk::CommandBuffer, kernel: &str, pc: &PushConsts, count: u32) {
        if let Some(limit) = self.max_dispatch {
            if self.dispatch_count >= limit {
                return;
            }
        }
        self.validate_dispatch(kernel, pc);
        self.context.record_dispatch(cmd, kernel, pc, count);
        self.dispatch_count += 1;
    }

    /// CPU-side bounds audit for every dispatch.  Any out-of-range access is
    /// reported before the command buffer ever reaches the GPU, so a mistake
    /// like the historical TimeMix memory index bug can never fault the
    /// driver again.
    fn validate_dispatch(&self, kernel: &str, pc: &PushConsts) {
        let work = self.context.work_size as u64;
        let state = self.context.state_count as u64;
        let weights = self.context.weights_count as u64;

        let check = |label: &str, offset: u32, count: u64, limit: u64, buffer: &str| {
            let end = offset as u64 + count;
            if end > limit {
                panic!(
                    "[{kernel}] {label} out of range: offset {offset} + {count} = {end} > \
                     {buffer} limit {limit}"
                );
            }
        };

        match kernel {
            "embed" => {
                check("token", pc.token_off, 1, work, "work");
                check("embedding", pc.weight_off, (self.config.vocab_size * self.config.d_model) as u64, weights, "weights");
                check("output", pc.out_off, pc.cols as u64, work, "work");
            }
            "linear" => {
                check("input", pc.in_off, (pc.k as u64).saturating_mul(pc.rows as u64), work, "work");
                check("weights", pc.weight_off, (pc.cols as u64) * (pc.k as u64), weights, "weights");
                if pc.flags & FLAG_HAS_BIAS != 0 {
                    check("bias", pc.bias_off, pc.cols as u64, weights, "weights");
                }
                if pc.flags & FLAG_USE_GATE != 0 {
                    check("gate", pc.gate_off, pc.k as u64, work, "work");
                }
                if pc.flags & FLAG_ADD_RESIDUAL != 0 {
                    check("residual", pc.residual_off, pc.cols as u64, work, "work");
                }
                check("output", pc.out_off, (pc.cols as u64) * (pc.rows as u64), work, "work");
            }
            "layernorm_reduce" => {
                check("input", pc.in_off, pc.cols as u64, work, "work");
                check("output", pc.out_off, 2, work, "work");
            }
            "layernorm_apply" => {
                check("input", pc.in_off, pc.cols as u64, work, "work");
                check("gamma", pc.weight_off, pc.cols as u64, weights, "weights");
                check("beta", pc.bias_off, pc.cols as u64, weights, "weights");
                check("sum", pc.gate_off, 1, work, "work");
                check("sumsq", pc.residual_off, 1, work, "work");
                check("output", pc.out_off, pc.cols as u64, work, "work");
                if pc.extra0 != PushConsts::NONE {
                    check("mirror", pc.extra0, pc.cols as u64, work, "work");
                }
            }
            "mix_inputs" => {
                check("input", pc.in_off, pc.cols as u64, work, "work");
                check("previous", pc.gate_off, pc.cols as u64, work, "work");
                check("mix params", pc.weight_off, 6 * pc.cols as u64, weights, "weights");
                check("output", pc.out_off, 6 * pc.cols as u64, work, "work");
            }
            "timemix_recurrence" => {
                let hs = pc.k as u64;
                let heads = (pc.cols as u64) / hs.max(1);
                check("r", pc.in_off, pc.cols as u64, work, "work");
                check("k", pc.weight_off, pc.cols as u64, work, "work");
                check("v", pc.bias_off, pc.cols as u64, work, "work");
                check("w", pc.gate_off, pc.cols as u64, work, "work");
                check("a", pc.residual_off, pc.cols as u64, work, "work");
                check("k_k", pc.extra0, pc.cols as u64, weights, "weights");
                check("k_a", pc.extra1, pc.cols as u64, weights, "weights");
                check("memory", pc.extra2, heads * hs * hs, state, "state");
                check("output", pc.out_off, pc.cols as u64, work, "work");
            }
            "rosa_update" | "rosa_read" => {
                check("memory", pc.in_off, pc.cols as u64, state, "state");
                check("direction", pc.weight_off, pc.cols as u64, weights, "weights");
                check("gate", pc.gate_off, pc.cols as u64, work, "work");
                check("output", pc.out_off, pc.cols as u64, if kernel == "rosa_read" { work } else { state }, "work/state");
                if kernel == "rosa_update" {
                    check("decay", pc.bias_off, pc.cols as u64, weights, "weights");
                    check("write", pc.residual_off, pc.cols as u64, work, "work");
                }
            }
            "ffn_combine" => {
                check("ffn_in", pc.in_off, pc.k as u64, work, "work");
                check("ffn_gate", pc.weight_off, pc.k as u64, work, "work");
                check("ffn_out", pc.bias_off, (pc.cols as u64) * (pc.k as u64), weights, "weights");
                check("residual", pc.gate_off, pc.cols as u64, work, "work");
                check("output", pc.out_off, pc.cols as u64, work, "work");
            }
            "noop" => {}
            "copy" => {
                check("input", pc.in_off, pc.cols as u64, work, "work");
                check("output", pc.out_off, pc.cols as u64, work, "work");
            }
            other => panic!("unknown kernel {other}"),
        }
    }

    fn barrier(&self, cmd: vk::CommandBuffer) {
        self.context.record_barrier(cmd);
    }

    fn layernorm(
        &mut self,
        cmd: vk::CommandBuffer,
        input: u32,
        output: u32,
        weight: u32,
        bias: u32,
        mirror_state: Option<u32>,
    ) {
        let mut reduce = PushConsts::new();
        reduce.in_off = input;
        reduce.out_off = self.work.reduce;
        reduce.cols = self.config.d_model as u32;
        self.dispatch(cmd, "layernorm_reduce", &reduce, 1);
        self.barrier(cmd);

        let mut apply = PushConsts::new();
        apply.in_off = input;
        apply.out_off = output;
        apply.weight_off = weight;
        apply.bias_off = bias;
        apply.gate_off = self.work.reduce;
        apply.residual_off = self.work.reduce + 1;
        apply.cols = self.config.d_model as u32;
        apply.eps = 1e-5;
        apply.extra0 = mirror_state.unwrap_or(PushConsts::NONE);
        let groups = div_ceil(self.config.d_model as u32, super::WORKGROUP);
        self.dispatch(cmd, "layernorm_apply", &apply, groups);
        self.barrier(cmd);
    }

    fn linear(
        &mut self,
        cmd: vk::CommandBuffer,
        input: u32,
        output: u32,
        weight: u32,
        bias: Option<u32>,
        cols: u32,
        k: u32,
        flags: u32,
        gate: Option<u32>,
        residual: Option<u32>,
    ) {
        let mut pc = PushConsts::new();
        pc.in_off = input;
        pc.out_off = output;
        pc.weight_off = weight;
        pc.bias_off = bias.unwrap_or(PushConsts::NONE);
        pc.gate_off = gate.unwrap_or(PushConsts::NONE);
        pc.residual_off = residual.unwrap_or(PushConsts::NONE);
        pc.cols = cols;
        pc.k = k;
        pc.flags = flags;
        let groups_x = div_ceil(cols, 8);
        self.dispatch(cmd, "linear", &pc, groups_x);
        self.barrier(cmd);
    }

    fn record_step(&mut self) -> Result<()> {
        let cmd = self.context.begin_step_record()?;
        let d = self.config.d_model as u32;
        let lr = self.config.low_rank() as u32;
        let r = self.config.rosa_size as u32;
        let vocab = self.config.vocab_size as u32;
        let head_total = (self.config.n_heads * self.config.head_size) as u32;

        // Embedding lookup.
        let mut embed = PushConsts::new();
        embed.token_off = self.work.token;
        embed.weight_off = self.weights.token_embedding;
        embed.out_off = self.work.hidden;
        embed.cols = d;
        self.dispatch(cmd, "embed", &embed, div_ceil(d, super::WORKGROUP));
        self.barrier(cmd);

        for layer in 0..self.config.n_layers {
            let lw = self.weights.layers[layer].clone();
            let state_l = self.state.layer_base(layer);
            let prev_off = self.work.prev_base + (layer as u32) * d;
            let mem_off = state_l.timemix_memory;
            let rosa_off = state_l.rosa_memory;

            // LN0.
            self.layernorm(
                cmd,
                self.work.hidden,
                self.work.ln0,
                lw.norm_time_w,
                lw.norm_time_b,
                None,
            );

            // Token mixing.
            let mut mix = PushConsts::new();
            mix.in_off = self.work.ln0;
            mix.out_off = self.work.mix;
            mix.weight_off = lw.mix;
            mix.gate_off = prev_off;
            mix.cols = d;
            self.dispatch(cmd, "mix_inputs", &mix, div_ceil(d, super::WORKGROUP));
            self.barrier(cmd);

            // Projections.
            self.linear(
                cmd,
                self.work.mix,
                self.work.r,
                lw.receptance,
                None,
                d,
                d,
                0,
                None,
                None,
            );
            self.linear(
                cmd,
                self.work.mix + 2 * d,
                self.work.k,
                lw.key,
                None,
                d,
                d,
                0,
                None,
                None,
            );
            self.linear(
                cmd,
                self.work.mix + 3 * d,
                self.work.v,
                lw.value,
                None,
                d,
                d,
                0,
                None,
                None,
            );
            self.linear(
                cmd,
                self.work.mix + d,
                self.work.t_w,
                lw.w1,
                None,
                lr,
                d,
                FLAG_TANH | FLAG_TRANSPOSE_W,
                None,
                None,
            );
            self.linear(
                cmd,
                self.work.t_w,
                self.work.w,
                lw.w2,
                Some(lw.w0),
                d,
                lr,
                FLAG_HAS_BIAS | FLAG_TRANSPOSE_W,
                None,
                None,
            );
            self.linear(
                cmd,
                self.work.mix + 4 * d,
                self.work.t_a,
                lw.a1,
                None,
                lr,
                d,
                FLAG_TRANSPOSE_W,
                None,
                None,
            );
            self.linear(
                cmd,
                self.work.t_a,
                self.work.a,
                lw.a2,
                Some(lw.a0),
                d,
                lr,
                FLAG_HAS_BIAS | FLAG_SIGMOID | FLAG_TRANSPOSE_W,
                None,
                None,
            );
            self.linear(
                cmd,
                self.work.mix + 5 * d,
                self.work.t_g,
                lw.g1,
                None,
                lr,
                d,
                FLAG_TRANSPOSE_W,
                None,
                None,
            );
            self.linear(
                cmd,
                self.work.t_g,
                self.work.g,
                lw.g2,
                None,
                d,
                lr,
                FLAG_SIGMOID | FLAG_TRANSPOSE_W,
                None,
                None,
            );

            // RWKV-7 style recurrence.
            let mut tm = PushConsts::new();
            tm.in_off = self.work.r;
            tm.out_off = self.work.o;
            tm.weight_off = self.work.k;
            tm.bias_off = self.work.v;
            tm.gate_off = self.work.w;
            tm.residual_off = self.work.a;
            tm.cols = head_total;
            tm.k = self.config.head_size as u32;
            tm.extra0 = lw.k_k;
            tm.extra1 = lw.k_a;
            tm.extra2 = mem_off;
            self.dispatch(cmd, "timemix_recurrence", &tm, div_ceil(head_total, super::WORKGROUP));
            self.barrier(cmd);

            // Output norm, gate, projection + residual.
            self.layernorm(cmd, self.work.o, self.work.ln_o, lw.out_norm_w, lw.out_norm_b, None);
            self.linear(
                cmd,
                self.work.ln_o,
                self.work.hidden,
                lw.output,
                None,
                d,
                d,
                FLAG_USE_GATE | FLAG_ADD_RESIDUAL,
                Some(self.work.g),
                Some(self.work.hidden),
            );

            // ROSA branch.
            self.layernorm(
                cmd,
                self.work.hidden,
                self.work.ln1,
                lw.norm_rosa_w,
                lw.norm_rosa_b,
                None,
            );
            self.linear(
                cmd,
                self.work.ln1,
                self.work.cand,
                lw.rosa_input_w,
                Some(lw.rosa_input_b),
                r,
                d,
                FLAG_HAS_BIAS,
                None,
                None,
            );
            self.linear(
                cmd,
                self.work.ln1,
                self.work.write,
                lw.rosa_write_w,
                Some(lw.rosa_write_b),
                r,
                d,
                FLAG_HAS_BIAS,
                None,
                None,
            );
            self.linear(
                cmd,
                self.work.ln1,
                self.work.read_gate,
                lw.rosa_read_w,
                Some(lw.rosa_read_b),
                r,
                d,
                FLAG_HAS_BIAS,
                None,
                None,
            );

            let mut rosa_update = PushConsts::new();
            rosa_update.in_off = rosa_off;
            rosa_update.out_off = rosa_off;
            rosa_update.weight_off = lw.rosa_householder;
            rosa_update.bias_off = lw.rosa_decay;
            rosa_update.gate_off = self.work.cand;
            rosa_update.residual_off = self.work.write;
            rosa_update.cols = r;
            self.dispatch(cmd, "rosa_update", &rosa_update, 1);
            self.barrier(cmd);

            let mut rosa_read = PushConsts::new();
            rosa_read.in_off = rosa_off;
            rosa_read.out_off = self.work.read;
            rosa_read.weight_off = lw.rosa_householder;
            rosa_read.gate_off = self.work.read_gate;
            rosa_read.cols = r;
            self.dispatch(cmd, "rosa_read", &rosa_read, 1);
            self.barrier(cmd);

            self.linear(
                cmd,
                self.work.read,
                self.work.hidden,
                lw.rosa_output,
                None,
                d,
                r,
                FLAG_ADD_RESIDUAL,
                None,
                Some(self.work.hidden),
            );

            // FFN.
            self.layernorm(cmd, self.work.hidden, self.work.ln2, lw.norm_ffn_w, lw.norm_ffn_b, None);
            self.linear(
                cmd,
                self.work.ln2,
                self.work.fin,
                lw.ffn_in,
                None,
                4 * d,
                d,
                0,
                None,
                None,
            );
            self.linear(
                cmd,
                self.work.ln2,
                self.work.fgate,
                lw.ffn_gate,
                None,
                4 * d,
                d,
                0,
                None,
                None,
            );

            let mut ffn = PushConsts::new();
            ffn.in_off = self.work.fin;
            ffn.out_off = self.work.hidden;
            ffn.weight_off = self.work.fgate;
            ffn.bias_off = lw.ffn_out;
            ffn.gate_off = self.work.hidden;
            ffn.cols = d;
            ffn.k = 4 * d;
            self.dispatch(cmd, "ffn_combine", &ffn, div_ceil(d, super::WORKGROUP));
            self.barrier(cmd);

            // Persist this token's LN0 output as the `previous` input for the
            // next step.  This must happen after `mix_inputs` has already read
            // the previous value.
            let mut copy_prev = PushConsts::new();
            copy_prev.in_off = self.work.ln0;
            copy_prev.out_off = prev_off;
            copy_prev.cols = d;
            self.dispatch(cmd, "copy", &copy_prev, div_ceil(d, super::WORKGROUP));
            self.barrier(cmd);
        }

        // Final norm + LM head.
        self.layernorm(
            cmd,
            self.work.hidden,
            self.work.ln_final,
            self.weights.final_norm_w,
            self.weights.final_norm_b,
            None,
        );
        let mut lm = PushConsts::new();
        lm.in_off = self.work.ln_final;
        lm.out_off = self.work.logits;
        lm.weight_off = self.weights.token_embedding;
        lm.cols = vocab;
        lm.k = d;
        self.dispatch(cmd, "linear", &lm, div_ceil(vocab, 8));

        self.context.end_step_record(cmd)?;
        Ok(())
    }
}

fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

#[derive(Clone, Copy)]
struct WorkLayout {
    hidden: u32,
    logits: u32,
    token: u32,
    reduce: u32,
    ln0: u32,
    mix: u32,
    t_w: u32,
    t_a: u32,
    t_g: u32,
    r: u32,
    k: u32,
    v: u32,
    w: u32,
    a: u32,
    g: u32,
    o: u32,
    ln_o: u32,
    ln1: u32,
    cand: u32,
    write: u32,
    read_gate: u32,
    read: u32,
    ln2: u32,
    fin: u32,
    fgate: u32,
    ln_final: u32,
    prev_base: u32,
    size: u32,
}

impl WorkLayout {
    fn build(config: &ModelConfig) -> Self {
        let d = config.d_model as u32;
        let lr = config.low_rank() as u32;
        let r = config.rosa_size as u32;
        let vocab = config.vocab_size as u32;
        let mut next = 0u32;
        let alloc = |count: u32, next: &mut u32| {
            let offset = *next;
            *next += count;
            offset
        };
        let hidden = alloc(d, &mut next);
        let logits = alloc(vocab, &mut next);
        let token = alloc(1, &mut next);
        let reduce = alloc(2, &mut next);
        let ln0 = alloc(d, &mut next);
        let mix = alloc(6 * d, &mut next);
        let t_w = alloc(lr, &mut next);
        let t_a = alloc(lr, &mut next);
        let t_g = alloc(lr, &mut next);
        let r_ = alloc(d, &mut next);
        let k = alloc(d, &mut next);
        let v = alloc(d, &mut next);
        let w = alloc(d, &mut next);
        let a = alloc(d, &mut next);
        let g = alloc(d, &mut next);
        let o = alloc(d, &mut next);
        let ln_o = alloc(d, &mut next);
        let ln1 = alloc(d, &mut next);
        let cand = alloc(r, &mut next);
        let write = alloc(r, &mut next);
        let read_gate = alloc(r, &mut next);
        let read = alloc(r, &mut next);
        let ln2 = alloc(d, &mut next);
        let fin = alloc(4 * d, &mut next);
        let fgate = alloc(4 * d, &mut next);
        let ln_final = alloc(d, &mut next);
        let prev_base = next;
        next += config.n_layers as u32 * d;
        Self {
            hidden,
            logits,
            token,
            reduce,
            ln0,
            mix,
            t_w,
            t_a,
            t_g,
            r: r_,
            k,
            v,
            w,
            a,
            g,
            o,
            ln_o,
            ln1,
            cand,
            write,
            read_gate,
            read,
            ln2,
            fin,
            fgate,
            ln_final,
            prev_base,
            size: next,
        }
    }
}

#[derive(Clone, Copy)]
struct StateLayout {
    layer_size: u32,
    timemix_memory: u32,
    rosa_memory: u32,
    size: u32,
}

impl StateLayout {
    fn build(config: &ModelConfig) -> Self {
        let memory_size = (config.n_heads * config.head_size * config.head_size) as u32;
        let r = config.rosa_size as u32;
        let timemix_memory = 0u32;
        let rosa_memory = timemix_memory + memory_size;
        let layer_size = rosa_memory + r;
        Self {
            layer_size,
            timemix_memory,
            rosa_memory,
            size: layer_size * config.n_layers as u32,
        }
    }

    fn layer_base(&self, layer: usize) -> LayerState {
        let base = (layer as u32) * self.layer_size;
        LayerState {
            timemix_memory: base + self.timemix_memory,
            rosa_memory: base + self.rosa_memory,
        }
    }
}

#[derive(Clone, Copy)]
struct LayerState {
    timemix_memory: u32,
    rosa_memory: u32,
}

#[derive(Clone)]
struct LayerWeights {
    norm_time_w: u32,
    norm_time_b: u32,
    mix: u32,
    w0: u32,
    w1: u32,
    w2: u32,
    a0: u32,
    a1: u32,
    a2: u32,
    g1: u32,
    g2: u32,
    k_k: u32,
    k_a: u32,
    receptance: u32,
    key: u32,
    value: u32,
    output: u32,
    out_norm_w: u32,
    out_norm_b: u32,
    norm_rosa_w: u32,
    norm_rosa_b: u32,
    rosa_input_w: u32,
    rosa_input_b: u32,
    rosa_write_w: u32,
    rosa_write_b: u32,
    rosa_read_w: u32,
    rosa_read_b: u32,
    rosa_householder: u32,
    rosa_decay: u32,
    rosa_output: u32,
    norm_ffn_w: u32,
    norm_ffn_b: u32,
    ffn_in: u32,
    ffn_gate: u32,
    ffn_out: u32,
}

struct WeightRefs {
    token_embedding: u32,
    final_norm_w: u32,
    final_norm_b: u32,
    layers: Vec<LayerWeights>,
}

impl WeightRefs {
    fn resolve(checkpoint: &Checkpoint, config: &ModelConfig) -> Result<Self> {
        let tensor_offset = |name: &str| -> Result<u32> { checkpoint.tensor_offset(name) };
        let mut layers = Vec::with_capacity(config.n_layers);
        for layer in 0..config.n_layers {
            let name = |suffix: &str| format!("blocks.{layer}.{suffix}");
            layers.push(LayerWeights {
                norm_time_w: tensor_offset(&name("norm_time.weight"))?,
                norm_time_b: tensor_offset(&name("norm_time.bias"))?,
                mix: tensor_offset(&name("time_mix.mix_r"))?,
                w0: tensor_offset(&name("time_mix.w0"))?,
                w1: tensor_offset(&name("time_mix.w1"))?,
                w2: tensor_offset(&name("time_mix.w2"))?,
                a0: tensor_offset(&name("time_mix.a0"))?,
                a1: tensor_offset(&name("time_mix.a1"))?,
                a2: tensor_offset(&name("time_mix.a2"))?,
                g1: tensor_offset(&name("time_mix.g1"))?,
                g2: tensor_offset(&name("time_mix.g2"))?,
                k_k: tensor_offset(&name("time_mix.k_k"))?,
                k_a: tensor_offset(&name("time_mix.k_a"))?,
                receptance: tensor_offset(&name("time_mix.receptance.weight"))?,
                key: tensor_offset(&name("time_mix.key.weight"))?,
                value: tensor_offset(&name("time_mix.value.weight"))?,
                output: tensor_offset(&name("time_mix.output.weight"))?,
                out_norm_w: tensor_offset(&name("time_mix.out_norm.weight"))?,
                out_norm_b: tensor_offset(&name("time_mix.out_norm.bias"))?,
                norm_rosa_w: tensor_offset(&name("norm_rosa.weight"))?,
                norm_rosa_b: tensor_offset(&name("norm_rosa.bias"))?,
                rosa_input_w: tensor_offset(&name("rosa.input.weight"))?,
                rosa_input_b: tensor_offset(&name("rosa.input.bias"))?,
                rosa_write_w: tensor_offset(&name("rosa.write_gate.weight"))?,
                rosa_write_b: tensor_offset(&name("rosa.write_gate.bias"))?,
                rosa_read_w: tensor_offset(&name("rosa.read_gate.weight"))?,
                rosa_read_b: tensor_offset(&name("rosa.read_gate.bias"))?,
                rosa_householder: tensor_offset(&name("rosa.householder"))?,
                rosa_decay: tensor_offset(&name("rosa.decay"))?,
                rosa_output: tensor_offset(&name("rosa.output.weight"))?,
                norm_ffn_w: tensor_offset(&name("norm_ffn.weight"))?,
                norm_ffn_b: tensor_offset(&name("norm_ffn.bias"))?,
                ffn_in: tensor_offset(&name("ffn_in.weight"))?,
                ffn_gate: tensor_offset(&name("ffn_gate.weight"))?,
                ffn_out: tensor_offset(&name("ffn_out.weight"))?,
            });
        }
        Ok(Self {
            token_embedding: tensor_offset("token_embedding.weight")?,
            final_norm_w: tensor_offset("final_norm.weight")?,
            final_norm_b: tensor_offset("final_norm.bias")?,
            layers,
        })
    }
}
