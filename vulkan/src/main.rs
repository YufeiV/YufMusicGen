use anyhow::{Result, bail};
use clap::Parser;

/// Switch the Windows console to UTF-8 (code page 65001) so Chinese/ASCII
/// progress output is displayed correctly and never trips the "non-UTF-8
/// byte sequence" panic in console mode.  No-op on other platforms.
#[cfg(windows)]
fn setup_windows_console() {
    extern "system" {
        fn SetConsoleOutputCP(w_code_page_id: u32) -> i32;
        fn SetConsoleCP(w_code_page_id: u32) -> i32;
    }
    const CP_UTF8: u32 = 65001;
    // SAFETY: passing a constant code page to the Win32 console API is safe;
    // failure is ignored because it only affects display, not correctness.
    unsafe {
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    }
}

#[cfg(not(windows))]
fn setup_windows_console() {}

mod cli {
    use std::path::PathBuf;

    #[derive(clap::Parser, Debug)]
    #[command(name = "yufmusicgen-vulkan", version, about = "Vulkan-based YufMusicGen client")]
    pub struct Cli {
        #[command(subcommand)]
        pub command: Command,
    }

    #[derive(clap::Subcommand, Debug)]
    pub enum Command {
        /// Generate a MIDI file from a text prompt (headless).
        Generate(GenerateArgs),
        /// Open the Vulkan piano-roll client window and generate live.
        Gui(GuiArgs),
        /// Print the General MIDI instrument table.
        ListInstruments,
    }

    #[derive(clap::Args, Debug)]
    pub struct GenerateArgs {
        /// Path to an exported .yuf checkpoint.
        #[arg(long)]
        pub checkpoint: PathBuf,
        /// Optional MidiTok tokenizer.json (overrides the embedded one).
        #[arg(long)]
        pub tokenizer: Option<PathBuf>,
        /// Text prompt (e.g. "cinematic piano, slow tempo").
        #[arg(long, default_value = "")]
        pub prompt: String,
        /// Instrument name, GM program number, or "drums".
        #[arg(long)]
        pub instrument: Option<String>,
        /// Mask every Program token except the requested instrument.
        #[arg(long)]
        pub instrument_only: bool,
        /// Condition on an existing MIDI file and continue from it.
        #[arg(long)]
        pub prompt_midi: Option<PathBuf>,
        /// Maximum MidiTok tokens kept from --prompt-midi (keeps the tail).
        #[arg(long, default_value_t = 512)]
        pub prompt_max_tokens: usize,
        /// Output .mid path.
        #[arg(long, default_value = "outputs/generated.mid")]
        pub output: PathBuf,
        /// Number of MIDI tokens to generate.
        #[arg(long)]
        pub steps: Option<usize>,
        /// Approximate target length in seconds (~20 tokens/second).
        #[arg(long)]
        pub seconds: Option<f64>,
        /// Sampling temperature (<= 0 uses greedy argmax).
        #[arg(long, default_value_t = 1.0)]
        pub temperature: f32,
        /// Nucleus sampling threshold.
        #[arg(long, default_value_t = 0.95)]
        pub top_p: f32,
        /// Random seed.
        #[arg(long, default_value_t = 1337)]
        pub seed: u64,
    }

    #[derive(clap::Args, Debug)]
    pub struct GuiArgs {
        #[arg(long)]
        pub checkpoint: PathBuf,
        /// Optional MidiTok tokenizer.json (overrides the embedded one).
        #[arg(long)]
        pub tokenizer: Option<PathBuf>,
        #[arg(long, default_value = "")]
        pub prompt: String,
        #[arg(long)]
        pub instrument: Option<String>,
        #[arg(long)]
        pub instrument_only: bool,
        #[arg(long)]
        pub prompt_midi: Option<PathBuf>,
        #[arg(long, default_value_t = 512)]
        pub prompt_max_tokens: usize,
        #[arg(long)]
        pub steps: Option<usize>,
        #[arg(long)]
        pub seconds: Option<f64>,
        #[arg(long, default_value_t = 1.0)]
        pub temperature: f32,
        #[arg(long, default_value_t = 0.95)]
        pub top_p: f32,
        #[arg(long, default_value_t = 1337)]
        pub seed: u64,
    }
}

fn main() -> Result<()> {
    setup_windows_console();
    let cli = cli::Cli::parse();
    match cli.command {
        cli::Command::Generate(args) => run_generate(args),
        cli::Command::ListInstruments => {
            yufmusicgen_vulkan::generation::list_instruments();
            Ok(())
        }
        cli::Command::Gui(args) => {
            let gui_args = yufmusicgen_vulkan::gui::GuiArgs {
                checkpoint: args.checkpoint,
                tokenizer: args.tokenizer,
                prompt: args.prompt,
                instrument: args.instrument,
                instrument_only: args.instrument_only,
                prompt_midi: args.prompt_midi,
                prompt_max_tokens: args.prompt_max_tokens,
                steps: args.steps,
                seconds: args.seconds,
                temperature: args.temperature,
                top_p: args.top_p,
                seed: args.seed,
            };
            yufmusicgen_vulkan::gui::run(gui_args)?;
            Ok(())
        }
    }
}

fn run_generate(args: cli::GenerateArgs) -> Result<()> {
    let params = yufmusicgen_vulkan::generation::GenerateParams {
        checkpoint: args.checkpoint,
        tokenizer: args.tokenizer,
        prompt: args.prompt,
        instrument: args.instrument,
        instrument_only: args.instrument_only,
        prompt_midi: args.prompt_midi,
        prompt_max_tokens: args.prompt_max_tokens,
        output: args.output,
        steps: args.steps,
        seconds: args.seconds,
        temperature: args.temperature,
        top_p: args.top_p,
        seed: args.seed,
    };
    if params.steps.is_none() && params.seconds.is_none() {
        eprintln!("[warn] no --steps or --seconds given; generating 512 tokens");
    }
    let mut last_status = String::new();
    let info = yufmusicgen_vulkan::generation::run_generation(&params, |fraction, status| {
        let percent = (fraction * 100.0) as u32;
        if status != last_status {
            last_status = status.to_string();
            eprintln!("[{percent:3}%] {status}");
        }
    })?;
    println!(
        "generated {} midi tokens (prompt {} + continuation) in {} steps -> {}",
        info.midi_tokens,
        info.prompt_tokens,
        info.steps_done,
        info.output.display()
    );
    println!(
        "tracks: {}, notes: {}, duration: {:.2}s{}",
        info.tracks,
        info.notes,
        info.duration_seconds,
        info.instrument
            .as_deref()
            .map(|value| format!(", instrument: {value}"))
            .unwrap_or_default()
    );
    Ok(())
}

fn _unused() -> Result<()> {
    bail!("unused")
}
