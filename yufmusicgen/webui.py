"""Gradio WebUI for YufMusicGen MIDI generation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
from symusic import Score

from .cli.generate import GenerationArgs, run_generation
from .instruments import GM_PROGRAMS
from .midi_io import read_midi


def _find_checkpoints() -> list[str]:
    root = Path("checkpoints")
    if not root.is_dir():
        return []
    return sorted(str(path) for path in root.rglob("*.pt"))


def _instrument_choices() -> list[str]:
    return [
        "auto",
        *(f"{index}: {name}" for index, name in enumerate(GM_PROGRAMS)),
        "-1: Drums",
    ]


def render_piano_roll(score: Score, path: str | Path) -> Path:
    """Render a piano roll PNG for a score (used as the preview image)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11, 6))
    notes = [note for track in score.tracks for note in track.notes]
    if not notes:
        axis.text(
            0.5,
            0.5,
            "no notes decoded",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=14,
        )
    else:
        ticks_per_quarter = score.ticks_per_quarter or 480
        palette = plt.cm.tab10.colors
        for index, track in enumerate(score.tracks):
            if not track.notes:
                continue
            color = palette[index % len(palette)]
            for note in track.notes:
                axis.barh(
                    note.pitch,
                    (note.end - note.time) / ticks_per_quarter,
                    left=note.time / ticks_per_quarter,
                    height=0.85,
                    color=color,
                    alpha=0.85,
                )
            name = track.name or f"program {track.program}"
            axis.plot([], [], color=color, label=name)
        axis.legend(loc="upper right", fontsize=8)
    axis.set_xlabel("time (beats)")
    axis.set_ylabel("pitch")
    axis.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _generate(
    checkpoint_dropdown: str,
    checkpoint_path: str,
    prompt: str,
    instrument: str,
    instrument_only: bool,
    prompt_midi: str | None,
    prompt_max_tokens: int,
    steps: int | None,
    seconds: float | None,
    temperature: float,
    top_p: float,
    seed: int,
    device: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, str | None, str]:
    checkpoint = (checkpoint_path or "").strip() or checkpoint_dropdown
    if not checkpoint:
        raise gr.Error("请选择或输入一个 checkpoint")
    if not Path(checkpoint).is_file():
        raise gr.Error(f"checkpoint 不存在: {checkpoint}")

    instrument_arg = None
    if instrument != "auto":
        instrument_arg = instrument.split(":", 1)[0].strip()

    output_dir = Path("outputs/webui")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output_dir / f"generated_{stamp}.mid"

    params = GenerationArgs(
        checkpoint=checkpoint,
        prompt=prompt,
        instrument=instrument_arg,
        instrument_only=instrument_only,
        prompt_midi=prompt_midi,
        prompt_max_tokens=prompt_max_tokens,
        output=str(output),
        steps=steps or None,
        seconds=seconds or None,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        device=device,
    )
    output_path, info = run_generation(params, progress=lambda f, s: progress(f, desc=s))
    score = read_midi(output_path)
    preview = output_dir / f"preview_{stamp}.png"
    render_piano_roll(score, preview)

    instrument_text = info["instrument"] or "auto"
    summary = (
        f"生成完成：{info['midi_tokens']} 个生成 token + "
        f"{info['prompt_tokens']} 个 prompt token，"
        f"{info['tracks']} 条轨道 / {info['notes']} 个音符，"
        f"时长约 {info['duration_seconds']:.1f}s，乐器 {instrument_text}\n"
        f"输出文件：{output_path}"
    )
    return str(output_path), str(preview), summary


def build_app() -> gr.Blocks:
    checkpoints = _find_checkpoints()
    with gr.Blocks(title="YufMusicGen WebUI") as demo:
        gr.Markdown(
            "# YufMusicGen MIDI 生成\n"
            "基于 RWKV-7 + MidiTok 的 MIDI 生成 WebUI。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                checkpoint_dropdown = gr.Dropdown(
                    label="checkpoint（自动扫描 checkpoints/）",
                    choices=checkpoints,
                    value=checkpoints[0] if checkpoints else None,
                )
                checkpoint_path = gr.Textbox(
                    label="或手动输入 checkpoint 路径",
                    placeholder="checkpoints/posttrain/posttrain-step-00013000.pt",
                )
                refresh_button = gr.Button("刷新 checkpoint 列表")
                prompt = gr.Textbox(
                    label="文本提示词",
                    placeholder="cinematic piano, slow tempo, emotional strings",
                )
                instrument = gr.Dropdown(
                    label="指定乐器",
                    choices=_instrument_choices(),
                    value="auto",
                )
                instrument_only = gr.Checkbox(
                    label="仅使用该乐器（屏蔽其他 Program token）",
                    value=False,
                )
                prompt_midi = gr.File(
                    label="Prompt MIDI（可选，作为续写条件）",
                    file_types=[".mid", ".midi", ".smf"],
                    type="filepath",
                )
                prompt_max_tokens = gr.Number(
                    label="Prompt 保留 token 数", value=512, precision=0, minimum=16
                )
            with gr.Column(scale=1):
                steps = gr.Slider(
                    label="生成 token 数（优先于秒数）",
                    minimum=16,
                    maximum=4096,
                    value=512,
                    step=16,
                )
                seconds = gr.Number(
                    label="目标时长（秒，约 20 token/s）", value=None, minimum=1
                )
                temperature = gr.Slider(
                    label="temperature", minimum=0.0, maximum=2.0, value=1.0, step=0.05
                )
                top_p = gr.Slider(
                    label="top-p", minimum=0.0, maximum=1.0, value=0.95, step=0.01
                )
                seed = gr.Number(label="seed", value=1337, precision=0, minimum=0)
                device = gr.Dropdown(
                    label="device",
                    choices=["auto", "cpu", "cuda"],
                    value="auto",
                )
                generate_button = gr.Button("生成", variant="primary")
            with gr.Column(scale=1):
                midi_file = gr.File(label="生成的 MIDI", interactive=False)
                preview = gr.Image(label="钢琴卷帘预览", interactive=False)
                summary = gr.Textbox(label="生成信息", lines=6, interactive=False)

        refresh_button.click(
            fn=lambda: gr.Dropdown(choices=_find_checkpoints()),
            outputs=checkpoint_dropdown,
        )
        generate_button.click(
            fn=_generate,
            inputs=[
                checkpoint_dropdown,
                checkpoint_path,
                prompt,
                instrument,
                instrument_only,
                prompt_midi,
                prompt_max_tokens,
                steps,
                seconds,
                temperature,
                top_p,
                seed,
                device,
            ],
            outputs=[midi_file, preview, summary],
        )
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the YufMusicGen Gradio WebUI")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public share link")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    demo = build_app()
    demo.queue()
    demo.launch(server_name=args.server_name, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
