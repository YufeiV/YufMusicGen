"""MIDI dataset preprocessing entry points."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

from .codec import MidiCodec
from .config import MidiCodecConfig
from .midi_io import midi_duration_seconds, read_midi, truncate_midi
from .tokenizer import MusicTokenizer, TokenSpec


MIDI_SUFFIXES = {".mid", ".midi", ".smf"}


def _iter_midi_paths(root: Path) -> Iterator[Path]:
    """Walk a directory tree with ``os.scandir``, yielding only MIDI files.

    Unlike ``Path.rglob``, this never builds a list of every file in the tree,
    so huge corpora (hundreds of thousands of files) are scanned quickly.
    """

    stack = [root]
    scanned = 0
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    scanned += 1
                    if scanned % 100_000 == 0:
                        print(f"  scanned {scanned:,} entries...", flush=True)
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            if Path(entry.name).suffix.lower() in MIDI_SUFFIXES:
                                yield Path(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue


def _records_from_input(input_path: Path) -> Iterator[tuple[Path, str]]:
    if input_path.is_file() and input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {input_path}:{line_number}") from exc
                source = Path(item.get("midi") or item.get("path") or "")
                if not source.is_absolute():
                    source = input_path.parent / source
                yield source, str(item.get("text") or item.get("caption") or "")
        return

    if input_path.is_file():
        sources: Iterator[Path] = iter((input_path,))
    else:
        sources = iter(sorted(_iter_midi_paths(input_path)))
    for source in sources:
        sidecar = source.with_suffix(".txt")
        text = sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else ""
        yield source, text


_WORKER: dict[str, Any] = {}


def _init_worker(codec_payload: dict[str, Any]) -> None:
    """Build one MidiTok codec per worker process."""

    _WORKER["codec"] = MidiCodec.from_config_dict(codec_payload)


def _encode_file(task: tuple[int, Path, str, float, float | None]) -> dict[str, Any]:
    """Load + tokenize one MIDI file inside a worker process."""

    index, source, text, min_seconds, max_seconds = task
    try:
        score = read_midi(source)
    except Exception as exc:
        return {"index": index, "skip": f"{source}: {exc}"}
    duration = midi_duration_seconds(score)
    if duration < min_seconds:
        return {
            "index": index,
            "skip": f"{source.name}: {duration:.2f}s is shorter than {min_seconds:.2f}s",
        }
    if max_seconds is not None and duration > max_seconds:
        score = truncate_midi(score, max_seconds)
        duration = midi_duration_seconds(score)
        if duration < min_seconds:
            return {
                "index": index,
                "skip": f"{source.name}: truncated to {duration:.2f}s",
            }
    try:
        midi_ids = _WORKER["codec"].encode(score)
    except Exception as exc:
        return {"index": index, "skip": f"{source.name}: {exc}"}
    if not midi_ids:
        return {"index": index, "skip": f"{source.name}: no tokenizable events"}
    return {
        "index": index,
        "source": source,
        "text": text,
        "duration": duration,
        "midi_ids": midi_ids,
    }


def _encode_sequentially(
    records: list[tuple[Path, str]],
    codec: MidiCodec,
    min_seconds: float,
    max_seconds: float | None,
) -> Iterator[dict[str, Any]]:
    _WORKER["codec"] = codec
    for index, (source, text) in enumerate(records):
        yield _encode_file((index, source, text, min_seconds, max_seconds))


def preprocess_dataset(
    input_path: str | Path,
    output_path: str | Path,
    codec_config: MidiCodecConfig | None = None,
    min_seconds: float = 0.5,
    max_seconds: float | None = None,
    overwrite: bool = False,
    workers: int = 1,
) -> Path:
    """Convert a MIDI directory/JSONL into a token dataset.

    When ``codec_config.vocab_size > 0`` a BPE vocabulary is trained on the
    full dataset before tokenizing, so every file benefits from the same
    learned tokenizer.  Tokenization is parallelized across ``workers``
    processes; pass ``workers <= 1`` to disable multiprocessing.
    """

    codec_config = codec_config or MidiCodecConfig()
    codec_config.validate()
    output_path = Path(output_path)
    token_path = output_path / "tokens"
    output_path.mkdir(parents=True, exist_ok=True)
    token_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.jsonl"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite to rebuild")

    records = list(_records_from_input(Path(input_path)))
    if not records:
        raise RuntimeError("no MIDI files found")
    print(f"found {len(records)} MIDI files in {input_path}")

    codec = MidiCodec(codec_config)
    if codec_config.vocab_size > 0:
        print(
            f"training BPE vocabulary (target {codec_config.vocab_size} tokens) "
            f"on {len(records)} MIDI files"
        )
        codec.train_vocab([source for source, _ in records], codec_config.vocab_size)
        print(f"BPE vocabulary ready: {codec.vocab_size} tokens")

    tokenizer = MusicTokenizer(TokenSpec(codec.vocab_size, codec.midi_offset))
    worker_count = max(1, workers)
    print(f"encoding with {worker_count} worker{'s' if worker_count > 1 else ''}")

    if worker_count > 1:
        codec_payload = codec.to_checkpoint_dict()
        tasks = (
            (index, source, text, min_seconds, max_seconds)
            for index, (source, text) in enumerate(records)
        )
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_worker,
                initargs=(codec_payload,),
            ) as executor:
                results = executor.map(_encode_file, tasks, chunksize=64)
                encoded = tqdm(results, total=len(records), desc="Encoding MIDI", unit="file")
                # Executor.map keeps task order, so the manifest stays sorted.
                result_list = list(encoded)
        except Exception as exc:
            print(f"multiprocessing failed ({exc}); falling back to sequential")
            result_list = list(
                tqdm(
                    _encode_sequentially(records, codec, min_seconds, max_seconds),
                    total=len(records),
                    desc="Encoding MIDI",
                    unit="file",
                )
            )
    else:
        result_list = list(
            tqdm(
                _encode_sequentially(records, codec, min_seconds, max_seconds),
                total=len(records),
                desc="Encoding MIDI",
                unit="file",
            )
        )

    manifest: list[dict[str, object]] = []
    for result in result_list:
        if "skip" in result:
            print(f"[skip] {result['skip']}")
            continue
        midi_ids = result["midi_ids"]
        tokens = tokenizer.build_sequence(result["text"], midi_ids)
        name = f"{len(manifest):08d}.npy"
        np.save(token_path / name, tokens, allow_pickle=False)
        manifest.append(
            {
                "id": Path(result["source"]).stem,
                "midi": str(result["source"].resolve()),
                "tokens": str(Path("tokens") / name),
                "text": result["text"],
                "duration": result["duration"],
                "midi_tokens": int(len(midi_ids)),
                "tokens_count": int(tokens.size),
            }
        )

    if not manifest:
        raise RuntimeError("no usable MIDI files found")
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    codec.save(output_path)
    print(f"wrote {len(manifest)} sequences -> {manifest_path}")
    return manifest_path
