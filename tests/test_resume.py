import json
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yufmusicgen.config import ModelConfig, TrainConfig
from yufmusicgen.data import ResumableShuffleSampler, TokenSequenceDataset
from yufmusicgen.training import train


def _full_epoch_stream(num_samples, batch_size, seed, epochs):
    """All permutations (including the drop_last tail) for ``epochs`` epochs."""

    sampler = ResumableShuffleSampler(list(range(num_samples)), batch_size, seed)
    stream = []
    for _ in range(epochs):
        stream.extend(iter(sampler))
    return stream


def test_resume_continues_the_same_permutation_stream():
    num_samples, batch_size, seed, epochs = 17, 3, 1234, 6
    batches_per_epoch = num_samples // batch_size
    per_epoch = batches_per_epoch * batch_size
    full = _full_epoch_stream(num_samples, batch_size, seed, epochs)

    for samples_seen in (
        0,
        per_epoch,
        2 * per_epoch,
        2 * per_epoch + 6,
        3 * per_epoch + 9,
        4 * per_epoch + 12,
        5 * per_epoch,
    ):
        epochs_skipped, offset = divmod(samples_seen, per_epoch)
        sampler = ResumableShuffleSampler(
            list(range(num_samples)), batch_size, seed, samples_seen
        )
        resumed = []
        for _ in range(epochs - epochs_skipped):
            resumed.extend(iter(sampler))
        expected = full[epochs_skipped * num_samples + offset :]
        assert resumed == expected, f"samples_seen={samples_seen}"


def test_resume_at_non_batch_boundary_replays_remainder(capsys):
    num_samples, batch_size, seed = 17, 3, 1234
    epochs = 2
    full = _full_epoch_stream(num_samples, batch_size, seed, epochs)
    consumed_per_epoch = (num_samples // batch_size) * batch_size
    consumed = [
        full[epoch * num_samples + i]
        for epoch in range(epochs)
        for i in range(consumed_per_epoch)
    ]

    # 16 is not a multiple of batch_size=3; the sampler rounds down to the
    # batch boundary (15) and the 16th consumed sample is replayed.
    sampler = ResumableShuffleSampler(
        list(range(num_samples)), batch_size, seed, samples_seen=16
    )
    resumed = list(iter(sampler))
    assert resumed[0] == consumed[15]
    assert "replaying 1 sample(s)" in capsys.readouterr().out


def _write_manifest(tmp_path, token_arrays):
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    lines = []
    for index, tokens in enumerate(token_arrays):
        path = tokens_dir / f"{index:08d}.npy"
        np.save(path, tokens)
        lines.append(
            json.dumps(
                {"id": str(index), "tokens": str(path.relative_to(tmp_path))}
            )
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_crops_are_deterministic_per_record_and_epoch(tmp_path):
    sequence_length = 64
    long_track = np.arange(500, dtype=np.int64)
    manifest = _write_manifest(tmp_path, [long_track])
    dataset = TokenSequenceDataset(
        manifest, sequence_length, random_crop=True, seed=42
    )

    crop = dataset[(0, 0)]["input_ids"]
    assert torch.equal(dataset[(0, 0)]["input_ids"], crop)
    # Different epochs get different windows of the same long track.
    windows = {tuple(dataset[(0, epoch)]["input_ids"].tolist()) for epoch in range(8)}
    assert len(windows) >= 2
    # Plain integer indexing is still supported (epoch 0).
    assert torch.equal(dataset[0]["input_ids"], dataset[(0, 0)]["input_ids"])


def _tiny_train_config(max_steps, **overrides):
    values = dict(
        sequence_length=64,
        batch_size=2,
        grad_accumulation=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=0,
        max_steps=max_steps,
        log_every=1,
        save_every=2,
        grad_clip=1.0,
        seed=1337,
        amp=False,
        amp_dtype="float16",
        num_workers=0,
        prefetch_factor=4,
    )
    values.update(overrides)
    return TrainConfig(**values)


def _tiny_model_config():
    return ModelConfig(
        vocab_size=300,
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_size=16,
        rosa_size=12,
        dropout=0.0,
        use_cuda_kernel=False,
        use_rosa_scan=False,
    )


def _parse_losses(output):
    losses = {}
    pattern = re.compile(r"pretrain step=(\d+)/(\d+) loss=([0-9.eE+-]+)")
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            losses[int(match.group(1))] = float(match.group(3))
    return losses


def test_resume_training_reproduces_original_losses(tmp_path, capsys):
    """A resumed run sees the exact same batches (and losses) as the original."""

    rng = np.random.default_rng(0)
    token_arrays = [
        rng.integers(0, 300, size=int(size), dtype=np.int64)
        for size in rng.integers(200, 500, size=7)
    ]
    manifest = _write_manifest(tmp_path, token_arrays)
    model_config = _tiny_model_config()

    # Original run: 5 steps, checkpoints at steps 2, 4 and 5.
    run1_dir = tmp_path / "run1"
    train(
        manifest,
        tmp_path,
        run1_dir,
        model_config,
        _tiny_train_config(5),
        "pretrain",
        device_name="cpu",
    )
    run1_losses = _parse_losses(capsys.readouterr().out)
    assert set(run1_losses) == {1, 2, 3, 4, 5}
    step2 = run1_dir / "pretrain-step-00000002.pt"
    assert step2.exists()

    # Resumed run from the step-2 checkpoint must reproduce steps 3-5 exactly.
    run2_dir = tmp_path / "run2"
    train(
        manifest,
        tmp_path,
        run2_dir,
        model_config,
        _tiny_train_config(3),
        "pretrain",
        device_name="cpu",
        init_checkpoint=step2,
    )
    run2_losses = _parse_losses(capsys.readouterr().out)
    # The log prints local steps (relative to the resumed run), so local step L
    # maps to global step 2 + L in the original run.
    for local_step in (1, 2, 3):
        assert run2_losses[local_step] == pytest.approx(
            run1_losses[2 + local_step], abs=1e-6
        )

    # Old-format checkpoints (no samples_seen) fall back to step*batch*accum.
    legacy = torch.load(step2, map_location="cpu")
    legacy.pop("samples_seen", None)
    legacy_step2 = tmp_path / "legacy-step-00000002.pt"
    torch.save(legacy, legacy_step2)
    run3_dir = tmp_path / "run3"
    train(
        manifest,
        tmp_path,
        run3_dir,
        model_config,
        _tiny_train_config(3),
        "pretrain",
        device_name="cpu",
        init_checkpoint=legacy_step2,
    )
    run3_losses = _parse_losses(capsys.readouterr().out)
    for local_step in (1, 2, 3):
        assert run3_losses[local_step] == pytest.approx(
            run1_losses[2 + local_step], abs=1e-6
        )

    # A different phase (post-train) starts a fresh data stream even though the
    # checkpoint carries a large samples_seen, so two post-train runs that both
    # init from the same pretrain checkpoint behave identically.
    post_run_a = tmp_path / "post-a"
    train(
        manifest,
        tmp_path,
        post_run_a,
        model_config,
        _tiny_train_config(2),
        "posttrain",
        device_name="cpu",
        init_checkpoint=step2,
        supervised=True,
    )
    output_a = capsys.readouterr().out
    assert "continuing data stream at sample" not in output_a
    post_run_b = tmp_path / "post-b"
    train(
        manifest,
        tmp_path,
        post_run_b,
        model_config,
        _tiny_train_config(2),
        "posttrain",
        device_name="cpu",
        init_checkpoint=step2,
        supervised=True,
    )
    output_b = capsys.readouterr().out
    losses_a = re.findall(r"loss=([0-9.eE+-]+)", output_a)
    losses_b = re.findall(r"loss=([0-9.eE+-]+)", output_b)
    assert losses_a == losses_b
