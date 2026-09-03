import json
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yufmusicgen.config import ModelConfig, TrainConfig
from yufmusicgen.model import YufMusicGen
from yufmusicgen.peft import (
    LoRALinear,
    MissLinear,
    PeftConfig,
    apply_peft,
    count_parameters,
    merge_adapters,
    unmerge_adapters,
)
from yufmusicgen.training import train


def _tiny_model_config() -> ModelConfig:
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


def _reference(model: YufMusicGen) -> YufMusicGen:
    reference = YufMusicGen(model.config)
    reference.load_state_dict(model.state_dict())
    return reference


@pytest.fixture()
def model():
    torch.manual_seed(7)
    return YufMusicGen(_tiny_model_config())


@pytest.fixture()
def tokens():
    torch.manual_seed(3)
    return torch.randint(0, 300, (2, 10))


def test_lora_zero_adapter_matches_base(model, tokens):
    reference = _reference(model)
    apply_peft(
        model,
        PeftConfig(method="lora", r=4, lora_alpha=16, lora_dropout=0.0),
    )
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_B.data.zero_()
    with torch.no_grad():
        logits, _ = model(tokens)
        expected, _ = reference(tokens)
    assert torch.allclose(logits, expected, atol=1e-5)


def test_lora_merge_roundtrip(model, tokens):
    apply_peft(
        model,
        PeftConfig(method="lora", r=4, lora_alpha=16, lora_dropout=0.0),
    )
    with torch.no_grad():
        before, _ = model(tokens)
    merged = merge_adapters(model)
    assert merged > 0
    with torch.no_grad():
        after, _ = model(tokens)
    assert torch.allclose(before, after, atol=1e-5)
    assert unmerge_adapters(model) == merged
    with torch.no_grad():
        restored, _ = model(tokens)
    assert torch.allclose(before, restored, atol=1e-5)


def test_lora_only_adapter_parameters_are_trainable(model):
    apply_peft(
        model,
        PeftConfig(method="lora", r=4, lora_alpha=16, lora_dropout=0.1),
    )
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    for name in trainable:
        assert name.endswith(".lora_A") or name.endswith(".lora_B")
    r, d = 4, model.config.d_model
    per_projection = r * d + d * r
    assert count_parameters(model) == per_projection * 4 * model.config.n_layers


def test_pissa_zero_step_with_unit_scaling(model, tokens):
    saved = model.state_dict()
    reference = YufMusicGen(model.config)
    reference.load_state_dict(saved)
    model.load_state_dict(saved)
    apply_peft(
        model,
        PeftConfig(
            method="lora",
            r=4,
            lora_alpha=4,
            lora_dropout=0.0,
            pissa_init=True,
        ),
    )
    with torch.no_grad():
        logits, _ = model(tokens)
        expected, _ = reference(tokens)
    # alpha == r means scaling == 1, so the residual base plus the principal
    # SVD adapter reproduces the original projection exactly.
    assert torch.allclose(logits, expected, atol=1e-5)


def test_pissa_reconstruction_matches_svd_reference():
    import numpy as np

    torch.manual_seed(11)
    for shape in [(32, 32), (40, 160), (160, 40)]:
        base = torch.nn.Linear(shape[1], shape[0], bias=False)
        original = base.weight.detach().clone()
        for r in (1, 4, min(shape)):
            lora = LoRALinear(
                base, r=r, alpha=r, dropout=0.0, pissa_init=True
            )
            effective = lora.base.weight.detach() + (
                lora.lora_B @ lora.lora_A
            )
            assert torch.allclose(effective, original, atol=1e-5)

            # Cross-check against an independent numpy SVD.  Individual
            # singular vectors are not unique (sign/rotation ambiguity on
            # near-degenerate values), so compare the unique invariants: the
            # principal reconstruction BA and the residual base weight.
            u, s, vh = np.linalg.svd(original.numpy(), full_matrices=False)
            a_ref = np.diag(np.sqrt(s[:r])) @ vh[:r]
            b_ref = u[:, :r] @ np.diag(np.sqrt(s[:r]))
            ba_ref = torch.from_numpy(b_ref @ a_ref)
            assert torch.allclose(lora.lora_B @ lora.lora_A, ba_ref, atol=1e-5)
            assert torch.allclose(
                lora.base.weight.detach(),
                torch.from_numpy(original.numpy() - b_ref @ a_ref),
                atol=1e-5,
            )
            base.weight.data.copy_(original)


def test_pissa_base_weight_is_frozen(model, tokens):
    apply_peft(
        model,
        PeftConfig(
            method="lora",
            r=4,
            lora_alpha=4,
            lora_dropout=0.0,
            pissa_init=True,
        ),
    )
    logits, _ = model(tokens)
    logits.sum().backward()
    for name, parameter in model.named_parameters():
        if name.endswith(".base.weight"):
            assert parameter.grad is None
        elif name.endswith(".lora_A") or name.endswith(".lora_B"):
            assert parameter.grad is not None
            assert parameter.grad.abs().sum() > 0


def test_pissa_rank_above_matrix_rank_raises(model):
    with pytest.raises(ValueError, match="pissa_init requires r"):
        apply_peft(
            model,
            PeftConfig(
                method="lora",
                r=999,
                lora_alpha=999,
                lora_dropout=0.0,
                pissa_init=True,
            ),
        )


def test_lora_merge_is_exact_in_eval_mode(model, tokens):
    # Dropout is training-only: after merge the adapter path is bypassed, so
    # the comparison must happen with dropout disabled (eval) or dropout=0.
    apply_peft(
        model,
        PeftConfig(
            method="lora",
            r=4,
            lora_alpha=16,
            lora_dropout=0.05,
            pissa_init=True,
        ),
    )
    model.eval()
    with torch.no_grad():
        before, _ = model(tokens)
    merge_adapters(model)
    with torch.no_grad():
        after, _ = model(tokens)
    assert torch.allclose(before, after, atol=1e-5)


def test_miss_zero_block_matches_base_and_merge(model, tokens):
    reference = _reference(model)
    apply_peft(model, PeftConfig(method="miss", r=4, miss_dropout=0.0))
    with torch.no_grad():
        before, _ = model(tokens)
        expected, _ = reference(tokens)
    assert torch.allclose(before, expected, atol=1e-5)
    assert merge_adapters(model) > 0
    with torch.no_grad():
        merged, _ = model(tokens)
    assert torch.allclose(before, merged, atol=1e-5)


def test_miss_mini_mode(model, tokens):
    reference = _reference(model)
    apply_peft(
        model,
        PeftConfig(
            method="miss",
            r=4,
            mini_r=4,
            miss_dropout=0.0,
            init_weights="mini",
        ),
    )
    with torch.no_grad():
        logits, _ = model(tokens)
        expected, _ = reference(tokens)
    assert torch.allclose(logits, expected, atol=1e-5)
    # mini mode stores (r, mini_r) per projection instead of (r, out_features).
    assert (
        count_parameters(model)
        == 4 * 4 * 4 * model.config.n_layers
    )


def test_state_tuning_freezes_base_and_trains_initial_states(model, tokens):
    saved = model.state_dict()
    apply_peft(model, PeftConfig(method="state"))
    trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable
    for name in trainable:
        assert "state_tuning" in name
    # Base weights must be byte-identical to the pre-PEFT checkpoint.
    for name, parameter in model.named_parameters():
        if "state_tuning" not in name:
            assert torch.equal(parameter, saved[name])

    logits, _ = model(tokens)
    total = logits.sum()
    total.backward()
    for name, parameter in model.named_parameters():
        if "state_tuning" in name:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None


def test_state_tuning_learned_initial_state_is_used(model, tokens):
    apply_peft(model, PeftConfig(method="state"))
    with torch.no_grad():
        zero_logits, _ = model(tokens)
    with torch.no_grad():
        for block in model.blocks:
            block.state_tuning.memory.data.fill_(0.5)
            block.state_tuning.rosa_memory.data.fill_(0.5)
        shifted_logits, _ = model(tokens)
    assert not torch.allclose(zero_logits, shifted_logits, atol=1e-5)


def test_peft_config_validation():
    with pytest.raises(ValueError):
        PeftConfig(method="lora", r=0).validate()
    with pytest.raises(ValueError):
        PeftConfig(method="miss", r=5).validate()
    with pytest.raises(ValueError):
        PeftConfig(method="bogus").validate()


def test_apply_peft_with_unknown_targets_raises(model):
    with pytest.raises(ValueError, match="no module matched"):
        apply_peft(model, PeftConfig(method="lora", target_modules=["nope"]))


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


@pytest.mark.parametrize(
    "method,extra",
    [
        ("lora", {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0}),
        (
            "lora",
            {
                "r": 4,
                "lora_alpha": 4,
                "lora_dropout": 0.0,
                "pissa_init": True,
            },
        ),
        ("miss", {"r": 4, "miss_dropout": 0.0}),
        ("state", {}),
    ],
)
def test_train_with_peft_saves_loadable_checkpoint(
    tmp_path, capsys, method, extra
):
    rng = np.random.default_rng(0)
    token_arrays = [
        rng.integers(0, 300, size=int(size), dtype=np.int64)
        for size in rng.integers(200, 400, size=4)
    ]
    manifest = _write_manifest(tmp_path, token_arrays)
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(max_steps=2, save_every=2)
    peft_config = PeftConfig(method=method, **extra)

    checkpoint = train(
        manifest,
        tmp_path,
        tmp_path / "out",
        model_config,
        train_config,
        phase="posttrain",
        device_name="cpu",
        supervised=False,
        peft_config=peft_config,
    )
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["peft_config"] == peft_config.to_dict()

    # Rebuild exactly like generation does: construct, apply PEFT, load.
    restored = YufMusicGen(
        ModelConfig(**payload["model_config"])
    )
    apply_peft(restored, PeftConfig.from_dict(payload["peft_config"]))
    restored.load_state_dict(payload["model"], strict=True)
    tokens = torch.randint(0, 300, (1, 12))
    with torch.no_grad():
        logits, _ = restored(tokens)
    assert logits.shape == (1, 12, 300)

    output = capsys.readouterr().out
    assert f"PEFT: method={method}" in output
    assert "trainable params:" in output


def test_state_tuning_training_actually_updates_states(tmp_path):
    rng = np.random.default_rng(1)
    token_arrays = [
        rng.integers(0, 300, size=int(size), dtype=np.int64)
        for size in rng.integers(200, 400, size=4)
    ]
    manifest = _write_manifest(tmp_path, token_arrays)
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(max_steps=3, save_every=3)

    checkpoint = train(
        manifest,
        tmp_path,
        tmp_path / "out",
        model_config,
        train_config,
        phase="posttrain",
        device_name="cpu",
        supervised=False,
        peft_config=PeftConfig(method="state"),
    )
    payload = torch.load(checkpoint, map_location="cpu")
    model = YufMusicGen(model_config)
    apply_peft(model, PeftConfig(method="state"))
    model.load_state_dict(payload["model"], strict=True)
    nonzero = [
        parameter
        for name, parameter in model.named_parameters()
        if "state_tuning" in name and parameter.abs().sum() > 0
    ]
    assert nonzero


def test_peft_resume_from_checkpoint_keeps_adapter_layout(tmp_path):
    rng = np.random.default_rng(2)
    token_arrays = [
        rng.integers(0, 300, size=int(size), dtype=np.int64)
        for size in rng.integers(200, 400, size=4)
    ]
    manifest = _write_manifest(tmp_path, token_arrays)
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(max_steps=2, save_every=2)
    peft_config = PeftConfig(method="lora", r=4, lora_alpha=8)

    checkpoint = train(
        manifest,
        tmp_path,
        tmp_path / "out",
        model_config,
        train_config,
        phase="posttrain",
        device_name="cpu",
        supervised=False,
        peft_config=peft_config,
    )
    # Resuming without CLI flags must restore the adapter layout from the
    # checkpoint so the state dict still matches.
    resumed = train(
        manifest,
        tmp_path,
        tmp_path / "out2",
        model_config,
        _tiny_train_config(max_steps=1, save_every=1),
        phase="posttrain",
        device_name="cpu",
        supervised=False,
        init_checkpoint=checkpoint,
    )
    payload = torch.load(resumed, map_location="cpu")
    assert payload["peft_config"] == peft_config.to_dict()


def _parse_trainable(output):
    match = re.search(r"trainable params: ([0-9,]+)", output)
    return int(match.group(1).replace(",", ""))


def test_state_tuning_uses_far_fewer_trainable_parameters(tmp_path, capsys):
    rng = np.random.default_rng(3)
    token_arrays = [
        rng.integers(0, 300, size=256, dtype=np.int64) for _ in range(2)
    ]
    manifest = _write_manifest(tmp_path, token_arrays)
    train_config = _tiny_train_config(max_steps=1, save_every=1)
    train(
        manifest,
        tmp_path,
        tmp_path / "out",
        _tiny_model_config(),
        train_config,
        phase="posttrain",
        device_name="cpu",
        supervised=False,
        peft_config=PeftConfig(method="state"),
    )
    trainable = _parse_trainable(capsys.readouterr().out)
    assert 0 < trainable < 3000


@pytest.mark.parametrize(
    "method,extra",
    [
        ("lora", {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0}),
        (
            "lora",
            {
                "r": 4,
                "lora_alpha": 4,
                "lora_dropout": 0.0,
                "pissa_init": True,
            },
        ),
        ("miss", {"r": 4, "miss_dropout": 0.0}),
        ("state", {}),
    ],
)
def test_peft_finetune_from_base_checkpoint(
    tmp_path, method, extra
):
    """Starting PEFT from a plain (non-PEFT) checkpoint must work."""

    rng = np.random.default_rng(4)
    token_arrays = [
        rng.integers(0, 300, size=int(size), dtype=np.int64)
        for size in rng.integers(200, 400, size=4)
    ]
    manifest = _write_manifest(tmp_path, token_arrays)
    model_config = _tiny_model_config()

    base_config = _tiny_train_config(max_steps=1, save_every=1, seed=11)
    base_checkpoint = train(
        manifest,
        tmp_path,
        tmp_path / "base",
        model_config,
        base_config,
        phase="pretrain",
        device_name="cpu",
        supervised=False,
    )
    payload = torch.load(base_checkpoint, map_location="cpu")
    assert payload.get("peft_config") is None

    peft_config = PeftConfig(method=method, **extra)
    train_config = _tiny_train_config(max_steps=2, save_every=2, seed=22)
    checkpoint = train(
        manifest,
        tmp_path,
        tmp_path / "ft",
        model_config,
        train_config,
        phase="posttrain",
        device_name="cpu",
        supervised=False,
        init_checkpoint=base_checkpoint,
        peft_config=peft_config,
    )
    saved = torch.load(checkpoint, map_location="cpu")
    assert saved["peft_config"] == peft_config.to_dict()


def test_pissa_checkpoint_reload_skips_svd_surgery(tmp_path):
    """Loading a PiSSA checkpoint must not re-run the SVD base mutation."""

    rng = np.random.default_rng(5)
    token_arrays = [
        rng.integers(0, 300, size=256, dtype=np.int64) for _ in range(2)
    ]
    manifest = _write_manifest(tmp_path, token_arrays)
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(max_steps=1, save_every=1)
    peft_config = PeftConfig(
        method="lora",
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        pissa_init=True,
    )
    checkpoint = train(
        manifest,
        tmp_path,
        tmp_path / "out",
        model_config,
        train_config,
        phase="posttrain",
        device_name="cpu",
        supervised=False,
        peft_config=peft_config,
    )
    payload = torch.load(checkpoint, map_location="cpu")

    model = YufMusicGen(model_config)
    apply_peft(
        model,
        PeftConfig.from_dict(payload["peft_config"]),
        init_adapters=False,
    )
    model.load_state_dict(payload["model"], strict=True)
    tokens = torch.randint(0, 300, (1, 12))
    with torch.no_grad():
        logits, _ = model(tokens)
    assert logits.shape == (1, 12, 300)

    # The restored base weight must be the saved residual, untouched by a
    # fresh PiSSA initialization.
    restored = dict(model.named_parameters())
    for name, parameter in restored.items():
        if name.endswith(".base.weight"):
            assert torch.equal(
                parameter, payload["model"][name]
            )
