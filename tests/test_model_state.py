import pytest

torch = pytest.importorskip("torch")

from yufmusicgen.config import ModelConfig
from yufmusicgen.model import YufMusicGen


def test_sequence_and_recurrent_step_paths_match():
    torch.manual_seed(7)
    config = ModelConfig(
        vocab_size=300,
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_size=16,
        rosa_size=12,
    )
    model = YufMusicGen(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 6))
    full_logits, _ = model(tokens)
    state = None
    step_logits = []
    for index in range(tokens.shape[1]):
        logits, state = model.step(tokens[:, index], state)
        step_logits.append(logits)
    stepped = torch.stack(step_logits, dim=1)
    assert torch.allclose(full_logits, stepped, atol=1e-5, rtol=1e-5)
