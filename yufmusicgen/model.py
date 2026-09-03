"""RWKV-7 style recurrent language model with a ROSA memory branch.

The TimeMix block follows the important RWKV-7 design constraints: token
mixing uses the previous token, attention is linear in sequence length, and the
state can be carried from one token to the next without a KV cache. ROSA here
means Recurrent Orthogonal State Augmentation: a small gated memory is updated
through a Householder reflection, which is exactly orthogonal and inexpensive.
"""

from __future__ import annotations

from typing import NamedTuple
import warnings

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .cuda_ops import cuda_kernel_enabled, report_cuda_fallback, rwkv7_cuda


_ROSA_SCAN_FAILED = False
_ROSA_SCAN_WARNED = False
_ROSA_CUDA_FAILED = False
_ROSA_CUDA_WARNED = False


class TimeMixState(NamedTuple):
    previous: Tensor
    memory: Tensor
    normalizer: Tensor


class BlockState(NamedTuple):
    timemix: TimeMixState
    rosa: Tensor


class RWKV7TimeMix(nn.Module):
    """RWKV-7 ``r/w/k/v/a/b`` TimeMix with a fused training recurrence."""

    def __init__(self, config: ModelConfig, layer_id: int = 0) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_size = config.head_size
        self.use_cuda_kernel = config.use_cuda_kernel

        ratio = 1.0 - layer_id / max(1, config.n_layers)
        ramp = torch.linspace(0.0, 1.0, config.d_model)
        self.mix_r = nn.Parameter(1.0 - ramp.pow(0.2 * ratio))
        self.mix_w = nn.Parameter(1.0 - ramp.pow(0.9 * ratio))
        self.mix_k = nn.Parameter(1.0 - ramp.pow(0.7 * ratio))
        self.mix_v = nn.Parameter(1.0 - ramp.pow(0.7 * ratio))
        self.mix_a = nn.Parameter(1.0 - ramp.pow(0.9 * ratio))
        self.mix_g = nn.Parameter(1.0 - ramp.pow(0.2 * ratio))

        low_rank = max(32, int(round(2.5 * (config.d_model**0.5) / 32) * 32))
        self.w0 = nn.Parameter(torch.linspace(-5.5, 0.0, config.d_model))
        self.w1 = nn.Parameter(torch.zeros(config.d_model, low_rank))
        self.w2 = nn.Parameter(torch.empty(low_rank, config.d_model))
        self.a0 = nn.Parameter(torch.full((config.d_model,), -0.19))
        self.a1 = nn.Parameter(torch.zeros(config.d_model, low_rank))
        self.a2 = nn.Parameter(torch.empty(low_rank, config.d_model))
        self.g1 = nn.Parameter(torch.zeros(config.d_model, low_rank))
        self.g2 = nn.Parameter(torch.empty(low_rank, config.d_model))
        nn.init.orthogonal_(self.w2, gain=0.1)
        nn.init.orthogonal_(self.a2, gain=0.1)
        nn.init.orthogonal_(self.g2, gain=0.1)

        self.k_k = nn.Parameter(torch.full((config.d_model,), 0.71))
        self.k_a = nn.Parameter(torch.full((config.d_model,), 1.02))
        self.receptance = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_norm = nn.LayerNorm(config.d_model)

        self.receptance.weight.data.uniform_(-0.5 / (config.d_model**0.5), 0.5 / (config.d_model**0.5))
        self.key.weight.data.uniform_(-0.05 / (config.d_model**0.5), 0.05 / (config.d_model**0.5))
        self.value.weight.data.uniform_(-0.5 / (config.d_model**0.5), 0.5 / (config.d_model**0.5))
        nn.init.zeros_(self.output.weight)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> TimeMixState:
        memory = torch.zeros(
            batch_size,
            self.n_heads,
            self.head_size,
            self.head_size,
            device=device,
            dtype=dtype,
        )
        previous = torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
        return TimeMixState(previous, memory, torch.empty(0, device=device, dtype=dtype))

    def _inputs_impl(self, x: Tensor, previous: Tensor) -> tuple[Tensor, ...]:
        shifted = torch.cat((previous[:, None], x[:, :-1]), dim=1)
        delta = x - shifted
        xr = x + delta * self.mix_r
        xw = x + delta * self.mix_w
        xk = x + delta * self.mix_k
        xv = x + delta * self.mix_v
        xa = x + delta * self.mix_a
        xg = x + delta * self.mix_g

        r = self.receptance(xr)
        w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid((xg @ self.g1) @ self.g2)

        batch, steps, _ = k.shape
        k_heads = (k * self.k_k).view(batch, steps, self.n_heads, self.head_size)
        k_heads = F.normalize(k_heads, dim=-1, p=2.0, eps=1e-12)
        a_heads = a.view(batch, steps, self.n_heads, self.head_size)
        k_adjusted = (k * (1.0 + (a - 1.0) * self.k_a)).view(
            batch, steps, self.n_heads, self.head_size
        )
        neg_kk = -k_heads
        kka = k_heads * a_heads
        return (
            r.view(batch, steps, self.n_heads, self.head_size),
            w.view(batch, steps, self.n_heads, self.head_size),
            k_adjusted,
            v.view(batch, steps, self.n_heads, self.head_size),
            neg_kk,
            kka,
            g,
        )

    def _inputs(self, x: Tensor, previous: Tensor) -> tuple[Tensor, ...]:
        # The fused RWKV-7 operator intentionally runs in FP32.  Under CUDA
        # autocast, keep this projection path in FP32 so BF16 training does not
        # silently fall back to the slow Python recurrence.
        use_cuda_inputs = (
            x.is_cuda and self.use_cuda_kernel and cuda_kernel_enabled()
        )
        if use_cuda_inputs:
            with torch.autocast(device_type="cuda", enabled=False):
                return self._inputs_impl(x.float(), previous.float())
        return self._inputs_impl(x, previous)

    @staticmethod
    def _torch_recurrence(
        r: Tensor,
        w_raw: Tensor,
        k: Tensor,
        v: Tensor,
        a: Tensor,
        b: Tensor,
        memory: Tensor,
    ) -> tuple[Tensor, Tensor]:
        outputs: list[Tensor] = []
        retention = torch.exp(-0.6065306597 * torch.sigmoid(w_raw))
        for index in range(r.shape[1]):
            state_a = torch.einsum("bhj,bhij->bhi", a[:, index], memory)
            memory = (
                memory * retention[:, index].unsqueeze(-2)
                + state_a.unsqueeze(-1) * b[:, index].unsqueeze(-2)
                + v[:, index].unsqueeze(-1) * k[:, index].unsqueeze(-2)
            )
            outputs.append(torch.einsum("bhij,bhj->bhi", memory, r[:, index]))
        return torch.stack(outputs, dim=1), memory

    def forward(self, x: Tensor, state: TimeMixState | None = None) -> tuple[Tensor, TimeMixState]:
        if x.ndim != 3:
            raise ValueError("TimeMix expects [batch, time, channels]")
        batch = x.shape[0]
        current = state or self.initial_state(batch, x.device, x.dtype)
        previous, memory, _ = current
        r, w, k, v, a, b, gate = self._inputs(x, previous)

        use_cuda = (
            state is None
            and self.use_cuda_kernel
            and cuda_kernel_enabled()
            and x.is_cuda
            and r.dtype == torch.float32
        )
        if use_cuda:
            try:
                mixed, memory = rwkv7_cuda(r.contiguous(), w.contiguous(), k.contiguous(), v.contiguous(), a.contiguous(), b.contiguous())
                # The fused kernel checkpoints the state with its two matrix
                # axes transposed; the PyTorch stateful path carries
                # [row, col], so rotate it back before passing it onward.
                memory = memory.transpose(-1, -2).contiguous()
            except Exception as exc:  # pragma: no cover - requires CUDA toolchain
                report_cuda_fallback(exc)
                mixed, memory = self._torch_recurrence(r, w, k, v, a, b, memory)
        else:
            mixed, memory = self._torch_recurrence(r, w, k, v, a, b, memory)

        mixed = mixed.view(x.shape)
        output = self.output(self.out_norm(mixed) * gate)
        next_state = TimeMixState(x[:, -1].detach(), memory, torch.empty(0, device=x.device, dtype=x.dtype))
        return output, next_state


class ROSAMemory(nn.Module):
    """Gated recurrent memory with an exactly orthogonal Householder update."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.rosa_size = config.rosa_size
        self.use_scan = config.use_rosa_scan
        self.use_cuda_kernel = config.use_cuda_kernel
        self.input = nn.Linear(config.d_model, config.rosa_size)
        self.write_gate = nn.Linear(config.d_model, config.rosa_size)
        self.read_gate = nn.Linear(config.d_model, config.rosa_size)
        self.output = nn.Linear(config.rosa_size, config.d_model, bias=False)
        self.householder = nn.Parameter(torch.randn(config.rosa_size))
        self.decay = nn.Parameter(torch.full((config.rosa_size,), 2.0))

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.rosa_size, device=device, dtype=dtype)

    def _orthogonal(self, value: Tensor) -> Tensor:
        direction = F.normalize(self.householder, dim=0)
        projection = torch.sum(value * direction, dim=-1, keepdim=True)
        return value - 2.0 * projection * direction

    def _scan_forward(self, x: Tensor, memory: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        from torch._higher_order_ops import scan

        batch, steps, _ = x.shape
        # Keep the recurrent accumulation in FP32.  A long BF16 scan can
        # otherwise lose enough mantissa bits to destabilize the optimizer.
        scan_dtype = torch.float32
        memory = memory.to(dtype=scan_dtype)
        candidate = self.input(x).to(dtype=scan_dtype)
        write = self.write_gate(x).to(dtype=scan_dtype)
        read_gate = self.read_gate(x).to(dtype=scan_dtype)
        decay = torch.sigmoid(self.decay).to(dtype=scan_dtype)
        direction = F.normalize(self.householder, dim=0).to(dtype=scan_dtype)
        scan_decay = decay.view(1, 1, -1).expand(batch, steps, -1)
        scan_direction = direction.view(1, 1, -1).expand(batch, steps, -1)

        def combine(previous: Tensor, item: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
            item_candidate_raw, item_write_raw, item_decay, item_direction = item
            item_candidate = torch.tanh(item_candidate_raw)
            item_write = torch.sigmoid(item_write_raw)
            projection = torch.sum(previous * item_direction, dim=-1, keepdim=True)
            rotated = previous - 2.0 * projection * item_direction
            next_memory = item_decay * rotated + item_write * item_candidate
            read = next_memory - 2.0 * torch.sum(
                next_memory * item_direction, dim=-1, keepdim=True
            ) * item_direction
            return next_memory, read.clone()

        final_memory, read = scan(
            combine,
            init=memory,
            xs=(candidate, write, scan_decay, scan_direction),
            dim=1,
        )
        return read, final_memory, read_gate

    @staticmethod
    def _reference_recurrence(
        candidate: Tensor,
        write: Tensor,
        read_gate: Tensor,
        decay: Tensor,
        direction: Tensor,
        memory: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Reference scan used when neither the fused kernel nor torch.scan run.

        ``decay`` is expected to already be post-sigmoid (matching the fused
        kernel and the precomputed activations in :meth:`forward`).
        """

        batch, steps, _ = candidate.shape
        if decay.dim() == 3 and decay.shape[1] == steps:
            decay = decay[:, 0]
        else:
            decay = decay.squeeze(1) if decay.dim() > 1 else decay
        direction = direction.squeeze(1) if direction.dim() > 1 else direction
        outputs: list[Tensor] = []
        for index in range(steps):
            rotated = memory - 2.0 * torch.sum(memory * direction, dim=-1, keepdim=True) * direction
            memory = decay * rotated + torch.sigmoid(write[:, index]) * torch.tanh(candidate[:, index])
            read = memory - 2.0 * torch.sum(memory * direction, dim=-1, keepdim=True) * direction
            read = read * torch.sigmoid(read_gate[:, index])
            outputs.append(read)
        return torch.stack(outputs, dim=1), memory

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch, steps, _ = x.shape
        memory = state if state is not None else self.initial_state(batch, x.device, x.dtype)
        global _ROSA_SCAN_FAILED, _ROSA_SCAN_WARNED, _ROSA_CUDA_FAILED, _ROSA_CUDA_WARNED

        # Pre-activations are computed once and shared by every backend; the
        # fused kernel and the reference scan apply identical activations.
        candidate = self.input(x)
        write = self.write_gate(x)
        read_gate = self.read_gate(x)
        decay = torch.sigmoid(self.decay).to(dtype=x.dtype)
        direction = F.normalize(self.householder, dim=0).to(dtype=x.dtype)

        if self.use_cuda_kernel and x.is_cuda and not _ROSA_CUDA_FAILED:
            try:
                from .rosa_cuda import rosa_cuda_enabled, rosa_householder_scan

                if rosa_cuda_enabled():
                    read, final_memory = rosa_householder_scan(
                        candidate, write, read_gate, decay, direction, memory
                    )
                    read = self.output(read * torch.sigmoid(read_gate))
                    return read, final_memory
            except Exception as exc:  # pragma: no cover - depends on CUDA toolchain
                _ROSA_CUDA_FAILED = True
                if not _ROSA_CUDA_WARNED:
                    warnings.warn(
                        "ROSA fused CUDA kernel is unavailable; using the torch scan "
                        f"({type(exc).__name__})",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _ROSA_CUDA_WARNED = True

        if self.use_scan and state is None and not _ROSA_SCAN_FAILED:
            try:
                read, final_memory, read_gate_post = self._scan_forward(x, memory)
                return self.output(read * read_gate_post), final_memory
            except Exception as exc:  # pragma: no cover - depends on PyTorch backend
                _ROSA_SCAN_FAILED = True
                if not _ROSA_SCAN_WARNED:
                    warnings.warn(
                        "ROSA torch scan is unavailable; using the reference recurrence "
                        f"({type(exc).__name__})",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _ROSA_SCAN_WARNED = True

        read, final_memory = self._reference_recurrence(
            candidate, write, read_gate, decay, direction, memory
        )
        return self.output(read), final_memory


class YufBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_time = nn.LayerNorm(config.d_model)
        self.time_mix = RWKV7TimeMix(config)
        self.norm_rosa = nn.LayerNorm(config.d_model)
        self.rosa = ROSAMemory(config)
        self.norm_ffn = nn.LayerNorm(config.d_model)
        hidden = config.d_model * 4
        self.ffn_in = nn.Linear(config.d_model, hidden, bias=False)
        self.ffn_gate = nn.Linear(config.d_model, hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> BlockState:
        return BlockState(
            self.time_mix.initial_state(batch_size, device, dtype),
            self.rosa.initial_state(batch_size, device, dtype),
        )

    def forward(self, x: Tensor, state: BlockState | None = None) -> tuple[Tensor, BlockState]:
        batch = x.shape[0]
        current = state or self.initial_state(batch, x.device, x.dtype)
        time_out, time_state = self.time_mix(
            self.norm_time(x), None if state is None else current.timemix
        )
        x = x + self.dropout(time_out)
        rosa_out, rosa_state = self.rosa(
            self.norm_rosa(x), None if state is None else current.rosa
        )
        x = x + self.dropout(rosa_out)
        ffn_input = self.norm_ffn(x)
        ffn = self.ffn_out(F.silu(self.ffn_in(ffn_input)) * self.ffn_gate(ffn_input))
        return x + self.dropout(ffn), BlockState(time_state, rosa_state)


class YufMusicGen(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList([YufBlock(self.config) for _ in range(self.config.n_layers)])
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> list[BlockState]:
        return [block.initial_state(batch_size, device, dtype) for block in self.blocks]

    def forward(
        self, input_ids: Tensor, state: list[BlockState] | None = None
    ) -> tuple[Tensor, list[BlockState]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        hidden = self.token_embedding(input_ids)
        next_states: list[BlockState] = []
        if state is None:
            for block in self.blocks:
                hidden, block_state = block(hidden, None)
                next_states.append(block_state)
        else:
            for block, block_state in zip(self.blocks, state):
                hidden, block_state = block(hidden, block_state)
                next_states.append(block_state)
        logits = self.lm_head(self.final_norm(hidden))
        return logits, next_states

    @torch.no_grad()
    def step(
        self, input_ids: Tensor, state: list[BlockState] | None = None
    ) -> tuple[Tensor, list[BlockState]]:
        if input_ids.ndim == 1:
            input_ids = input_ids[:, None]
        logits, state = self.forward(input_ids, state)
        return logits[:, -1], state

    @staticmethod
    def detach_state(state: list[BlockState]) -> list[BlockState]:
        return [
            BlockState(
                TimeMixState(
                    block.timemix.previous.detach(),
                    block.timemix.memory.detach(),
                    block.timemix.normalizer.detach(),
                ),
                block.rosa.detach(),
            )
            for block in state
        ]
