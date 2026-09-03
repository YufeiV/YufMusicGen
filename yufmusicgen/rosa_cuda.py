"""Lazy loader and autograd wrapper for the fused ROSA Householder scan.

The recurrent part of ``ROSAMemory`` (the Householder-orthogonal state update)
is a sequential dependency, so a host-side loop pays one kernel launch per
token plus Python overhead.  This module mirrors the repository's RWKV-7 CUDA
pattern: the whole recurrence is fused into one small kernel per batch element
(forward and backward), so training runs a single pass and stateful inference
keeps the fused scan instead of falling back to the reference Python loop.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load


_LOAD_LOCK = threading.Lock()
_LOAD_ERROR: Exception | None = None
_LOADED = False


def _source_paths() -> tuple[str, str]:
    root = Path(__file__).resolve().parent / "cuda"
    return str(root / "rosa_householder.cpp"), str(root / "rosa_householder.cu")


def _load_extension() -> None:
    global _LOADED, _LOAD_ERROR
    if _LOADED:
        return
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR
    with _LOAD_LOCK:
        if _LOADED:
            return
        if _LOAD_ERROR is not None:
            raise _LOAD_ERROR
        try:
            # The RWKV-7 extension already ensured MSVC/CUDA environment
            # variables are visible for plain `uv run`; reuse that helper so
            # both extensions build under the same toolchain.
            from .cuda_ops import _ensure_windows_toolchain

            _ensure_windows_toolchain()
            if os.name == "nt":
                # PyTorch versions that default to OEM decoding can fail on
                # non-ASCII Windows locales while parsing `cl --version`.
                cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
            cpp_source, cuda_source = _source_paths()
            load(
                name="yufmusicgen_rosa",
                sources=[cpp_source, cuda_source],
                is_python_module=False,
                verbose=os.environ.get("YUFMUSICGEN_CUDA_VERBOSE", "0") == "1",
                extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
            )
            _LOADED = True
        except Exception as exc:
            _LOAD_ERROR = exc
            raise


def rosa_cuda_enabled() -> bool:
    return torch.cuda.is_available() and os.environ.get("YUFMUSICGEN_DISABLE_CUDA", "0") != "1"


class _ROSAHouseholderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cand: torch.Tensor,
        write: torch.Tensor,
        read_raw: torch.Tensor,
        decay: torch.Tensor,
        direction: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not all(x.is_cuda for x in (cand, write, read_raw, decay, direction, memory)):
            raise ValueError("ROSA CUDA inputs must be on CUDA")
        if not all(x.is_contiguous() for x in (cand, write, read_raw, decay, direction, memory)):
            raise ValueError("ROSA CUDA inputs must be contiguous")
        if len({tuple(x.shape) for x in (cand, write, read_raw)}) != 1:
            raise ValueError("ROSA CUDA per-step inputs must have identical shapes")

        _load_extension()
        batch, steps, dim = cand.shape
        read_out = torch.empty_like(cand)
        memory_out = torch.empty_like(memory)
        cand_saved = torch.empty_like(cand)
        write_saved = torch.empty_like(cand)
        read_saved = torch.empty_like(cand)
        gate_saved = torch.empty_like(cand)
        decay_saved = torch.empty_like(decay.expand(batch, steps, dim))
        mem_saved = torch.empty(
            (batch, steps + 1, dim), device=cand.device, dtype=cand.dtype
        )
        torch.ops.yufmusicgen_rosa.forward(
            cand, write, read_raw, decay, direction, memory,
            read_out, memory_out, cand_saved, write_saved, read_saved,
            gate_saved, decay_saved, mem_saved,
        )

        ctx.save_for_backward(
            cand_saved, write_saved, read_saved, gate_saved, decay_saved,
            mem_saved, direction, memory_out
        )
        ctx.mark_non_differentiable(memory_out)
        return read_out, memory_out

    @staticmethod
    def backward(
        ctx,
        d_read: torch.Tensor,
        _d_memory_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        (
            cand_saved,
            write_saved,
            read_saved,
            gate_saved,
            decay_saved,
            mem_saved,
            direction,
            memory_out,
        ) = ctx.saved_tensors
        batch, steps, dim = cand_saved.shape
        d_cand = torch.empty_like(cand_saved)
        d_write = torch.empty_like(cand_saved)
        d_read_raw = torch.empty_like(cand_saved)
        d_decay = torch.empty_like(decay_saved)
        d_direction_partial = torch.empty((batch, dim), device=direction.device, dtype=direction.dtype)
        d_memory_out = torch.empty((batch, dim), device=direction.device, dtype=direction.dtype)
        workspace = torch.empty(
            (batch, steps, dim * 5), device=direction.device, dtype=torch.float32
        )
        torch.ops.yufmusicgen_rosa.backward(
            cand_saved, write_saved, read_saved, gate_saved, decay_saved,
            mem_saved, direction, memory_out,
            d_read.contiguous(),
            d_cand, d_write, d_read_raw, d_decay, d_direction_partial,
            d_memory_out, workspace,
        )
        # `direction` is shared across the batch; the kernel accumulates one
        # partial vector per batch element, so reduce here.
        d_direction = d_direction_partial.sum(dim=0)
        # The kernel reports p_0 = dL/dm_0 directly.
        d_memory = d_memory_out
        return d_cand, d_write, d_read_raw, d_decay, d_direction, d_memory


def rosa_householder_scan(
    cand: torch.Tensor,
    write: torch.Tensor,
    read_raw: torch.Tensor,
    decay: torch.Tensor,
    direction: torch.Tensor,
    memory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused Householder scan and return ``(read, final_memory)``.

    ``decay`` must already be post-sigmoid; ``direction`` must be a unit
    vector.  The scan itself always runs in FP32 (matching the reference), so
    inputs are converted here and the read is converted back to the input
    dtype.
    """

    return _ROSAHouseholderFunction.apply(
        cand.float().contiguous(),
        write.float().contiguous(),
        read_raw.float().contiguous(),
        decay.float().contiguous(),
        direction.float().contiguous(),
        memory.float().contiguous(),
    )
