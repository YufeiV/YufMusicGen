"""Lazy loader and autograd wrapper for the vendored RWKV-7 CUDA kernel."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load


CHUNK_LENGTH = 16
_LOAD_LOCK = threading.Lock()
_LOADED_HEAD_SIZE: int | None = None
_LOAD_ERROR: Exception | None = None
_WARNED = False


def _source_paths() -> tuple[str, str]:
    root = Path(__file__).resolve().parent / "cuda"
    return str(root / "rwkv7_clampw.cpp"), str(root / "rwkv7_clampw.cu")


def _ensure_windows_toolchain() -> None:
    """Import MSVC/CUDA variables so plain ``uv run`` can build the extension."""

    if os.name != "nt":
        return

    if shutil.which("cl") is None:
        vcvars_candidates: list[Path] = []
        program_files_x86 = Path(
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        )
        vswhere = program_files_x86 / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.exists():
            result = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-find",
                    r"VC\Auxiliary\Build\vcvars64.bat",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            vcvars_candidates.extend(
                Path(line.strip())
                for line in result.stdout.splitlines()
                if line.strip()
            )

        for root_name in ("ProgramFiles", "ProgramFiles(x86)"):
            root = os.environ.get(root_name)
            if root:
                vcvars_candidates.extend(
                    Path(root).glob(
                        "Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat"
                    )
                )

        vcvars = next((path for path in vcvars_candidates if path.exists()), None)
        if vcvars is not None:
            command = f'call "{vcvars}" >nul 2>&1 && set'
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            for line in result.stdout.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key and " " not in key:
                        os.environ[key] = value

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        cuda_roots = list(
            (Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA").glob("v*")
        )
        if cuda_roots:
            cuda_home = str(sorted(cuda_roots)[-1])
            os.environ["CUDA_HOME"] = cuda_home
            os.environ["CUDA_PATH"] = cuda_home
    if cuda_home:
        cuda_bin = str(Path(cuda_home) / "bin")
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
        # cpp_extension caches this value at import time, before the helper runs.
        cpp_extension.CUDA_HOME = cuda_home


def _load_for_head_size(head_size: int) -> None:
    global _LOADED_HEAD_SIZE, _LOAD_ERROR
    if _LOADED_HEAD_SIZE == head_size:
        return
    if _LOADED_HEAD_SIZE is not None and _LOADED_HEAD_SIZE != head_size:
        raise RuntimeError(
            f"RWKV-7 CUDA extension is already loaded for head_size={_LOADED_HEAD_SIZE}; "
            f"cannot load a second head_size={head_size} in the same process"
        )
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR
    with _LOAD_LOCK:
        if _LOADED_HEAD_SIZE == head_size:
            return
        if _LOAD_ERROR is not None:
            raise _LOAD_ERROR
        try:
            if os.name == "nt":
                _ensure_windows_toolchain()
                # PyTorch versions that default to OEM decoding can fail on
                # non-ASCII Windows locales while parsing `cl --version`.
                cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
                venv_scripts = Path(sys.executable).resolve().parent
                if (venv_scripts / "ninja.exe").exists():
                    os.environ["PATH"] = str(venv_scripts) + os.pathsep + os.environ.get("PATH", "")
            cpp_source, cuda_source = _source_paths()
            load(
                name=f"yufmusicgen_rwkv7_{head_size}",
                sources=[cpp_source, cuda_source],
                is_python_module=False,
                verbose=os.environ.get("YUFMUSICGEN_CUDA_VERBOSE", "0") == "1",
                extra_cflags=(
                    ["/O2", "/D_FP32_"]
                    if os.name == "nt"
                    else ["-O3", "-D_FP32_"]
                ),
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-D_FP32_",
                    f"-D_N_={head_size}",
                    f"-D_CHUNK_LEN_={CHUNK_LENGTH}",
                ],
            )
            _LOADED_HEAD_SIZE = head_size
        except Exception as exc:  # pragma: no cover - compiler/platform dependent
            _LOAD_ERROR = exc
            raise


def _pad_inputs(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], int]:
    length = r.shape[1]
    padded = ((length + CHUNK_LENGTH - 1) // CHUNK_LENGTH) * CHUNK_LENGTH
    if padded == length:
        return (r, w, k, v, a, b), length
    extra = padded - length
    zeros = lambda: torch.zeros(
        (r.shape[0], extra, r.shape[2], r.shape[3]), device=r.device, dtype=r.dtype
    )
    # A very negative raw w gives retention ~= 1; zero k/v/a/b makes padding
    # preserve the real final state while its output is discarded.
    pad_w = torch.full_like(zeros(), -80.0)
    return (
        torch.cat((r, zeros()), dim=1),
        torch.cat((w, pad_w), dim=1),
        torch.cat((k, zeros()), dim=1),
        torch.cat((v, zeros()), dim=1),
        torch.cat((a, zeros()), dim=1),
        torch.cat((b, zeros()), dim=1),
    ), length


class _RWKV7CUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if r.ndim != 4:
            raise ValueError("RWKV-7 CUDA inputs must have shape [B, T, H, N]")
        if not all(x.is_cuda for x in (r, w, k, v, a, b)):
            raise ValueError("RWKV-7 CUDA inputs must be on CUDA")
        if not all(x.dtype == torch.float32 for x in (r, w, k, v, a, b)):
            raise TypeError("the bundled default kernel is compiled for float32")
        if not all(x.is_contiguous() for x in (r, w, k, v, a, b)):
            raise ValueError("RWKV-7 CUDA inputs must be contiguous")
        if len({tuple(x.shape) for x in (r, w, k, v, a, b)}) != 1:
            raise ValueError("RWKV-7 CUDA inputs must have identical shapes")

        _load_for_head_size(r.shape[-1])
        (rp, wp, kp, vp, ap, bp), original_length = _pad_inputs(r, w, k, v, a, b)
        batch, length, heads, head_size = rp.shape
        output = torch.empty_like(vp)
        states = torch.empty(
            batch,
            heads,
            length // CHUNK_LENGTH,
            head_size,
            head_size,
            device=r.device,
            dtype=torch.float32,
        )
        sa = torch.empty_like(rp, dtype=torch.float32)
        torch.ops.yufmusicgen_rwkv7.forward(rp, wp, kp, vp, ap, bp, output, states, sa)

        ctx.save_for_backward(rp, wp, kp, vp, ap, bp, states, sa)
        ctx.original_length = original_length
        # The upstream kernel checkpoints each thread's state with the two
        # matrix axes transposed; the PyTorch stateful path uses [row, col].
        final_state = states[:, :, -1].transpose(-1, -2).contiguous()
        ctx.mark_non_differentiable(final_state)
        return output[:, :original_length].contiguous(), final_state

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_state: torch.Tensor | None = None):
        rp, wp, kp, vp, ap, bp, states, sa = ctx.saved_tensors
        grad_full = torch.zeros_like(vp)
        grad_full[:, : ctx.original_length] = grad_output
        grads = [torch.empty_like(x) for x in (rp, wp, kp, vp, ap, bp)]
        torch.ops.yufmusicgen_rwkv7.backward(
            rp,
            wp,
            kp,
            vp,
            ap,
            bp,
            grad_full,
            states,
            sa,
            *grads,
        )
        return tuple(grad[:, : ctx.original_length].contiguous() for grad in grads)


def rwkv7_cuda(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused RWKV-7 recurrence and return output plus final state."""

    return _RWKV7CUDAFunction.apply(r, w, k, v, a, b)


def cuda_kernel_enabled() -> bool:
    return torch.cuda.is_available() and os.environ.get("YUFMUSICGEN_DISABLE_CUDA", "0") != "1"


def report_cuda_fallback(exc: Exception) -> None:
    global _WARNED
    if _WARNED or os.environ.get("YUFMUSICGEN_SILENCE_CUDA_WARNING", "0") == "1":
        return
    _WARNED = True
    warnings.warn(
        "YufMusicGen RWKV-7 CUDA kernel unavailable; using the PyTorch recurrence. "
        f"Reason: {exc}",
        RuntimeWarning,
        stacklevel=3,
    )
