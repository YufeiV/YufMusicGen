"""Dump layer-0 intermediates for a single token from the PyTorch reference.

Mirrors the region names/offsets of the Vulkan ``dump_step`` example so the
two implementations can be compared numerically:

    python dump_reference_step.py <checkpoint.pt> [token]

Prints ``label count v1 v2 ...`` lines (all values, 9 decimals) for the layer-0
work regions plus layer-0 TimeMix/ROSA memory and the final logits.
"""

from __future__ import annotations

import sys

import torch
from torch.nn import functional as F

from yufmusicgen.config import ModelConfig, dataclass_from_dict
from yufmusicgen.model import YufMusicGen


def dump(label: str, values: torch.Tensor) -> None:
    flat = values.detach().float().flatten()
    print(f"{label} {flat.numel()} " + " ".join(f"{v:.9f}" for v in flat.tolist()))


def main() -> None:
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
    token = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dataclass_from_dict(ModelConfig, payload["model_config"])
    cfg.use_rosa_scan = False
    cfg.use_cuda_kernel = False
    model = YufMusicGen(cfg)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    block = model.blocks[0]
    tm = block.time_mix
    rosa = block.rosa

    x0 = model.token_embedding.weight[token]
    dump("embed", x0)

    ln0 = F.layer_norm(
        x0, (cfg.d_model,), block.norm_time.weight, block.norm_time.bias, 1e-5
    )
    dump("ln0", ln0)

    prev = torch.zeros_like(ln0)
    delta = ln0 - prev
    mr, mw, mk, mv, ma, mg = (
        tm.mix_r,
        tm.mix_w,
        tm.mix_k,
        tm.mix_v,
        tm.mix_a,
        tm.mix_g,
    )
    xr = ln0 + delta * mr
    xw = ln0 + delta * mw
    xk = ln0 + delta * mk
    xv = ln0 + delta * mv
    xa = ln0 + delta * ma
    xg = ln0 + delta * mg
    mix = torch.stack([xr, xw, xk, xv, xa, xg], dim=0)
    dump("mix", mix)

    r = xr @ tm.receptance.weight.t()
    k = xk @ tm.key.weight.t()
    v = xv @ tm.value.weight.t()
    w = tm.w0 + torch.tanh(xw @ tm.w1) @ tm.w2
    a = torch.sigmoid(tm.a0 + (xa @ tm.a1) @ tm.a2)
    g = torch.sigmoid((xg @ tm.g1) @ tm.g2)
    for name, tensor in (("r", r), ("k", k), ("v", v), ("w", w), ("a", a), ("g", g)):
        dump(name, tensor)

    heads, hs = cfg.n_heads, cfg.head_size
    k_heads = F.normalize((k * tm.k_k).view(heads, hs), dim=-1, p=2.0, eps=1e-12)
    a_heads = a.view(heads, hs)
    k_adj = (k * (1.0 + (a - 1.0) * tm.k_a)).view(heads, hs)
    memory = torch.zeros(heads, hs, hs)
    retention = torch.exp(-0.6065306597 * torch.sigmoid(w)).view(heads, hs)
    state_a = torch.einsum("hj,hij->hi", -k_heads, memory)
    memory = (
        memory * retention.unsqueeze(-2)
        + state_a.unsqueeze(-1) * (k_heads * a_heads).unsqueeze(-2)
        + v.view(heads, hs).unsqueeze(-1) * k_adj.unsqueeze(-2)
    )
    o = torch.einsum("hij,hj->hi", memory, r.view(heads, hs))
    dump("o", o)
    dump("tm_mem", memory)

    ln_o = F.layer_norm(
        o.flatten(), (cfg.d_model,), tm.out_norm.weight, tm.out_norm.bias, 1e-5
    )
    dump("ln_o", ln_o)
    h1 = x0 + (ln_o * g) @ tm.output.weight.t()
    dump("h1", h1)

    ln1 = F.layer_norm(h1, (cfg.d_model,), block.norm_rosa.weight, block.norm_rosa.bias, 1e-5)
    dump("ln1", ln1)
    cand = ln1 @ rosa.input.weight.t() + rosa.input.bias
    write = ln1 @ rosa.write_gate.weight.t() + rosa.write_gate.bias
    read_gate = ln1 @ rosa.read_gate.weight.t() + rosa.read_gate.bias
    for name, tensor in (("cand", cand), ("write", write), ("read_gate", read_gate)):
        dump(name, tensor)

    decay = torch.sigmoid(rosa.decay)
    direction = F.normalize(rosa.householder, dim=0)
    rosa_mem = decay * torch.zeros_like(cand) + torch.sigmoid(write) * torch.tanh(cand)
    dump("rosa_mem", rosa_mem)
    proj = torch.sum(rosa_mem * direction, dim=-1, keepdim=True)
    read = (rosa_mem - 2.0 * proj * direction) * torch.sigmoid(read_gate)
    dump("read", read)
    h2 = h1 + read @ rosa.output.weight.t()
    dump("h2", h2)

    ln2 = F.layer_norm(h2, (cfg.d_model,), block.norm_ffn.weight, block.norm_ffn.bias, 1e-5)
    dump("ln2", ln2)
    fin = ln2 @ block.ffn_in.weight.t()
    fgate = ln2 @ block.ffn_gate.weight.t()
    dump("fin", fin)
    dump("fgate", fgate)
    h3 = h2 + (F.silu(fin) * fgate) @ block.ffn_out.weight.t()
    dump("h3", h3)

    # Per-layer outputs through the full network (first token, zero states).
    hidden = x0
    for i, blk in enumerate(model.blocks):
        tm_in = F.layer_norm(
            hidden.unsqueeze(0).unsqueeze(0),
            (cfg.d_model,),
            blk.norm_time.weight,
            blk.norm_time.bias,
            1e-5,
        )
        time_out = blk.time_mix(tm_in, None)[0]
        hidden = hidden + time_out.squeeze()
        rosa_in = F.layer_norm(
            hidden.unsqueeze(0).unsqueeze(0),
            (cfg.d_model,),
            blk.norm_rosa.weight,
            blk.norm_rosa.bias,
            1e-5,
        )
        rosa_out = blk.rosa(rosa_in, None)[0]
        hidden = hidden + rosa_out.squeeze()
        ffn_in = F.layer_norm(
            hidden.unsqueeze(0).unsqueeze(0),
            (cfg.d_model,),
            blk.norm_ffn.weight,
            blk.norm_ffn.bias,
            1e-5,
        )
        hidden = hidden + blk.ffn_out(F.silu(blk.ffn_in(ffn_in)) * blk.ffn_gate(ffn_in)).squeeze()
        dump(f"h3_L{i}", hidden)

    final = F.layer_norm(hidden, (cfg.d_model,), model.final_norm.weight, model.final_norm.bias, 1e-5)
    dump("final", final)
    logits = final @ model.lm_head.weight.t()
    dump("logits", logits)


if __name__ == "__main__":
    main()
