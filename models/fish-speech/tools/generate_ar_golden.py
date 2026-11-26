import math
from pathlib import Path

import torch


def precompute_freqs_cis(seq_len, dim, base):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len)
    complex_values = torch.polar(torch.ones_like(torch.outer(positions, freqs)), torch.outer(positions, freqs))
    return torch.stack([complex_values.real, complex_values.imag], -1).to(torch.bfloat16)


def apply_rotary_emb(x, freqs_cis):
    source = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs = freqs_cis.view(1, source.size(1), 1, source.size(3), 2)
    output = torch.stack([
        source[..., 0] * freqs[..., 0] - source[..., 1] * freqs[..., 1],
        source[..., 1] * freqs[..., 0] + source[..., 0] * freqs[..., 1],
    ], -1)
    return output.flatten(3).to(x.dtype)


def main():
    torch.set_num_threads(1)
    batch, frames, qheads, kvheads, head_dim, position = 2, 3, 4, 2, 6, 37
    width = (qheads + 2 * kvheads) * head_dim
    numbers = torch.arange(batch * frames * width, dtype=torch.float32)
    packed = (numbers.sin() * 1.25 + numbers.cos() * 0.11).reshape(batch, frames, width)
    q, k, v = packed.split([qheads * head_dim, kvheads * head_dim, kvheads * head_dim], -1)
    q = q.view(batch, frames, qheads, head_dim)
    k = k.view(batch, frames, kvheads, head_dim)
    v = v.view(batch, frames, kvheads, head_dim)
    freq = precompute_freqs_cis(position + frames, head_dim, 10000)[position:position + frames]
    q_rot = apply_rotary_emb(q, freq).transpose(1, 2).contiguous()
    k_rot = apply_rotary_emb(k, freq).transpose(1, 2).contiguous()
    outputs = {
        "packed": packed,
        "q": q.transpose(1, 2).contiguous(),
        "k": k.transpose(1, 2).contiguous(),
        "v": v.transpose(1, 2).contiguous(),
        "q_rot": q_rot,
        "k_rot": k_rot,
    }
    target = Path(__file__).resolve().parents[1] / "core" / "fixtures" / "ar_qkv_rope_pytorch.txt"
    target.write_text("\n".join(f"{name}:" + ",".join(f"{x:.9g}" for x in value.flatten().tolist()) for name, value in outputs.items()) + "\n")
    print(target)


if __name__ == "__main__":
    main()
