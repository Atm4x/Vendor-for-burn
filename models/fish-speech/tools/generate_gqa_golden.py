from pathlib import Path

import torch
import torch.nn.functional as F


def compute(batch, query_heads, kv_heads, q_len, kv_len, head_dim):
    q_nums = torch.arange(batch * query_heads * q_len * head_dim, dtype=torch.float32)
    kv_nums = torch.arange(batch * kv_heads * kv_len * head_dim, dtype=torch.float32)
    q = (q_nums.sin() * 0.55 + q_nums.cos() * 0.05).reshape(batch, query_heads, q_len, head_dim)
    k = (kv_nums.sin() * 0.25 + kv_nums.cos() * 0.1).reshape(batch, kv_heads, kv_len, head_dim).bfloat16().float()
    v = (kv_nums.cos() * 0.8 - kv_nums.sin() * 0.04).reshape(batch, kv_heads, kv_len, head_dim).bfloat16().float()
    start = kv_len - q_len
    mask = torch.full((q_len, kv_len), float('-inf'))
    for pos in range(q_len):
        mask[pos, :start + pos + 1] = 0
    out = F.scaled_dot_product_attention(
        q, k.repeat_interleave(query_heads // kv_heads, 1),
        v.repeat_interleave(query_heads // kv_heads, 1), attn_mask=mask,
        dropout_p=0.0,
    )
    return q, k, v, out.transpose(1, 2).contiguous().flatten(2)


def main():
    torch.set_num_threads(1)
    scenarios = [(2, 4, 2, 3, 3, 6), (2, 4, 2, 2, 5, 6), (1, 8, 2, 1, 7, 8)]
    lines = []
    for i, args in enumerate(scenarios):
        lines.append(f'shape{i}:' + ','.join(str(n) for n in args))
        for name, tensor in zip(('q', 'k', 'v', 'output'), compute(*args)):
            lines.append(f'{name}{i}:' + ','.join(f'{x:.9g}' for x in tensor.flatten().tolist()))
    path = Path(__file__).resolve().parents[1] / 'core' / 'fixtures' / 'gqa_pytorch.txt'
    path.write_text('\n'.join(lines) + '\n')
    print(path)


if __name__ == '__main__':
    main()
