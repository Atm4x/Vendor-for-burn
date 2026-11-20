from pathlib import Path
import json
import math
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

torch.set_num_threads(1)
torch.manual_seed(2606)
HERE = Path(__file__).resolve().parents[1] / 'fixtures'
HERE.mkdir(parents=True, exist_ok=True)
DIM = 16
TEXT_DIM = 8
MEL = 2
DEPTH = 2
HEADS = 4
HEAD_DIM = DIM // HEADS
FF = 2
VOCAB = 8
FRAMES = 4

weights = {}

def add(name, shape, scale=0.07, bias=False):
    data = torch.randn(*shape, dtype=torch.float32) * scale
    if bias:
        data *= 0.5
    weights[name] = data.contiguous()
    return data

def linear(prefix, din, dout):
    add(prefix + '.weight', (dout, din))
    add(prefix + '.bias', (dout,), bias=True)

def conv(prefix, channels, groups, width):
    add(prefix + '.weight', (channels, channels // groups, width), scale=0.03)
    add(prefix + '.bias', (channels,), bias=True)

linear('transformer.time_embed.time_mlp.0', 256, DIM)
linear('transformer.time_embed.time_mlp.2', DIM, DIM)
add('transformer.text_embed.text_embed.weight', (VOCAB + 1, TEXT_DIM))
for i in range(1):
    root = f'transformer.text_embed.text_blocks.{i}'
    conv(root + '.dwconv', TEXT_DIM, TEXT_DIM, 7)
    add(root + '.norm.weight', (TEXT_DIM,), scale=0.1)
    weights[root + '.norm.weight'] += 1
    add(root + '.norm.bias', (TEXT_DIM,), scale=0.03)
    linear(root + '.pwconv1', TEXT_DIM, TEXT_DIM * 2)
    linear(root + '.pwconv2', TEXT_DIM * 2, TEXT_DIM)
    add(root + '.grn.gamma', (1, 1, TEXT_DIM * 2), scale=0.07)
    add(root + '.grn.beta', (1, 1, TEXT_DIM * 2), scale=0.07)
linear('transformer.input_embed.proj', MEL * 2 + TEXT_DIM, DIM)
conv('transformer.input_embed.conv_pos_embed.conv1d.0', DIM, 16, 31)
conv('transformer.input_embed.conv_pos_embed.conv1d.2', DIM, 16, 31)
for i in range(DEPTH):
    root = f'transformer.transformer_blocks.{i}'
    linear(root + '.attn_norm.linear', DIM, DIM * 6)
    for suffix in ('to_q', 'to_k', 'to_v', 'to_out.0'):
        linear(root + '.attn.' + suffix, DIM, DIM)
    linear(root + '.ff.ff.0.0', DIM, DIM * FF)
    linear(root + '.ff.ff.2', DIM * FF, DIM)
linear('transformer.norm_out.linear', DIM, 2 * DIM)
linear('transformer.proj_out', DIM, MEL)

@torch.no_grad()
def lin(x, key):
    return F.linear(x, weights[key + '.weight'], weights[key + '.bias'])

@torch.no_grad()
def conv1(x, key, groups, padding):
    return F.conv1d(x, weights[key + '.weight'], weights[key + '.bias'], groups=groups, padding=padding)

def norm(x):
    return F.layer_norm(x, (x.shape[-1],), eps=1e-6)

def time_embedding(t):
    half = torch.arange(128, dtype=torch.float32)
    freqs = torch.exp(half * (-math.log(10000.0) / 127))
    sinusoids = t.reshape(-1, 1) * 1000 * freqs.reshape(1, -1)
    x = torch.cat([sinusoids.sin(), sinusoids.cos()], dim=-1)
    return lin(F.silu(lin(x, 'transformer.time_embed.time_mlp.0')),
               'transformer.time_embed.time_mlp.2')

def text_embedding(token_ids, uncond):
    ids = token_ids[:, :FRAMES] + 1
    ids = F.pad(ids, (0, FRAMES - ids.shape[1]))
    pad = ids == 0
    if uncond:
        ids = torch.zeros_like(ids)
    x = F.embedding(ids, weights['transformer.text_embed.text_embed.weight'])
    pos = torch.arange(FRAMES, dtype=torch.float32)[:, None]
    bands = torch.exp(torch.arange(0, TEXT_DIM, 2, dtype=torch.float32) * -math.log(10000) / TEXT_DIM)
    angle = pos * bands
    positions = torch.cat((angle.cos(), angle.sin()), -1)
    x = (x + positions.unsqueeze(0)).masked_fill(pad.unsqueeze(-1), 0)
    for i in range(1):
        root = f'transformer.text_embed.text_blocks.{i}'
        residual = x
        x = conv1(x.transpose(1, 2), root + '.dwconv', TEXT_DIM, 3).transpose(1, 2)
        x = F.layer_norm(x, (TEXT_DIM,), weights[root + '.norm.weight'], weights[root + '.norm.bias'], eps=1e-6)
        x = F.gelu(lin(x, root + '.pwconv1'))
        response = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        response = response / (response.mean(-1, keepdim=True) + 1e-6)
        x = x + weights[root + '.grn.gamma'] * x * response + weights[root + '.grn.beta']
        x = (residual + lin(x, root + '.pwconv2')).masked_fill(pad.unsqueeze(-1), 0)
    return x

def input_embedding(noisy, cond, text, uncond):
    if uncond:
        cond = torch.zeros_like(cond)
    x = lin(torch.cat((noisy, cond, text), -1), 'transformer.input_embed.proj')
    hidden = F.mish(conv1(x.transpose(1, 2), 'transformer.input_embed.conv_pos_embed.conv1d.0', 16, 15))
    hidden = F.mish(conv1(hidden, 'transformer.input_embed.conv_pos_embed.conv1d.2', 16, 15))
    return x + hidden.transpose(1, 2)

def rope(x):
    bands = torch.exp(torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) * -math.log(10000) / HEAD_DIM)
    angle = torch.arange(FRAMES, dtype=torch.float32)[:, None] * bands[None, :]
    cos = torch.cat((angle.cos(), angle.cos()), dim=-1)[None, None]
    sin = torch.cat((angle.sin(), angle.sin()), dim=-1)[None, None]
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), -1) * sin

def backbone(x, time):
    for i in range(DEPTH):
        root = f'transformer.transformer_blocks.{i}'
        mod = lin(F.silu(time), root + '.attn_norm.linear').chunk(6, 1)
        att_in = norm(x) * (1 + mod[1].unsqueeze(1)) + mod[0].unsqueeze(1)
        q = rope(lin(att_in, root + '.attn.to_q').reshape(2, FRAMES, HEADS, HEAD_DIM).transpose(1, 2))
        k = rope(lin(att_in, root + '.attn.to_k').reshape(2, FRAMES, HEADS, HEAD_DIM).transpose(1, 2))
        v = lin(att_in, root + '.attn.to_v').reshape(2, FRAMES, HEADS, HEAD_DIM).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, FRAMES, DIM)
        x = x + mod[2].unsqueeze(1) * lin(att, root + '.attn.to_out.0')
        ff_in = norm(x) * (1 + mod[4].unsqueeze(1)) + mod[3].unsqueeze(1)
        ff = lin(F.gelu(lin(ff_in, root + '.ff.ff.0.0'), approximate='tanh'), root + '.ff.ff.2')
        x = x + mod[5].unsqueeze(1) * ff
    final_mod = lin(F.silu(time), 'transformer.norm_out.linear').chunk(2, dim=1)
    x = norm(x) * (1 + final_mod[0].unsqueeze(1)) + final_mod[1].unsqueeze(1)
    return lin(x, 'transformer.proj_out')

noisy = torch.randn(1, FRAMES, MEL) * 0.4
condition = torch.randn(1, FRAMES, MEL) * 0.2
tokens = torch.tensor([[2, 4, 1]], dtype=torch.int64)
t = 0.37
cfg = 2.1
conditional_text = text_embedding(tokens, False)
unconditional_text = text_embedding(tokens, True)
conditional_input = input_embedding(noisy, condition, conditional_text, False)
unconditional_input = input_embedding(noisy, condition, unconditional_text, True)
both = torch.cat((conditional_input, unconditional_input), 0)
times = time_embedding(torch.tensor([t], dtype=torch.float32)).repeat(2, 1)
outputs = backbone(both, times)
combined = outputs[0:1] + cfg * (outputs[0:1] - outputs[1:2])

fixture = {
    'config': {'dim':DIM,'depth':DEPTH,'heads':HEADS,'head_dim':HEAD_DIM,'text_dim':TEXT_DIM,
               'ff_mult':FF,'conv_layers':1,'mel_channels':MEL},
    'noisy': noisy.flatten().tolist(), 'reference': condition.flatten().tolist(),
    'token_ids': tokens.flatten().tolist(), 'timestep': t, 'cfg_strength': cfg,
    'conditional_text': conditional_text.flatten().tolist(),
    'unconditional_text': unconditional_text.flatten().tolist(),
    'conditional_input': conditional_input.flatten().tolist(),
    'unconditional_input': unconditional_input.flatten().tolist(),
    'conditional_output': outputs[0:1].flatten().tolist(),
    'unconditional_output': outputs[1:2].flatten().tolist(),
    'cfg_output': combined.flatten().tolist(),
}
save_file({key: tensor.detach().clone() for key, tensor in weights.items()}, HERE / 'dit_full_small.safetensors')
(HERE / 'dit_full_expected.json').write_text(json.dumps(fixture, indent=2) + '\n', encoding='utf-8')
print('F5 full DiT PyTorch fixture generated:', len(weights), 'weight tensors; output:', fixture['cfg_output'])
