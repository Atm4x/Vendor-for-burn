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

def backbone(x, time, trace=None):
    for i in range(DEPTH):
        root = f'transformer.transformer_blocks.{i}'
        mod = lin(F.silu(time), root + '.attn_norm.linear').chunk(6, 1)
        att_in = norm(x) * (1 + mod[1].unsqueeze(1)) + mod[0].unsqueeze(1)
        q = rope(lin(att_in, root + '.attn.to_q').reshape(x.shape[0], FRAMES, HEADS, HEAD_DIM).transpose(1, 2))
        k = rope(lin(att_in, root + '.attn.to_k').reshape(x.shape[0], FRAMES, HEADS, HEAD_DIM).transpose(1, 2))
        v = lin(att_in, root + '.attn.to_v').reshape(x.shape[0], FRAMES, HEADS, HEAD_DIM).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(x.shape[0], FRAMES, DIM)
        x = x + mod[2].unsqueeze(1) * lin(att, root + '.attn.to_out.0')
        ff_in = norm(x) * (1 + mod[4].unsqueeze(1)) + mod[3].unsqueeze(1)
        ff = lin(F.gelu(lin(ff_in, root + '.ff.ff.0.0'), approximate='tanh'), root + '.ff.ff.2')
        x = x + mod[5].unsqueeze(1) * ff
        if trace is not None:
            trace.append(x.flatten().tolist())
    final_mod = lin(F.silu(time), 'transformer.norm_out.linear').chunk(2, dim=1)
    x = norm(x) * (1 + final_mod[0].unsqueeze(1)) + final_mod[1].unsqueeze(1)
    return lin(x, 'transformer.proj_out')

def case(batch, frames, tokens, timestep, cfg_strength, *, reference_frames=None, steps=None, timeline=None, trace_layers=False):
    global FRAMES
    FRAMES = frames
    noisy = torch.randn(batch, frames, MEL) * 0.4
    reference = torch.randn(batch, frames, MEL) * 0.2
    token_ids = torch.tensor(tokens, dtype=torch.int64)
    assert token_ids.shape[0] == batch

    trace = [] if trace_layers else None

    def forward(noisy_tensor, time, reference_input=reference):
        text_cond = text_embedding(token_ids, False)
        text_uncond = text_embedding(token_ids, True)
        input_cond = input_embedding(noisy_tensor, reference_input, text_cond, False)
        input_uncond = input_embedding(noisy_tensor, reference_input, text_uncond, True)
        both = torch.cat((input_cond, input_uncond), 0)
        time_vec = time_embedding(torch.full((batch,), float(time), dtype=torch.float32))
        outputs = backbone(both, torch.cat((time_vec, time_vec), 0), trace)
        cond = outputs[:batch]
        uncond = outputs[batch:]
        combined = cond + float(cfg_strength) * (cond - uncond)
        return text_cond, text_uncond, input_cond, input_uncond, cond, uncond, combined

    a, b, c, d, e, f, g = forward(noisy, timestep)
    fixture = {
        'config': {'dim': DIM, 'depth': DEPTH, 'heads': HEADS, 'head_dim': HEAD_DIM,
                   'text_dim': TEXT_DIM, 'ff_mult': FF, 'conv_layers': 1, 'mel_channels': MEL},
        'batch':batch, 'frames': frames, 'token_length':len(tokens[0]),
        'noisy':noisy.flatten().tolist(), 'reference':reference.flatten().tolist(),
        'token_ids':token_ids.flatten().tolist(), 'timestep': timestep, 'cfg_strength': cfg_strength,
        'conditional_text':a.flatten().tolist(), 'unconditional_text':b.flatten().tolist(),
        'conditional_input':c.flatten().tolist(), 'unconditional_input':d.flatten().tolist(),
        'conditional_output':e.flatten().tolist(), 'unconditional_output':f.flatten().tolist(),
        'cfg_output':g.flatten().tolist(),
    }
    if trace_layers:
        fixture['layer_states'] = trace
    if steps is not None:
        assert 0 < reference_frames < frames
        reference_conditioned = reference.clone()
        reference_conditioned[:, reference_frames:, :] = 0.0
        state = noisy.clone()
        if timeline is None:
            timeline = [float(index) / float(steps) for index in range(steps + 1)]
        assert len(timeline) == steps + 1
        snapshots = []
        for index in range(steps):
            _, _, _, _, _, _, velocity = forward(state, timeline[index], reference_conditioned)
            state = state + (timeline[index + 1] - timeline[index]) * velocity
            snapshots.append(state.flatten().tolist())
        state = torch.cat((reference_conditioned[:, :reference_frames, :],
                           state[:, reference_frames:, :]), dim=1)
        fixture.update({
            'sampling_steps':steps,
            'reference_frames':reference_frames,
            'conditioned_reference':reference_conditioned.flatten().tolist(),
            'trajectory_timesteps':timeline,
            'trajectory_states':snapshots,
            'trajectory_final':state.flatten().tolist(),
        })
    return fixture


if __name__ == '__main__':
    legacy = case(1, 4, [[2, 4, 1]], 0.37, 2.1)
    save_file({key: tensor.detach().clone() for key, tensor in weights.items()}, HERE / 'dit_full_small.safetensors')
    (HERE / 'dit_full_expected.json').write_text(json.dumps(legacy, indent=2) + '\n', encoding='utf-8')

    cases = [
        case(1, 1, [[1]], 0.0, 0.0),
        case(1, 7, [[1, 3]], 0.91, 3.5, reference_frames=2, steps=4),
        case(2, 5, [[1, 3, 2], [0, 4, -1]], 0.125, 0.75, reference_frames=2, steps=3),
        case(2, 8, [[2, 4, 1, 3, 7, 0, -1, -1, -1], [1, 5, 3, 2, 4, 7, 0, -1, -1]], 0.51, 1.75),
        case(1, 9, [[0, 3, 4, 6]], 1.0, 1.0),
    ]
    (HERE / 'dit_full_grid_expected.json').write_text(json.dumps(cases, indent=2) + '\n', encoding='utf-8')

    torch.manual_seed(22122)
    template_prefix = 'transformer.transformer_blocks.0.'
    block_tensors = [(name[len(template_prefix):], value) for name, value in weights.items()
                     if name.startswith(template_prefix)]
    for block in range(2, 22):
        prefix = f'transformer.transformer_blocks.{block}.'
        for suffix, template in block_tensors:
            weights[prefix + suffix] = torch.randn_like(template) * (0.025 if suffix.endswith('bias') else 0.07)
    DEPTH = 22
    deep = case(1, 3, [[1, 2]], 0.61, 1.9, trace_layers=True)
    save_file({name: value.detach().contiguous() for name, value in weights.items()}, HERE / 'dit_full_22_blocks.safetensors')
    (HERE / 'dit_full_22_expected.json').write_text(json.dumps(deep, indent=2) + '\n', encoding='utf-8')
    print('F5 DiT: legacy +', len(cases), 'dynamic shape/CFG cases,', sum('sampling_steps' in c for c in cases),
          'Euler trajectories, and', len(deep['layer_states']), 'layer-deep oracle')
