import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def make_text_front_fixture(checkpoint: Path | None):
    torch.set_num_threads(1)
    if checkpoint is None:
        width, hidden, batch, time = 4, 3, 2, 3
        phone_weight = torch.tensor(
            [[0.1, -0.3, 0.2, 0.05], [-0.5, 0.2, 0.4, -0.1], [0.3, 0.15, -0.2, 0.45]],
            dtype=torch.float32,
        )
        phone_bias = torch.tensor([0.2, -0.12, 0.05], dtype=torch.float32)
        pitch_weight = (torch.arange(256 * hidden).reshape(256, hidden) % 19 - 9).float() / 150
    else:
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weight = data["weight"]
        phone_weight = weight["enc_p.emb_phone.weight"].float().contiguous()
        phone_bias = weight["enc_p.emb_phone.bias"].float().contiguous()
        pitch_weight = weight["enc_p.emb_pitch.weight"].float().contiguous()
        hidden, width = phone_weight.shape
        batch, time = 1, 4

    phone = torch.arange(batch * time * width, dtype=torch.float32).reshape(batch, time, width)
    phone = ((phone % 29) - 14) / 17
    pitch = (torch.arange(batch * time, dtype=torch.int64).reshape(batch, time) * 7 + 1) % 256
    lengths = torch.tensor([time] + [time - 2] * (batch - 1), dtype=torch.int64)
    x = F.leaky_relu(
        (F.linear(phone, phone_weight, phone_bias) + F.embedding(pitch, pitch_weight))
        * math.sqrt(hidden),
        negative_slope=0.1,
    ).transpose(1, 2)
    mask = (torch.arange(time).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).float()

    return {
        "batch": batch,
        "time": time,
        "in_channels": width,
        "hidden_channels": hidden,
        "phone": phone.flatten().tolist(),
        "pitch": pitch.flatten().tolist(),
        "lengths": lengths.tolist(),
        "phone_weight": phone_weight.flatten().tolist(),
        "phone_bias": phone_bias.tolist(),
        "pitch_weight": pitch_weight.flatten().tolist(),
        "expected_features": x.contiguous().flatten().tolist(),
        "expected_mask": mask.flatten().tolist(),
    }


def make_gate_fixture():
    a = torch.tensor(
        [[[0.3, -0.5, 1.1], [0.8, -0.9, 0.2], [-0.6, 0.4, 0.7], [0.5, -0.3, -0.2]]]
    )
    b = torch.tensor(
        [[[-0.7, 0.8, -0.1], [0.2, 0.6, -0.3], [0.1, -0.1, 0.2], [0.3, -0.5, 0.9]]]
    )
    combined = a + b
    expected = torch.tanh(combined[:, :2, :]) * torch.sigmoid(combined[:, 2:, :])
    return {"a": a.flatten().tolist(), "b": b.flatten().tolist(), "expected": expected.flatten().tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--encoder-layer", action="store_true")
    args = parser.parse_args()
    output = args.output or Path(__file__).parent.parent / "tests" / "fixtures" / "text_front.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture = (make_encoder_layer_fixture(args.checkpoint) if args.encoder_layer else
               make_attention_fixture(args.checkpoint) if args.attention else
               make_text_front_fixture(args.checkpoint))
    output.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    if args.checkpoint is None and not args.attention and not args.encoder_layer:
        (output.parent / "wavenet_gate.json").write_text(json.dumps(make_gate_fixture(), indent=2), encoding="utf-8")
    print(output)



def make_attention_fixture(checkpoint: Path | None):
    torch.set_num_threads(1)
    torch.manual_seed(371)
    if checkpoint is None:
        batch, time, channels, heads, window = 2, 7, 8, 2, 2
        weight = {}
        for name in ("q", "k", "v", "o"):
            weight[f"conv_{name}.weight"] = torch.randn(channels, channels, 1) * 0.12
            weight[f"conv_{name}.bias"] = torch.randn(channels) * 0.06
        weight["emb_rel_k"] = torch.randn(1, window * 2 + 1, channels // heads) * 0.15
        weight["emb_rel_v"] = torch.randn(1, window * 2 + 1, channels // heads) * 0.15
        lengths = [time, time - 3]
    else:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)["weight"]
        prefix = "enc_p.encoder.attn_layers.0."
        channels, heads, window, batch, time = 192, 2, 10, 1, 14
        weight = {
            name: tensor.float().contiguous()
            for key, tensor in raw.items()
            if key.startswith(prefix) and (name := key[len(prefix):])
        }
        lengths = [time - 2]
    features = torch.randn(batch, channels, time, dtype=torch.float32) * 0.3
    mask = (torch.arange(time)[None, :] < torch.tensor(lengths)[:, None]).float()[:, None, :]
    features = features * mask
    k_channels = channels // heads

    def conv(name, values):
        return F.conv1d(values, weight[f"conv_{name}.weight"], weight[f"conv_{name}.bias"])

    def reshaped(name):
        return conv(name, features).reshape(batch, heads, k_channels, time).transpose(2, 3)

    q, k, v = reshaped("q"), reshaped("k"), reshaped("v")
    scaled = q / math.sqrt(k_channels)
    scores = torch.matmul(scaled, k.transpose(-1, -2))

    def get_relative_embeddings(values):
        pad = max(time - window - 1, 0)
        start = max(window + 1 - time, 0)
        padded = F.pad(values, [0, 0, pad, pad]) if pad else values
        return padded[:, start:start + 2 * time - 1]

    relative_k = get_relative_embeddings(weight["emb_rel_k"])
    relative_logits = torch.matmul(scaled, relative_k.unsqueeze(0).transpose(-1, -2))
    p = F.pad(relative_logits, [0, 1])
    flat = F.pad(p.reshape(batch, heads, time * 2 * time), [0, time - 1])
    relative_scores = flat.reshape(batch, heads, time + 1, 2 * time - 1)[:, :, :time, time - 1:]
    scores = scores + relative_scores
    attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
    scores = scores.masked_fill(attn_mask == 0, -1e4)
    attention = F.softmax(scores, dim=-1)
    outputs = torch.matmul(attention, v)
    relative_v = get_relative_embeddings(weight["emb_rel_v"])
    padded = F.pad(attention, [0, time - 1])
    padded = F.pad(padded.reshape(batch, heads, time * (2 * time - 1)), [time, 0])
    weights = padded.reshape(batch, heads, time, 2 * time)[:, :, :, 1:]
    outputs = outputs + torch.matmul(weights, relative_v.unsqueeze(0))
    outputs = outputs.transpose(2, 3).contiguous().reshape(batch, channels, time)
    outputs = conv("o", outputs)
    result = dict(batch=batch, time=time, channels=channels, heads=heads, window=window,
                  input=features.flatten().tolist(), lengths=lengths, expected=outputs.flatten().tolist())
    for name, long_name in (("q", "query"), ("k", "key"), ("v", "value"), ("o", "output")):
        result[f"{long_name}_weight"] = weight[f"conv_{name}.weight"].flatten().tolist()
        result[f"{long_name}_bias"] = weight[f"conv_{name}.bias"].flatten().tolist()
    result["relative_keys"] = weight["emb_rel_k"].flatten().tolist()
    result["relative_values"] = weight["emb_rel_v"].flatten().tolist()
    return result


def make_encoder_layer_fixture(checkpoint: Path | None):
    fixture = make_attention_fixture(checkpoint)
    torch.manual_seed(181)
    channels = fixture["channels"]
    filter_channels = 768 if checkpoint else 16
    kernel = 3
    if checkpoint:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)["weight"]
        prefix = "enc_p.encoder."
        load = lambda key: raw[prefix + key].float().contiguous()
        conv1_weight = load("ffn_layers.0.conv_1.weight")
        conv1_bias = load("ffn_layers.0.conv_1.bias")
        conv2_weight = load("ffn_layers.0.conv_2.weight")
        conv2_bias = load("ffn_layers.0.conv_2.bias")
        norm1_gamma = load("norm_layers_1.0.gamma")
        norm1_beta = load("norm_layers_1.0.beta")
        norm2_gamma = load("norm_layers_2.0.gamma")
        norm2_beta = load("norm_layers_2.0.beta")
    else:
        conv1_weight = torch.randn(filter_channels, channels, kernel) * 0.10
        conv1_bias = torch.randn(filter_channels) * 0.03
        conv2_weight = torch.randn(channels, filter_channels, kernel) * 0.10
        conv2_bias = torch.randn(channels) * 0.03
        norm1_gamma = torch.ones(channels) + torch.randn(channels) * 0.05
        norm1_beta = torch.randn(channels) * 0.03
        norm2_gamma = torch.ones(channels) + torch.randn(channels) * 0.05
        norm2_beta = torch.randn(channels) * 0.03
    batch,time = fixture["batch"],fixture["time"]
    x = torch.tensor(fixture["input"], dtype=torch.float32).reshape(batch, channels, time)
    mask = (torch.arange(time)[None, :] < torch.tensor(fixture["lengths"])[:, None]).float()[:, None, :]
    attn = torch.tensor(fixture["expected"], dtype=torch.float32).reshape(batch, channels, time)
    x = F.layer_norm((x + attn).transpose(1, 2), (channels,), norm1_gamma, norm1_beta, 1e-5).transpose(1, 2)
    hidden = F.conv1d(x * mask, conv1_weight, conv1_bias, padding=(kernel - 1) // 2)
    hidden = torch.relu(hidden)
    y = F.conv1d(hidden * mask, conv2_weight, conv2_bias, padding=(kernel - 1) // 2) * mask
    output = F.layer_norm((x + y).transpose(1, 2), (channels,), norm2_gamma, norm2_beta, 1e-5).transpose(1, 2)
    output = output * mask
    fixture.update(
        filter_channels=filter_channels,
        kernel=kernel,
        norm_attention_gamma=norm1_gamma.tolist(),
        norm_attention_beta=norm1_beta.tolist(),
        norm_feed_forward_gamma=norm2_gamma.tolist(),
        norm_feed_forward_beta=norm2_beta.tolist(),
        conv1_weight=conv1_weight.flatten().tolist(),
        conv1_bias=conv1_bias.tolist(),
        conv2_weight=conv2_weight.flatten().tolist(),
        conv2_bias=conv2_bias.tolist(),
        expected_encoder=output.contiguous().flatten().tolist(),
    )
    return fixture


if __name__ == "__main__":
    main()
