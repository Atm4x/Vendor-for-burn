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
    args = parser.parse_args()
    output = args.output or Path(__file__).parent.parent / "tests" / "fixtures" / "text_front.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(make_text_front_fixture(args.checkpoint), indent=2), encoding="utf-8")
    if args.checkpoint is None:
        (output.parent / "wavenet_gate.json").write_text(json.dumps(make_gate_fixture(), indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
