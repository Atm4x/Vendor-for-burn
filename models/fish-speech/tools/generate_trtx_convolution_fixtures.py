import argparse
import importlib.util
from pathlib import Path

import torch


def load_firefly(source):
    spec = importlib.util.spec_from_file_location("fish_firefly_original", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def values(count, divisor, offset):
    return torch.tensor([((i * 17 + offset) % 37 - 18) / divisor for i in range(count)], dtype=torch.float32)


def build_fixtures(firefly):
    cases = [
        ("grouped_dilated", "conv", 1, 4, 6, 9, 3, 1, 2, 2),
        ("odd_stride", "conv", 2, 2, 2, 7, 2, 2, 1, 1),
        ("depthwise_short", "conv", 1, 3, 3, 4, 7, 1, 1, 3),
        ("transconv_stride2", "transconv", 1, 2, 3, 5, 4, 2, 1, 1),
        ("transconv_stride8", "transconv", 1, 1, 2, 3, 16, 8, 1, 1),
        ("linear_channels", "linear", 2, 4, 8, 7, 1, 1, 1, 1),
    ]
    lines = []
    for name, kind, batch, cin, cout, frames, kernel, stride, dilation, groups in cases:
        if kind == "linear":
            layer = torch.nn.Linear(cin, cout)
            weight, bias = layer.weight, layer.bias
        else:
            if kind == "conv":
                layer = firefly.FishConvNet(cin, cout, kernel, dilation=dilation, stride=stride, groups=groups)
            else:
                layer = firefly.FishTransConvNet(cin, cout, kernel, dilation=dilation, stride=stride)
            weight, bias = layer.conv.weight, layer.conv.bias
        w = values(weight.numel(), 29.0, 3).reshape_as(weight)
        b = values(bias.numel(), 19.0, 11)
        x = values(batch * cin * frames, 13.0, 7).reshape(batch, cin, frames)
        with torch.no_grad():
            weight.copy_(w)
            bias.copy_(b)
            y = layer(x.transpose(1, 2)).transpose(1, 2) if kind == "linear" else layer(x)
        lines.append(" ".join(map(str, (name, kind, batch, cin, cout, frames, kernel, stride, dilation, groups, y.shape[-1]))))
        for label, arr in [("input", x), ("weight", w), ("bias", b), ("output", y)]:
            lines.append(label + " " + " ".join(f"{float(v):.9g}" for v in arr.flatten()))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(build_fixtures(load_firefly(args.source)), encoding="utf-8")
