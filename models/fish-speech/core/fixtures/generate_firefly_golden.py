"""Reproduce numerical CPU fixtures from the unmodified Fish Speech 1.5 firefly.py."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vendor/Fish-speech-pipeline-main/fish_speech_lib/fish_speech/models/vqgan/modules/firefly.py"
if not SOURCE.exists():
    raise RuntimeError(f"missing original source: {SOURCE}")
spec = importlib.util.spec_from_file_location("fish_firefly_reference", SOURCE)
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)
torch.set_num_threads(1)


def values(n: int, seed: int, weight: bool = False) -> torch.Tensor:
    if weight:
        return torch.tensor([((i * 13 + seed) % 37) / 40.0 - .45 for i in range(n)], dtype=torch.float32)
    return torch.tensor([((i * 19 + seed) % 29) / 11.0 - 1.2 for i in range(n)], dtype=torch.float32)


def bias_values(n: int, seed: int) -> torch.Tensor:
    return torch.tensor([((i * 7 + seed) % 13) / 30.0 - .2 for i in range(n)], dtype=torch.float32)


def assign_conv(layer, seed: int) -> None:
    with torch.no_grad():
        layer.weight.copy_(values(layer.weight.numel(), seed, True).view_as(layer.weight))
        if layer.bias is not None:
            layer.bias.copy_(bias_values(layer.bias.numel(), seed + 1))


def write_fixture(lines: list[str], label: str, x: torch.Tensor) -> None:
    lines.append(f"{label} {','.join(map(str, x.shape))} " + " ".join(f"{v:.9g}" for v in x.flatten().tolist()))


def main() -> None:
    torch.manual_seed(31)
    cases = [
        ("conv_regular", 1, 2, 3, 7, 3, 2, 1, 1, 1),
        ("conv_depthwise", 1, 3, 3, 5, 7, 1, 1, 3, 2),
        ("conv_dilated", 1, 2, 2, 6, 3, 1, 3, 1, 3),
        ("conv_short", 2, 1, 1, 1, 7, 1, 1, 1, 4),
        ("conv_grouped", 2, 4, 6, 9, 4, 2, 1, 2, 5),
    ]
    lines = ["# Label batch,channels,frames flat values. Generated from original fish_speech/modules/firefly.py."]
    for name, batch, inc, outc, time, kernel, stride, dilation, groups, seed in cases:
        layer = ref.FishConvNet(inc, outc, kernel, stride=stride, dilation=dilation, groups=groups).eval()
        assign_conv(layer.conv, seed)
        x = values(batch * inc * time, 3).view(batch, inc, time)
        write_fixture(lines, name, layer(x).detach())
    t_cases = [
        ("transconv_regular", 1, 3, 2, 5, 4, 2, 1, 6),
        ("transconv_upsample", 2, 1, 1, 3, 16, 8, 1, 7),
        ("transconv_dilated", 1, 2, 3, 4, 5, 2, 2, 8),
    ]
    for name, batch, inc, outc, time, kernel, stride, dilation, seed in t_cases:
        layer = ref.FishTransConvNet(inc, outc, kernel, stride=stride, dilation=dilation).eval()
        assign_conv(layer.conv, seed)
        x = values(batch * inc * time, 3).view(batch, inc, time)
        write_fixture(lines, name, layer(x).detach())

    # Exercise exact ConvNeXt module equations, including GELU and gamma scale.
    for channels, frames, seed in [(3, 4, 10), (2, 1, 11)]:
        block = ref.ConvNeXtBlock(dim=channels, mlp_ratio=2, kernel_size=7, drop_path=0.0).eval()
        with torch.no_grad():
            assign_conv(block.dwconv.conv, seed)
            block.norm.weight.copy_(values(channels, seed + 1, True))
            block.norm.bias.copy_(bias_values(channels, seed + 2))
            block.pwconv1.weight.copy_(values(block.pwconv1.weight.numel(), seed + 3, True).view_as(block.pwconv1.weight))
            block.pwconv1.bias.copy_(bias_values(block.pwconv1.bias.numel(), seed + 4))
            block.pwconv2.weight.copy_(values(block.pwconv2.weight.numel(), seed + 5, True).view_as(block.pwconv2.weight))
            block.pwconv2.bias.copy_(bias_values(block.pwconv2.bias.numel(), seed + 6))
            block.gamma.copy_(values(channels, seed + 7, True))
        x = values(channels * frames, 3).view(1, channels, frames)
        write_fixture(lines, f"convnext_c{channels}_t{frames}", block(x).detach())

    for channels, frames, seed in [(2, 6, 12)]:
        block = ref.ResBlock1(channels, kernel_size=3, dilation=(1, 3, 5)).eval()
        # weight_norm wrappers carry parametrizations; set the effective normalized kernel.
        with torch.no_grad():
            for i, (conv1, conv2) in enumerate(zip(block.convs1, block.convs2)):
                for j, conv in enumerate((conv1, conv2)):
                    module = conv.conv
                    module.parametrizations.weight.original1.fill_(1.0)
                    target = values(module.weight.numel(), seed + i * 2 + j, True).view_as(module.weight)
                    # Removal with leave_parametrized=False would restore original weights rather than our target.
                    # Set g to row L2 norm and v to target; then effective weight is target.
                    module.parametrizations.weight.original0.copy_(target.norm(dim=(1,2), keepdim=True))
                    module.parametrizations.weight.original1.copy_(target)
                    module.bias.copy_(bias_values(module.bias.numel(), seed + i * 2 + j + 1))
        x = values(channels * frames, 3).view(1, channels, frames)
        write_fixture(lines, "resblock_c2_t6", block(x).detach())

    generator = ref.HiFiGANGenerator(
        hop_length=4, upsample_rates=(2, 2), upsample_kernel_sizes=(4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        num_mels=3, upsample_initial_channel=8,
        pre_conv_kernel_size=7, post_conv_kernel_size=7,
    ).eval()
    next_seed = 20
    with torch.no_grad():
        for module in generator.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                target = values(module.weight.numel(), next_seed, True).view_as(module.weight)
                if hasattr(module, "parametrizations") and hasattr(module.parametrizations, "weight"):
                    module.parametrizations.weight.original0.copy_(
                        target.norm(dim=(1, 2), keepdim=True)
                    )
                    module.parametrizations.weight.original1.copy_(target)
                else:
                    module.weight.copy_(target)
                module.bias.copy_(bias_values(module.bias.numel(), next_seed + 1))
                next_seed += 1
    assert next_seed == 60, next_seed
    x = values(3 * 3, 3).view(1, 3, 3)
    write_fixture(lines, "hifigan_toy_full", generator(x).detach())

    path = pathlib.Path(__file__).with_suffix(".txt")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"created {path} with {len(lines) - 1} fixtures")


if __name__ == "__main__":
    main()
