"""Generate deterministic FP32 parity fixtures directly from a GigaAM source checkout."""
import argparse
import json
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/zapformer")
    args = parser.parse_args()
    package = types.ModuleType("gigaam")
    package.__path__ = [str(args.upstream.resolve() / "gigaam")]
    sys.modules["gigaam"] = package
    from gigaam.encoder import ZapformerEncoder

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for causal, activation in [(True, "swish"), (False, "silu")]:
        torch.manual_seed(19 if causal else 23)
        config = dict(
            in_channels=8, frontend_channels=[2, 2, 4], embed_dim=8,
            encoder_dim=[8, 12], downsampling_factor=[1, 2],
            num_encoder_layers=[1, 2], num_heads=[2, 3], query_head_dim=[4, 4],
            value_head_dim=[3, 2], feedforward_multiple=2, conv_params=[3, 0],
            causal_frontend=causal, dual_mode=True, chunk_size=[-1, 4, 8, 16],
            window_size=[[-1, -1], [32, 0], [32, 0], [32, 0]], max_seq_len=32,
            activation=activation,
        )
        model = ZapformerEncoder(**config).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("weighted_mean.weights"):
                    parameter.copy_(torch.linspace(-0.4, 0.4, parameter.numel()))
                elif name.endswith("input_scale"):
                    parameter.fill_(0.85)
                elif name.endswith("residual_scales"):
                    parameter.copy_(torch.linspace(-0.9, 0.6, parameter.numel()))
            x = torch.randn(3, 35, 8)
            lengths = torch.tensor([35, 29, 0])
            x[1, 29:] = 100.0
            x[2] = 100.0
            frontend, frontend_lengths = model.encoder_embed(x, lengths)
            encoded, output_lengths = model(x.transpose(1, 2), lengths)
            altered = x.clone()
            altered[1, 29:] = -100.0
            altered[2] = -100.0
            torch.testing.assert_close(model(altered.transpose(1, 2), lengths)[0], encoded)
        assert torch.isfinite(encoded).all()
        tensors = {f"encoder.{k}": v.detach().contiguous().clone() for k, v in model.state_dict().items() if ".rope_bank." not in k or k.startswith("encoder.rope_bank.")}
        tensors.update({"reference.input": x, "reference.frontend": frontend, "reference.output": encoded.transpose(1, 2).contiguous()})
        filename = f"{'causal' if causal else 'noncausal'}_{activation}.safetensors"
        save_file(tensors, args.output / filename)
        cases.append(dict(file=filename, config={"_target_": "gigaam.encoder.ZapformerEncoder", **config}, lengths=lengths.tolist(), output_lengths=output_lengths.tolist()))
        assert frontend_lengths.tolist() == output_lengths.tolist()
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
