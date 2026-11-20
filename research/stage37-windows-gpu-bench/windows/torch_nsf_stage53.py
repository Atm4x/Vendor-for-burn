"""Matched synthetic NSF reference: PyTorch/cuDNN eager and CUDA Graph replay."""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def reference_inputs(fixture, frames, step, device):
    original = fixture["input_shape"][2]
    channels = fixture["input_shape"][1]
    upsample = fixture["source_shape"][2] // original
    gain = (1.0, -0.75, 0.45, 1.35)[step % 4]
    source_gain = (1.0, 0.6, -0.8, 1.2)[step % 4]
    values = [
        gain * fixture["input"][channel * original + i % original]
        + ((i + channel + step) % 9) * 0.0003
        for channel in range(channels) for i in range(frames)
    ]
    source = [source_gain * fixture["harmonic_source"][i % fixture["source_shape"][2]]
              for i in range(frames * upsample)]
    condition = None
    if fixture["condition"] is not None:
        condition = [v * gain + step * 0.0001 for v in fixture["condition"]]
    x = torch.tensor(values, dtype=torch.float32, device=device).reshape(1, channels, frames)
    s = torch.tensor(source, dtype=torch.float32, device=device).reshape(1, 1, frames * upsample)
    g = None if condition is None else torch.tensor(condition, dtype=torch.float32, device=device).reshape(fixture["condition_shape"])
    return x, s, g


class NsfReference:
    def __init__(self, checkpoint, fixture, device):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.weights = payload["weight"]
        self.device = device
        self.rates = fixture["upsample_rates"]
        self.kernels = fixture["kernels_per_stage"]
        self.dilations = fixture["dilations"]
        self.pre = self.conv("dec.conv_pre")
        self.cond = self.conv("dec.cond") if "dec.cond.weight" in self.weights else None
        self.post = self.conv("dec.conv_post")
        self.stages = []
        for stage, rate in enumerate(self.rates):
            up = self.conv(f"dec.ups.{stage}", normalized=True)
            noise = self.conv(f"dec.noise_convs.{stage}")
            blocks = []
            for branch in range(self.kernels):
                idx = stage * self.kernels + branch
                blocks.append((
                    [self.conv(f"dec.resblocks.{idx}.convs1.{i}", normalized=True) for i in range(3)],
                    [self.conv(f"dec.resblocks.{idx}.convs2.{i}", normalized=True) for i in range(3)],
                ))
            self.stages.append((rate, up, noise, blocks))

    def conv(self, prefix, normalized=False):
        if normalized:
            v = self.weights[f"{prefix}.weight_v"].float().contiguous()
            g = self.weights[f"{prefix}.weight_g"].float().reshape(-1, 1, 1)
            norm = torch.sqrt(torch.sum(v * v, dim=(1, 2), keepdim=True))
            w = v * (g / norm)
        else:
            w = self.weights[f"{prefix}.weight"].float().contiguous()
        bias = self.weights.get(f"{prefix}.bias")
        return (w.to(self.device), None if bias is None else bias.float().to(self.device))

    @staticmethod
    def conv1d(x, weight_bias, stride=1, dilation=1, pad=0):
        weight, bias = weight_bias
        return F.conv1d(x, weight, bias, stride=stride, dilation=dilation, padding=pad)

    def forward(self, x, harmonic, g):
        x = self.conv1d(x, self.pre, pad=3)
        if self.cond is not None:
            if g is None:
                raise ValueError("conditioning expected")
            x = x + self.conv1d(g, self.cond)
        elif g is not None:
            raise ValueError("unexpected conditioning")
        for stage, (rate, up, noise, blocks) in enumerate(self.stages):
            up_weight, up_bias = up
            kernel = up_weight.shape[2]
            x = F.conv_transpose1d(F.leaky_relu(x, 0.1), up_weight, up_bias,
                                   stride=rate, padding=(kernel - rate) // 2)
            future_stride = math.prod(self.rates[stage + 1:])
            prepared = x + self.conv1d(harmonic, noise, stride=future_stride, pad=future_stride // 2)
            total = torch.zeros_like(prepared)
            for first, second in blocks:
                branch = prepared
                for idx, dilation in enumerate(self.dilations):
                    k1 = first[idx][0].shape[2]
                    k2 = second[idx][0].shape[2]
                    h = self.conv1d(F.leaky_relu(branch, 0.1), first[idx], dilation=dilation,
                                    pad=(k1 - 1) * dilation // 2)
                    h = self.conv1d(F.leaky_relu(h, 0.1), second[idx], pad=(k2 - 1) // 2)
                    branch = branch + h
                total = total + branch
            x = total / len(blocks)
        x = F.leaky_relu(x, 0.01)
        return torch.tanh(self.conv1d(x, self.post, pad=3))


def stats(values):
    return statistics.median(values)


def write_floats(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    result.detach().to(torch.float32).cpu().contiguous().numpy().astype("<f4", copy=False).tofile(path)


def timed_read(output):
    return output.detach().cpu().contiguous()


def run_case(model, fixture, frames, steps, warmup, dump_prefix, use_cuda):
    device = model.device
    prepared = [reference_inputs(fixture, frames, i, device) for i in range(steps)]
    prepared_host = [reference_inputs(fixture, frames, i, torch.device("cpu")) for i in range(steps)]
    static = tuple(t.clone() if t is not None else None for t in prepared[0])
    with torch.no_grad():
        for _ in range(warmup):
            timed_read(model.forward(*prepared[0]))
        if use_cuda:
            torch.cuda.synchronize()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(max(3, warmup)):
                    model.forward(*static)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            capture_start = time.perf_counter()
            with torch.cuda.graph(graph):
                captured = model.forward(*static)
            torch.cuda.synchronize()
            capture_ms = (time.perf_counter() - capture_start) * 1000
        else:
            graph, captured, capture_ms = None, None, None
        measurements = []
        changed = set()
        for i, ((x, source, condition), host) in enumerate(zip(prepared, prepared_host)):
            if use_cuda:
                t = time.perf_counter()
                static[0].copy_(host[0])
                static[1].copy_(host[1])
                if static[2] is not None:
                    static[2].copy_(host[2])
                refresh_ms = (time.perf_counter() - t) * 1000
                t = time.perf_counter()
                graph.replay()
                output = timed_read(captured)
                replay_ms = (time.perf_counter() - t) * 1000
            else:
                refresh_ms, replay_ms, output = None, None, None
            t = time.perf_counter()
            eager = timed_read(model.forward(x, source, condition))
            eager_ms = (time.perf_counter() - t) * 1000
            if use_cuda:
                delta = (eager - output).abs().max().item()
                if not torch.equal(eager, output) and delta > 0.0002:
                    raise AssertionError(f"PyTorch Graph/eager mismatch: frames={frames} step={i} max_abs={delta}")
            else:
                delta = None
            changed.add(eager.numpy().tobytes())
            if dump_prefix is not None and i < 4:
                write_floats(Path(f"{dump_prefix}-{frames}-{i}.f32le"), eager)
            measurements.append({
                "step": i, "refresh_ms": refresh_ms, "graph_replay_and_readback_ms": replay_ms,
                "eager_and_readback_ms": eager_ms, "graph_eager_max_abs": delta,
            })
        if len(changed) < 2:
            raise AssertionError(f"NSF outputs did not vary: frames={frames}")
        return {
            "frames": frames,
            "capture_ms": capture_ms,
            "hardware_graph": bool(use_cuda),
            "distinct_outputs": len(changed),
            "median_eager_ms": stats([m["eager_and_readback_ms"] for m in measurements]),
            "median_refresh_ms": stats([m["refresh_ms"] for m in measurements]) if use_cuda else None,
            "median_graph_replay_ms": stats([m["graph_replay_and_readback_ms"] for m in measurements]) if use_cuda else None,
            "measurements": measurements,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--frames", default="64,256,1024")
    p.add_argument("--iterations", type=int, default=16)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--dump-prefix", type=Path)
    p.add_argument("--allow-cpu-smoke", action="store_true")
    args = p.parse_args()
    if args.iterations < 4 or args.warmup < 0:
        p.error("at least 4 iterations and non-negative warmup required")
    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.allow_cpu_smoke:
        p.error("CUDA GPU required; refusing misleading CPU vs CUDA comparison")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    model = NsfReference(args.checkpoint, fixture, device)
    cases = []
    for raw_frames in args.frames.split(","):
        frames = int(raw_frames)
        case = run_case(model, fixture, frames, args.iterations, args.warmup, args.dump_prefix, use_cuda)
        cases.append(case)
        print(f"{frames} frames: eager={case['median_eager_ms']:.4f}ms, graph={case['median_graph_replay_ms']}ms", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "stage": 53,
        "backend": "pytorch-cuda" if use_cuda else "pytorch-cpu-smoke",
        "torch_version": torch.__version__,
        "cuda_wheel_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if use_cuda else None,
        "device": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        "tf32_allowed": False,
        "cudnn_enabled": True,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "semantics": "Same synthetic NSF checkpoint and dynamic steps as Stage52 Rust, full FP32 output readback per iteration; preloaded eager GPU inputs; CPU-to-GPU CUDA Graph persistent input refresh measured separately; TF32 disabled.",
        "cases": cases,
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
