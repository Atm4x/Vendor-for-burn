import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Matched Conv1d PyTorch/cuDNN benchmark")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--report", type=Path, default=Path("torch-cudnn.json"))
    parser.add_argument("--tf32", choices=("allow", "disable"), default="disable")
    parser.add_argument("--parity-dir", type=Path)
    args = parser.parse_args()
    if not (1 <= args.iterations <= 1000 and 0 <= args.warmup <= 100):
        parser.error("Invalid number of repetitions")
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.init()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = args.tf32 == "allow"
    torch.backends.cuda.matmul.allow_tf32 = args.tf32 == "allow"
    report = {
        "backend": "pytorch-" + device,
        "status": "pass",
        "torch_version": torch.__version__,
        "cuda_wheel_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if device == "cuda" else None,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "tf32_allowed": bool(torch.backends.cudnn.allow_tf32),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "semantics": "PyTorch conv1d + synchronous CPU readback; warmed median (same as CubeCL)",
        "results": [],
    }
    shapes = [("NSF K3", 256, 256, 240, 3, 1), ("NSF K11 D5", 128, 128, 2400, 11, 5), ("NSF K11", 32, 32, 9600, 11, 1)]
    for name, cin, cout, length, k, dilation in shapes:
        # Integer arithmetic first, then float32 division, matching Rust benchmark inputs.
        x = (((torch.arange(length * cin, dtype=torch.int64) * 29) % 311).to(torch.float32) - 150.0) / 211.0
        w = (((torch.arange(cout * k * cin, dtype=torch.int64) * 13) % 199).to(torch.float32) - 100.0) / 379.0
        x = x.reshape(1, length, cin).permute(0, 2, 1).contiguous().to(device)
        w = w.reshape(cout, k, cin).permute(0, 2, 1).contiguous().to(device)
        pad = (k - 1) * dilation // 2
        measurements = []
        result = None
        for i in range(args.warmup + args.iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                out = F.conv1d(x, w, stride=1, padding=pad, dilation=dilation)
                # Include host readback in timing, exactly like the CubeCL benchmark.
                result = out.permute(0, 2, 1).contiguous().cpu()
            elapsed = (time.perf_counter() - start) * 1000.0
            if i >= args.warmup:
                measurements.append(elapsed)
        if args.parity_dir:
            args.parity_dir.mkdir(parents=True, exist_ok=True)
            filename = name.lower().replace(" ", "_") + "_pytorch.f32le"
            result.reshape(-1).to(torch.float32).numpy().astype("<f4", copy=False).tofile(args.parity_dir / filename)
        flat = result.reshape(-1).to(torch.float64)
        probes = [flat[0].item(), flat[flat.numel() // 2].item(), flat[-1].item()]
        report["results"].append({
            "name": name,
            "cin": cin, "cout": cout, "length": length, "kernel": k, "dilation": dilation,
            "conv_readback_ms": statistics.median(measurements),
            "direct_probe": probes,
            "output_sum": flat.sum().item(),
        })
        print(f"{name}: {statistics.median(measurements):.4f} ms (PyTorch {device})", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
