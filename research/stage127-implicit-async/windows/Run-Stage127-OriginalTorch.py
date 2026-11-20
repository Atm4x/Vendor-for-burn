#!/usr/bin/env python3
"""Stage126 PyTorch CUDA full-WAV reference: original pipeline from trusted local source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, help="Unpacked tts-with-rvc-main Python repository")
    parser.add_argument("--assets-dir", default="embed", help="Directory containing hubert_base.pt and rmvpe.pt")
    parser.add_argument("--voice", default="CrazyMita.pth")
    parser.add_argument("--hubert", default="hubert_base.pt")
    parser.add_argument("--rmvpe", default="rmvpe.pt")
    parser.add_argument("--index", default="CrazyMita.index")
    parser.add_argument("--input-wav", default="MitaKind.wav")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--strict-fp32", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="original_torch_stage80")
    args = parser.parse_args()
    if not 2 <= args.runs <= 16 or not 0 <= args.index_rate <= 1:
        parser.error("Expected runs 2..16 and index rate in [0, 1]")
    root, assets = Path(args.original_root).resolve(), Path(args.assets_dir).resolve()
    sys.path.insert(0, str(root))
    import importlib.machinery
    import types
    package = types.ModuleType("tts_with_rvc")
    package.__file__ = str(root / "tts_with_rvc" / "__init__.py")
    package.__path__ = [str(root / "tts_with_rvc")]
    package.__package__ = "tts_with_rvc"
    package.__spec__ = importlib.machinery.ModuleSpec("tts_with_rvc", loader=None, is_package=True)
    sys.modules["tts_with_rvc"] = package

    if not (root / "tts_with_rvc/infer/vc/pipeline.py").exists():
        parser.error("Original source tree not found")
    for filename in (args.voice, args.hubert, args.rmvpe, args.input_wav):
        if not (assets / filename).is_file():
            parser.error(f"Missing original model/input: {assets / filename}")
    if args.index_rate > 0 and not (assets / args.index).is_file():
        parser.error(f"Missing FAISS index: {assets / args.index}")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from scipy.io import wavfile
    from tts_with_rvc.infer.lib.audio import load_audio
    from tts_with_rvc.infer.vc.config import Config
    from tts_with_rvc.infer.vc.modules import VC
    imported_source = Path(sys.modules["tts_with_rvc.infer.vc.config"].__file__).resolve()
    if not imported_source.is_relative_to(root):
        raise RuntimeError(f"Imported wrong RVC Config implementation: {imported_source}")

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires the same CUDA RTX device as Rust")
    if args.precision == "fp32" and args.strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # The supplied 2024 upstream code expects PyTorch's pre-2.6 torch.load default.
    # Only run this benchmark with locally trusted voice/HuBERT/RMVPE checkpoints.
    _torch_load = torch.load

    def load_trusted_local_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load(*args, **kwargs)

    torch.load = load_trusted_local_checkpoint
    old_dir = Path.cwd()
    os.chdir(assets)
    try:
        config = Config()
        config.device = "cuda:0"
        config.is_half = args.precision == "fp16"
        if hasattr(config, "device_config") and callable(config.device_config):
            config.x_pad, config.x_query, config.x_center, config.x_max = config.device_config()
        elif not all(hasattr(config, k) for k in ("x_pad", "x_query", "x_center", "x_max")):
            raise RuntimeError("Original Config has neither device_config() nor populated x_* parameters")
        else:
            print("WARNING: using installed Config x_* values; device_config unavailable; source fingerprints must be checked", flush=True)
        started = time.perf_counter()
        vc = VC(config)
        vc.get_vc(str(assets / args.voice))
        vc.hubert_model = vc.load_hubert(config, str(assets / args.hubert))
        vc.hubert_is_half = config.is_half
        if args.precision == "fp32" and args.strict_fp32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.cuda.synchronize()
        load_ms = (time.perf_counter() - started) * 1000
        audio = load_audio(str(assets / args.input_wav), 16000)
        peak = np.max(np.abs(audio)) / 0.95
        if peak > 1:
            audio = audio / peak
        file_index = str(assets / args.index) if args.index_rate > 0 else ""
        durations, stage_times, output_hashes, outputs = [], [], [], []
        for iteration in range(args.runs):
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.synchronize()
            times = [0.0, 0.0, 0.0]
            started = time.perf_counter()
            if iteration > 0:
                torch.cuda.nvtx.range_push("stage127_torch_warm")
            try:
                with torch.no_grad():
                    samples = vc.pipeline.pipeline(
                        vc.hubert_model, vc.net_g, 0, audio, str(assets / args.input_wav),
                        times, 0, "rmvpe", file_index, args.index_rate,
                        vc.if_f0, 3, vc.tgt_sr, 0, 0.5, vc.version, 0.33, None,
                    )
                torch.cuda.synchronize()
            finally:
                if iteration > 0:
                    torch.cuda.nvtx.range_pop()
            elapsed_ms = (time.perf_counter() - started) * 1000
            wav_path = out / f"original_{args.precision}_{iteration}.wav"
            wavfile.write(str(wav_path), vc.tgt_sr, samples)
            durations.append(elapsed_ms)
            stage_times.append(times)
            output_hashes.append(hashlib.sha256(wav_path.read_bytes()).hexdigest())
            outputs.append(int(len(samples)))
            print(f"Original {args.precision} iteration {iteration}: {elapsed_ms:.1f} ms (audio {len(samples)/vc.tgt_sr:.2f}s)", flush=True)
        result = {
            "stage": 127, "original_upstream": str(root),
            "source_root_not_modified": True,
            "precision": args.precision, "device": torch.cuda.get_device_name(), "strict_fp32": bool(args.precision == "fp32" and args.strict_fp32), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "source_seconds": len(audio) / 16000, "model_input_sample_rate": 16000,
            "output_sr": vc.tgt_sr, "output_samples": outputs,
            "load_ms": load_ms, "inference_ms": durations,
            "warm_median_ms": statistics.median(durations[1:]),
            "warm_rtf": statistics.median(durations[1:]) / (len(audio) / 16000 * 1000),
            "phase_times_s": stage_times, "sha256": output_hashes,
            "original_config": {"device_config_available": hasattr(config, "device_config"), "x_pad": config.x_pad, "x_query": config.x_query,
                                "x_center": config.x_center, "x_max": config.x_max},
            "index_rate": args.index_rate,
            "input_note": "Source is decoded by original ffmpeg loader and normalised by original vc_single policy",
            "parity_note": "Different chunk selection, FP16 arithmetic, RNG and reference audio resampling prohibit PCM-bitwise parity with current Rust 300-frame pipeline",
            "timing_note": "PyTorch: includes original highpass and RMS normalization, unlike Rust infer-wav-full-benchmark core time. Compare RTF with documented scope difference.",
        }
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Original PyTorch warm median: {result['warm_median_ms']:.1f} ms; output: {out / 'summary.json'}")
    finally:
        os.chdir(old_dir)


if __name__ == "__main__":
    main()
