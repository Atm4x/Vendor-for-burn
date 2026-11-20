"""Deterministic, bounded Nsight Compute target matrix for RVC Stage 104."""

import argparse
import json
from pathlib import Path

CASES = ("dense3x3", "dense3x3_wide", "pointwise1x1", "depthwise3x3")
MICRO_STRATEGIES = ("direct", "reuse", "pair")
PIPELINE_STRATEGIES = ("direct", "reuse", "pair", "quad", "gemm", "tf32_batched", "winograd")
BLOCKS = {
    "audio_h2d": "stage104_rust_audio_h2d",
    "hubert": "stage102_rust_hubert",
    "hubert_readback": "stage104_rust_hubert_readback",
    "rmvpe": "stage102_rust_rmvpe",
    "retrieval_pitch": "stage104_rust_retrieval_pitch",
    "encoder": "stage104_rust_encoder",
    "latent": "stage104_rust_latent",
    "flow": "stage104_rust_flow",
    "nsf_prepare": "stage104_rust_nsf_prepare",
    "nsf": "stage102_rust_nsf",
    "nsf_readback": "stage104_rust_nsf_readback",
    "rmvpe_unet": "stage104_rmvpe_unet",
    "rmvpe_gru": "stage104_rmvpe_gru",
    "rmvpe_head": "stage104_rmvpe_head",
    "nsf_upsample": "stage104_nsf_upsample",
    "nsf_noise_conv": "stage104_nsf_noise_conv",
    "nsf_residuals": "stage104_nsf_residuals",
}
FAMILIES = {
    "all": None,
    "conv": "regex:(conv|depthwise|transpose|deconv)",
    "gemm": "regex:(gemm|matmul|cutlass|mma)",
    "recurrent": "regex:(gru|lstm|recurrent)",
    "elementwise": "regex:(elementwise|unary|binary|mask|copy|reduce|gather|scatter|permute)",
}


def chosen(values, valid, label):
    items = [s.strip() for s in values.split(",") if s.strip()]
    if not items or len(items) != len(set(items)) or any(x not in valid for x in items):
        raise ValueError(f"Invalid {label}: {values}; allowed: {', '.join(valid)}")
    return items


def make_plan(scope="micro", cases="dense3x3,dense3x3_wide,pointwise1x1,depthwise3x3",
              micro_strategies=",".join(MICRO_STRATEGIES), blocks=",".join(BLOCKS),
              pipeline_strategies="direct", families="all", include_torch=True,
              include_torch_full=False, max_targets=80):
    if scope not in ("micro", "pipeline", "all"):
        raise ValueError(f"Unknown scope {scope}")
    cs = chosen(cases, CASES, "cases")
    ms = chosen(micro_strategies, MICRO_STRATEGIES, "micro strategies")
    bs = chosen(blocks, BLOCKS, "blocks")
    ps = chosen(pipeline_strategies, PIPELINE_STRATEGIES, "pipeline strategies")
    fs = chosen(families, FAMILIES, "families")
    targets = []
    if scope in ("micro", "all"):
        for c in cs:
            for strategy in ms:
                targets.append(dict(id=f"micro-rust-{c}-{strategy}", kind="micro_rust", case=c,
                                    strategy=strategy, block=None, family="all", kernel_filter=None,
                                    nvtx=f"stage104_conv2d_{c}/"))
            if include_torch:
                targets.append(dict(id=f"micro-torch-{c}", kind="micro_torch", case=c,
                                    strategy=None, block=None, family="all", kernel_filter=None,
                                    nvtx=f"stage104_conv2d_{c}/"))
    if scope in ("pipeline", "all"):
        for strategy in ps:
            for block in bs:
                for family in fs:
                    targets.append(dict(id=f"pipeline-rust-{strategy}-{block}-{family}",
                                        kind="pipeline_rust", case=None, strategy=strategy,
                                        block=block, family=family, kernel_filter=FAMILIES[family],
                                        nvtx=f"{BLOCKS[block]}/"))
        if include_torch_full:
            for family in fs:
                targets.append(dict(id=f"pipeline-torch-{family}", kind="pipeline_torch",
                                    case=None, strategy=None, block="whole_model", family=family,
                                    kernel_filter=FAMILIES[family], nvtx="stage102_torch_measured/"))
    if len(targets) > max_targets:
        raise ValueError(f"Requested {len(targets)} NCU captures exceeds cap {max_targets}; narrow the matrix")
    assert len(targets) == len({x["id"] for x in targets})
    return {"stage": 104, "scope": scope, "target_count": len(targets), "targets": targets,
            "notes": ["Each capture profiles only selected launches, not every kernel in its block",
                      "Missing matched NVTX kernels are recorded explicitly as no_kernels",
                      "NCU replay timings are not ordinary inference latency",
                      "No WAV, model or index is ever included in result archives"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", choices=("micro", "pipeline", "all"), default="micro")
    p.add_argument("--cases", default=",".join(CASES))
    p.add_argument("--micro-strategies", default=",".join(MICRO_STRATEGIES))
    p.add_argument("--blocks", default=",".join(BLOCKS))
    p.add_argument("--pipeline-strategies", default="direct")
    p.add_argument("--families", default="all")
    p.add_argument("--no-torch", action="store_true")
    p.add_argument("--include-torch-full", action="store_true")
    p.add_argument("--max-targets", type=int, default=80)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    try:
        plan = make_plan(args.scope, args.cases, args.micro_strategies, args.blocks,
                         args.pipeline_strategies, args.families, not args.no_torch,
                         args.include_torch_full, args.max_targets)
    except ValueError as e:
        p.error(str(e))
    doc = json.dumps(plan, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(doc, encoding="utf-8")
    else:
        print(doc)


if __name__ == "__main__":
    main()
