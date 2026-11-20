"""Global NSF spectral/direct planner under a checkpoint-wide GPU weight budget."""

from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import torch

from spectral_conv import candidate_plans, SpectralPlan


@dataclass(frozen=True)
class Layer:
    name: str
    direct: float
    candidates: tuple[SpectralPlan, ...]


STAGE_FRAMES = (3840, 38400, 76800, 153600)


def inspect_checkpoint(path: Path) -> list[Layer]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    weight_map = payload["weight"]
    layers = []
    for block in range(12):
        stage, _ = divmod(block, 3)
        for family in ("convs1", "convs2"):
            for layer in range(3):
                name = f"dec.resblocks.{block}.{family}.{layer}"
                o, i, k = weight_map[name + ".weight_v"].shape
                dilation = (1, 3, 5)[layer] if family == "convs1" else 1
                plans = candidate_plans(i, o, k, dilation, STAGE_FRAMES[stage])
                layers.append(Layer(name, plans[0].direct_flops, tuple(plans)))
    return layers


def select_global(layers: list[Layer], budget_mib: int = 512,
                  bucket_mib: int = 2) -> list[tuple[Layer, SpectralPlan | None]]:
    """Multiple-choice knapsack; direct conv is always legal and uses no spectral storage.

    Each transition maximizes saved analytic FLOPs at a fixed *rounded-up* memory
    cost. This ensures the returned plan never exceeds the actual byte budget.
    """
    if budget_mib <= 0 or bucket_mib <= 0:
        raise ValueError("budgets and buckets must be positive")
    slots = budget_mib // bucket_mib
    best = {0: (0.0, [])}
    for layer in layers:
        opts = [(0, 0.0, None)]
        for plan in layer.candidates:
            saved = layer.direct - plan.spectral_flops
            if saved <= 0:
                continue
            mi = (plan.spectral_weight_bytes + (bucket_mib * 1024**2 - 1)) // (bucket_mib * 1024**2)
            if mi <= slots:
                opts.append((int(mi), saved, plan))
        next_best = {}
        for used, (saved_total, route) in best.items():
            for size, saved, option in opts:
                next_used = used + size
                if next_used > slots:
                    continue
                candidate = saved_total + saved
                if next_used not in next_best or candidate > next_best[next_used][0]:
                    next_best[next_used] = (candidate, [*route, (layer, option)])
        best = next_best
    _, route = max(best.values(), key=lambda item: item[0])
    return route


def summarize(layers: list[Layer], selection: list[tuple[Layer, SpectralPlan | None]]) -> dict:
    original = sum(l.direct for l in layers)
    chosen = sum(p.spectral_flops if p is not None else l.direct for l, p in selection)
    memory = sum(p.spectral_weight_bytes for _, p in selection if p is not None)
    details = []
    for layer, plan in selection:
        if plan is not None:
            details.append({
                "layer":layer.name,
                "channels":plan.channels_in,
                "kernel":plan.kernel,
                "dilation":plan.dilation,
                "nfft":plan.nfft,
                "valid_frames":plan.valid_frames,
                "blocks":plan.blocks,
                "spectral_weight_mib":round(plan.spectral_weight_bytes/1024**2, 4),
                "analytic_direct_to_spectral_flops":round(plan.ideal_flop_ratio, 4),
            })
    return {
        "original_direct_tflops_per_384_latent_input": original/1e12,
        "selected_analytic_tflops_per_384_latent_input": chosen/1e12,
        "analytic_flop_ratio":original/chosen,
        "spectrum_weight_mib":memory/1024**2,
        "spectral_layers":len(details),
        "direct_layers":len(layers)-len(details),
        "selected":details,
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--budgets-mib',type=int,nargs='+',default=[64,128,256,512,1024])
    args=p.parse_args()
    layers=inspect_checkpoint(args.checkpoint)
    result={}
    for budget in args.budgets_mib:
        plan=select_global(layers,budget)
        summary=summarize(layers,plan)
        result[str(budget)]=summary
        print(f"budget={budget:>4} MiB: spectral={summary['spectral_layers']:>2}, "
              f"actual_weights={summary['spectrum_weight_mib']:>7.1f} MiB, "
              f"analytic_flop_ratio={summary['analytic_flop_ratio']:.2f}x")
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
