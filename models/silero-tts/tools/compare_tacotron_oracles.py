#!/usr/bin/env python3
"""Compare two independent original Tacotron oracle runs with identical masks."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def compare(reference, candidate):
    ref = load_file(str(reference))
    got = load_file(str(candidate))
    if ref.keys() != got.keys():
        raise ValueError(f"Tensor set differs for {candidate.name}")
    result = {"model": reference.stem, "tensors": {}}
    for name, a in ref.items():
        b = got[name]
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Nonfinite {name} in {candidate.name}")
        if name.endswith('_masks'):
            if not torch.equal(a, b):
                raise ValueError(f"Dropout masks differ for {candidate.name}; not a paired experiment")
        if a.shape != b.shape:
            result["tensors"][name] = {"reference_shape": list(a.shape), "candidate_shape": list(b.shape)}
            continue
        result["tensors"][name] = {"count": a.numel(), "max_abs": (a-b).abs().max().item()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    candidates = sorted(args.candidate.glob('*.safetensors'))
    if not candidates:
        raise ValueError('No candidate fixtures')
    for path in candidates:
        print(json.dumps(compare(args.reference / path.name, path)), flush=True)


if __name__ == '__main__':
    main()
