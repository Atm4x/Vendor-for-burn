"""Verify the Rust Firefly IR weight contract against original Python module definitions.

Uses meta tensors and a shape-compatible FSQ parameter scaffold. This checks
checkpoint parameter names/shapes, not numerical inference or FSQ behavior.
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from normalize_firefly_config import normalize_config


class ResidualFsqParameterScaffold(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.project_in = nn.Linear(width, 4)
        self.project_out = nn.Linear(4, width)


class GroupedFsqParameterScaffold(nn.Module):
    def __init__(self, dim, levels, num_quantizers, groups):
        super().__init__()
        if len(levels) != 4 or num_quantizers != 1:
            raise ValueError("FSQ parameter scaffold requires four levels and one quantizer")
        if dim % groups:
            raise ValueError("FSQ dimension must be divisible by groups")
        self.groups = groups
        self.rvqs = nn.ModuleList(
            ResidualFsqParameterScaffold(dim // groups) for _ in range(groups)
        )


def load_reference(source: Path):
    pkg_name = "fish_reference_shape_audit"
    package = types.ModuleType(pkg_name)
    package.__path__ = [str(source)]
    sys.modules[pkg_name] = package

    def load(module_name: str):
        name = f"{pkg_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(name, source / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    firefly = load("firefly")
    external_fsq = types.ModuleType("vector_quantize_pytorch")
    external_fsq.GroupedResidualFSQ = GroupedFsqParameterScaffold
    with patch.dict(sys.modules, {"vector_quantize_pytorch": external_fsq}):
        fsq = load("fsq")
    return firefly, fsq


def source_weights(firefly, fsq, config):
    original_linspace = torch.linspace
    with torch.device("meta"), patch(
        "torch.linspace",
        lambda *args, **kwargs: original_linspace(
            *args, **{**kwargs, "device": "cpu"}
        ),
    ):
        def kwargs(component):
            return {key: value for key, value in config[component].items() if key != "_target_"}

        backbone = firefly.ConvNeXtEncoder(**kwargs("backbone"))
        head = firefly.HiFiGANGenerator(**kwargs("head"))
        quantizer = fsq.DownsampleFiniteScalarQuantize(**kwargs("quantizer"))
        model = firefly.FireflyArchitecture(
            backbone=backbone,
            head=head,
            quantizer=quantizer,
            spec_transform=nn.Identity(),
        )

    parameters = {}
    for key, tensor in model.state_dict().items():
        if key.endswith(".parametrizations.weight.original0"):
            continue
        if key.endswith(".parametrizations.weight.original1"):
            key = key.removesuffix(".parametrizations.weight.original1") + ".weight"
        if key in parameters:
            raise ValueError(f"Duplicate normalized parameter: {key}")
        parameters[key] = tuple(tensor.shape)
    return parameters


def expected_weights(path: Path):
    expected = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        stage, key, shape = line.split("\t")
        if stage not in ("reference", "waveform"):
            raise ValueError(f"Unexpected stage: {stage}")
        dims = tuple(int(x) for x in shape.split(","))
        if key in expected and expected[key] != dims:
            raise ValueError(f"Conflicting IR contract: {key}")
        expected[key] = dims
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Python vqgan/modules directory")
    parser.add_argument("--expected", type=Path, required=True, help="Rust --firefly-weight-shapes TSV")
    parser.add_argument("--config", type=Path, required=True, help="Original static Python config.py")
    args = parser.parse_args()
    firefly, fsq = load_reference(args.source)
    config = normalize_config(args.config.read_text(encoding="utf-8"))
    source = source_weights(firefly, fsq, config)
    expected = expected_weights(args.expected)
    missing = {key: value for key, value in expected.items() if key not in source}
    extra = {key: value for key, value in source.items() if key not in expected}
    mismatch = {
        key: (expected[key], source[key])
        for key in expected.keys() & source.keys()
        if expected[key] != source[key]
    }
    print(
        f"source={len(source)} ir={len(expected)} missing={len(missing)} "
        f"extra={len(extra)} mismatched={len(mismatch)}"
    )
    for name, values in (("missing", missing), ("extra", extra), ("mismatch", mismatch)):
        if values:
            print(f"{name}: {list(values.items())[:30]}")
    return 1 if missing or extra or mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
