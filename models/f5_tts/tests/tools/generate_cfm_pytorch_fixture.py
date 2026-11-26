from __future__ import annotations

import argparse
import ast
import math
import zipfile
from pathlib import Path
import torch


def load_original_function(archive: Path):
    with zipfile.ZipFile(archive) as source:
        python_source = source.read('F5-TTS-main/src/f5_tts/model/utils.py').decode('utf-8')
    node = ast.parse(python_source)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'get_epss_timesteps')
    scope = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(archive), 'exec'), scope)
    return scope['get_epss_timesteps']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('original_archive', type=Path)
    args = ap.parse_args()
    original = load_original_function(args.original_archive)
    dst = Path(__file__).resolve().parents[1] / 'fixtures/cfm_pytorch_expected.tsv'
    with dst.open('w', encoding='utf-8') as writer:
        for steps in (1, 2, 5, 6, 7, 10, 12, 16, 20, 32, 48, 64, 128):
            for sway in (None, -1.0, -0.3, 0.5):
                t = original(steps, device='cpu', dtype=torch.float32)
                if sway is not None:
                    t = t + sway * (torch.cos(torch.pi / 2 * t) - 1 + t)
                coefficient = 'none' if sway is None else str(sway)
                writer.write(f'{steps}\t{coefficient}\t{",".join(str(v.item()) for v in t)}\n')
    print('Generated 52 PyTorch ODE time grids from original F5 source:', dst)


if __name__ == '__main__':
    main()
