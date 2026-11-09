import argparse
import statistics
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

CASES = [
    ('K3_256_L240', 'resblocks_0_convs1_0', 3, 1, 240),
    ('K11_D5_128_L2400', 'resblocks_5_convs1_2', 11, 5, 2400),
    ('K11_D1_32_L9600', 'resblocks_11_convs2_2', 11, 1, 9600),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    print(f'torch={torch.__version__} threads={args.threads} mkldnn={torch.backends.mkldnn.enabled}')
    for name, dirname, kernel, dilation, length in CASES:
        p = args.root / dirname
        x = torch.from_numpy(np.fromfile(p / 'x.bin', dtype='<f4').copy()).reshape(1, -1, length)
        yref = np.fromfile(p / 'out.bin', dtype='<f4')
        weights = torch.from_numpy(np.fromfile(p / 'w.bin', dtype='<f4').copy()).reshape(-1, x.shape[1], kernel)
        bias = torch.from_numpy(np.fromfile(p / 'bias.bin', dtype='<f4').copy())
        with torch.inference_mode():
            for _ in range(5):
                y = F.conv1d(x, weights, bias, padding=kernel//2*dilation, dilation=dilation)
            err = float(np.max(np.abs(y.numpy().reshape(-1) - yref)))
            durations = []
            for _ in range(15):
                t0 = time.perf_counter()
                result = F.conv1d(x, weights, bias, padding=kernel//2*dilation, dilation=dilation)
                durations.append((time.perf_counter()-t0)*1000)
                assert result.numel()==yref.size
        print(f'{name} torch_ms={statistics.median(durations):.4f} min_ms={min(durations):.4f} max_ms={max(durations):.4f} max_ref_error={err:.8f}',flush=True)

if __name__ == '__main__': main()
