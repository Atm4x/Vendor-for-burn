import argparse
import json
import math
import statistics
import time


def run(case, mode, repetitions):
    import torch
    import torch.nn.functional as nnf

    shapes = {
        'small': (1, 3, 5, 25, 2, 6, 2),
        'upsample8': (2, 64, 32, 512, 8, 16, 4),
        'upsample10': (2, 128, 64, 256, 10, 20, 5),
        'upsample4': (2, 32, 16, 1024, 4, 8, 2),
    }
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch CUDA not available')
    batch, ci, co, t, stride, kernel, padding = shapes[case]
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.sin(torch.arange(batch * ci * t, dtype=torch.float32) * 0.031).mul_(0.2).reshape(batch, ci, t).cuda()
    w = torch.cos(torch.arange(ci * co * kernel, dtype=torch.float32) * 0.027).mul_(0.1).reshape(ci, co, kernel).cuda()
    bias = torch.arange(co, dtype=torch.float32).mul_(0.001).cuda()
    reference = nnf.conv_transpose1d(x, w, bias, stride=stride, padding=padding)
    torch.cuda.synchronize()
    if mode != 'f32':
        w, bias = w.half(), bias.half()
        if mode == 'f16':
            x = x.half()
    def operation():
        x_run = x.half() if mode == 'f16_bridge' else x
        y = nnf.conv_transpose1d(x_run, w, bias, stride=stride, padding=padding)
        return y.float() if mode == 'f16_bridge' else y

    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    host = []
    device = []
    last = None
    for _ in range(repetitions):
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter_ns()
        begin.record()
        last = operation()
        end.record()
        torch.cuda.synchronize()
        host.append((time.perf_counter_ns() - t0) / 1_000_000)
        device.append(begin.elapsed_time(end))
    actual = last.float()
    delta = actual - reference
    max_abs = delta.abs().max().item()
    mse = torch.mean(delta.square()).item()
    signal = torch.mean(reference.square()).item()
    snr = 10.0 * math.log10(signal / mse) if mse else None
    result = {
        'stage': 111, 'implementation': 'torch_cudnn', 'case': case, 'dtype': mode,
        'host_median_ms': statistics.median(host),
        'device_median_ms': statistics.median(device),
        'max_abs_vs_torch_f32': max_abs, 'snr_db_vs_torch_f32': snr,
        'input_shape': [batch, ci, t], 'weight_shape': [ci, co, kernel],
        'torch_version': torch.__version__, 'cudnn_version': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0), 'repetitions': repetitions,
        'dtype_is_tensor_core_eligible': mode in ('f16','f16_bridge'),
        'bridge_cast_included': mode == 'f16_bridge',
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case', choices=['small', 'upsample8', 'upsample10', 'upsample4'])
    p.add_argument('mode', choices=['f32', 'f16', 'f16_bridge'])
    p.add_argument('--runs', type=int, default=12)
    args = p.parse_args()
    print(json.dumps(run(args.case, args.mode, args.runs)))


if __name__ == '__main__':
    main()
