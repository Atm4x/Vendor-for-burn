#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def parse_lines(lines):
    sessions = {}
    current = None
    for line in lines:
        begin = re.search(r'CUBECL_NSF_BEGIN frames=(\d+)', line)
        if begin:
            current = int(begin.group(1))
            sessions[current] = {
                'aligned_materializations': 0,
                'contiguous_materializations': 0,
                'aligned_logical_elems': 0,
                'aligned_shapes': Counter(),
            }
            continue
        if 'CUBECL_NSF_END frames=' in line:
            current = None
            continue
        if current is None:
            continue
        stats = sessions[current]
        if 'CUBECL_ALIGN' in line and 'materialize=true' in line:
            stats['aligned_materializations'] += 1
            shape = re.search(r'shape=\[([^]]+)\]', line)
            if shape:
                dims = [int(v.strip()) for v in shape.group(1).split(',')]
                prod = 1
                for d in dims:
                    prod *= d
                stats['aligned_logical_elems'] += prod
                stats['aligned_shapes'][str(tuple(dims))] += 1
        if 'CUBECL_CONTIGUOUS' in line and 'materialize=true' in line:
            stats['contiguous_materializations'] += 1
    for stats in sessions.values():
        stats['aligned_shapes'] = dict(stats['aligned_shapes'])
        stats['aligned_estimated_f32_mib'] = round(stats['aligned_logical_elems'] * 4 / (1024 * 1024), 4)
    return sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = parse_lines(args.trace.read_text().splitlines())
    result = json.dumps({'stage': 48, 'measurement': 'host layout conversions at Burn ConvNd callsites', 'frames': data}, indent=2)
    if args.output:
        args.output.write_text(result + '\n')
    print(result)


if __name__ == '__main__':
    main()
