import argparse
import json
from pathlib import Path


def compare(direct, reuse):
    if direct.get('strategy') != 'direct' or reuse.get('strategy') != 'reuse':
        raise ValueError('Invalid Stage103 strategy pairing')
    if direct.get('backend') != reuse.get('backend') or direct.get('backend') != 'cuda':
        raise ValueError('Stage103 requires matched CUDA backends')
    lhs = {c['name']: c for c in direct['cases']}
    rhs = {c['name']: c for c in reuse['cases']}
    if lhs.keys() != rhs.keys():
        raise ValueError('Mismatched shapes/cases')
    comparisons = []
    for name, base in lhs.items():
        test = rhs[name]
        for field in ('shape', 'weight_shape', 'groups', 'padding', 'stride', 'dilation', 'dtype', 'layout'):
            if base[field] != test[field]:
                raise ValueError(f'{name}: mismatched {field}')
        if test['groups'] == 1 and test['weight_shape'][2:] == [3, 3]:
            full = test.get('full_parity')
            if not full or full.get('elements') != test['output_elements'] or not full.get('passed'):
                raise ValueError(f'{name}: full tensor parity unavailable')
            if full['max_abs'] > 1e-4:
                raise ValueError(f'{name}: numerical drift beyond 1e-4')
        else:
            full = None
        if base['median_ms'] <= 0 or test['median_ms'] <= 0:
            raise ValueError('Invalid timing')
        comparisons.append({'name': name, 'direct_ms': base['median_ms'], 'reuse_ms': test['median_ms'],
                            'speedup': base['median_ms'] / test['median_ms'], 'full_parity': full})
    return {'stage': 103, 'algorithm': '2x2-spatial-reuse-4x4-footprint', 'cases': comparisons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--direct', type=Path, required=True)
    ap.add_argument('--reuse', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = compare(json.loads(args.direct.read_text(encoding='utf-8-sig')),
                     json.loads(args.reuse.read_text(encoding='utf-8-sig')))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    for case in result['cases']:
        print(f"{case['name']}: direct={case['direct_ms']:.4f} ms reuse={case['reuse_ms']:.4f} ms speedup={case['speedup']:.3f}x")


if __name__ == '__main__':
    main()
