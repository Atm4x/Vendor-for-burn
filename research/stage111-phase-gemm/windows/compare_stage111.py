import json
import sys
from pathlib import Path


def analyze(folder, cases=('upsample8', 'upsample10')):
    results = []
    for case in cases:
        rows = {}
        for mode in ('direct', 'phase_f32', 'phase_f16'):
            path = folder / f'{case}-{mode}.benchmark.json'
            if not path.exists():
                raise ValueError(f'missing {path.name}')
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            if data['case'] != case or data['mode'] != mode or not data['full_output_parity']:
                raise ValueError(f'invalid numerical result: {path}')
            rows[mode] = data
        base = float(rows['direct']['median_ms'])
        for mode in ('phase_f32', 'phase_f16'):
            row = rows[mode]
            ms = float(row['median_ms'])
            results.append({
                'case': case, 'mode': mode, 'direct_ms': base,
                'candidate_ms': ms, 'speedup': base / ms,
                'max_abs': row['max_abs'], 'snr_db': row['snr_db'],
                'dtype': row['dtype'], 'algorithm': 'phase_batched_cubecl_matmul',
            })
    (folder / 'comparison.json').write_text(json.dumps({
        'stage': 111, 'comparisons': results,
        'note': 'synchronized host runtimes; FP16 changes numerical precision',
    }, indent=2), encoding='utf-8')
    return results


if __name__ == '__main__':
    root = Path(sys.argv[1])
    cases = tuple(sys.argv[2].split(',')) if len(sys.argv) > 2 else ('upsample8', 'upsample10')
    print(json.dumps(analyze(root, cases), indent=2))
