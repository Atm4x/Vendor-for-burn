import json
import sys
from pathlib import Path


def analyze(root: Path, cases=('upsample8', 'upsample10'), modes=('direct', 'ilp4', 'fused2tap')):
    table = []
    for case in cases:
        timings = {}
        for mode in modes:
            path = root / f'{case}-{mode}.benchmark.json'
            if not path.is_file():
                raise ValueError(f'Missing benchmark: {path}')
            row = json.loads(path.read_text(encoding='utf-8-sig'))
            if not row.get('full_output_parity') or row['max_abs'] > 1e-4:
                raise ValueError(f'Invalid numerical parity: {path}')
            if row.get('case') != case or row.get('mode') != mode:
                raise ValueError(f'Mismatched case/mode: {path}')
            timings[mode] = row['median_ms']
        baseline = timings.get('ilp4', timings.get('direct'))
        table.append({
            'case': case,
            'median_ms': timings,
            'fused_vs_ilp4': baseline / timings['fused2tap'] if 'fused2tap' in timings else None,
        })
    (root / 'summary.json').write_text(json.dumps(table, indent=2), encoding='utf-8')
    print(json.dumps(table))
    return table


if __name__ == '__main__':
    analyze(Path(sys.argv[1]), tuple(sys.argv[2].split(',')), tuple(sys.argv[3].split(',')))
