import json
import sys
from pathlib import Path


def analyze(folder):
    p = Path(folder)
    results = {}
    for file in p.glob('transpose-*.benchmark.json'):
        info = json.loads(file.read_text(encoding='utf-8-sig'))
        if info.get('full_output_parity') is not True or info.get('stage') != 106:
            raise ValueError('Numerical verification missing: ' + file.name)
        if float(info['max_abs']) > 1e-4:
            raise ValueError('Numerical tolerance violated: ' + file.name)
        results[(info['case'], info['mode'])] = info
    cases = sorted(set(case for case, mode in results))
    if not cases:
        raise ValueError('No benchmark results')
    output = []
    for case in cases:
        required = ('single', 'kic', 'kic_ntc')
        actual = {m for c, m in results if c == case}
        if actual != set(required):
            raise ValueError(f'Incomplete candidate matrix for {case}: {actual}')
        base = float(results[(case, 'single')]['median_ms'])
        if base <= 0:
            raise ValueError('Invalid baseline')
        for mode in required:
            data = results[(case, mode)]
            ms = float(data['median_ms'])
            if ms <= 0:
                raise ValueError('Invalid latency')
            output.append({'case': case, 'mode': mode, 'median_ms': ms,
                           'speedup_vs_single': base / ms,
                           'max_abs': float(data['max_abs']),
                           'snr_db': data.get('snr_db')})
    (p / 'comparison.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    print(json.dumps(analyze(sys.argv[1]), indent=2))
