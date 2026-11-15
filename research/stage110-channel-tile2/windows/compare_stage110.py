import json
import sys
from pathlib import Path

MODES = ('single', 'ilp4', 'tile2', 'tile2_split')
CASES = ('upsample8', 'upsample10')


def analyze(directory: Path, cases=CASES, modes=MODES):
    if not modes or modes[0] != 'single':
        raise ValueError('single must be first baseline')
    rows = []
    for case in cases:
        baseline = None
        for mode in modes:
            path = directory / f'transpose-{case}-{mode}.benchmark.json'
            if not path.is_file():
                raise ValueError(f'missing benchmark: {path.name}')
            result = json.loads(path.read_text(encoding='utf-8-sig'))
            if result.get('case') != case or result.get('mode') != mode:
                raise ValueError(f'mismatched benchmark: {path.name}')
            if result.get('full_output_parity') is not True:
                raise ValueError(f'parity not confirmed: {path.name}')
            if not isinstance(result.get('max_abs'), (float, int)) or not 0 <= result['max_abs'] <= 1e-4:
                raise ValueError(f'invalid max_abs: {path.name}')
            if not isinstance(result.get('median_ms'), (float, int)) or not 0 < result['median_ms'] < float('inf'):
                raise ValueError(f'invalid median timing: {path.name}')
            if baseline is None:
                baseline = result['median_ms']
            rows.append({'case': case, 'mode': mode, 'median_ms': result['median_ms'],
                         'speedup_vs_single': baseline / result['median_ms'],
                         'max_abs': result['max_abs'], 'snr_db': result.get('snr_db')})
    (directory / 'comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return rows


if __name__ == '__main__':
    directory = Path(sys.argv[1]); cases = tuple(sys.argv[2].split(',')) if len(sys.argv) > 2 else CASES
    modes = tuple(sys.argv[3].split(',')) if len(sys.argv) > 3 else MODES
    print(json.dumps(analyze(directory, cases, modes), indent=2))
