import json
import sys
from pathlib import Path

MODES = ('single', 'kic', 'pipe4', 'ilp4', 'pipe4_kic', 'ilp4_kic')


def analyze(directory: Path, cases=('upsample8', 'upsample10'), modes=MODES):
    results = []
    for case in cases:
        baseline = None
        for mode in modes:
            path = directory / f'transpose-{case}-{mode}.benchmark.json'
            if not path.is_file():
                raise ValueError(f'missing required benchmark: {path.name}')
            entry = json.loads(path.read_text(encoding='utf-8-sig'))
            if entry['mode'] != mode or entry['case'] != case:
                raise ValueError(f'case/mode mismatch: {path.name}')
            if entry['full_output_parity'] is not True or entry['max_abs'] > 1e-4:
                raise ValueError(f'full output parity failed: {path.name}')
            if entry['median_ms'] <= 0:
                raise ValueError(f'invalid timing: {path.name}')
            if mode == 'single':
                baseline = entry['median_ms']
            if baseline is None:
                raise ValueError('single must precede candidates')
            results.append({
                'case': case, 'mode': mode,
                'median_ms': entry['median_ms'],
                'speedup_vs_single': baseline / entry['median_ms'],
                'max_abs': entry['max_abs'],
                'snr_db': entry.get('snr_db'),
            })
    (directory / 'comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def main():
    directory = Path(sys.argv[1])
    cases = sys.argv[2].split(',') if len(sys.argv) > 2 else ('upsample8', 'upsample10')
    modes = sys.argv[3].split(',') if len(sys.argv) > 3 else MODES
    print(json.dumps(analyze(directory, cases, modes), indent=2))


if __name__ == '__main__':
    main()
