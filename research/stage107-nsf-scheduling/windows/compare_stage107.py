import argparse
import json
from pathlib import Path


def compare(folder):
    base = Path(folder)
    cases = {}
    for f in sorted(base.glob('*.benchmark.json')):
        record = json.loads(f.read_text(encoding='utf-8-sig'))
        if record.get('stage') != 106 or record.get('full_output_parity') is not True:
            raise ValueError(f'unverified Stage106 probe output: {f}')
        op = record.get('operator', 'conv_transpose1d')
        cases.setdefault((op, record['case']), {})[record['mode']] = record

    if not cases:
        raise ValueError('no numerical benchmarks')
    comparisons = []
    for (operator, name), modes in sorted(cases.items()):
        required = ('direct', 'tiled', 'interior') if operator == 'single_input_conv1d' else (
            'direct', 'polyphase', 'single')
        if set(modes) != set(required):
            raise ValueError(f'missing/unexpected modes for {operator}/{name}: {sorted(modes)}')
        baseline = float(modes['direct']['median_ms'])
        for candidate in required[1:]:
            record = modes[candidate]
            ms = float(record['median_ms'])
            if baseline <= 0 or ms <= 0:
                raise ValueError('nonpositive benchmark timing')
            comparisons.append({
                'operator': operator, 'case': name, 'candidate': candidate,
                'direct_ms': baseline, 'candidate_ms': ms,
                'speedup': baseline / ms,
                'max_abs': float(record['max_abs']), 'snr_db': record.get('snr_db'),
            })
    (base / 'comparison.json').write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'stage': 107, 'comparisons': comparisons}, ensure_ascii=False, indent=2))
    return comparisons


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    compare(parser.parse_args().folder)
