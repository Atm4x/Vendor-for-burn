from pathlib import Path
from tempfile import TemporaryDirectory
import json
from compare_stage112 import analyze


def test_contract():
    with TemporaryDirectory() as d:
        p = Path(d)
        for case in ('upsample8', 'upsample10'):
            for mode in ('direct', 'ilp4', 'fused2tap'):
                (p / f'{case}-{mode}.benchmark.json').write_text(json.dumps({
                    'case': case, 'mode': mode, 'full_output_parity': True,
                    'max_abs': 1e-7, 'median_ms': 0.1,
                }), encoding='utf-8')
        assert len(analyze(p)) == 2
        (p / 'upsample10-fused2tap.benchmark.json').unlink()
        try:
            analyze(p)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing result was accepted')


if __name__ == '__main__':
    test_contract()
    print('Stage112 analyzer contract PASS')
