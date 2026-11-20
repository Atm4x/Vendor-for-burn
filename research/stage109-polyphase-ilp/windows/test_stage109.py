import json
from pathlib import Path
from tempfile import TemporaryDirectory

from compare_stage109 import MODES, analyze


def test_contract():
    with TemporaryDirectory() as tmp:
        p = Path(tmp)
        for case in ('upsample8', 'upsample10'):
            for i, mode in enumerate(MODES):
                (p / f'transpose-{case}-{mode}.benchmark.json').write_text(json.dumps({
                    'case': case, 'mode': mode, 'full_output_parity': True,
                    'max_abs': 1e-7, 'median_ms': .05 + i * .001,
                    'snr_db': 130,
                }), encoding='utf-8')
        assert len(analyze(p)) == 12
        path = p / 'transpose-upsample10-ilp4_kic.benchmark.json'
        path.unlink()
        try:
            analyze(p)
        except ValueError:
            pass
        else:
            raise AssertionError('partial test matrix accepted')


if __name__ == '__main__':
    test_contract()
    print('Stage109 compare contract PASS')
