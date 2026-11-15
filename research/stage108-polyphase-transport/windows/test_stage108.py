import json
from pathlib import Path
from tempfile import TemporaryDirectory
from compare_stage108 import analyze


def test_analyzer():
    with TemporaryDirectory() as temp:
        p = Path(temp)
        for case in ('upsample8', 'upsample10'):
            for i, mode in enumerate(('single', 'kic', 'kic_ntc')):
                (p / f'transpose-{case}-{mode}.benchmark.json').write_text(json.dumps({
                    'stage': 106, 'full_output_parity': True, 'case': case, 'mode': mode,
                    'median_ms': .1 + .01*i, 'max_abs': 1e-7, 'snr_db': 130.
                }), encoding='utf8')
        assert len(analyze(p)) == 6
        (p / 'transpose-upsample8-kic.benchmark.json').unlink()
        try:
            analyze(p)
        except ValueError:
            pass
        else:
            raise AssertionError('partial matrix accepted')


if __name__ == '__main__':
    test_analyzer()
    print('Stage108 analyzer PASS')
