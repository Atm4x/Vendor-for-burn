import json
from pathlib import Path
from tempfile import TemporaryDirectory

from compare_stage110 import MODES, analyze


def run():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for case in ('upsample8', 'upsample10'):
            for i, mode in enumerate(MODES):
                (root / f'transpose-{case}-{mode}.benchmark.json').write_text(json.dumps({
                    'case': case, 'mode': mode, 'full_output_parity': True,
                    'max_abs': 1e-7, 'median_ms': 0.05 + i * .001,
                    'snr_db': 130,
                }), encoding='utf-8')
        assert len(analyze(root)) == 8
        missing = root / 'transpose-upsample10-tile2_split.benchmark.json'
        missing.unlink()
        try:
            analyze(root)
        except ValueError:
            pass
        else:
            raise AssertionError('partial profile matrix accepted')
        missing.write_text(json.dumps({'case': 'upsample10','mode': 'tile2_split',
            'full_output_parity': False, 'max_abs': 0, 'median_ms': 0.05}), encoding='utf-8')
        try:
            analyze(root)
        except ValueError:
            pass
        else:
            raise AssertionError('parity failure accepted')


if __name__ == '__main__':
    run()
    print('Stage110 contracts PASS')
