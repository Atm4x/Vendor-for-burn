import json
from pathlib import Path
from tempfile import TemporaryDirectory
from compare_stage111 import analyze


def test_comparison():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for case in ('upsample8', 'upsample10'):
            for mode in ('direct', 'phase_f32', 'phase_f16'):
                data = {
                    'case': case, 'mode': mode, 'full_output_parity': True,
                    'median_ms': 0.1, 'max_abs': 0.0, 'snr_db': None,
                    'dtype': 'f16' if mode == 'phase_f16' else 'f32',
                }
                (root / f'{case}-{mode}.benchmark.json').write_text(json.dumps(data))
        assert len(analyze(root)) == 4
        (root / 'upsample10-phase_f16.benchmark.json').unlink()
        try:
            analyze(root)
        except ValueError:
            pass
        else:
            raise AssertionError('incomplete matrix was accepted')


if __name__ == '__main__':
    test_comparison()
    print('Stage111 analyzer PASS')
