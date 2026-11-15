import json
import tempfile
import unittest
from pathlib import Path

from compare_stage107 import compare

ROOT = Path(__file__).resolve().parents[3]


class Stage107Contracts(unittest.TestCase):
    def test_single_and_interior_have_full_output_parity(self):
        for path, mode in [
            ('models/tts-with-rvc-rs/src/bin/stage106_polyphase_probe.rs', 'single'),
            ('models/tts-with-rvc-rs/src/bin/stage106_noise_probe.rs', 'interior'),
        ]:
            content = (ROOT / path).read_text(encoding='utf-8')
            self.assertIn('full_output_parity', content)
            self.assertIn(f'"{mode}"', content)

    def test_new_kernels_are_independent_from_old(self):
        poly = (ROOT / 'vendor/burn-cubecl-0.22.1/src/kernel/conv/conv_transpose2d/polyphase1d.rs').read_text(encoding='utf-8')
        noise = (ROOT / 'vendor/burn-cubecl-0.22.1/src/kernel/conv/noise1d.rs').read_text(encoding='utf-8')
        self.assertIn('polyphase_single_kernel', poly)
        self.assertIn('conv_transpose1d_polyphase_single_packed', poly)
        self.assertIn('fast_interior', noise)
        self.assertIn('noise_temporal_pair_kernel', noise)

    def test_comparator_rejects_missing_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'transpose-upsample8-direct.benchmark.json'
            p.write_text(json.dumps(dict(stage=106, full_output_parity=True,
                                         operator='conv_transpose1d', case='upsample8',
                                         mode='direct', median_ms=0.1)))
            with self.assertRaises(ValueError):
                compare(directory)

    def test_comparator_computes_pair_and_single(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode, timing in [('direct', 0.2), ('polyphase', 0.1), ('single', 0.05)]:
                (Path(directory) / f'{mode}.benchmark.json').write_text(json.dumps(dict(
                    stage=106, full_output_parity=True, operator='conv_transpose1d',
                    case='upsample8', mode=mode, median_ms=timing, max_abs=0, snr_db=150)))
            rows = compare(directory)
            self.assertEqual([r['candidate'] for r in rows], ['polyphase', 'single'])
            self.assertEqual(rows[1]['speedup'], 4.0)

    def test_runner_nvtx_and_bounds(self):
        script = (ROOT / 'research/stage107-nsf-scheduling/windows/Run-RVC-Stage107-NSF-Scheduling.ps1').read_text(encoding='utf-8')
        for key in ['nvtx_available=true', '--nvtx-push-pop-scope', '--launch-count',
                    'No kernels were profiled', 'full_output_parity', 'RVC_STAGE107_NSF_SCHEDULING_RESULTS.zip']:
            self.assertIn(key, script)
        self.assertNotIn('Copy-Item *.wav', script)


if __name__ == '__main__':
    unittest.main()
