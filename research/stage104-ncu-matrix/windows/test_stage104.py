"""Unit tests that enforce bounded, non-misleading NCU capture plans."""

import unittest
from stage104_matrix import make_plan, BLOCKS, CASES, MICRO_STRATEGIES


class Stage104PlanTests(unittest.TestCase):
    def test_micro_all_cases_all_engines(self):
        p = make_plan()
        self.assertEqual(p['target_count'], 12)
        self.assertEqual(len(set(t['id'] for t in p['targets'])), 12)
        self.assertTrue(all(t['nvtx'].endswith('/') for t in p['targets']))

    def test_full_all_blocks(self):
        p = make_plan(scope='pipeline', include_torch=False)
        self.assertEqual(p['target_count'], 17)
        self.assertEqual(p['targets'][0]['nvtx'], 'stage104_rust_audio_h2d/')
        self.assertEqual(p['targets'][-1]['nvtx'], 'stage104_nsf_residuals/')

    def test_both_and_expanded_categories(self):
        p = make_plan(scope='all', blocks='rmvpe,nsf', families='all,conv,gemm',
                      pipeline_strategies='direct,reuse', include_torch_full=True)
        self.assertEqual(p['target_count'], 12 + 2 * 2 * 3 + 3)
        self.assertEqual(p['targets'][-1]['nvtx'], 'stage102_torch_measured/')

    def test_invalid_or_duplicate_name_rejected(self):
        for cases in ('dense3x3,dense3x3', 'blah'):
            with self.assertRaises(ValueError):
                make_plan(cases=cases)

    def test_huge_matrix_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exceeds cap'):
            make_plan(scope='pipeline', pipeline_strategies='direct,reuse,quad,gemm,tf32_batched,winograd',
                      families='all,conv,gemm,recurrent,elementwise', max_targets=60)

    def test_nvtx_names_match_rust_sources(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        rust = '\n'.join((root / f).read_text(encoding='utf-8') for f in (
            'models/tts-with-rvc-rs/src/rvc_inference.rs',
            'models/tts-with-rvc-rs/src/rmvpe_gru.rs',
            'models/tts-with-rvc-rs/src/decoder.rs',
        ))
        for name in BLOCKS.values():
            self.assertIn('"' + name + '"', rust, name)

    def test_micro_cases_match_probe(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        rust = (root / 'models/tts-with-rvc-rs/src/bin/stage103_conv2d_probe.rs').read_text()
        for case in CASES:
            self.assertIn('name: "' + case + '"', rust)
        for strategy in MICRO_STRATEGIES:
            self.assertIn('"' + strategy + '"', rust)

    def test_no_torch(self):
        p = make_plan(scope='micro', include_torch=False)
        self.assertEqual(p['target_count'], 8)


if __name__ == '__main__':
    unittest.main()
