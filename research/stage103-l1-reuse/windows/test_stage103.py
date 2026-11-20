import copy
import unittest
from compare_stage103 import compare


class TestStage103(unittest.TestCase):
    def setUp(self):
        c = {'name': 'dense3x3', 'shape': [1, 16, 32, 32], 'weight_shape': [32, 16, 3, 3],
             'groups': 1, 'padding': 1, 'stride': 1, 'dilation': 1,
             'dtype': 'fp32', 'layout': 'nchw', 'output_elements': 32768, 'median_ms': 0.5}
        self.direct = {'strategy': 'direct', 'backend': 'cuda', 'cases': [c]}
        self.reuse = {'strategy': 'reuse', 'backend': 'cuda', 'cases': [copy.deepcopy(c)]}
        self.reuse['cases'][0]['median_ms'] = 0.25
        self.reuse['cases'][0]['full_parity'] = {'passed': True, 'elements': 32768, 'max_abs': 1e-6}

    def test_speedup_and_full_parity(self):
        self.assertEqual(compare(self.direct, self.reuse)['cases'][0]['speedup'], 2)

    def test_reject_missing_full_parity(self):
        self.reuse['cases'][0]['full_parity'] = None
        with self.assertRaisesRegex(ValueError, 'full tensor'):
            compare(self.direct, self.reuse)

    def test_reject_precision_mismatch(self):
        self.reuse['cases'][0]['dtype'] = 'fp16'
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            compare(self.direct, self.reuse)

    def test_reject_numerical_regression(self):
        self.reuse['cases'][0]['full_parity']['max_abs'] = 0.1
        with self.assertRaisesRegex(ValueError, 'drift'):
            compare(self.direct, self.reuse)

    def test_reject_backend_mismatch(self):
        self.reuse['backend'] = 'cpu'
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            compare(self.direct, self.reuse)


if __name__ == '__main__':
    unittest.main()
