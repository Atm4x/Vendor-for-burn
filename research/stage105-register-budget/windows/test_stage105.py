import unittest
from compare_stage105_pair import compare


class PairContract(unittest.TestCase):
    def setUp(self):
        self.case = {"name": "dense3x3", "shape": [2, 32, 64, 64], "weight_shape": [32, 32, 3, 3],
                     "groups": 1, "padding": 1, "stride": 1, "dilation": 1,
                     "dtype": "fp32", "layout": "nchw", "median_ms": 0.4,
                     "output_elements": 262144}
        self.base = {"strategy": "direct", "backend": "cuda", "cases": [self.case.copy()]}
        tested = {**self.case, "median_ms": 0.2,
                  "full_parity": {"passed": True, "elements": 262144, "max_abs": 1e-6}}
        self.candidate = {"strategy": "pair", "backend": "cuda", "cases": [tested]}

    def test_matching(self):
        self.assertAlmostEqual(compare(self.base, self.candidate)["results"][0]["speedup"], 2.0)

    def test_wrong_strategy(self):
        self.candidate["strategy"] = "reuse"
        with self.assertRaises(ValueError): compare(self.base, self.candidate)

    def test_missing_full_parity(self):
        self.candidate["cases"][0].pop("full_parity")
        with self.assertRaises(ValueError): compare(self.base, self.candidate)

    def test_shape_mismatch(self):
        self.candidate["cases"][0]["shape"] = [2, 64, 64, 64]
        with self.assertRaises(ValueError): compare(self.base, self.candidate)

    def test_reject_drift(self):
        self.candidate["cases"][0]["full_parity"]["max_abs"] = 0.001
        with self.assertRaises(ValueError): compare(self.base, self.candidate)


if __name__ == "__main__": unittest.main()
