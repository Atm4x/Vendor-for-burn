"""Offline validation for Stage 102 matched contracts and WAV-free archive policy."""

import copy
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compare_stage102_conv2d", ROOT / "compare_stage102_conv2d.py")
compare_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_module)


class ProfileContractTests(unittest.TestCase):
    def setUp(self):
        self.a = {
            "engine": "burn", "backend": "cuda", "cases": [{
                "name": "3x3", "shape": [1, 16, 64, 64], "weight_shape": [32, 16, 3, 3],
                "stride": 1, "padding": 1, "dilation": 1, "groups": 1,
                "output_elements": 131072, "sample_values": [0.25, 1, -1, 0],
                "mean": 0.1, "median_ms": 2, "dtype": "fp32", "layout": "nchw",
            }]
        }
        self.b = {"engine": "torch", "backend": "cuda", "cases": [copy.deepcopy(self.a["cases"][0])]}
        self.b["cases"][0]["median_ms"] = 1.2

    def test_valid_matched(self):
        c = compare_module.compare(self.a, self.b)
        self.assertTrue(c["all_sample_probes_correct"])
        self.assertAlmostEqual(c["cases"][0]["burn_over_torch"], 2/1.2)

    def test_precision_is_not_implicitly_matched(self):
        self.b["cases"][0]["dtype"] = "fp16"
        with self.assertRaisesRegex(ValueError, "precision/layout"):
            compare_module.compare(self.a, self.b)

    def test_tensor_shape_mismatch_is_rejected(self):
        self.b["cases"][0]["shape"] = [1, 16, 32, 32]
        with self.assertRaisesRegex(ValueError, "incompatible shape"):
            compare_module.compare(self.a, self.b)

    def test_small_numerical_drift_detected(self):
        self.b["cases"][0]["sample_values"][0] = 9.0
        c = compare_module.compare(self.a, self.b)
        self.assertFalse(c["all_sample_probes_correct"])

    def test_no_wavs_in_archive(self):
        runner = (ROOT / "Run-RVC-Stage102-Nsight.ps1").read_text(encoding="utf-8")
        self.assertIn("Join-Path $reports '*'", runner)
        self.assertNotIn("Compress-Archive -Path (Join-Path $root '*')", runner)
        self.assertIn("output_audio_in_zip=$false", runner)


if __name__ == "__main__":
    unittest.main()
