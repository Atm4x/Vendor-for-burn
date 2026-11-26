"""Static-configuration normalization regression tests without model dependencies."""

import json
import unittest
from pathlib import Path

from normalize_firefly_config import normalize_config


class FireflySourceTests(unittest.TestCase):
    def test_reference_literal_is_stable(self):
        source = Path(__file__).parents[1] / "tests" / "fixtures" / "firefly_1_5.json"
        reference = json.loads(source.read_text(encoding="utf-8"))
        normalized = normalize_config(f"FIREFLY_GAN_VQ_CONFIG = {reference!r}")
        self.assertEqual(normalized, reference)

    def test_never_executes_python_expressions(self):
        with self.assertRaises((ValueError, TypeError)):
            normalize_config("FIREFLY_GAN_VQ_CONFIG = __import__('os').system('echo bad')")

    def test_rejects_ambiguous_and_nonliteral_configs(self):
        for source in (
            "FIREFLY_GAN_VQ_CONFIG = {}\nFIREFLY_GAN_VQ_CONFIG = {}",
            "FIREFLY_GAN_VQ_CONFIG = {1: 5}",
            "FIREFLY_GAN_VQ_CONFIG = {}\nFIREFLY_GAN_VQ_CONFIG['a'] = 1",
            "FIREFLY_GAN_VQ_CONFIG = int()",
            "FIREFLY_GAN_VQ_CONFIG = {'x': 1 + 2}",
        ):
            with self.subTest(source=source), self.assertRaises((ValueError, TypeError)):
                normalize_config(source)


if __name__ == "__main__":
    unittest.main()
