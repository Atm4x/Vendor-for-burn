"""CPU parity fixtures for the offline Fish Speech checkpoint converter."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn.utils.parametrizations import weight_norm

from export_checkpoint import canonicalize, export, flatten_weight_norm, pack_qkv


class ExportCheckpointTests(unittest.TestCase):
    def test_original_weight_norm_conv1d_matches_flattened(self):
        torch.manual_seed(11)
        module = weight_norm(torch.nn.Conv1d(3, 4, 7, bias=True), dim=0)
        input_value = torch.randn(2, 3, 32)
        before = module(input_value)
        state = {"conv." + key: value for key, value in module.state_dict().items()}
        converted = flatten_weight_norm(state)
        self.assertEqual(set(converted), {"conv.weight", "conv.bias"})
        raw = torch.nn.Conv1d(3, 4, 7, bias=True)
        raw.load_state_dict({key.removeprefix("conv."): tensor for key, tensor in converted.items()})
        torch.testing.assert_close(module.weight, raw.weight, rtol=0, atol=0)
        torch.testing.assert_close(before, raw(input_value), rtol=0, atol=0)

    def test_original_weight_norm_transconv1d_matches_flattened(self):
        torch.manual_seed(17)
        module = weight_norm(torch.nn.ConvTranspose1d(3, 4, 8, stride=2), dim=0)
        input_value = torch.randn(2, 3, 24)
        before = module(input_value)
        state = {"deconv." + key: value for key, value in module.state_dict().items()}
        converted = flatten_weight_norm(state)
        raw = torch.nn.ConvTranspose1d(3, 4, 8, stride=2)
        raw.load_state_dict({key.removeprefix("deconv."): tensor for key, tensor in converted.items()})
        torch.testing.assert_close(module.weight, raw.weight, rtol=0, atol=0)
        torch.testing.assert_close(before, raw(input_value), rtol=0, atol=0)

    def test_qkv_fish_packing_and_alias(self):
        state = {"model.layers.0.attention." + key: torch.ones(2, 4) * value
                 for key, value in (("wq.weight", 1), ("wk.weight", 2), ("wv.weight", 3))}
        packed = canonicalize(state, "semantic")
        self.assertEqual(list(packed), ["layers.0.attention.wqkv.weight"])
        self.assertEqual(packed["layers.0.attention.wqkv.weight"].shape, (6, 4))
        self.assertEqual(packed["layers.0.attention.wqkv.weight"][:, 0].tolist(), [1, 1, 2, 2, 3, 3])
        with self.assertRaises(ValueError):
            pack_qkv({"one.wq.weight": torch.ones(2, 4)})

    def test_generator_prefixed_checkpoint_and_roundtrip_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoint = root / "firefly.pth"
            config = root / "config.json"
            output = root / "exported"
            config.write_bytes((Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "firefly_1_5.json").read_bytes())
            torch.save({"state_dict": {"generator.head.conv_pre.conv.weight": torch.randn(2, 3, 7),
                                      "optimizer.weight": torch.randn(2, 3)}}, checkpoint)
            manifest = export(checkpoint, config, output, "firefly", "test-fixture-license")
            self.assertEqual(list(manifest["tensors"]), ["head.conv_pre.conv.weight"])
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "firefly.safetensors").is_file())
            self.assertEqual(len(manifest["source_checkpoint_sha256"]), 64)

    def test_unrecognized_firefly_variant_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoint = root / "firefly.pth"
            config = root / "config.json"
            torch.save({"generator.conv.weight": torch.ones(1, 1, 1)}, checkpoint)
            config.write_text('{"_target_":"unexpected"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not match"):
                export(checkpoint, config, root / "exported", "firefly", "fixture")
            self.assertFalse((root / "exported").exists())

    def test_unmatched_norm_is_rejected(self):
        with self.assertRaises(ValueError):
            flatten_weight_norm({"conv.parametrizations.weight.original0": torch.ones(3, 1, 1)})


if __name__ == "__main__":
    unittest.main()
