import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3]


class Stage106Structure(unittest.TestCase):
    def test_polyphase_does_not_allocate_im2col(self):
        path = ROOT / 'vendor/burn-cubecl-0.22.1/src/kernel/conv/conv_transpose2d/polyphase1d.rs'
        code = path.read_text(encoding='utf-8')
        self.assertIn('polyphase_pair_kernel', code)
        self.assertIn('conv_transpose1d_prepack_weight', code)
        self.assertIn('conv_transpose1d_polyphase_packed', code)
        self.assertNotIn('col2im(', code)

    def test_noise_tiled_is_specialized_for_single_input(self):
        path = ROOT / 'vendor/burn-cubecl-0.22.1/src/kernel/conv/noise1d.rs'
        code = path.read_text(encoding='utf-8')
        self.assertIn('noise_temporal_pair_kernel', code)
        self.assertIn('cin != 1', code)
        self.assertIn('conv1d_single_input_prepack_weight', code)

    def test_diagnostics_are_bounded_and_audio_free(self):
        path = ROOT / 'research/stage106-nsf-polyphase/windows/Run-RVC-Stage106-NSF-Dataflow.ps1'
        code = path.read_text(encoding='utf-8')
        for part in ('--nvtx-push-pop-scope', '--nvtx-include', '--launch-count', 'full_output_parity', 'RVC_STAGE106_NSF_DATAFLOW_RESULTS.zip'):
            self.assertIn(part, code)
        self.assertNotIn('Copy-Item *.wav', code)

    def test_inference_is_explicit_opt_in(self):
        code = (ROOT / 'models/tts-with-rvc-rs/src/decoder.rs').read_text(encoding='utf-8')
        self.assertIn('RVC_NSF_POLYPHASE', code)
        self.assertIn('RVC_NSF_NOISE_TILED', code)
        self.assertIn('prepack_transpose1d_weight', code)


if __name__ == '__main__':
    unittest.main()
