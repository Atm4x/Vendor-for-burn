from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import math
import json
import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parents[1] / 'fixtures'
TOOLS = Path(__file__).resolve().parent


def load_oracle(name, filename):
    spec = spec_from_file_location(name, TOOLS / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    dit = load_oracle('dit_oracle', 'generate_full_dit_fixture.py')
    vocos = load_oracle('vocos_oracle', 'generate_vocos_decoder_fixture.py')
    torch.manual_seed(11809)
    dit.MEL = 100
    dit.weights['transformer.input_embed.proj.weight'] = torch.randn(dit.DIM, 2 * dit.MEL + dit.TEXT_DIM) * 0.07
    dit.weights['transformer.proj_out.weight'] = torch.randn(dit.MEL, dit.DIM) * 0.07
    dit.weights['transformer.proj_out.bias'] = torch.randn(dit.MEL) * 0.035
    save_file({name: value.contiguous() for name, value in dit.weights.items()}, HERE / 'e2e_dit_small.safetensors')

    network = vocos.Toy().eval()
    network.load_state_dict(load_file(HERE / 'vocos_decoder_small.safetensors'))

    def render(dit_fixture, reference_frames, gain, suffix, sway=None, empirical=False):
        frames = dit_fixture['frames']
        mel = torch.tensor(dit_fixture['trajectory_final'], dtype=torch.float32).reshape(1, frames, 100)
        generated = mel[:, reference_frames:, :]
        with torch.inference_mode():
            head = network(generated)
            magnitude, phase = head[0].T.chunk(2, 0)
            spectrum = torch.polar(magnitude.exp(), phase)
            spectrum = torch.cat((spectrum, spectrum[:, -1:]), dim=1)
            waveform = torch.istft(spectrum, n_fft=1024, hop_length=256,
                                   center=True, window=torch.hann_window(1024)) * gain
        fixture = {
            'config': dit_fixture['config'],
            'batch': 1, 'frames': frames, 'reference_frames': reference_frames,
            'token_ids': dit_fixture['token_ids'],
            'noise': dit_fixture['noisy'],
            'conditioned_reference': dit_fixture['conditioned_reference'],
            'cfg_strength': dit_fixture['cfg_strength'],
            'steps': dit_fixture['sampling_steps'],
            'sway_sampling_coef': sway,
            'empirically_pruned_steps': empirical,
            'restoration_gain': gain,
            'trajectory_states': dit_fixture['trajectory_states'],
            'full_mel': dit_fixture['trajectory_final'],
            'generated_mel': generated.flatten().tolist(),
            'vocos_head': head.flatten().tolist(),
            'wave': waveform.flatten().tolist(),
            'pcm16': [max(-32768, min(32767, round(float(v) * 32768))) for v in waveform.clamp(-1, 1)],
        }
        (HERE / suffix).write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
        print(suffix, len(fixture['wave']), 'PCM samples')

    basic = dit.case(1, 9, [[2, 4, 1, 3]], 0.4, 1.7, reference_frames=3, steps=3)
    render(basic, 3, 0.75, 'e2e_audio_pytorch_expected.json')

    epss = torch.tensor([0, 2, 4, 8, 16, 32], dtype=torch.float32) / 32.0
    sway = -1.0
    times = epss + sway * ((math.pi * 0.5 * epss).cos() - 1 + epss)
    pruned = dit.case(1, 8, [[3, 1, 7]], 0.65, 2.2, reference_frames=2,
                      steps=5, timeline=times.tolist())
    render(pruned, 2, 0.6, 'e2e_audio_sway_pytorch_expected.json',
           sway=sway, empirical=True)



if __name__ == '__main__':
    main()
