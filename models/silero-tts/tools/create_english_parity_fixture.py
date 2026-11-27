"""Developer-only oracle for the official English package; never used by the worker."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.package import PackageImporter


def run(checkpoint: Path, output: Path) -> None:
    torch.set_num_threads(4)
    package = PackageImporter(str(checkpoint)).load_pickle('tts_models', 'model')
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    data = {}
    with torch.inference_mode():
        for index, (text, speaker) in enumerate([('Hello, world!', 'en_0'), ('This is a native voice.', 'en_1')]):
            model_audio = package.apply_tts(text=text, speaker=speaker, sample_rate=8000)
            model = package.model.tts_model
            sentence, _, valid = package.prepare_text_input(text)
            assert valid
            tokens = package.model.preprocess_tacotron(sentence).unsqueeze(0)
            speakers = torch.tensor([package.speaker_to_id[speaker]])
            mask = tokens.eq(0)
            duration = model.dur_predictor(tokens, speakers, mask, 1.0)
            duration_frames = (duration.exp() - 1).clamp_min(0).round()
            pitch = model.pitch_predictor(tokens, speakers, mask)
            mel = model.tacotron(tokens, speakers, mask, duration_frames, pitch)
            prefix = f'case{index}'
            data[f'{prefix}_duration'] = duration_frames
            data[f'{prefix}_log_duration'] = duration
            data[f'{prefix}_pitch'] = pitch
            data[f'{prefix}_mel'] = mel
            for rate in (8000, 24000, 48000):
                pcm = model.vocoder(mel, rate).squeeze(1)
                assert pcm.shape[-1] == int(duration_frames.sum()) * (rate // 80)
                actual = model_audio if rate == 8000 else package.apply_tts(text=text, speaker=speaker, sample_rate=rate)
                torch.testing.assert_close(pcm.squeeze(0), actual, atol=1e-6, rtol=1e-6)
                data[f'{prefix}_pcm{rate}'] = pcm
            cases.append(dict(prefix=prefix, text=text, speaker=speaker, speaker_id=int(speakers[0]), tokens=tokens[0].tolist()))
    save_file({k: v.detach().cpu().contiguous().clone().float() for k, v in data.items()}, str(output / 'english.safetensors'))
    (output / 'english.json').write_text(json.dumps(cases, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
