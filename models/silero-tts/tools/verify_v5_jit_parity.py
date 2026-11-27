#!/usr/bin/env python3
"""Independent PyTorch oracle for the exact original CIS standalone graph."""
import argparse
import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.package import PackageImporter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('package', type=Path)
    p.add_argument('jit', type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    package = PackageImporter(str(args.package)).load_pickle('tts_models', 'model')
    graph = torch.jit.load(str(args.jit), map_location='cpu').eval()
    frontend = json.loads(Path(__file__).resolve().parents[1].joinpath('v5_cis_jit_frontend.json').read_text())
    cases = [('Привет.', 'ru_aigul'), ('Բարեւ աշխարհ։', 'hye_zara'), ('Salam dünya.', 'aze_gamat')]
    with tempfile.TemporaryDirectory() as folder:
        renamed = Path(folder) / 'arbitrary.bin'
        os.link(args.jit.resolve(), renamed)
        requests = [dict(command='load', request_id='load', checkpoint=str(renamed), backend='flex')]
        for i, (text, speaker) in enumerate(cases):
            requests.append(dict(command='infer', request_id=str(i), speaker=speaker, sample_rate=48000, input=dict(kind='text', text=text)))
        requests += [dict(command='unload', request_id='unload'), dict(command='shutdown', request_id='shutdown')]
        process = subprocess.run([str(args.binary.resolve()), '--stdio'], input=''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in requests), capture_output=True, text=True, encoding='utf-8', check=True, timeout=120)
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(replies) == len(requests) and all(x['ok'] for x in replies), replies
    assert replies[0]['data']['model_variant'] == 'v5_convnext_jit'
    part = package.packages[0]
    for (text, speaker), response in zip(cases, replies[1:]):
        speaker_ids, _ = part.get_speakers(speaker)
        s, _, b, r, p, _ = part.prepare_tts_model_input(text, False, speaker_ids, speaker.split('_')[0])
        seq, _, _, _ = part.merge_batch_model(s,b,r,p)
        sp = torch.tensor([frontend['speakers'][speaker]], dtype=torch.long)
        with torch.inference_mode():
            expected = graph(seq, sp, sr=torch.tensor(48000)).cpu().numpy().ravel()
            assert torch.equal(graph(seq, sp, sr=torch.tensor(8000)), graph(seq, sp, sr=torch.tensor(24000)))
        actual = np.frombuffer(base64.b64decode(response['data']['audio']['data_base64']), dtype='<f4')
        assert actual.shape == expected.shape, (speaker, actual.shape, expected.shape)
        assert np.isfinite(actual).all()
        error = float(np.max(np.abs(actual-expected)))
        print(json.dumps(dict(speaker=speaker, samples=len(actual), max_abs=error)), flush=True)
        assert error <= 1e-3, error


if __name__ == '__main__':
    main()
