#!/usr/bin/env python3
"""Developer-only independent oracle for multilingual checkpoints. Runtime never imports torch."""
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
from smoke_native_worker import run as smoke

V3_CASES = {
    'v3_en': ('Hello, world!', 'en_0'),
    'v3_en_indic': ('This is a voice.', 'tamil_female'),
    'v3_de': ('Fröhliche Grüße, schön!', 'eva_k'),
    'v3_es': ('¡Buenos días! ¿Cómo estás?', 'es_0'),
    'v3_fr': ('Bonjour, ça va très bien.', 'fr_0'),
    'v3_indic': ('namastē duniyā.', 'hindi_female'),
}


V4_CASES = {'v4_ru': [('Привет, мир!', 'aidar'), ('Съешь ещё этих мягких французских булок.', 'aidar'), ('Кто-то на двери увидел замок.', 'aidar'), ('Я люблю русский язык.', 'aidar'), ('Прив+ет, всё хорошо.', 'aidar')], 'v4_ua': [('Привіт, світе!', 'mykyta')], 'v4_uz': [('Salom dunyo!', 'dilnavoz')], 'v4_indic': [('namastē duniyā.', 'hindi_female')], 'v4_cyrillic': [('Исәнмесез.', 'b_tat')]}

V5_CASES = {
    'v5_cis_base': [('Прив+ет.', 'ru_aigul'), ('Բարեւ աշխարհ։', 'hye_zara'), ('Salam dünya.', 'aze_gamat')],
    'v5_cis_base_nostress': [('Привет.', 'ru_aigul')],
    'v5_turkic': [('Исәнмесез.', 'tat_0')],
}


def compare(binary, path, text, speaker, variant="v3_hifigan"):
    requests = [dict(command='load', request_id='load', checkpoint=str(path.resolve()), backend='flex')]
    requests += [dict(command='infer', request_id=str(rate), speaker=speaker, sample_rate=rate,
                      input=dict(kind='text', text=text)) for rate in (8000, 24000, 48000)]
    requests += [dict(command='unload', request_id='unload'), dict(command='shutdown', request_id='shutdown')]
    with tempfile.TemporaryDirectory() as folder:
        renamed = Path(folder) / 'arbitrary-name.bin'
        os.link(path.resolve(), renamed)
        requests[0]['checkpoint'] = str(renamed)
        process = subprocess.run([str(binary.resolve()), '--stdio'], input=''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in requests),
                                 capture_output=True, text=True, encoding='utf-8', check=True, timeout=120)
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(replies) == len(requests), process.stderr
    assert all(x['ok'] for x in replies), [(x.get('request_id'), x.get('error')) for x in replies if not x['ok']]
    assert replies[0]['data']['model_variant'] == variant
    oracle = PackageImporter(str(path)).load_pickle('tts_models', 'model')
    assert set(replies[0]['data']['speaker_names']) == (set(oracle.speaker_to_id if variant == "v3_hifigan" else oracle.speaker_to_model if variant == "v4_hifigan" else oracle.speaker_to_package) - {"random"})
    for rate, response in zip((8000, 24000, 48000), replies[1:4]):
        samples = np.frombuffer(base64.b64decode(response['data']['audio']['data_base64']), dtype='<f4')
        with torch.inference_mode():
            expected = oracle.apply_tts(text=text, speaker=speaker, sample_rate=rate).cpu().numpy().ravel()
        assert samples.shape == expected.shape, (path.name, rate, samples.shape, expected.shape)
        assert np.isfinite(samples).all() and np.any(np.abs(samples) > 0.01)
        max_abs = np.max(np.abs(samples - expected))
        snr = 10 * np.log10(np.sum(expected.astype(np.float64)**2) / np.sum((samples.astype(np.float64) - expected)**2))
        print(json.dumps(dict(model=path.stem, speaker=speaker, rate=rate, count=len(samples), max_abs=float(max_abs), snr=float(snr))), flush=True)
        assert max_abs <= 1e-3, (path.name, rate, max_abs)
    smoke(binary, path, text, speaker, variant)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('checkpoints', type=Path)
    parser.add_argument('--model', choices=(*V3_CASES, *V4_CASES, *V5_CASES), action='append')
    parser.add_argument('--reject-unimplemented', action='store_true', help='Check the acquired unsupported neural-stress V5 graph')
    args = parser.parse_args()
    torch.set_num_threads(4)
    for model in args.model or (*V3_CASES, *V4_CASES, *V5_CASES):
        path = args.checkpoints / (model + '.pt')
        if model in V3_CASES:
            compare(args.binary, path, *V3_CASES[model])
        elif model in V4_CASES:
            for text, speaker in V4_CASES[model]:
                compare(args.binary, path, text, speaker, "v4_hifigan")
        else:
            for text, speaker in V5_CASES[model]:
                compare(args.binary, path, text, speaker, 'v5_convnext_plain')

    if args.reject_unimplemented:
        for model in ('v5_6_ru',):
            requests = [dict(command='load', request_id='unsupported', checkpoint=str((args.checkpoints / (model + '.pt')).resolve()), backend='flex'), dict(command='shutdown', request_id='shutdown')]
            process = subprocess.run([str(args.binary.resolve()), '--stdio'], input=''.join(json.dumps(x)+'\n' for x in requests), capture_output=True, text=True, check=True, timeout=120)
            replies = [json.loads(line) for line in process.stdout.splitlines()]
            assert len(replies) == 2 and not replies[0]['ok'] and replies[1]['ok'], replies
            expected = 'unsupported neural-stress V5 graph' if model == 'v5_6_ru' else 'unsupported V4 graph'
            assert expected in str(replies[0]['error']), replies[0]
            print(json.dumps(dict(model=model, expected_rejection=True)), flush=True)
