"""Independent original F5-TTS chunk_text + NumPy multi-chunk and voice stitching oracle."""
import ast
import os
import json
import re
import struct
import zipfile
from pathlib import Path

import numpy as np


UPSTREAM = Path(os.environ.get('F5_TTS_ORIGINAL_ZIP', '/mnt/data/F5-TTS-main (1).zip'))
SOURCE_NAME = 'F5-TTS-main/src/f5_tts/infer/utils_infer.py'


def original_chunker():
    with zipfile.ZipFile(UPSTREAM) as archive:
        source = archive.read(SOURCE_NAME).decode('utf-8')
    parsed = ast.parse(source)
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'chunk_text')
    scope = {'re': re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), SOURCE_NAME, 'exec'), scope)
    return scope['chunk_text']


def float32(x):
    return struct.unpack('f', struct.pack('f', float(x)))[0]


def chunk_plan(text, transcript, ref_samples, speed, fixed, fade, chunker):
    ref_text = transcript.strip()
    if not ref_text.endswith('. ') and not ref_text.endswith('。'):
        ref_text += ' ' if ref_text.endswith('.') else '. '
    ref_seconds = ref_samples / 24000
    max_chars = int(len(ref_text.encode()) / ref_seconds * (22 - ref_seconds) * speed)
    chunks = chunker(text, max_chars=max_chars)
    ref_frames = ref_samples // 256
    ref_effective = ref_text + (' ' if ref_text[-1].isascii() else '')
    sum_bytes = sum(len(x.encode()) for x in chunks)
    frames = []
    for chunk in chunks:
        if fixed is not None:
            allocation = float(fixed) - ref_frames * 256 / 24000 + float(fade) * (len(chunks)-1)
            allocation = ref_frames * 256 / 24000 + allocation * len(chunk.encode()) / sum_bytes
            duration = int(float32(allocation) * 24000 / 256)
        else:
            local_speed = speed * 0.3 if len(chunk.encode()) < 10 else speed
            duration = ref_frames + int(ref_frames / len(ref_effective.encode()) * len(chunk.encode()) / local_speed)
        frames.append(duration)
    return chunks, frames


def combine(waves, fade):
    joined = waves[0]
    fade_length = int(fade * 24000)
    for nxt in waves[1:]:
        overlap = min(fade_length, len(joined), len(nxt))
        if overlap:
            fade_out = np.linspace(1.0, 0.0, overlap)
            fade_in = np.linspace(0.0, 1.0, overlap)
            middle = joined[-overlap:] * fade_out + nxt[:overlap] * fade_in
            joined = np.concatenate([joined[:-overlap], middle, nxt[overlap:]])
        else:
            joined = np.concatenate([joined, nxt])
    return joined


def value_wave(index):
    return np.full(8192, (index+1)/8, dtype=np.float32)


def run(case, chunker):
    serial = 0
    calls = []
    if case['kind'] == 'standard':
        chunks, frames = chunk_plan(case['text'], 'Hi', 120_000, 1.0, case['fixed'], case['fade'], chunker)
        waves = []
        for chunk, frame in zip(chunks, frames):
            waves.append(value_wave(serial))
            calls.append({'voice': 'main', 'text': chunk, 'frames': frame})
            serial += 1
        result = combine(waves, case['fade'])
    else:
        script = case['script']
        parts = re.split(r'(\[[a-zA-Z0-9_]+\])', script)
        tagged = []
        voice = 'main'
        for part in parts:
            if re.fullmatch(r'\[[a-zA-Z0-9_]+\]', part):
                voice = part[1:-1]
            elif part.strip():
                tagged.append((voice, part.strip()))
        output = []
        for voice, text in tagged:
            selected = voice if voice in ('main', 'other') else 'main'
            transcript = 'Hi' if selected == 'main' else 'Hey'
            chunks, frames = chunk_plan(text, transcript, 120_000, 1.0, None, 0.15, chunker)
            waves = []
            for chunk, frame in zip(chunks, frames):
                waves.append(value_wave(serial))
                calls.append({'voice': selected, 'text': chunk, 'frames': frame})
                serial += 1
            output.append(combine(waves, 0.15))
        result = np.concatenate(output)
    return {'kind':case['kind'], 'input':case, 'calls':calls}, result.astype('<f4')


def main():
    chunker = original_chunker()
    cases = [
        {'kind':'standard','text':'One sentence. Another sentence. Yet another sentence.', 'fixed':None, 'fade':0.15},
        {'kind':'standard','text':'First sentence. Second sentence. A third sentence.', 'fixed':12.0, 'fade':0.15},
        {'kind':'multi','script':'[main] One sentence. Another sentence. [other] Hello there. More words. [other] And more! [unknown] Finally.'},
    ]
    result = {'source':'F5-TTS-main/src/f5_tts/infer/utils_infer.py chunk_text and infer_batch_process crossfade',
              'cases':[]}
    packed = bytearray()
    for case in cases:
        record, samples = run(case, chunker)
        record['offset_samples'] = len(packed) // 4
        record['sample_count'] = len(samples)
        packed.extend(samples.tobytes())
        result['cases'].append(record)
    path = Path(__file__).resolve().parent.parent/'fixtures'/'trtx_worker_python_expected.json'
    path.write_text(json.dumps(result, separators=(',',':')),encoding='utf-8')
    path.with_suffix('.f32le').write_bytes(packed)
    print('cases',len(result['cases']),'wave_samples',len(packed)//4,'calls',[len(c['calls']) for c in result['cases']])

if __name__ == '__main__':
    main()
