"""Independent frame-domain speech-edit oracle executing the source AST from original F5-TTS."""
import ast
import json
import os
from pathlib import Path
import random
import struct
import zipfile

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path(os.environ.get('F5_TTS_ORIGINAL_ZIP', '/mnt/data/F5-TTS-main (1).zip'))
SOURCE = 'F5-TTS-main/src/f5_tts/infer/speech_edit.py'


def extract_upstream_edit():
    with zipfile.ZipFile(ORIGINAL) as archive:
        parsed = ast.parse(archive.read(SOURCE).decode('utf8'))
    nodes = parsed.body
    def assigned(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i, n in enumerate(nodes) if assigned(n, 'offset_frame'))
    assert assigned(nodes[start + 1], 'mel_cond')
    assert assigned(nodes[start + 2], 'edit_mask')
    end = next(i for i in range(start, len(nodes)) if assigned(nodes[i], 'edit_mask')
               and isinstance(nodes[i].value, ast.Call) and isinstance(nodes[i].value.func, ast.Attribute)
               and nodes[i].value.func.attr == 'pad')
    isolated = ast.fix_missing_locations(ast.Module(body=nodes[start:end + 1], type_ignores=[]))
    return compile(isolated, SOURCE, 'exec')


def fnv64(values):
    digest = 0xcbf29ce484222325
    for byte in values:
        digest = ((digest ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    return f'{digest:016x}'


def frame_seconds(frame):
    return frame * 256.0 / 24000.0


def generate():
    code = extract_upstream_edit()
    rng = random.Random(175023)
    cases = []
    for iteration in range(320):
        original_frames = rng.randrange(12, 145)
        original = torch.arange(original_frames * 100, dtype=torch.float32).reshape(1, original_frames, 100)
        parts = []
        cursor = 1
        for _ in range(rng.randrange(1, 5)):
            remaining = original_frames - cursor
            if remaining < 4:
                break
            start = rng.randrange(cursor, original_frames - 2)
            end = rng.randrange(start + 1, min(original_frames, start + 17) + 1)
            parts.append((start, end))
            cursor = end + rng.randrange(0, 3)
        if not parts:
            continue
        seconds = [[frame_seconds(start), frame_seconds(end)] for start, end in parts]
        replacements = [frame_seconds(rng.randrange(1, 20)) for _ in parts] if iteration % 3 != 0 else None
        # Values close to half-frame exercise the Python round() tie-to-even rule.
        if iteration % 9 == 0:
            for item in seconds:
                item[0] += (0.5 * 256 / 24000) if (item[1] - item[0]) > frame_seconds(2) else 0.0
        scope = {
            'torch': torch, 'F': F, 'device': torch.device('cpu'),
            'original_mel': original, 'n_mel_channels': 100,
            'target_sample_rate': 24000, 'hop_length': 256,
            'parts_to_edit': seconds, 'fix_duration': replacements,
        }
        exec(code, scope)
        condition = scope['mel_cond']
        mask = scope['edit_mask']
        assert condition.shape[1] == mask.numel()
        if not condition.shape[1] or condition.shape[1] >= 65536:
            continue
        text_len = rng.randrange(1, 180)
        total = max(text_len, condition.shape[1]) + 1
        padded = F.pad(condition, (0, 0, 0, total - condition.shape[1]))
        padded_mask = F.pad(mask, (0, total - mask.numel()), value=False)
        raw = padded.contiguous().numpy().tobytes()
        def bits(x):
            return None if x is None else int.from_bytes(struct.pack('<d', x), 'little')
        cases.append({
            'original_frames': original_frames,
            'spans': [{'start_seconds': part[0], 'end_seconds': part[1],
                       'replacement_seconds': None if replacements is None else replacements[i],
                       'start_bits': bits(part[0]), 'end_bits': bits(part[1]),
                       'replacement_bits': bits(None if replacements is None else replacements[i])}
                      for i, part in enumerate(seconds)],
            'text_tokens': text_len,
            'edited_frames': condition.shape[1],
            'total_frames': total,
            'mask': ''.join('1' if x else '0' for x in padded_mask.flatten().tolist()),
            'mel_fnv64': fnv64(raw),
        })
    output = ROOT / 'fixtures' / 'speech_edit_upstream_expected.json'
    output.write_text(json.dumps({'source': SOURCE, 'cases': cases}, separators=(',', ':'))+'\n', encoding='utf8')
    print(output, len(cases), 'cases')


if __name__ == '__main__':
    generate()
