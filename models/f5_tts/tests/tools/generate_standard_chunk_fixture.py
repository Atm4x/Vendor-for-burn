from __future__ import annotations

import argparse
import ast
import random
import re
import zipfile
from pathlib import Path


SOURCES = ('F5-TTS-main/src/f5_tts/infer/utils_infer.py', 'src/f5_tts/infer/utils_infer.py')


def python_reference(archive: Path):
    with zipfile.ZipFile(archive) as z:
        original = next(z.read(path).decode('utf-8') for path in SOURCES if path in z.namelist())
    module = ast.parse(original)
    function = next(item for item in module.body if isinstance(item, ast.FunctionDef) and item.name == 'chunk_text')
    target = ast.Module(body=[function], type_ignores=[])
    scope = {"re": re}
    exec(compile(ast.fix_missing_locations(target), str(archive), 'exec'), scope)
    return scope['chunk_text']


def make_cases():
    rng = random.Random(20261120)
    tokens = ['hello', 'world', 'a', 'sample', '测试', '语音', '你好', 'words', '.', '?', '!', ',', ';', ':',
              '，', '。', '！', '？', '；', '：', ' ', '\t', '\n', ' ', ' ', 'abc123', '你好hello', 'Русский']
    cases = [('', 0), ('hello', 4), ('hello. world', 6), ('你好，世界。', 9),
             ('Hello! World. Again?', 12), ('UnbrokenVeryLongSentence', 2)]
    for _ in range(1_000):
        text = ''.join(rng.choices(tokens, k=rng.randrange(1, 35)))
        max_chars = rng.choice([0, 1, 2, 4, 8, 12, 16, 30, 50, 90, 135, 256])
        cases.append((text, max_chars))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_archive', type=Path)
    args = parser.parse_args()
    func = python_reference(args.original_archive)
    target = Path(__file__).resolve().parents[1] / 'fixtures/standard_chunk_python_expected.tsv'
    with target.open('w', encoding='utf-8') as output:
        for text, max_chars in make_cases():
            expected = func(text, max_chars)
            output.write(f'{max_chars}\t{text.encode("utf8").hex()}\t{len(expected)}\t{"|".join(segment.encode("utf8").hex() for segment in expected)}\n')
    print(f'Created {len(make_cases())} differential records at {target}')


if __name__ == '__main__':
    main()
