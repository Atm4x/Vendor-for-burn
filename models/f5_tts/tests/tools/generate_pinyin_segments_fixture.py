from __future__ import annotations

import argparse
import ast
import random
import zipfile
from pathlib import Path

HAN = {'你': 'ni3', '好': 'hao3', '世': 'shi4', '界': 'jie4', '测': 'ce4', '试': 'shi4', '中': 'zhong1', '国': 'guo2'}


class MockJieba:
    current = ()

    def cut(self, text):
        assert ''.join(self.current) == text, (self.current, text)
        return self.current


def mock_pinyin(text, style, tone_sandhi):
    if text.startswith('你好'):
        return ['ni2', 'hao3'] + [HAN.get(ch, ch) for ch in text[2:]]
    return [HAN.get(ch, ch) for ch in text]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_archive', type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.original_archive) as source:
        original = source.read('F5-TTS-main/src/f5_tts/model/utils.py').decode()
    tree = ast.parse(original)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'convert_char_to_pinyin')
    jieba = MockJieba()
    scope = {'rjieba': jieba, 'lazy_pinyin': mock_pinyin, 'Style': type('Style', (), {'TONE3': 1})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])), str(args.original_archive), 'exec'), scope)
    reference = scope['convert_char_to_pinyin']
    seed = random.Random(1234567)
    tokens = ['你好', '世界', '测试', '中国', '你', '好', 'abc', 'hello', 'world', '!', '?', ',', ';', '.', ':', "'", ' ', '，', '。', 'a', 'é', '你abc', '你好world', '你好，']
    paths = [tokens[:], ['你好', 'world', '!'], ['你abc', '你好'], ['你好', ':', 'hello'], ['测试', '。', '世界']]
    paths += [seed.choices(tokens, k=seed.randrange(1, 25)) for _ in range(500)]
    target = Path(__file__).resolve().parents[1] / 'fixtures/pinyin_segments_python_expected.tsv'
    with target.open('w', encoding='utf-8') as output:
        for parts in paths:
            text = ''.join(parts)
            normalized = text.translate(str.maketrans({';': ',', '“': '"', '”': '"', '‘': "'", '’': "'"}))
            normalized_parts = [part.translate(str.maketrans({';': ',', '“': '"', '”': '"', '‘': "'", '’': "'"})) for part in parts]
            assert ''.join(normalized_parts) == normalized
            jieba.current = normalized_parts
            expected = reference([text])[0]
            output.write(f'{"|".join(s.encode().hex() for s in normalized_parts)}\t{"|".join(t.encode().hex() for t in expected)}\n')
    print(f'Generated {len(paths)} original Python token streams, using explicit segmentation & mocked dictionary:', target)


if __name__ == '__main__':
    main()
