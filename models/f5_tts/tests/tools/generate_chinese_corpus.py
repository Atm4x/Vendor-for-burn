"""Rebuild CONT12 Chinese parity data using pinned Python language libraries."""

import json
import random
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import rjieba
from pypinyin import Style, lazy_pinyin
from pypinyin.phrases_dict import phrases_dict

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'assets'
FIXTURES = ROOT / 'tests' / 'fixtures'


def is_han(symbol):
    return '\u3100' <= symbol <= '\u9fff'


def valid_reading(value):
    prefix = value.rstrip('12345')
    return value.isascii() and prefix.islower() and len(value) - len(prefix) <= 1


def main():
    assert version('rjieba') == '0.2.2'
    assert version('pypinyin') == '0.55.0'

    chars = []
    for code in range(0x3100, 0xa000):
        symbol = chr(code)
        if not is_han(symbol):
            continue
        readings = lazy_pinyin(symbol, style=Style.TONE3, tone_sandhi=True)
        if len(readings) == 1 and valid_reading(readings[0]):
            chars.append(f'{symbol}\t{readings[0]}')
    (ASSETS / 'pypinyin_chars.tsv').write_text('\n'.join(chars) + '\n', encoding='utf-8')

    phrases = []
    for word in sorted(phrases_dict):
        if not 2 <= len(word) <= 32 or not all(map(is_han, word)):
            continue
        readings = lazy_pinyin(word, style=Style.TONE3, tone_sandhi=True)
        if len(readings) == len(word) and all(map(valid_reading, readings)):
            phrases.append(f'{word}\t{" ".join(readings)}')
    (ASSETS / 'pypinyin_phrases.tsv').write_text('\n'.join(phrases) + '\n', encoding='utf-8')

    random_source = random.Random(39281)
    keys = [row.split('\t', 1)[0] for row in phrases]
    samples = [
        '你好，世界！', '我是中国人。', '银行行长今天来到重庆。',
        '你好吗？Yes, I am fine!', '今天的天气怎么样？',
        '第一，不要去。', '绿茶和咖啡', 'Hello中国world!',
        '中英文 Mixed text 123.',
    ]
    for _ in range(1800):
        separator = random_source.choice(['', '，', '。', ' ', ' hello '])
        samples.append(separator.join(random_source.choice(keys) for _ in range(random_source.randint(1, 5))))

    with (FIXTURES / 'rjieba_reference.jsonl').open('w', encoding='utf-8') as output:
        for text in samples:
            segments = list(rjieba.cut(text))
            pronunciations = [
                lazy_pinyin(segment, style=Style.TONE3, tone_sandhi=True)
                for segment in segments
            ]
            output.write(json.dumps({
                'text': text,
                'segments': segments,
                'syllables': pronunciations,
            }, ensure_ascii=False) + '\n')

    subprocess.run([sys.executable, str(Path(__file__).with_name('generate_rjieba_token_fixture.py'))], check=True)
    print(f'chars={len(chars)} phrases={len(phrases)} cases={len(samples)}')


if __name__ == '__main__':
    main()
