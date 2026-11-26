import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'fixtures'
ASSETS = Path(__file__).resolve().parents[2] / 'assets'

readings = {}
for line in (ASSETS / 'pypinyin_chars.tsv').read_text(encoding='utf-8').splitlines():
    han, phoneme = line.split('\t', 1)
    readings[han] = phoneme

custom = str.maketrans({';': ',', '“': '"', '”': '"', '‘': "'", '’': "'"})


def is_han(char):
    return '\u3100' <= char <= '\u9fff'


def normalize(entry):
    result = []
    for word, pronunciations in zip(entry['segments'], entry['syllables']):
        word = word.translate(custom)
        byte_length = len(word.encode('utf-8'))
        if byte_length == len(word):
            if result and byte_length > 1 and result[-1] not in " :'\"":
                result.append(' ')
            result.extend(word)
        elif byte_length == 3 * len(word):
            for char, syllable in zip(word, pronunciations):
                if is_han(char):
                    result.append(' ')
                result.append(syllable)
        else:
            for char in word:
                if ord(char) < 256:
                    result.append(char)
                elif is_han(char):
                    result.append(' ')
                    result.append(readings[char])
                else:
                    result.append(char)
    return result


source = ROOT / 'rjieba_reference.jsonl'
target = ROOT / 'rjieba_f5_tokens_expected.jsonl'
with source.open(encoding='utf-8') as reader, target.open('w', encoding='utf-8') as writer:
    for line in reader:
        entry = json.loads(line)
        tokens = normalize(entry)
        writer.write(json.dumps({'text': entry['text'], 'tokens': tokens}, ensure_ascii=False, separators=(',', ':')) + '\n')
print(target, target.stat().st_size)
