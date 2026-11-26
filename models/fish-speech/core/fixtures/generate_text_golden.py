from pathlib import Path
import random
import sys

source = Path(sys.argv[1])
clean_ns = {}
exec((source / 'fish_speech_lib/fish_speech/text/clean.py').read_text(), clean_ns)
split_ns = {'clean_text': clean_ns['clean_text']}
exec((source / 'fish_speech_lib/fish_speech/text/spliter.py').read_text().replace('from fish_speech_lib.fish_speech.text.clean import clean_text', ''), split_ns)

cases = [
    ('', 50), ('  ', 5), (',,,...!!!', 3), ('a', 1), ('a,aaaaaa3.14', 10),
    ('3.14. 15.92. check', 10), ('Привет, как дела? Погода хорошая.', 15),
    ('Это 3.14 градуса. А это 10.005, и ещё 123.456.', 20),
    ('这是一段很长的中文文本,而且没有句号,也没有感叹号,也没有问号,也没有换行符。', 50),
    ('hello‘world’ 🚀,,,,a.,,! Testing!', 12),
    ('foo\nbar  \tbaz', 4), ('5.7', 2), ('<3_f_14> 42.88', 7),
    ('my🤖 robot🌍 and 🚀 rocket', 16), ('🇷🇺🇨🇳😀🙉Hello', 5),
    ('проверка‘кавычек’', 5), ('中。文。！語。', 5), (',,,,,', 4),
    ('a b c d e f g h i', 7), ('end.', 3), ('...', 2),
    ('one, two, three, four.', 7), ('a..b..c', 4),
    ('0.0001 12.002 3.10000', 11),
]
random.seed(13)
words = ['слово', 'мир', '🐱', '🤖', 'Hello', '中文', '3.1415', '0.001', '!!!', ',', '’', '‘', '😀', '  ', '.', 'one', 'two', '\n', 'пять', '，', '。']
for _ in range(120):
    text = ''.join(random.choice(words) + random.choice([' ', '', ',', '.']) for _ in range(random.randrange(1, 40)))
    cases.append((text, random.randrange(1, 96)))

def hexs(text):
    return text.encode('utf-8').hex()

out = Path(__file__).with_name('text_split_original_python.tsv')
with out.open('w') as f:
    for raw, size in cases:
        parts = split_ns['split_text'](raw, size)
        f.write(str(size) + '\t' + hexs(raw) + '\t' + '|'.join(hexs(part) for part in parts) + '\n')
print(f'{len(cases)} original Python fixtures written: {out}')
