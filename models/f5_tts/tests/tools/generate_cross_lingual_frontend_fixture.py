"""Create deterministic Python parity cases for the Cross-Lingual F5-TTS frontend."""

import random
import re
from pathlib import Path

import pyphen


HYPHENATOR = pyphen.Pyphen(lang="en_US")
PUNCTUATION = {
    ".": 1.5, "!": 1.5, "?": 1.5, ",": 0.7, ";": 1.0, ":": 1.0,
    "。": 1.3, "！": 1.3, "？": 1.3, "，": 0.6, "；": 0.8, "：": 0.8,
}


def count_units(text):
    units = 0.0
    for token in re.findall(r"[a-zA-Z']+|[\u4e00-\u9fff]", text):
        units += 1 if len(token) == 1 and "\u4e00" <= token <= "\u9fff" else len(
            HYPHENATOR.inserted(token.lower()).split("-")
        )
    units += sum(PUNCTUATION.get(char, 0.0) for char in text)
    return max(1, round(units))


def chunk_text(text, max_units):
    chunks = []
    current = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if current and count_units(candidate) > max_units:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def main():
    rng = random.Random(0xF52009)
    words = ["Hello", "world", "don't", "it's", "we'll", "transformer", "crosslingual", "synthesis", "I", "a", "yes", "no", "temperature", "characteristically", "orange", "one", "look", "happiness", "goodbye", "really"]
    cjk = ["你好", "中国", "重庆", "行", "不", "一", "世界", "测试", "汉字", "今天", "是"]
    punc = [".", "!", "?", ",", ";", ":", "。", "！", "？", "，", "；", "：", "", " ", "  ", "\n", "—", "…"]
    corpus = [
        "Hello world! This is a test.", "你好，世界。今天怎么样？", "Hello, 中国！ 你好 world?", "don't  it's we'll",
        "How are you? I'm fine. Thank you!", "    ", "...", "!!!", "你好，world. 再见!", "中国测试。OK!", "I ?", "No, no, no!",
        "  Hello!    World?  ", "a,b", "a, b", "你好！！世界。", "éclair 大家好", "hello 123 world", "", "中文\\nHello", "one、two?",
    ]
    for _ in range(2500):
        terms = []
        for _ in range(rng.randint(1, 19)):
            term = rng.choice(words) if rng.random() < .7 else rng.choice(cjk)
            terms.append(term + rng.choice(punc))
        corpus.append(" ".join(terms) if rng.random() < .7 else "".join(terms))
    output = []
    for index, text in enumerate(corpus):
        max_units = rng.randint(1, 95) if index % 11 else rng.randint(1, 8)
        output.append({"text":text,"units":count_units(text),"max_units":max_units,"chunks":chunk_text(text,max_units)})
    def encode(value):
        return value.encode("utf-8").hex()

    output_path = Path(__file__).resolve().parents[1] / "fixtures" / "cross_lingual_frontend_python_expected.tsv"
    output_path.write_text("".join(
        f"{encode(case['text'])}\t{case['units']}\t{case['max_units']}\t{len(case['chunks'])}\t"
        + "|".join(map(encode, case['chunks'])) + "\n"
        for case in output
    ), encoding="utf-8")
    print(f"{len(output)} Python reference cases -> {output_path}")


if __name__ == "__main__":
    main()
