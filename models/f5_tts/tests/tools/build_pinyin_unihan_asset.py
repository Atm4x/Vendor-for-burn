"""Build the offline F5-TTS Chinese reading table from the LPPL xpinyin database.

The source is xpinyin-database.def, Qing Lee, LPPL 1.3c+ (2022).
This build-time tool is not needed by the Rust runtime.
"""

import argparse
import re
import unicodedata
from pathlib import Path


TONE_MARKS = {"\u0304": "1", "\u0301": "2", "\u030c": "3", "\u0300": "4"}
RECORD = re.compile(r"\\XPYU\{(.+?)\}\{(\d+)\}\{(.+?)\}")


def tone3(syllable: str) -> str:
    decomposed = unicodedata.normalize("NFD", syllable)
    tone = ""
    result = []
    for ch in decomposed:
        if ch in TONE_MARKS:
            if tone and tone != TONE_MARKS[ch]:
                raise ValueError(f"Conflicting tones in {syllable!r}")
            tone = TONE_MARKS[ch]
        elif ch == "\u0308":
            if not result or result[-1] != "u":
                raise ValueError(f"Unexpected dieresis in {syllable!r}")
            result[-1] = "v"
        elif not unicodedata.combining(ch):
            result.append(ch.lower())
        else:
            raise ValueError(f"Unrecognized combining mark in {syllable!r}")
    rendered = "".join(result).replace("ü", "v") + tone
    if not rendered.isascii() or not rendered.replace("v", "a").rstrip("12345").isalpha():
        raise ValueError(f"Invalid pinyin {syllable!r}: {rendered!r}")
    return rendered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    entries = {}
    for line in args.source.read_text(encoding="utf8").splitlines():
        match = RECORD.search(line)
        if match is None:
            continue
        han, codepoint, reading = match.groups()
        if len(han) != 1 or ord(han) != int(codepoint):
            raise ValueError(f"Invalid character mapping: {match.group(0)}")
        if not 0x3100 <= ord(han) <= 0x9FFF:
            continue
        if han in entries:
            raise ValueError(f"Duplicate character: {han}")
        entries[han] = tone3(reading)
    args.destination.write_text(
        "".join(f"{han}\t{reading}\n" for han, reading in sorted(entries.items())) ,
        encoding="utf-8",
    )
    print(f"Generated {len(entries)} Unicode codepoints into {args.destination}")


if __name__ == "__main__":
    main()
