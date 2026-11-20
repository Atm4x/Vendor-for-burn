# Chinese pronunciation assets

`han_pinyin_unihan.tsv` is a transformed Unicode Han character reading table derived from
`xpinyin-database.def` (Qing Lee, copyright 2012–2022), itself built from Unihan 14.0
readings (2021). The upstream source is available through the CTAN `xpinyin` package.

The upstream `xpinyin-database.def` is distributed under the **LaTeX Project Public
License 1.3c or later (LPPL)**. This transformed data file carries the same LPPL
terms; it is not relicensed under the Rust crate's MIT license.
This derivative is renamed, restricted to the F5-compatible U+3100–U+9FFF range,
and converts diacritical tone marks to ASCII TONE3 numerals. The reproducible
transformation is `tests/tools/build_pinyin_unihan_asset.py`, accepting the original
`xpinyin-database.def` as its input. No external Python modules are used by the
converter. The Rust runtime requires neither Python nor the original TeX files.

`han_pinyin_words.tsv` contains a small curated list of commonly encountered
polyphonic words and base readings, allowing context-specific pronunciations and
word-local sandhi. It is **not** equivalent to `rjieba` segmentation or `pypinyin`
with `tone_sandhi=True` on arbitrary Chinese text. Consumers may supply their own
word lexicon through `ChinesePinyin::extend_words` to extend coverage.
