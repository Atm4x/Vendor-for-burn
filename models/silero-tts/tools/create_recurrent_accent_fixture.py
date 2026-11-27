#!/usr/bin/env python3
"""Developer-only reference for original quantized V2/V3 accentors."""
import argparse
import json
from pathlib import Path
import torch
from torch.package import PackageImporter

TEXTS = [
    'Привет, мир!', 'Съешь ещё этих мягких французских булок.',
    'Я люблю русский язык.', 'На двери замок.', 'Всё прив+ычно.',
    'Периодов много.', 'Кто-то идёт домой.', 'Привет. Ты дома?', 'а', 'Прив+ет, всё!',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    for name in ['aidar_v2', 'ru_v3']:
        p = PackageImporter(str(args.checkpoints / (name+'.pt'))).load_pickle('tts_models','model')
        model = p.model.eval()
        cases = []
        for text in TEXTS:
            if name == 'aidar_v2':
                sentence, clean = p.prepare_text_input(text, p.symbols)
            else:
                sentence, clean, _ = p.prepare_text_input(text)
            with torch.inference_mode():
                g, lengths, boundaries = model.preprocess_batch_accentor([clean])
                stress, yo = model.accentor(g, lengths)
                predictions = model.get_accentor_preds(g, lengths)
                args_ = ([sentence], *predictions, boundaries)
                accented = model.postprocess_batch_accentor(*args_, True)[0] if name == 'ru_v3' else model.postprocess_batch_accentor(*args_)[0]
            cases.append(dict(text=text, sentence=sentence, words=clean.split(), accented=accented,
                              stress=stress.tolist(), yo=yo.tolist()))
        (args.output / (name+'.json')).write_text(json.dumps(cases, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
