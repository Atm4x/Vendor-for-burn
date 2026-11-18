"""Capture the actual Silero V5 SSML segments, merged token IDs and modifiers."""

import argparse
import json
import warnings
from pathlib import Path

import torch
from torch.package import PackageImporter

SAMPLES = [
    '<speak>Привет, <break time="350ms"/> мир. <prosody rate="slow" pitch="high">Тест</prosody> конец</speak>',
    '<speak><prosody rate="90%" pitch="+15%">Сначала <prosody rate="fast">потом</prosody> снова</prosody>.</speak>',
    '<speak><s>Это первое.</s><s>А это второе.</s></speak>',
    '<speak><p>Первая часть.</p><p>Вторая часть!</p></speak>',
    '<speak>Начало<break strength="weak"/> продолжение.</speak>',
    '<speak><break time="1s"/>Привет.</speak>',
    '<speak>Первый. <break strength="x_weak"/> Второй.</speak>',
    '<speak>Тест <unsupported>текст</unsupported> после.</speak>',
    '<speak>Текст <prosody rate="x-fast" pitch="robot">и ещё.</prosody> Конец.</speak>',
    '<speak>Один <prosody pitch="-20%">два</prosody> три <break time="6000ms"/> четыре.</speak>',
    '<speak>Строка<break strength="strong"/></speak>',
    '<speak>Здесь <prosody rate="fast">быстро, <break time="150ms"/> теперь дальше</prosody> конец.</speak>',
    '<speak>Да<break time="75ms"/>.</speak>',
]


def run(checkpoint: Path, destination: Path):
    torch.set_grad_enabled(False)
    package = PackageImporter(str(checkpoint)).load_pickle('tts_models', 'model').packages[0]
    package.unpack_q_model()
    result = []
    for xml in SAMPLES:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prepared = package.process_ssml(xml)
            sentences = [x['text'] for x in prepared]
            accented = package.accentor(' '.join(sentences))
            sequence, pauses, rates, pitches = package.merge_batch_model(
                sentences,
                [x['break_time'] for x in prepared],
                [x['prosody_rate'] for x in prepared],
                [x['prosody_pitch'] for x in prepared],
            )
        result.append({
            'ssml': xml,
            'segments': prepared,
            'accented': accented,
            'sequence': sequence.squeeze().tolist(),
            'pauses': {str(k): v for k, v in pauses.items()},
            'rates': rates.squeeze().tolist(),
            'pitches': pitches.squeeze().tolist(),
        })
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'ssml-parity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SSML reference cases:', len(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
