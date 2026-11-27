#!/usr/bin/env python3
"""Capture the actual inference-time prenet dropout masks and original TorchScript outputs."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.package import PackageImporter

CASES = {
    **{f'{name}_v2': [('Привет, мир!', 'default', 0)] for name in ('aidar','baya','irina','kseniya','natasha','ruslan')},
    'thorsten_v2': [('Guten Tag!', 'default', 0)],
    'tux_v2': [('Buenos días.', 'default', 0)],
    'gilles_v2': [('Bonjour.', 'default', 0)],
    'aigul_v2': [('Һаумы.', 'default', 0)],
    'erdni_v2': [('Мендвт.', 'default', 0)],
    'dilyara_v2': [('Исәнмесез.', 'default', 0)],
    'dilnavoz_v2': [('Salom.', 'default', 0)],
    'lj_v2': [('Hello, world!', 'default', 0)],
    'multi_v2': [('Hello, world!', 'lj', 9), ('Guten Tag!', 'thorsten', 6), ('Прив+ет.', 'aidar', 0)],
    'mykyta_v2': [('Привіт, світе!', 'default', 0)],
}


def create(path, output):
    p = PackageImporter(str(path)).load_pickle('tts_models', 'model')
    m = p.model.eval()
    multi = hasattr(p, 'speaker_to_id')
    rates = [8000, 24000, 48000] if hasattr(m.vocoder, 'ups_u') else [8000, 16000]
    cases = []
    tensors = {}
    for index, (text, speaker, sid) in enumerate(CASES[path.stem]):
        torch.manual_seed(1847 + index)
        prepared = p.prepare_text_input(text, p.symbols)
        if hasattr(m, 'accentor'):
            sentence, clean = prepared
            graphemes, lengths, boundaries = m.preprocess_batch_accentor([clean])
            predictions = m.get_accentor_preds(graphemes, lengths)
            accented = m.postprocess_batch_accentor([sentence], *predictions, boundaries)[0]
            ids = m.preprocess_tacotron(accented).unsqueeze(0)
        else:
            ids = prepared.unsqueeze(0)
        tac = m.tacotron
        x = tac.embedding(ids)
        if multi:
            x = torch.cat((x, tac.speaker_embedding(torch.tensor([sid])).unsqueeze(1).repeat(1, ids.shape[1], 1)), 2)
        memory = tac.encoder(x.transpose(1, 2))
        decoder = tac.decoder
        decoder.initialize_decoder_states(memory)
        inp = decoder.get_go_frame(memory)
        masks = []
        frames = []
        for _ in range(decoder.max_decoder_steps):
            state = torch.random.get_rng_state()
            pre = decoder.prenet(inp)
            after = torch.random.get_rng_state()
            torch.random.set_rng_state(state)
            for _ in range(2):
                masks.append(torch.nn.functional.dropout(torch.ones(1, 256), 0.5, True))
            assert torch.equal(torch.random.get_rng_state(), after)
            mel, gate, _ = decoder.decode(pre)
            if torch.sigmoid(gate).item() > decoder.gate_threshold:
                break
            frames.append(mel)
            inp = mel
        assert frames
        mel = torch.cat(frames, 0).transpose(0, 1).unsqueeze(0)
        mel = mel + tac.postnet(mel)
        key = f'case{index}'
        tensors[f'{key}_memory'] = memory.squeeze(0).contiguous()
        tensors[f'{key}_masks'] = torch.cat(masks, 0).contiguous()
        tensors[f'{key}_mel'] = mel.contiguous()
        for rate in rates:
            wav = m.vocoder(mel, rate)[:, :, :mel.shape[-1] * (rate // 80)]
            tensors[f'{key}_pcm{rate}'] = wav.reshape(-1).contiguous()
        cases.append(dict(key=key, text=text, speaker=speaker, speaker_id=sid, ids=ids.flatten().tolist(), rates=rates))
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / (path.stem + '.safetensors')))
    (output / (path.stem + '.json')).write_text(json.dumps(cases, ensure_ascii=False), encoding='utf8')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoints', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--models', nargs='+', choices=CASES, default=list(CASES))
    args = cli.parse_args()
    torch.set_num_threads(4)
    with torch.inference_mode():
        for name in args.models:
            create(args.checkpoints / (name + '.pt'), args.output)
