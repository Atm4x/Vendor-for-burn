#!/usr/bin/env python3
"""Developer-only proof of the standalone CIS alphabet and voice permutation.

No executable code or tensor values from this script are used by Rust inference.
The emitted JSON is bound to the exact original JIT digest in verified_models.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.package import PackageImporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('jit', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    package = PackageImporter(str(args.package)).load_pickle('tts_models', 'model')
    part = package.packages[0]
    a = part.models[0].state_dict()
    b = torch.jit.load(str(args.jit), map_location='cpu').state_dict()
    pairs = [('tacotron', 'mel_encoder'), ('dur_predictor.dur_pred', 'dur_predictor.dur_pred'), ('pitch_predictor.pitch_pred', 'pitch_predictor.pitch_pred')]
    perm = None
    for x, y in pairs:
        assert torch.equal(a[x+'.embedding.weight'], b[y+'.embedding.weight']), x
        left, right = a[x+'.speaker_embedding.weight'], b[y+'.speaker_embedding.weight']
        assert left.shape == right.shape
        distances = (left[:, None, :] - right[None, :, :]).abs().amax(dim=2)
        values, mapping = distances.min(dim=1)
        assert values.max().item() == 0 and mapping.unique().numel() == len(left)
        assert (distances == 0).sum(dim=1).eq(1).all()
        if perm is None: perm = mapping
        else: assert torch.equal(perm, mapping)
    frontend = dict(input_symbols=part.symbols, symbol_ids=part.symbol_to_id,
                    alphabet=sorted(part.alphabet), sos=part.sos_token, eos=part.eos_token,
                    speakers={key: int(perm[value]) for key, value in part.speaker_to_ids[0].items()},
                    transliteration={lang: {k: v for k, v in table.items() if len(k)==1} for lang, table in (part.ext_alph or {}).items()})
    args.output.write_text(json.dumps(frontend, ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf8')
    for path in (args.package,args.jit):
        print(path.name, hashlib.file_digest(path.open('rb'),'sha256').hexdigest())
    print('Exact bijection across all three embeddings:', perm.tolist())

if __name__ == '__main__': main()
