"""Generate small native PyTorch checkpoint and exact dual-head inference references."""
import argparse
import json
import sys
import types
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'tests/fixtures/zapformer')
    args = parser.parse_args()
    package = types.ModuleType('gigaam')
    package.__path__ = [str(args.upstream.resolve() / 'gigaam')]
    sys.modules['gigaam'] = package
    from gigaam.encoder import ZapformerEncoder
    from gigaam.language import LanguageEmbedding, LanguageFiLM, LanguageLateFusion
    from gigaam.decoder import RNNTHead, CTCHead
    from gigaam.decoding import RNNTGreedyDecoding, CTCGreedyDecoding

    torch.set_num_threads(1)
    torch.manual_seed(65)
    cases = json.loads((args.output / 'cases.json').read_text())
    config = cases[0]['config'].copy()
    kwargs = dict(config)
    del kwargs['_target_']
    language = dict(languages=['ru','en','kk'], num_slots=4, embedding_dim=4,
                    film=dict(enabled=True, placement='all'),
                    late_fusion=dict(enabled=True, hidden=6))
    model = torch.nn.Module()
    model.encoder = ZapformerEncoder(**kwargs)
    model.language_embedding = LanguageEmbedding(language)
    model.encoder.language_film = LanguageFiLM(model.encoder.layer_dims,4,language['film'])
    model.language_fusion = LanguageLateFusion(8,4,language['late_fusion'])
    vocabs = {'raw':['a','b','c'], 'e2e':['A','B','C','D']}
    heads = {}
    decodings = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'language_' in name:
                parameter.uniform_(-0.15,0.15)
        x=torch.randn(3,35,8)
        lengths=torch.tensor([35,29,0])
        tags=['ru-en',{'kz':2.0,'en':1.0},'auto']
        model.eval()
        emb=model.language_embedding(tags,batch_size=3)
        encoded,lens=model.encoder(x.transpose(1,2),lengths,lang_embedding=emb)
        encoded=model.language_fusion(encoded,emb)
        for kind in ('rnnt','ctc'):
            for name,vocab in vocabs.items():
                classes=len(vocab)+1
                if kind=='rnnt':
                    decoder=dict(pred_hidden=5 if name=='raw' else 7,pred_rnn_layers=2,num_classes=classes)
                    joint=dict(enc_hidden=8,pred_hidden=decoder['pred_hidden'],joint_hidden=6,num_classes=classes)
                    head=RNNTHead(decoder,joint)
                    # Ensure at least one token and one blank in the fixture; don't fix the entire output.
                    if name=='raw':
                        head.joint.joint_net[1].bias[-1]-=0.4
                    else:
                        head.joint.joint_net[1].bias[-1]+=0.1
                    heads[name]=dict(_target_='gigaam.decoder.RNNTHead',decoder=decoder,joint=joint)
                    decodings[name]=dict(_target_='gigaam.decoding.RNNTGreedyDecoding',vocabulary=vocab,max_symbols_per_step=3)
                else:
                    head=CTCHead(8,classes)
                    heads[name]=dict(_target_='gigaam.decoder.CTCHead',feat_in=8,num_classes=classes)
                    decodings[name]=dict(_target_='gigaam.decoding.CTCGreedyDecoding',vocabulary=vocab)
                if name=='raw': model.heads=torch.nn.ModuleDict()
                model.heads[name]=head.eval()
            results={}
            for name,vocab in vocabs.items():
                decoding=RNNTGreedyDecoding(vocab,max_symbols_per_step=3) if kind=='rnnt' else CTCGreedyDecoding(vocab)
                results[name]=decoding.decode(model.heads[name],encoded,lens)
            cfg=dict(encoder=config,preprocessor=dict(features=8,n_fft=320,win_length=320,center=False),heads=heads,decodings=decodings,language_conditioning=language)
            # Save aliases once only; native loading reads the canonical rotary table.
            state={k:v for k,v in model.state_dict().items() if '.rope_bank.' not in k or k.startswith('encoder.encoder.rope_bank.')}
            torch.save(dict(cfg=OmegaConf.create(cfg),state_dict=state),args.output / f'dual_{kind}.ckpt')
            (args.output / f'dual_{kind}.json').write_text(json.dumps(results)+'\n')
        reference=dict(input=x.flatten().tolist(),encoded=encoded.transpose(1,2).contiguous().flatten().tolist(),lengths=lengths.tolist(),output_lengths=lens.tolist(),shape=list(encoded.transpose(1,2).shape),tags=tags)
        (args.output / 'dual_input.json').write_text(json.dumps(reference)+'\n')


if __name__=='__main__':
    main()
