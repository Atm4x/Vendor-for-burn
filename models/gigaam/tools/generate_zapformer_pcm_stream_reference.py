"""Generate end-to-end PCM packet references using the original GigaAMStreaming."""
import argparse
import dataclasses
import json
import sys
import types
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[1] / 'tests/fixtures/zapformer')
    args = parser.parse_args()
    package = types.ModuleType('gigaam')
    package.__path__ = [str(args.upstream.resolve() / 'gigaam')]
    sys.modules['gigaam'] = package
    from gigaam.encoder import ZapformerEncoder
    from gigaam.language import LanguageEmbedding, LanguageFiLM, LanguageLateFusion
    from gigaam.decoder import RNNTHead, CTCHead
    from gigaam.decoding import RNNTGreedyDecoding, CTCGreedyDecoding
    from gigaam.preprocess import FeatureExtractor
    from gigaam.streaming import GigaAMStreaming

    class Model(torch.nn.Module):
        @property
        def _device(self):
            return next(self.parameters()).device

        def encode_language(self, tag, batch):
            return self.language_embedding(tag, batch_size=batch)

    torch.set_num_threads(1)
    packets = [9, 311, 1600, 640, 2700, 17, 4223, 5200, 7800, 3300]
    # Exact representable PCM, identical arithmetic in Rust and Python.
    pcm = ((torch.arange(sum(packets)) * 1297 % 4096 - 2048).float() / 16384)
    cases = []
    for kind in ('ctc','rnnt'):
        source = torch.load(args.fixtures / f'dual_{kind}.ckpt', weights_only=False)
        cfg = source['cfg']
        cfg.preprocessor.sample_rate = 16000
        cfg.streaming = dict(contexts=[dict(future_context_ms=ms,left_context=8) for ms in (160,320,640)])
        model = Model()
        model.cfg = cfg
        model.preprocessor = FeatureExtractor(**cfg.preprocessor)
        model.encoder = ZapformerEncoder(**{k:v for k,v in cfg.encoder.items() if k!='_target_'})
        model.encoder.frame_shift_ms = 40
        language = cfg.language_conditioning
        model.language_embedding = LanguageEmbedding(language)
        model.encoder.language_film = LanguageFiLM(model.encoder.layer_dims,4,language.film)
        model.language_fusion = LanguageLateFusion(8,4,language.late_fusion)
        model.heads = torch.nn.ModuleDict()
        model.decodings = {}
        for name in ('raw','e2e'):
            h,d = cfg.heads[name],cfg.decodings[name]
            if kind=='rnnt':
                model.heads[name]=RNNTHead(h.decoder,h.joint)
                model.decodings[name]=RNNTGreedyDecoding(d.vocabulary,max_symbols_per_step=d.max_symbols_per_step)
            else:
                model.heads[name]=CTCHead(h.feat_in,h.num_classes)
                model.decodings[name]=CTCGreedyDecoding(d.vocabulary)
        model.load_state_dict(source['state_dict'],strict=False)
        model.eval()
        for ms in (160,320,640):
            for timestamps in (False,True):
                stream = GigaAMStreaming(model,ms,language='ru-en',word_timestamps=timestamps)
                pos = 0
                result = []
                offsets = dict(raw=0,e2e=0)
                for count in packets+[0]:
                    output = stream.transcribe_partial(pcm[pos:pos+count],final=count==0)
                    pos += count
                    packet = {}
                    for name,item in output.items():
                        state = stream.decoder_states[name]
                        begin = offsets[name]
                        packet[name] = dict(text=item.text,words=[dataclasses.asdict(w) for w in item.words] if item.words is not None else None,token_ids=state.tokens[begin:],frames=state.frames[begin:])
                        offsets[name]=len(state.tokens)
                    result.append(packet)
                cases.append(dict(kind=kind,ms=ms,timestamps=timestamps,packets=result))
    (args.fixtures/'pcm_stream_reference.json').write_text(json.dumps(dict(packets=packets,cases=cases),separators=(',',':'))+'\n')


if __name__=='__main__':
    main()
