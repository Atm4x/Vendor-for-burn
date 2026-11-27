#!/usr/bin/env python3
"""Exercise the shared IPC executable, including all verified original checkpoints when supplied."""
import argparse
import hashlib
import json
import subprocess
import tempfile
import uuid
import wave
from pathlib import Path


def run(binary, checkpoint, text=None, speaker=None, variant=None, sample_rate=8000):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder).resolve()
        if checkpoint is None:
            checkpoint = root / 'invalid.pt'
            checkpoint.write_bytes(b'not a model')
        with subprocess.Popen(
            [str(binary.resolve()), '--ipc-stdio', '--backend', 'flex',
             '--checkpoint', str(checkpoint.resolve()), '--input-dir', str(root),
             '--output-dir', str(root)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding='utf-8', cwd=root,
        ) as worker:
            def ask(command, payload=None, ok=True):
                request = {'version': 1, 'id': str(uuid.uuid4()), 'command': command,
                           'payload': {} if payload is None else payload}
                worker.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
                worker.stdin.flush()
                reply = json.loads(worker.stdout.readline())
                assert reply['version'] == 1 and reply['id'] == request['id'], reply
                assert reply['ok'] is ok, reply
                return reply['result']

            try:
                assert ask('capabilities')['service'] == 'tts'
                assert ask('status')['phase'] == 'started'
                ask('warmup', ok=False)
                if text is None:
                    ask('load', ok=False)
                    assert ask('status')['phase'] == 'started'
                else:
                    loaded = ask('load')
                    assert loaded['model_variant'] == variant
                    assert sample_rate in loaded['sample_rates']
                    assert speaker in loaded['speaker_names']
                    request = {'silero_version': 1, 'input': {'kind': 'text', 'text': text},
                               'speaker': speaker, 'sample_rate': sample_rate}
                    ask('infer', request, ok=False)
                    assert ask('warmup')['phase'] == 'ready'
                    outputs = [ask('infer', request) for _ in range(2)]
                    assert outputs[0]['audio']['file_name'] != outputs[1]['audio']['file_name']
                    if variant not in ('v1_tacotron2', 'v2_tacotron2'):
                        assert outputs[0]['audio']['sha256'] == outputs[1]['audio']['sha256']
                    for result in outputs:
                        desc = result['audio']
                        assert Path(desc['file_name']).name == desc['file_name']
                        path = root / desc['file_name']
                        data = path.read_bytes()
                        assert len(data) == desc['size_bytes']
                        assert hashlib.sha256(data).hexdigest() == desc['sha256']
                        with wave.open(str(path), 'rb') as wav:
                            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, sample_rate)
                            assert wav.getnframes() == result['sample_count'] > 0
                            assert any(wav.readframes(wav.getnframes()))
                    assert ask('unload')['phase'] == 'started'
                    print(json.dumps({'variant': variant, 'sample_rate': sample_rate,
                                      'sample_count': outputs[0]['sample_count'], 'compute_ms': [x['compute_ms'] for x in outputs]}), flush=True)
                ask('shutdown')
                assert worker.wait(timeout=10) == 0
            finally:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('--en', type=Path)
    parser.add_argument('--ru', type=Path)
    parser.add_argument('--matrix-dir', type=Path)
    args = parser.parse_args()
    run(args.binary, None)
    if args.matrix_dir:
        catalogue = json.loads((Path(__file__).resolve().parent.parent / 'verified_models.json').read_text(encoding='utf-8'))
        for model in catalogue:
            run(args.binary, args.matrix_dir / (model['model_id'] + '-renamed.pt'), **model['smoke'])
    if args.en:
        run(args.binary, args.en, 'Hello, world!', 'en_0', 'v3_hifigan')
    if args.ru:
        run(args.binary, args.ru, 'Привет.', 'aidar', 'v5_ru')


if __name__ == '__main__':
    main()
