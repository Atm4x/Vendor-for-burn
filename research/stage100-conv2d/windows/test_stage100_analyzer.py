import importlib.util
import json
import tempfile
import wave
from array import array
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / 'windows' / 'Analyze-Stage100-Long-CUDA.py'
spec = importlib.util.spec_from_file_location('stage100_analysis', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

CASES = [
    ('reference_direct', 'direct'),
    ('candidate_direct', 'direct'),
    ('fused_direct', 'direct'),
    ('fused_spatial', 'spatial_quad'),
    ('fused_tf32x2_batched', 'compensated_tf32x2_batched'),
    ('fused_tf32x2_batched', 'compensated_tf32x2_batched'),
    ('fused_spatial', 'spatial_quad'),
    ('fused_direct', 'direct'),
    ('candidate_direct', 'direct'),
    ('reference_direct', 'direct'),
]

def write_wav(path: Path, values):
    a = array('h', values)
    if __import__('sys').byteorder != 'little': a.byteswap()
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(40000)
        f.writeframes(a.tobytes())

def fixture(root):
    (root/'manifest.json').write_text(json.dumps({
        'order':[x[0] for x in CASES], 'runs_per_process':3,
        'tensor_batch_size':2,'core_frames':900,'input_duration_seconds':90.0,
        'torch_fp16_reference_ms':2500.0,
    }), encoding='utf-8')
    for idx,(name,mode) in enumerate(CASES,1):
        p=root/f'{idx:02}_{name}.wav'
        write_wav(p,[10,20,30,40] if mode == 'direct' else [10,20,31,40])
        (root/f'{idx:02}_{name}.wav.benchmark.json').write_text(json.dumps({
            'backend':'cuda','tensor_batch_size':2,'core_frames':900,
            'iterations_exact_f32':True,'inference_ms':[900,700,750],
            'rmvpe_unet_conv_execution':mode,
            'rmvpe_unet_bn_folded': name.startswith('fused_'),
            'planned_cut_frames':[10,20],'chunk_counts':[3,3,3],
        }))

def run():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);fixture(root)
        result=module.analyze(root)
        assert len(result['modes'])==5
        assert result['modes']['fused_spatial']['median_ms']==725
        assert result['modes']['fused_spatial']['pcm_vs_direct']['max_abs_pcm16']==1
        module.write_report(result,root/'analysis')
        assert 'long' in (root/'analysis'/'comparison.md').read_text().lower()
        print('PASS normal-output-and-math')
        p=root/'04_fused_spatial.wav.benchmark.json'
        obj=json.loads(p.read_text());obj['rmvpe_unet_conv_execution']='direct';p.write_text(json.dumps(obj))
        try: module.analyze(root)
        except ValueError as e: assert 'ignored' in str(e)
        else: raise AssertionError('did not fail ignored setting')
        print('PASS ignore-guard')
        obj['rmvpe_unet_conv_execution']='spatial_quad';obj['planned_cut_frames']=[10,21]
        p.write_text(json.dumps(obj))
        try: module.analyze(root)
        except ValueError as e: assert 'chunk' in str(e)
        else: raise AssertionError('did not fail cut drift')
        print('PASS boundaries-guard')
        obj['planned_cut_frames']=[10,20];obj['rmvpe_unet_bn_folded']=False;p.write_text(json.dumps(obj))
        try: module.analyze(root)
        except ValueError as e: assert 'BatchNorm' in str(e)
        else: raise AssertionError('did not reject missing BN folding')
        print('PASS BN-fold-guard')
        obj['rmvpe_unet_bn_folded']=True;p.write_text(json.dumps(obj))
        obj['planned_cut_frames']=[10,20];p.write_text(json.dumps(obj))
        m=root/'manifest.json';manifest=json.loads(m.read_text());manifest['input_duration_seconds']=11.9
        m.write_text(json.dumps(manifest))
        try: module.analyze(root)
        except ValueError as e: assert 'long-WAV' in str(e)
        else: raise AssertionError('did not reject short input')
        print('PASS long-duration-guard')

if __name__=='__main__':run()
