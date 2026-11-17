"""Check Stage131 routing, profiling output parsing and fail-closed diagnostics."""
import csv
import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).parent
SOURCE=ROOT.parents[1]/'models'/'tts-with-rvc-rs'/'src'


def load(name, file):
    spec=importlib.util.spec_from_file_location(name, ROOT/file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage131Contract(unittest.TestCase):
    def test_gate_parallel_is_separate_coalesced_route(self):
        rust=(SOURCE/'rmvpe_persistent.rs').read_text()
        self.assertIn('fn bigru_gate_parallel_kernel(',rust)
        self.assertIn('CubeDim::new_1d((hidden * 3) as u32)',rust)
        self.assertIn('if gate_parallel && (column_split != 1 || !weight_coalesced)',rust)
        self.assertIn('bigru_kernel::launch_unchecked(',rust)
        self.assertIn('bigru_split_kernel::launch_unchecked(',rust)
        self.assertIn('bigru_gate_parallel_kernel::launch_unchecked(',rust)
        self.assertIn('rmvpe_gru_gate_parallel', (SOURCE/'main.rs').read_text())

    def test_windows_runner_measures_all_five_and_checks_gate_pcm(self):
        ps=(ROOT/'Run-Stage131-GRU-AB.ps1').read_text()
        for fragment in ('RVC_RMVPE_GRU_GATE_PARALLEL', "@('control','coalesced','gate_parallel','split2','split4')", '--import', "'--page','raw','--csv'", "$mode -in @('coalesced','gate_parallel') -and -not $metrics.bit_exact", 'Analyze-Stage131-GRU-Nsight.py'):
            self.assertIn(fragment, ps)

    def test_ncu_rejects_wrong_cuda_kernel(self):
        analyzer=load('stage131_analyzer','Analyze-Stage131-GRU-Nsight.py')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'actual.ncu.csv'
            with path.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['Range','ActionID','KernelName','gpu__time_duration.sum']);w.writerow([0,1,'direct_conv2d_kernel',100.0])
            with self.assertRaises(ValueError):
                analyzer.inspect_ncu(path,'bigru_gate_parallel_kernel')
            with path.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['Range','ActionID','KernelName','gpu__time_duration.sum']);w.writerow([0,1,'bigru_gate_parallel_kernel_77',100.0])
            self.assertEqual(analyzer.inspect_ncu(path,'bigru_gate_parallel_kernel')['capture_count'],1)

    def test_full_wav_guard_rejects_short_audio(self):
        import array
        import wave
        probe=load('stage131_wav_probe','Validate-Stage131-WAV.py')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'short.wav'
            with wave.open(str(path),'wb') as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(array.array('h',[0]*16000).tobytes())
            with self.assertRaises(ValueError):
                probe.inspect(path)

    def test_nsys_requires_matching_block_and_warm_nvtx(self):
        analyzer=load('stage131_analyzer','Analyze-Stage131-GRU-Nsight.py')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.nsys.sqlite'
            db=sqlite3.connect(path)
            db.execute('CREATE TABLE StringIds(id INTEGER,value TEXT)')
            db.execute('INSERT INTO StringIds VALUES(1,?)',('bigru_gate_parallel_kernel_aaa',))
            db.execute('CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT)')
            db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?)',(0,1000,'stage102_rust_measured'))
            db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER,registersPerThread INTEGER,staticSharedMemory INTEGER,dynamicSharedMemory INTEGER,shortName INTEGER)')
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(10,900,2,2,1,768,1,1,33,4096,0,1))
            db.commit();db.close()
            self.assertEqual(analyzer.inspect_nsys(path,768,'bigru_gate_parallel_kernel')['warm_passes'][0]['gpu_kernel_sum_ms'],0.00089)
            with self.assertRaises(ValueError):analyzer.inspect_nsys(path,512,'bigru_gate_parallel_kernel')

if __name__=='__main__':unittest.main()
