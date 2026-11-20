"""Validate the low-storage CUDA profiler contracts and fail-closed evidence parser."""

import csv
from contextlib import closing
import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parents[1] / "models" / "tts-with-rvc-rs" / "src"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage132Contracts(unittest.TestCase):
    def test_gru_inference_warm_nvtx_is_synchronized(self):
        src = (SRC / "main.rs").read_text()
        segment = src.split('if command == "infer-wav-segment"')[1].split('if command == "infer-wav-full-diagnostic"')[0]
        self.assertIn('Range::new("stage102_rust_measured")', segment)
        self.assertLess(segment.index('Range::new("stage102_rust_measured")'), segment.index('converted = inferencer.infer_16khz'))
        self.assertLess(segment.index('post-inference synchronization failed'), segment.index('drop(_nvtx)'))

    def test_full90_measurement_and_short_nsight_scope(self):
        ps = (ROOT / "Run-Stage132-GRU-AB.ps1").read_text()
        self.assertIn("@('control','coalesced','gate_parallel')", ps)
        self.assertIn("ArgsFor $wav $Runs $false", ps)
        self.assertIn("ArgsFor $wav 2 $true", ps)
        self.assertIn("ArgsFor \"$stem.wav\" 2 $true", ps)
        self.assertIn("Export-Stage132-NCU-Metrics.py", ps)
        self.assertIn("ZipFileExtensions]::CreateEntryFromFile", ps)
        self.assertIn("AvailableFreeSpace -lt 2GB", ps)
        self.assertNotIn("'--page','raw'", ps)

    def test_ncu_numeric_export_and_wrong_kernel_gate(self):
        analyzer = load("stage132_analyzer", "Analyze-Stage132-GRU-Nsight.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Range", "ActionID", "KernelName", "gpu__time_duration.avg", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"])
                writer.writerow([0, 0, "bigru_gate_parallel_kernel_abc", 4780928, 15790272])
            report = analyzer.inspect_ncu(path, "bigru_gate_parallel_kernel")
            self.assertEqual(report["capture_count"], 1)
            self.assertEqual(report["metrics"][0]["gpu__time_duration.avg"], "4780928")
            with self.assertRaises(ValueError):
                analyzer.inspect_ncu(path, "bigru_kernel")

    def test_nsys_short_segment_with_and_without_strings(self):
        analyzer = load("stage132_analyzer", "Analyze-Stage132-GRU-Nsight.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            with closing(sqlite3.connect(path)) as db:
                db.execute("CREATE TABLE StringIds(id INTEGER,value TEXT)")
                db.execute("INSERT INTO StringIds VALUES(1,'bigru_gate_parallel_kernel_a')")
                db.execute("CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT)")
                db.execute("INSERT INTO NVTX_EVENTS VALUES(0,10000000,'stage102_rust_measured')")
                db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER,registersPerThread INTEGER,staticSharedMemory INTEGER,dynamicSharedMemory INTEGER,shortName INTEGER)")
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,1000100,2,2,1,768,1,1,40,4096,0,1)")
                db.commit()
            result = analyzer.inspect_nsys(path, 768, "bigru_gate_parallel_kernel")
            self.assertEqual(result["warm_passes"][0]["kernel_count"], 1)
            with closing(sqlite3.connect(path)) as db:
                db.execute("DELETE FROM StringIds")
                db.commit()
            result = analyzer.inspect_nsys(path, 768, "bigru_gate_parallel_kernel")
            self.assertIn("geometry only", result["kernel_identity"])
            with self.assertRaises(ValueError):
                analyzer.inspect_nsys(path, 256, "bigru_kernel")
            with closing(sqlite3.connect(path)) as db:
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(200,2000200,2,2,1,768,1,1,40,4096,0,1)")
                db.commit()
            with self.assertRaisesRegex(ValueError, "Expected one"):
                analyzer.inspect_nsys(path, 768, "bigru_gate_parallel_kernel")


if __name__ == "__main__":
    unittest.main()
