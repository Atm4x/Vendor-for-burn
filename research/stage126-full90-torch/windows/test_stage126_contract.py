"""Stage126 route and PCM acceptance contract tests."""
import array
import importlib.util
import pathlib
import tempfile
import unittest
import wave
p=pathlib.Path(__file__).with_name("Run-Stage126-Full90-Compare.py")
spec=importlib.util.spec_from_file_location("stage126_driver",p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
class Stage126Tests(unittest.TestCase):
    def test_modes(self):
        for route in ("control","linear","residue","combined","legacy"):
            e=m.native_env(route)
            self.assertEqual(e["RVC_NSF_RESBLOCK_TRANSPORT"],"1")
            self.assertEqual(e["RVC_NSF_RESBLOCK_TEMPORAL_TILE"],"4")
            self.assertNotIn("RVC_NSF_IMPLICIT_GEMM",e)
            self.assertNotIn("RVC_NSF_RESBLOCK_PLANE_BROADCAST",e)
            self.assertNotIn("RVC_NSF_RESBLOCK_WEIGHT_PREFETCH",e)
            self.assertEqual("RVC_NSF_RESBLOCK_ACTIVATION_PLACEMENT" in e,route!="legacy")
            self.assertEqual("RVC_NSF_RESBLOCK_LINEAR_INTERIOR" in e,route in ("linear","combined"))
            self.assertEqual("RVC_NSF_RESBLOCK_RESIDUE_ORDER" in e,route in ("residue","combined"))
    def test_pcm(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=(pathlib.Path(d)/n for n in ("a.wav","b.wav"))
            for path,data in ((a,[100,-200,300]),(b,[100,-201,300])):
                with wave.open(str(path),"wb") as w:
                    w.setnchannels(1);w.setsampwidth(2);w.setframerate(40000);w.writeframes(array.array("h",data).tobytes())
            r=m.pcm(a,b)
            self.assertTrue(r["compatible"])
            self.assertEqual(r["different_samples"],1)
            self.assertEqual(r["max_abs_pcm_lsb"],1)
            self.assertEqual(m.pcm(a,a)["different_samples"],0)
    def test_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);source=root/"data";source.mkdir()
            (source/"summary.json").write_text("{}")
            (source/"weight.pth").write_bytes(b"private")
            (source/"audio.wav").write_bytes(b"private")
            dest=root/"evidence.zip";m.archive(source,dest)
            import zipfile
            with zipfile.ZipFile(dest) as z:self.assertEqual(z.namelist(),["summary.json"])
if __name__=="__main__":unittest.main()
