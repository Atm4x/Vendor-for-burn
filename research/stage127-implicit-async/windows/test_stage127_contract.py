import importlib.util
from pathlib import Path
import types
import unittest

p = Path(__file__).with_name('Run-Stage127-Full90-Profile.py')
spec = importlib.util.spec_from_file_location('stage127', p)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class RouteContract(unittest.TestCase):
    def test_sync_and_async_always_change_only_conv1d_loader(self):
        sync = suite.native_env('implicit_sync')
        async_ = suite.native_env('implicit_async')
        diff = {key for key in set(sync) | set(async_) if sync.get(key) != async_.get(key)}
        self.assertEqual(diff, {'RVC_NSF_CONV1D_GEMM_ASYNC'})
        self.assertEqual(sync['RVC_NSF_IMPLICIT_GEMM'], '1')
        self.assertEqual(async_['RVC_NSF_CONV1D_GEMM_ASYNC'], '1')

    def test_ntc_and_hybrid_flags(self):
        ntc = suite.native_env('ntc_combined')
        hybrid = suite.native_env('hybrid_sync')
        self.assertEqual(ntc['RVC_NSF_RESBLOCK_TRANSPORT'], '1')
        self.assertEqual(hybrid['RVC_NSF_RESBLOCK_TRANSPORT'], '1')
        self.assertEqual(ntc['RVC_NSF_RESBLOCK_SHARED_NTC'], '1')
        self.assertEqual(hybrid['RVC_NSF_RESBLOCK_SHARED_NTC'], '1')
        self.assertEqual(ntc.get('RVC_NSF_TRANSPOSE_GEMM'), '0')
        self.assertEqual(hybrid['RVC_NSF_TRANSPOSE_GEMM'], '1')
        self.assertNotIn('RVC_NSF_IMPLICIT_GEMM', hybrid)

    def test_fp16_island_does_not_enable_ntc(self):
        mode = suite.native_env('implicit_fp16')
        self.assertEqual(mode['RVC_NSF_RESBLOCK_FP16'], '1')
        self.assertNotIn('RVC_NSF_RESBLOCK_TRANSPORT', mode)

    def test_nsys_kernel_selection_uses_real_summary(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'implicit_sync.nsys.summary_cuda_gpu_kern_sum.csv'
            source.write_text('Total Time (ns),Name\n1000,other_kernel\n9000,simple_conv_kernel_a\n3000,gemm_kernel_b\n')
            name,path,actual=suite.nsys_target_name(Path(tmp),'implicit_sync')
            self.assertEqual(actual,'simple_conv_kernel_a')
            self.assertEqual(name,'simple_conv_kernel_a')
            self.assertEqual(path,str(source))

    def test_requested_mode_audit(self):
        args = types.SimpleNamespace(runs=3, core_frames=300, batch_size=2)
        for mode in suite.MODES:
            env = suite.native_env(mode)
            transport = mode.startswith('hybrid') or mode == 'ntc_combined'
            report = dict(backend='cuda',graph_replay=False,
                nsf_conv_execution='direct' if mode=='ntc_combined' else 'mixed' if mode.startswith('hybrid') else 'implicit_gemm',
                nsf_conv1d_gemm_loading='async_cyclic' if mode.endswith('async') else 'sync_cyclic',
                nsf_execution='sequential',rmvpe_gru_execution='persistent',
                core_frames=300,tensor_batch_size=2,iterations_exact_f32=True,
                inference_ms=[100,90,90],chunk_counts=[30,30,30],source_seconds=90,
                nsf_resblock_precision='fp16_island_fp32_stage' if mode=='implicit_fp16' else 'fp32',
                nsf_transpose_execution='gemm' if mode.startswith(('implicit','hybrid')) else 'direct',
                nsf_resblock_benchmark_requested=dict(transport=transport,shared_ntc=transport,
                    temporal_tile='4' if transport else None,activation_placement=transport,
                    linear_interior=mode=='ntc_combined',residue_order=mode=='ntc_combined',
                    plane_broadcast=False,weight_prefetch=False))
            suite.validate_native(report,mode,args,90.0)
            report['nsf_conv1d_gemm_loading'] = 'wrong'
            with self.assertRaises(ValueError):suite.validate_native(report,mode,args,90.0)


if __name__ == '__main__':unittest.main()
