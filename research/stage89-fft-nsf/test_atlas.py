"""Knapsack planner proof against exhaustive small cases."""
from itertools import product
import math
import unittest
from atlas import Layer,select_global,summarize
from spectral_conv import SpectralPlan

class AtlasTests(unittest.TestCase):
    def test_global_knapsack_matches_bruteforce(self):
        for cap in (2,4,8,16,24):
            ls=[]
            for i in range(5):
                plans=tuple(SpectralPlan(4+i,4+i,3 if i%2==0 else 7,1,500,64 * 2**e)
                            for e in (0,1,2))
                ls.append(Layer(str(i),float(plans[0].direct_flops),plans))
            selected=select_global(ls,cap,bucket_mib=2)
            observed = sum(layer.direct - (plan.spectral_flops if plan is not None else layer.direct)
                           for layer,plan in selected)
            best=-float('inf')
            for choices in product((-1,0,1,2),repeat=5):
                cost=0;saving=0.0
                for layer,index in zip(ls,choices):
                    if index>=0:
                        plan=layer.candidates[index]
                        if layer.direct-plan.spectral_flops<=0:break
                        cost+=math.ceil(plan.spectral_weight_bytes/(2*1024**2)) * 2
                        saving+=layer.direct-plan.spectral_flops
                else:
                    if cost<=cap:best=max(best,saving)
            self.assertAlmostEqual(observed,best,places=4)

    def test_large_plan_real_budget(self):
        p=SpectralPlan(256,256,11,5,3840,512)
        self.assertGreater(p.ideal_flop_ratio,1)
        self.assertEqual(p.effective_kernel,51)
        self.assertEqual(p.valid_frames,462)
        self.assertEqual(p.bins,257)
        self.assertLessEqual(p.spectral_weight_bytes,140*1024**2)

if __name__=='__main__':unittest.main()
