import unittest
import numpy as np

from team7_quant.module7 import RiskState, put_delta, update_risk_level, historical_es


class Module7Tests(unittest.TestCase):
    def test_put_delta_sign(self):
        d=put_delta(100,95,.2,30/365,.03,.02)
        self.assertLess(d,0); self.assertGreater(d,-1)

    def test_jump_and_release(self):
        cfg={"drawdown_thresholds":[.08,.12,.15],"release_thresholds":[.06,.10,.13],"release_days":5}
        s,reason=update_risk_level(RiskState(),.16,cfg)
        self.assertEqual(s.level,3)
        for _ in range(5): s,reason=update_risk_level(s,.12,cfg)
        self.assertEqual(s.level,2)

    def test_es_tail(self):
        x=np.arange(-252,0)/10000
        es,n=historical_es(x)
        self.assertEqual(n,13); self.assertGreater(es,0)


if __name__ == "__main__": unittest.main()
