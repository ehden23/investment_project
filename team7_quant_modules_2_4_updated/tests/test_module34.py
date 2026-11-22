import unittest
import pandas as pd

from team7_quant.module34 import RegimeState, adapt_weights_for_module2, classify_observed_regime, update_regime_state


class Module34Tests(unittest.TestCase):
    def test_four_regimes(self):
        self.assertEqual(classify_observed_regime(100.6,100.2),"확장")
        self.assertEqual(classify_observed_regime(100.6,101.0),"둔화")
        self.assertEqual(classify_observed_regime(99.2,99.7),"수축")
        self.assertEqual(classify_observed_regime(99.2,98.8),"회복")
        self.assertIsNone(classify_observed_regime(100,100))

    def test_two_confirmations(self):
        cfg={"confirmations":2}
        s=RegimeState(confirmed_regime="확장",last_processed_observation_month=pd.Timestamp("2015-12-31"))
        s,changed,_=update_regime_state(s,{"observation_month":pd.Timestamp("2016-01-31"),"observed_regime":"둔화"},cfg)
        self.assertFalse(changed); self.assertEqual(s.confirmation_count,1)
        s,changed,_=update_regime_state(s,{"observation_month":pd.Timestamp("2016-02-29"),"observed_regime":"둔화"},cfg)
        self.assertTrue(changed); self.assertEqual(s.confirmed_regime,"둔화")

    def test_weight_adapter_keeps_momentum_at_twenty_percent(self):
        out = adapt_weights_for_module2({"w_V": .6, "w_Q": .4}, {"factor_profile": "weighted_momentum", "fixed_momentum_weight": .2})
        self.assertAlmostEqual(out["w_V"], .48)
        self.assertAlmostEqual(out["w_Q"], .32)
        self.assertAlmostEqual(out["w_M"], .20)
        self.assertAlmostEqual(out["w_V"] + out["w_Q"] + out["w_M"], 1.0)


if __name__ == "__main__": unittest.main()

