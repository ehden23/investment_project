import unittest
import pandas as pd

from team7_quant.module6 import full_rebalance, mandatory_replacement, portfolio_drift


CFG={"buy_commission":.00015,"sell_commission":.00015,"sell_tax":.002,"slippage_each_side":.0014,
     "partial_fraction":.5,"drift_threshold":.05}


class Module6Tests(unittest.TestCase):
    def test_initial_build_golden(self):
        ids=[f"S{i}" for i in range(30)]
        out=full_rebalance(pd.Series(dtype=float),1.0,pd.Series(1/30,index=ids),None,CFG)
        self.assertAlmostEqual(out.final_values.sum(),.9984523987818882,12)
        self.assertAlmostEqual(out.costs["total"],.001547601218111816,12)

    def test_replacement_golden(self):
        current=pd.Series({"OLD":.1,"KEEP":.9}); target=pd.Series({"KEEP":.9,"NEW":.1})
        executable=pd.Series(True,index=["OLD","KEEP","NEW"])
        out=mandatory_replacement(current,0,{"KEEP","NEW"},target,executable,CFG)
        self.assertAlmostEqual(out.diagnostics["net_exit_proceeds"],.099645,12)
        self.assertAlmostEqual(out.diagnostics["replacement_buy"],.09949078927662124,12)
        self.assertAlmostEqual(out.final_values.sum()+out.final_cash,.9994907892766213,12)

    def test_strict_drift_boundary(self):
        self.assertAlmostEqual(portfolio_drift(pd.Series({"A":.95}),.05,pd.Series({"A":1.0})),.05)


if __name__ == "__main__": unittest.main()

