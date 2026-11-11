import unittest
import numpy as np
import pandas as pd

from team7_quant.module2 import percentile_rank, select_stocks


class Module2Tests(unittest.TestCase):
    def setUp(self):
        self.codes = [f"{i:06d}" for i in range(1, 32)]
        dates = pd.to_datetime(["2015-03-31", "2015-12-31", "2016-03-31", "2016-06-30"])
        self.prices = pd.DataFrame([(d,c,100+int(c),1_000_000+int(c)*1000) for d in dates for c in self.codes],
                                   columns=["date","stock_code","adj_close","market_cap"])
        rows=[]
        for c in self.codes:
            for y in [2013,2014,2015]:
                rows.append([c,y,f"{y}-12-31",10+y,100+y,20+y,30+y,300+y,200,"연결"])
        self.fin=pd.DataFrame(rows,columns=["stock_code","fiscal_year","fiscal_month_end","ni_parent","equity_parent","cfo","gross_profit","total_assets","total_liabilities","statement_type"])
        self.sec=pd.DataFrame({"stock_code":self.codes,"date":pd.Timestamp("2010-01-01"),"sector_code":"A","is_financial":False})
        self.uni=pd.DataFrame({"date":pd.Timestamp("2016-03-31"),"stock_code":self.codes})
        self.cfg={"profile":"weighted_momentum","selected_count":30,"financial_lag_months":4,"missing_factor_score":0.5,"unit_error_abs_ratio":100}

    def test_percentile_bounds(self):
        p=percentile_rank(pd.Series([1,2,2,4,np.nan]))
        self.assertEqual(p.min(),0); self.assertEqual(p.max(),1); self.assertAlmostEqual(p.iloc[1],p.iloc[2])

    def test_financial_lag_and_selection(self):
        r=select_stocks("2016-03-31",self.uni,{"prices":self.prices,"financials":self.fin,"sectors":self.sec},None,self.cfg)
        self.assertTrue((r.scores["fiscal_year"]==2014).all())
        self.assertEqual(len(r.selected),30)
        self.assertTrue(r.selected[["S_V","S_Q","S_M","CoreScore"]].apply(lambda x:x.between(0,1).all()).all())

    def test_invalid_momentum_weight(self):
        with self.assertRaises(ValueError):
            select_stocks("2016-03-31",self.uni,{"prices":self.prices,"financials":self.fin,"sectors":self.sec},
                          {"value":.45,"quality":.45,"momentum":.10},self.cfg)


if __name__ == "__main__": unittest.main()

