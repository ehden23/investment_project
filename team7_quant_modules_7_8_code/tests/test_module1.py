import unittest

import pandas as pd

from team7_quant.module1 import build_universe


class Module1Tests(unittest.TestCase):
    def setUp(self):
        self.codes = [f"{i:06d}" for i in range(1, 11)]
        self.master = pd.DataFrame({
            "stock_code": self.codes, "listing_date": "2010-01-01", "delisting_date": pd.NaT,
            "market_type": "KOSPI", "security_type": "주권", "share_class": "보통주",
        })
        dates = pd.bdate_range("2015-12-01", "2016-03-31")
        self.prices = pd.DataFrame(
            [(d, c, int(c) * 1000, int(c) * 1_000_000) for d in dates for c in self.codes],
            columns=["date", "stock_code", "trading_value", "market_cap"],
        )
        self.financials = pd.DataFrame({
            "stock_code": self.codes, "fiscal_month_end": "2014-12-31", "equity_parent": 100,
            "ni_parent": 10, "cfo": 12, "gross_profit": 30, "total_assets": 200,
            "total_liabilities": 100, "statement_type": "연결",
        })
        self.cfg = {
            "minimum_listing_months": 12, "liquidity_lookback_sessions": 63,
            "bottom_percentile_cutoff": 0.20, "financial_lag_months": 4,
            "kospi_labels": ["KOSPI"], "common_share_labels": ["보통주"],
            "allowed_security_types": ["주권"],
            "required_financial_columns": ["ni_parent", "equity_parent", "cfo", "gross_profit", "total_assets", "total_liabilities"],
        }

    def test_filters_bottom_twenty_percent(self):
        result = build_universe("2016-03-31", self.master, self.prices, self.financials, self.cfg)
        self.assertEqual(len(result.eligible), 8)
        excluded = result.audit[~result.audit["is_eligible"]]
        self.assertTrue(excluded["exclusion_reason"].str.contains("bottom_20pct").all())

    def test_future_delisting_does_not_remove_past_stock(self):
        self.master.loc[0, "delisting_date"] = "2017-01-10"
        result = build_universe("2016-03-31", self.master, self.prices, self.financials, self.cfg)
        row = result.audit[result.audit["stock_code"] == "000001"].iloc[0]
        self.assertTrue(row["listed_at_asof"])


if __name__ == "__main__":
    unittest.main()
