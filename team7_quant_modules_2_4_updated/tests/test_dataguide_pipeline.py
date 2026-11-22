import unittest
import tempfile
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from team7_quant.dataguide import load_dataguide_transposed_market, load_market_auto
from team7_quant.io import normalize_stock_code
from team7_quant.pipeline24 import add_dividend_reinvestment_proxy, build_factor_returns


class DataGuidePipelineTests(unittest.TestCase):
    def test_dataguide_a_prefix_is_removed(self):
        got = normalize_stock_code(pd.Series(["A005930", "5930", "000660"])).tolist()
        self.assertEqual(got, ["005930", "005930", "000660"])

    def test_factor_return_uses_next_month(self):
        scores = pd.DataFrame({
            "asof_date": [pd.Timestamp("2020-01-31")] * 3,
            "stock_code": ["000001", "000002", "000003"],
            "S_V": [1.0, 0.5, 0.0],
            "S_Q": [0.0, 0.5, 1.0],
            "market_cap": [3, 2, 1],
            "all_factors_missing": [False, False, False],
        })
        prices = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-31"] * 3 + ["2020-02-28"] * 3),
            "stock_code": ["000001", "000002", "000003"] * 2,
            "adj_close": [100, 100, 100, 110, 100, 90],
        })
        result = build_factor_returns(scores, prices, count=1).iloc[0]
        self.assertAlmostEqual(result["value_return"], 0.10)
        self.assertAlmostEqual(result["quality_return"], -0.10)

    def test_dividend_proxy_is_reinvested_multiplicatively(self):
        prices = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "stock_code": ["000001", "000001"],
            "adj_close": [100.0, 110.0],
            "market_cap": [1_000.0, 1_000.0],
        })
        dividends = pd.DataFrame({
            "stock_code": ["000001"],
            "available_at": pd.to_datetime(["2020-02-29"]),
            "dividend_cash_paid": [100.0],
        })
        result, logs = add_dividend_reinvestment_proxy(prices, dividends)
        self.assertTrue(logs.empty)
        self.assertAlmostEqual(result.iloc[1]["total_return_proxy"], 0.21)
        self.assertAlmostEqual(result.iloc[1]["total_return_index_proxy"], 121.0)

    def test_dps_is_added_to_holding_period_return(self):
        prices = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "stock_code": ["000001", "000001"],
            "adj_close": [100.0, 110.0], "market_cap": [1_000.0, 1_100.0],
        })
        dividends = pd.DataFrame({
            "stock_code": ["000001"], "available_at": pd.to_datetime(["2020-02-29"]),
            "dividend_per_share": [5.0],
        })
        result, _ = add_dividend_reinvestment_proxy(prices, dividends)
        self.assertAlmostEqual(result.iloc[1]["total_return_proxy"], 0.15)
        self.assertAlmostEqual(result.iloc[1]["total_return_index_proxy"], 115.0)

    def test_transposed_market_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "market.xlsx"
            wb = Workbook(); ws = wb.active
            headers = ["코드", "코드명", "유형", "아이템코드", "아이템명", "집계주기"]
            header_row = 12  # DataGuide 설정/메타데이터 행 수가 달라도 자동 탐지해야 한다.
            for c, value in enumerate(headers, 1): ws.cell(header_row, c, value)
            dates = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-28"])
            for c, value in enumerate(dates, 7): ws.cell(header_row, c, value.to_pydatetime())
            rows = [
                ("A000001", "테스트", "SSC", "S410000700", "수정주가", [90, 100, 110]),
                ("A000001", "테스트", "SSC", "S410000900", "거래대금", [10, 20, 30]),
                ("A000001", "테스트", "SSC", "S410001200", "시가총액", [1, 2, 3]),
            ]
            for r, (code, name, typ, item, label, values) in enumerate(rows, header_row + 1):
                for c, value in enumerate([code, name, typ, item, label, "일간", *values], 1): ws.cell(r, c, value)
            wb.save(path)
            wb.close()
            result, meta = load_dataguide_transposed_market(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(meta["security_count"], 1)
            self.assertAlmostEqual(result.iloc[0]["adj_close"], 100)
            self.assertAlmostEqual(result.iloc[0]["market_cap"], 2_000_000)
            self.assertAlmostEqual(result.iloc[1]["avg_trading_value_63d"], 20)
            auto_result, auto_meta = load_market_auto(path)
            self.assertEqual(len(auto_result), 2)
            self.assertIn("transposed", auto_meta["source_format"])


if __name__ == "__main__":
    unittest.main()
