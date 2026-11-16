"""백테스트 엔진 단위시험: 미래 정보 미사용, 비용 차감 후 NAV 보존, 목표비중 합 1 (합성자료)."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from team7_quant import backtest as bt
from team7_quant.config import load_config
from team7_quant.daily_panel import DailyPanel

ROOT = Path(__file__).resolve().parents[1]
EXEC_CFG = {"buy_commission": 0.00015, "sell_commission": 0.00015, "sell_tax": 0.0020, "slippage_each_side": 0.0005}
ALLOC_CFG = {"selected_count": 30, "lookback_sessions": 252, "min_lookback_sessions": 126, "annual_factor": 252,
             "top_count": 5, "delta": 2.5, "alpha": 0.03, "tau": 0.025, "max_weight": 0.10}


def make_panel(n_days: int = 400, n_stocks: int = 35, seed: int = 0) -> DailyPanel:
    """평일 달력의 합성 패널. 종목마다 다른 변동성의 로그정규 주가."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]
    vol = rng.uniform(0.01, 0.03, n_stocks)
    rets = rng.normal(0.0003, vol, size=(n_days, n_stocks))
    px = pd.DataFrame(1000 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=codes)
    cap = px * rng.uniform(1e6, 1e8, n_stocks)
    volume = pd.DataFrame(1000.0, index=dates, columns=codes)
    return DailyPanel(px, cap, volume, {})


def make_selection(signal_date: pd.Timestamp, codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": signal_date, "stock_code": codes, "rank": np.arange(1, len(codes) + 1)})


class CalendarTests(unittest.TestCase):
    def test_signal_on_holiday_executes_next_session(self):
        sessions = pd.DatetimeIndex(["2015-12-29", "2015-12-30", "2016-01-04", "2016-01-05"])
        s = pd.Timestamp("2015-12-31")
        self.assertEqual(bt.signal_cutoff(sessions, s), pd.Timestamp("2015-12-30"))
        self.assertEqual(bt.execution_session(sessions, s), pd.Timestamp("2016-01-04"))
        self.assertIsNone(bt.execution_session(sessions, pd.Timestamp("2016-01-05")))


class NoLookaheadTests(unittest.TestCase):
    def setUp(self):
        self.panel = make_panel()
        self.signal = self.panel.adj_close.index[300]
        self.codes = list(self.panel.adj_close.columns[:30])
        self.sel = make_selection(self.signal, self.codes)

    def _perturb_after(self, day: pd.Timestamp) -> DailyPanel:
        """day 이후의 가격·시총·거래량을 크게 바꾼 패널."""
        p = copy.deepcopy(self.panel)
        later = p.adj_close.index > day
        rng = np.random.default_rng(99)
        p.adj_close.loc[later] *= rng.uniform(0.3, 3.0, size=p.adj_close.loc[later].shape)
        p.market_cap.loc[later] *= 5.0
        p.volume.loc[later] = 0.0
        return p

    def test_targets_unchanged_when_future_prices_change(self):
        base, _ = bt.build_signal_targets(self.signal, self.sel, self.panel, self.panel.returns(), ALLOC_CFG)
        moved = self._perturb_after(self.signal)
        new, _ = bt.build_signal_targets(self.signal, self.sel, moved, moved.returns(), ALLOC_CFG)
        for aid in base:
            pd.testing.assert_series_equal(base[aid].sort_index(), new[aid].sort_index())

    def test_nav_path_unchanged_before_perturbation(self):
        targets = {self.signal: bt.build_signal_targets(self.signal, self.sel, self.panel,
                                                        self.panel.returns(), ALLOC_CFG)[0]["BL_ALPHA_003"]}
        end = self.panel.adj_close.index[-1]
        cut = self.panel.adj_close.index[350]
        a = bt.run_strategy("T", targets, self.panel, EXEC_CFG, 1e10, end).daily_nav.set_index("date")
        b = bt.run_strategy("T", targets, self._perturb_after(cut), EXEC_CFG, 1e10, end).daily_nav.set_index("date")
        pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])
        self.assertFalse(np.allclose(a.loc[cut:].iloc[1:]["nav"], b.loc[cut:].iloc[1:]["nav"]))

    def test_trades_execute_on_next_session(self):
        targets = {self.signal: pd.Series(1 / 30, index=self.codes)}
        res = bt.run_strategy("T", targets, self.panel, EXEC_CFG, 1e10, self.panel.adj_close.index[-1])
        self.assertEqual(set(res.trades["date"]), {self.panel.adj_close.index[301]})


class NavConservationTests(unittest.TestCase):
    def test_post_trade_nav_equals_pre_trade_nav_minus_cost(self):
        panel = make_panel()
        idx = panel.adj_close.index
        codes = list(panel.adj_close.columns)
        rng = np.random.default_rng(1)
        targets = {}
        for i in [260, 280, 300, 320, 340]:   # 리밸런싱 5번, 매번 다른 30종목·비중
            chosen = list(rng.choice(codes, 30, replace=False))
            w = rng.uniform(0.5, 1.5, 30)
            targets[idx[i]] = pd.Series(w / w.sum(), index=chosen)
        res = bt.run_strategy("T", targets, panel, EXEC_CFG, 1e10, idx[-1])
        self.assertEqual(res.rebalance_count, 5)
        for d in res.diagnostics:
            cost = res.daily_nav.set_index("date").loc[d["execution_date"], "cost"]
            self.assertGreater(cost, 0)
            self.assertAlmostEqual(d["pre_nav"] - cost, d["post_nav"], delta=1e-12 * d["pre_nav"])  # 상대오차 1e-12
        self.assertAlmostEqual(res.trades["cost"].sum(), res.daily_nav["cost"].sum(), delta=1e-6)
        self.assertTrue((res.daily_nav["cash"] >= 0).all())

    def test_constant_prices_lose_exactly_the_costs(self):
        panel = make_panel()
        flat = DailyPanel(panel.adj_close * 0 + 1000.0, panel.market_cap, panel.volume, {})
        idx = flat.adj_close.index
        codes = list(flat.adj_close.columns)
        targets = {idx[260]: pd.Series(1 / 30, index=codes[:30]), idx[300]: pd.Series(1 / 30, index=codes[5:35])}
        nav = bt.run_strategy("T", targets, flat, EXEC_CFG, 1e10, idx[-1]).daily_nav
        self.assertAlmostEqual(nav["nav"].iloc[-1], 1e10 - nav["cost"].sum(), delta=1e-3)
        # 첫 매수 비용 ≈ 1e10 × (수수료+슬리피지) / (1+매수비용률)
        buy = EXEC_CFG["buy_commission"] + EXEC_CFG["slippage_each_side"]
        self.assertAlmostEqual(nav["cost"].iloc[1], 1e10 * buy / (1 + buy), delta=1.0)

    def test_zero_volume_stock_is_not_traded_and_logged(self):
        panel = make_panel()
        idx = panel.adj_close.index
        codes = list(panel.adj_close.columns[:30])
        panel.volume.loc[idx[261], codes[0]] = 0.0   # 체결일 거래량 0
        res = bt.run_strategy("T", {idx[260]: pd.Series(1 / 30, index=codes)}, panel, EXEC_CFG, 1e10, idx[-1])
        self.assertNotIn(codes[0], set(res.trades["stock_code"]))
        self.assertEqual(list(res.untradable["stock_code"]), [codes[0]])
        self.assertEqual(res.untradable.iloc[0]["reason"], "zero_volume")
        self.assertAlmostEqual(res.daily_nav.iloc[1]["cash"] / res.daily_nav.iloc[1]["nav"], 1 / 30, delta=1e-3)


class TargetWeightTests(unittest.TestCase):
    def setUp(self):
        self.panel = make_panel()
        self.signal = self.panel.adj_close.index[300]
        self.codes = list(self.panel.adj_close.columns[:30])

    def _check(self, targets):
        for aid, w in targets.items():
            self.assertAlmostEqual(float(w.sum()), 1.0, places=8, msg=aid)
            self.assertTrue((w >= -1e-10).all(), aid)
            self.assertEqual(len(w), 30)
        self.assertLessEqual(float(targets["BL_ALPHA_003"].max()), ALLOC_CFG["max_weight"] + 1e-8)

    def test_full_history_weights_sum_to_one(self):
        t, log = bt.build_signal_targets(self.signal, make_selection(self.signal, self.codes),
                                         self.panel, self.panel.returns(), ALLOC_CFG)
        self.assertEqual(log["bl_status"], "OK")
        self.assertEqual(log["lookback_used"], 252)
        self._check(t)

    def test_short_history_shortens_lookback(self):
        self.panel.adj_close.iloc[:300 - 150, 0] = np.nan   # 첫 종목은 신호일까지 151개 가격 = 150개 수익률
        t, log = bt.build_signal_targets(self.signal, make_selection(self.signal, self.codes),
                                         self.panel, self.panel.returns(), ALLOC_CFG)
        self.assertEqual(log["bl_status"], "SHORTENED")
        self.assertEqual(log["lookback_used"], 150)
        self._check(t)

    def test_too_short_history_falls_back_to_equal_weight(self):
        self.panel.adj_close.iloc[:300 - 100, 0] = np.nan
        t, log = bt.build_signal_targets(self.signal, make_selection(self.signal, self.codes),
                                         self.panel, self.panel.returns(), ALLOC_CFG)
        self.assertEqual(log["bl_status"], "EW_FALLBACK")
        pd.testing.assert_series_equal(t["BL_ALPHA_003"], t["EW"])
        self._check(t)


class BenchmarkProxyTests(unittest.TestCase):
    def test_previous_day_cap_weighting_by_hand(self):
        dates = pd.bdate_range("2016-01-04", periods=3)
        px = pd.DataFrame({"A": [100.0, 110.0, 99.0], "B": [50.0, 50.0, 55.0], "C": [np.nan, np.nan, 10.0]}, index=dates)
        cap = pd.DataFrame({"A": [300.0, 330.0, 297.0], "B": [100.0, 100.0, 110.0], "C": [np.nan, np.nan, 999.0]}, index=dates)
        panel = DailyPanel(px, cap, px * 0 + 1, {})
        b = bt.build_benchmark_proxy(panel, dates[0], dates[-1], 100.0)
        # 1일차: (300×0.10 + 100×0) / 400 = 0.075. 2일차: (330×−0.10 + 100×0.10) / 430. C는 전일 시총이 없어 제외.
        r2 = (330 * -0.10 + 100 * 0.10) / 430
        np.testing.assert_allclose(b["daily_return"], [0.0, 0.075, r2])
        np.testing.assert_allclose(b["level"], [100.0, 107.5, 107.5 * (1 + r2)])
        self.assertEqual(list(b["n_constituents"]), [0, 2, 2])


class ConfigTests(unittest.TestCase):
    def test_round_trip_cost_floor(self):
        cfg = json.loads((ROOT / "config" / "base.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(bt.round_trip_cost(cfg["execution"]), 0.0023)
        cfg["execution"]["sell_tax"] = 0.0005   # 왕복 0.03+0.05+0.10 = 0.18% < 0.23%
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cfg.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
