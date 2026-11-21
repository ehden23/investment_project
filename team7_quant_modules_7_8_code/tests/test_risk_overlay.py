"""Accounting, causality, rollover, funding and historical contract tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from team7_quant.risk_data import RiskMarket, observed_dividends, contract_multiplier
from team7_quant.risk_inputs import CostBook, attach_cash_limits
from team7_quant.risk_features import rolling_stock_betas
from team7_quant.risk_overlay import run_overlay as engine_run, DEFAULT_RISK
from team7_quant.module8 import performance


EXEC = {"buy_commission": 0.00015, "sell_commission": 0.00015,
        "sell_tax": 0.002, "slippage_each_side": 0.0005}
TEST_COSTS = {"option_commission_rate": .0015, "option_slippage_ticks": .5,
              "future_commission_rate": .00003, "future_slippage_points": .025,
              "cost_source": "SYNTHETIC_TEST_ONLY"}


def run_overlay(f, m, execution=None, cfg=None, **kwargs):
    return engine_run(f, m, execution, {**TEST_COSTS, **(cfg or {})}, **kwargs)


def fixture(start="2016-01-04", n=18, declining=False, rollover=False):
    dates = pd.bdate_range(start, periods=n)
    levels = np.linspace(100, 65 if declining else 100, n)
    index = pd.DataFrame({"open": levels, "close": levels}, index=dates)
    macro = pd.DataFrame({"q": 0.0, "r_continuous": 0.01, "cd_rate": .01,
                          "q_status": "TEST", "available_at": dates[0]}, index=dates)
    opts, futs = [], []
    expiry = dates[-1] + pd.Timedelta(days=35)
    for d, level in zip(dates, levels):
        for code, strike in [("P1", 95), ("P2", 90)]:
            opts.append({"date": d, "contract_code": code, "contract_name": code,
                         "close": 1.0, "volume": 1000, "open_interest": 2000, "iv": .2,
                         "expiry_date": expiry, "strike": strike, "multiplier": contract_multiplier(d)})
        for code in ["F1", "F2"]:
            futs.append({"date": d, "contract_code": code, "contract_name": code,
                         "close": level, "volume": 1000, "open_interest": 2000,
                         "expiry_date": expiry, "multiplier": contract_multiplier(d)})
    schedule = [{"signal_date": dates[0], "execution_date": dates[1], "kind": kind,
                 "contract_code": code, "expiry_date": expiry} for kind, code in [("PUT", "P1"), ("FUTURES", "F1")]]
    if rollover:
        schedule += [{"signal_date": dates[7], "execution_date": dates[8], "kind": kind,
                      "contract_code": code, "expiry_date": expiry} for kind, code in [("PUT", "P2"), ("FUTURES", "F2")]]
    features = pd.DataFrame({"date": dates, "strategy_id": "STATIC_EW", "base_nav": 1e9 * levels / 100,
                             "base_cash": .4e9 * levels / 100, "base_cost": 0.0, "base_equity_weight": .6,
                             "beta_current": .6, "beta_forecast": .6,
                             "beta_fallback_count": 0, "beta_min_pairs": 252,
                             "equity_rebalance_signal": False, "funding_tradable": True})
    margin = pd.DataFrame({"eod_initial_rate": .09,"eod_maintenance_rate": .06,
                           "session_initial_rate": .09,"session_maintenance_rate": .06,
                           "eod_date_status": "SYNTHETIC_TEST_ONLY"}, index=dates)
    m = RiskMarket(index, pd.DataFrame(opts), pd.DataFrame(futs), macro, pd.DataFrame(schedule),
                   {"dividend_policy": "observed_index_yield"}, margin)
    return features, m


class FundingAndAccountingTests(unittest.TestCase):
    def test_no_hedge_exactly_preserves_upstream_nav(self):
        f, m = fixture(declining=True)
        r = run_overlay(f, m, EXEC, funding="strict", mode="NONE")
        np.testing.assert_allclose(r.nav.nav, f.base_nav, rtol=0, atol=1e-6)
        self.assertTrue(r.trades.empty)

    def test_strict_does_not_invent_funding_or_sell_upstream(self):
        f, m = fixture(declining=True)
        f['base_cash'] = 0.
        r = run_overlay(f, m, EXEC, funding="strict")
        self.assertTrue(r.nav.put_quantity.eq(0).all())
        self.assertTrue(r.nav.futures_short_quantity.eq(0).all())
        np.testing.assert_allclose(r.nav.stock_value, f.base_nav)
        self.assertGreater(r.nav.blocked_orders.sum(), 0)
        np.testing.assert_allclose(r.nav.nav, f.base_nav)

    def test_flat_market_loses_only_explicit_costs(self):
        f, m = fixture()
        r = run_overlay(f, m, EXEC, funding="upstream_cash", mode="PUT")
        self.assertGreater(r.nav.put_quantity.max(), 0)
        self.assertAlmostEqual(r.nav.nav.iloc[-1], 1e9 - r.nav.overlay_cost.sum(), places=5)
        self.assertLess(r.nav.self_financing_error.abs().max(), 1e-5)
        self.assertTrue(r.nav.hedge_cash_available.ge(-1e-5).all())

    def test_short_futures_gain_is_daily_cash_not_notional(self):
        f, m = fixture(declining=True)
        r = run_overlay(f, m, EXEC, funding="upstream_cash")
        self.assertGreater(r.nav.futures_short_quantity.max(), 0)
        self.assertGreater(r.nav.futures_pnl.sum(), 0)
        # Margin is a lock, not an expense or separately added asset.
        np.testing.assert_allclose(r.nav.nav, r.nav.stock_value + r.nav.total_cash + r.nav.put_value, atol=1e-5)
        self.assertLess(r.nav.self_financing_error.abs().max(), 1e-5)

    def test_same_size_roll_closes_old_contract(self):
        f, m = fixture(rollover=True)
        r = run_overlay(f, m, EXEC, funding="upstream_cash", mode="PUT")
        roll = m.index.index[8]
        t = r.trades[(r.trades.date == roll) & (r.trades.instrument == "PUT")]
        self.assertEqual(set(t.contract_code), {"P1", "P2"})
        self.assertLess(t[t.contract_code == "P1"].quantity_change.sum(), 0)
        self.assertGreater(t[t.contract_code == "P2"].quantity_change.sum(), 0)

    def test_contract_split_has_no_free_pnl(self):
        f, m = fixture(start="2017-03-20", n=12)
        zero = {"option_commission_rate": 0, "option_slippage_ticks": 0,
                "future_commission_rate": 0, "future_slippage_points": 0}
        ex = {k: 0 for k in EXEC}
        r = run_overlay(f, m, ex, zero, funding="upstream_cash", mode="PUT")
        d = r.nav.set_index("date")
        self.assertEqual(d.loc['2017-03-27', 'put_quantity'], 2 * d.loc['2017-03-24', 'put_quantity'])
        self.assertAlmostEqual(d.loc['2017-03-27', 'put_value'], d.loc['2017-03-24', 'put_value'])
        self.assertAlmostEqual(d.loc['2017-03-27', 'nav'], d.loc['2017-03-24', 'nav'])

    def test_no_hedge_preserves_the_cash_path(self):
        f, m = fixture(declining=True)
        r = run_overlay(f, m, EXEC, funding="upstream_cash", mode="NONE")
        self.assertTrue(r.nav.put_quantity.eq(0).all())
        self.assertTrue(r.nav.futures_short_quantity.eq(0).all())
        np.testing.assert_allclose(r.nav.total_cash,f.base_cash)


class CausalityTests(unittest.TestCase):
    def test_future_quotes_do_not_change_past_nav_or_orders(self):
        f, m = fixture(declining=True, rollover=True)
        cut = m.index.index[9]
        changed = copy.deepcopy(m)
        changed.options.loc[changed.options.date > cut, 'close'] *= 1.7
        changed.futures.loc[changed.futures.date > cut, 'close'] *= 1.1
        a = run_overlay(f, m, EXEC, funding="upstream_cash")
        b = run_overlay(f, changed, EXEC, funding="upstream_cash")
        pd.testing.assert_frame_equal(a.nav[a.nav.date <= cut], b.nav[b.nav.date <= cut])
        pd.testing.assert_frame_equal(a.trades[a.trades.date <= cut], b.trades[b.trades.date <= cut])

    def test_daily_dividend_is_observed_not_a_yearly_lag(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'div.csv'
            d=pd.DataFrame({'일자':['2020/01/02','2020/01/03'],'배당수익률':[2.,3.]})
            d.to_csv(path,index=False,encoding='cp949')
            a=observed_dividends(path)
            self.assertEqual(a.loc['2020-01-02','q'],.02)
            d.loc[1,'배당수익률']=4.;d.to_csv(path,index=False,encoding='cp949')
            b=observed_dividends(path)
            pd.testing.assert_frame_equal(a.loc[:'2020-01-02'],b.loc[:'2020-01-02'])

    def test_beta_uses_only_past_prices(self):
        idx = pd.bdate_range('2015-01-01', periods=300)
        rng = np.random.default_rng(4)
        r = rng.normal(0, .01, len(idx))
        market = pd.Series(100 * np.cumprod(1 + r), index=idx)
        prices = pd.DataFrame({'A': 100 * np.cumprod(1 + 1.4 * r)}, index=idx)
        a, _ = rolling_stock_betas(prices, market)
        self.assertAlmostEqual(a.A.iloc[-1], 1.4, places=10)
        cut = idx[200]; prices.loc[prices.index > cut, 'A'] *= 5
        b, _ = rolling_stock_betas(prices, market)
        pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])


class PerformanceTests(unittest.TestCase):
    def test_cd_accrual_uses_previous_quote_and_calendar_days(self):
        dates = pd.to_datetime(['2020-01-03', '2020-01-06', '2020-01-07'])
        nav = pd.Series([100, 101, 100.5], index=dates)
        cd = pd.Series([.02, .03, .90], index=dates)
        index = pd.Series([100, 100, 101], index=dates)
        result = performance(nav, index, cd)
        expected = np.array([.01 - .02 * 3 / 365, 100.5 / 101 - 1 - .03 / 365])
        self.assertAlmostEqual(result['sharpe'], expected.mean() / expected.std(ddof=1) * np.sqrt(252))
        self.assertAlmostEqual(result['mdd'], 100.5 / 101 - 1)


class ExplicitInputTests(unittest.TestCase):
    def test_old_fixed_margin_override_is_rejected(self):
        f,m=fixture()
        with self.assertRaisesRegex(ValueError,'제거된 임의 가정'):run_overlay(f,m,cfg={'margin_rate':.15})
    def test_missing_costs_do_not_become_zero(self):
        f,m=fixture()
        with self.assertRaisesRegex(ValueError,'미확정 파생 거래비용'):
            engine_run(f,m)

    def test_explicit_zero_costs_are_allowed(self):
        c=CostBook.from_config({**{k:0 for k in TEST_COSTS if k!='cost_source'},'cost_source':'TEST_EXPLICIT_ZERO'})
        self.assertEqual(c.fee('PUT',3,2,250000,pd.Timestamp('2020-01-01')),0)

    def test_cost_schedule_cannot_backfill_from_future(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'cost.csv'
            pd.DataFrame([{'effective_date':'2020-01-03','source':'TEST',**{k:0 for k in TEST_COSTS if k!='cost_source'}}]).to_csv(p,index=False)
            c=CostBook.from_config({'cost_file':str(p)})
            with self.assertRaisesRegex(ValueError,'이전 적용'):c.at(pd.Timestamp('2020-01-02'))

    def test_cash_pool_cannot_add_external_money(self):
        f,m=fixture();f['hedge_cash_available']=f.base_cash+100
        with self.assertRaisesRegex(ValueError,'이미 포함된 현금'):attach_cash_limits(f)

    def test_consumed_cash_is_not_replenished_next_day(self):
        f,m=fixture();r=run_overlay(f,m,mode='PUT')
        self.assertGreater(r.nav.put_quantity.max(),0)
        used=-r.nav.overlay_cash_delta.iloc[1]
        self.assertGreater(used,0)
        self.assertAlmostEqual(r.nav.hedge_cash_available.iloc[2],f.base_cash.iloc[2]-used)

    def test_upstream_cannot_reuse_money_paid_as_option_premium(self):
        f,m=fixture();f.loc[4:,'base_cash']=0
        with self.assertRaisesRegex(ValueError,'현금 부족'):run_overlay(f,m,mode='PUT')

    def test_proportional_funding_is_rejected(self):
        f,m=fixture()
        with self.assertRaisesRegex(ValueError,'비례 매도'):run_overlay(f,m,funding='proportional')

    def test_expiry_has_no_invented_settlement_fallback(self):
        f,m=fixture();expiry=m.index.index[5]
        m.options['expiry_date']=expiry
        m.options.loc[m.options.date.eq(expiry),'volume']=0
        with self.assertRaisesRegex(ValueError,'실제 최종결제'):run_overlay(f,m,mode='PUT')


class HistoricalMarginTests(unittest.TestCase):
    def test_prepared_march_2020_boundary_matches_original_notice(self):
        path=Path(__file__).resolve().parents[1]/'data/risk_prepared/margin_rates.csv'
        if not path.exists():self.skipTest('Audited market inputs not present')
        rates=pd.read_csv(path,index_col='date')
        self.assertAlmostEqual(rates.loc['2020-03-17','session_initial_rate'],.093)
        self.assertAlmostEqual(rates.loc['2020-03-17','eod_initial_rate'],.111)
        self.assertAlmostEqual(rates.loc['2020-03-18','session_maintenance_rate'],.074)
        self.assertAlmostEqual(rates.loc['2020-03-23','eod_initial_rate'],.138)
    def test_margin_change_uses_published_history_and_is_not_an_expense(self):
        f,m=fixture(declining=True);f['beta_current']=1.4;f['beta_forecast']=1.4
        f['base_cash']=.3*f.base_nav
        m.margin.loc[m.index.index[10]:,'eod_initial_rate']=.4
        r=run_overlay(f,m,cfg={'put_coverage':0})
        held=r.nav[r.nav.futures_short_quantity>0]
        self.assertFalse(held.empty)
        day=held[held.date>=m.index.index[10]].iloc[0]
        self.assertAlmostEqual(day.margin_initial_required,day.futures_short_quantity*contract_multiplier(day.date)*m.index.loc[day.date,'close']*.4)
        self.assertLess(r.nav.self_financing_error.abs().max(),1e-5)

    def test_maintenance_breach_queues_reduction_next_session(self):
        f,m=fixture(declining=True);f['beta_current']=1.4;f['beta_forecast']=1.4;f['base_cash']=.15*f.base_nav
        m.margin.loc[m.index.index[10]:,'eod_initial_rate']=.8
        m.margin.loc[m.index.index[10]:,'eod_maintenance_rate']=.6
        r=run_overlay(f,m,cfg={'put_coverage':0})
        calls=r.events[r.events.reason.eq('MARGIN_CALL')]
        reductions=r.trades[r.trades.reason.eq('MARGIN_CALL_REDUCTION')]
        self.assertFalse(calls.empty);self.assertFalse(reductions.empty)
        self.assertTrue((reductions.signal_date<reductions.date).all())
        self.assertEqual(reductions.date.iloc[0],m.index.index[m.index.index.get_loc(calls.date.iloc[0])+1])

    def test_future_margin_change_does_not_change_past_trades(self):
        f,m=fixture(declining=True);other=copy.deepcopy(m);cut=m.index.index[10]
        other.margin.loc[other.margin.index>cut,'eod_initial_rate']=.5
        a=run_overlay(f,m);b=run_overlay(f,other)
        pd.testing.assert_frame_equal(a.nav[a.nav.date<=cut],b.nav[b.nav.date<=cut])
        pd.testing.assert_frame_equal(a.trades[a.trades.date<=cut],b.trades[b.trades.date<=cut])


class EndToEndEvaluationTests(unittest.TestCase):
    def test_three_modes_write_reports_with_cash_accounting(self):
        from team7_quant.module8 import run_risk_evaluation
        f,m=fixture(declining=True,rollover=True)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'prepared';p.mkdir();out=Path(folder)/'results'
            f.to_csv(p/'baseline_features.csv',index=False)
            m.index.to_csv(p/'kospi200.csv',index_label='date')
            m.options.to_csv(p/'puts.csv',index=False);m.futures.to_csv(p/'futures.csv',index=False)
            m.macro.to_csv(p/'risk_macro.csv',index_label='date')
            m.margin.to_csv(p/'margin_rates.csv',index_label='date')
            m.schedule.to_csv(p/'hedge_schedule.csv',index=False)
            (p/'risk_input_metadata.json').write_text(json.dumps(m.metadata))
            result=run_risk_evaluation(p,out,{'risk':TEST_COSTS})
            self.assertEqual(set(result.scenario),{'NO_HEDGE','PUT_ONLY','PUT_FUTURES'})
            daily=pd.read_csv(out/'risk_daily_nav.csv')
            self.assertLess(daily.self_financing_error.abs().max(),1e-5)
            self.assertTrue((out/'report.html').exists())
            trades=pd.read_csv(out/'risk_trades.csv')
            self.assertEqual(set(trades.instrument),{'PUT','FUTURES'})
            self.assertEqual(len(pd.read_csv(out/'performance_summary.csv')),3)


if __name__ == '__main__':
    unittest.main()
