"""Module 7. NAV=upstream NAV+derivative cash movements+put market value.

The funding pool is already included in upstream cash. No equity liquidation,
external cash, synthetic funding, flat margin rates, or zero cost fallbacks.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor
import numpy as np
import pandas as pd
from .module7 import RiskState, put_delta, stress_test, update_risk_level
from .risk_data import MULTIPLIER_CHANGE, RiskMarket, contract_multiplier
from .risk_inputs import COST_FIELDS, CostBook, attach_cash_limits
DEFAULT_RISK = {'drawdown_thresholds': [0.08, 0.12, 0.15], 'release_thresholds': [0.06, 0.1, 0.13], 'release_days': 5, 'beta_caps': [0.8, 0.5, 0.2], 'put_coverage': 0.5, 'vol_window': 60, 'es_window': 252, 'es_confidence': 0.95, 'roll_sessions': 5, 'beta_window': 252, 'beta_minimum': 126, 'put_resize_on_equity_rebalance': True, 'margin_policy': 'standalone_futures_no_offsets', 'dividend_input_policy': 'observed_index_yield', 'cost_file': None, 'cost_source': None, 'cash_file': None, **{k: None for k in COST_FIELDS}}

@dataclass
class OverlayResult:
    nav: pd.DataFrame
    trades: pd.DataFrame
    events: pd.DataFrame
    stress: pd.DataFrame
    assumptions: dict

def run_overlay(features: pd.DataFrame, market: RiskMarket, upstream_execution=None, cfg=None, *, funding='upstream_cash', mode='PUT_FUTURES', scenario='PUT_FUTURES'):
    retired = {'margin_rate', 'minimum_cash_ratio', 'reinvest_threshold', 'force_q_zero', 'dividend_lag_months'} & set(cfg or {})
    if retired:
        raise ValueError('제거된 임의 가정 설정: ' + ', '.join(sorted(retired)))
    cfg = {**DEFAULT_RISK, **(cfg or {})}
    if funding not in {'upstream_cash', 'strict'}:
        raise ValueError('앞 모듈 우선: 주식 비례 매도/외부 자금 조달은 지원하지 않습니다.')
    if mode not in {'NONE', 'PUT', 'PUT_FUTURES'}:
        raise ValueError('Unknown hedge mode')
    f = attach_cash_limits(features, cfg.get('cash_file')).sort_values('date').set_index('date')
    if not isinstance(f.index, pd.DatetimeIndex) or f.index.has_duplicates or f.empty or (f.strategy_id.nunique() != 1):
        raise ValueError('한 전략의 고유한 일별 스냅샷이 필요합니다.')
    required = ['base_nav', 'base_cost', 'beta_current', 'beta_forecast', 'beta_fallback_count']
    if f[required].isna().any().any() or not np.isfinite(f[required].to_numpy(float)).all() or (f.base_nav <= 0).any():
        raise ValueError('Invalid upstream NAV/beta/cost inputs')
    if mode != 'NONE' and (f.beta_fallback_count > 0).any():
        raise ValueError('베타 이력 부족. beta=1로 대체하지 않습니다.')
    costs = CostBook.from_config(cfg) if mode != 'NONE' else None
    if costs:
        costs.at(f.index[min(1, len(f) - 1)])
    if mode == 'PUT_FUTURES' and cfg['margin_policy'] != 'standalone_futures_no_offsets':
        raise ValueError('지원 증거금 방식: standalone_futures_no_offsets')
    if cfg['dividend_input_policy'] not in {'observed_index_yield', 'provided_forward_yield'}:
        raise ValueError('배당 입력 정의가 필요합니다.')
    if mode != 'NONE' and market.metadata.get('dividend_policy') == 'legacy_annual_proxy':
        raise ValueError('연간 배당 시차 대용치를 일별 배당 자료로 교체하세요.')
    strategy = str(f.strategy_id.iloc[0])
    run_id = f'{strategy}__{scenario}'
    books = {'PUT': market.options.set_index(['date', 'contract_code']), 'FUTURES': market.futures.set_index(['date', 'contract_code'])}
    for book in books.values():
        if book.index.has_duplicates:
            raise ValueError('Duplicate derivative quotes')

    @lru_cache(maxsize=None)
    def quote(kind, code, date):
        try:
            q = books[kind].loc[date, code]
        except KeyError as e:
            raise ValueError(f'Missing {kind} quote {code} {date.date()}') from e
        if not np.isfinite(q.close) or q.close < 0:
            raise ValueError(f'Invalid quote {code} {date.date()}')
        return q
    schedule = {d: {r.kind: r.contract_code for r in g.itertuples()} for d, g in market.schedule.groupby('signal_date')}
    cash_delta = 0.0
    put_qty = future_qty = 0
    put_code = future_code = candidate_put = candidate_future = None
    last_put = last_future = 0.0
    state = RiskState()
    pending = margin_pending = None
    prev_nav = prev_base = peak = float(f.base_nav.iloc[0])
    navs = []
    trades = []
    events = []
    for i, (date, b) in enumerate(f.iterrows()):
        mult = contract_multiplier(date)
        if i and f.index[i - 1] < MULTIPLIER_CHANGE <= date:
            put_qty *= 2
            future_qty *= 2
            if pending:
                pending['put_target'] *= 2
                pending['future_target'] *= 2
            if margin_pending:
                margin_pending['target'] *= 2
        put_pnl = 0.0 if not put_qty else put_qty * mult * (float(quote('PUT', put_code, date).close) - last_put)
        future_pnl = 0.0 if not future_qty else -future_qty * mult * (float(quote('FUTURES', future_code, date).close) - last_future)
        cash_delta += future_pnl
        cost_today = 0.0
        blocked = 0

        def event(reason, **kw):
            events.append({'date': date, 'run_id': run_id, 'reason': reason, **kw})

        def available():
            return float(b.hedge_cash_available) + cash_delta

        def put_value():
            return 0.0 if not put_qty else put_qty * mult * float(quote('PUT', put_code, date).close)

        def value():
            return float(b.base_nav) + cash_delta + put_value()

        def rate_row():
            if market.margin is None or date not in market.margin.index:
                raise ValueError(f'증거금 이력 없음: {date.date()}')
            r = market.margin.loc[date]
            if not np.isfinite([r.eod_initial_rate, r.eod_maintenance_rate]).all() or not 0 < r.eod_maintenance_rate <= r.eod_initial_rate <= 1:
                raise ValueError(f'증거금 요율/단위 오류 {date.date()}')
            return r

        def margin(qty=None, maintenance=False):
            q = future_qty if qty is None else qty
            if not q:
                return 0.0
            r = rate_row()
            return q * mult * float(market.index.loc[date, 'close']) * (r.eod_maintenance_rate if maintenance else r.eod_initial_rate)

        def executable(kind, code):
            q = quote(kind, code, date)
            return bool(np.isfinite(q.volume) and q.volume > 0 and (q.close > 0))

        def trade(kind, code, diff, price, fee, reason, signal):
            trades.append({'date': date, 'signal_date': signal, 'run_id': run_id, 'instrument': kind, 'contract_code': code, 'quantity_change': diff, 'price': price, 'multiplier': mult, 'cost': fee, 'reason': reason, 'cost_source': costs.at(date).source if costs else ''})

        def change_put(target, code, reason, signal):
            nonlocal put_qty, put_code, cash_delta, cost_today, blocked
            diff = int(target) - put_qty
            if not diff:
                return
            if put_qty and code != put_code:
                raise RuntimeError('Close old put first')
            if not executable('PUT', code):
                blocked += 1
                event('PUT_FILL_BLOCKED', contract_code=code)
                return
            price = float(quote('PUT', code, date).close)
            fee1 = costs.fee('PUT', 1, price, mult, date)
            requested = diff
            if diff > 0:
                diff = min(diff, max(0, floor((available() - margin() + 1e-07) / (price * mult + fee1))))
                if diff < requested:
                    blocked += 1
                    event('PUT_CASH_LIMIT', requested=requested, executed=diff)
            if not diff:
                return
            fee = abs(diff) * fee1
            cash_delta -= diff * mult * price + fee
            cost_today += fee
            put_qty += diff
            put_code = code if put_qty else None
            trade('PUT', code, diff, price, fee, reason, signal)

        def change_future(target, code, reason, signal):
            nonlocal future_qty, future_code, cash_delta, cost_today, blocked
            diff = int(target) - future_qty
            if not diff:
                return
            if future_qty and code != future_code:
                raise RuntimeError('Close old future first')
            if not executable('FUTURES', code):
                blocked += 1
                event('FUTURES_FILL_BLOCKED', contract_code=code)
                return
            price = float(quote('FUTURES', code, date).close)
            fee1 = costs.fee('FUTURES', 1, price, mult, date)
            requested = diff
            if diff > 0:
                capacity = max(0, floor((available() - margin() + 1e-07) / (margin(1) + fee1)))
                diff = min(diff, capacity)
                if diff < requested:
                    blocked += 1
                    event('FUTURES_MARGIN_LIMIT', requested=requested, executed=diff)
            if not diff:
                return
            fee = abs(diff) * fee1
            cash_delta -= fee
            cost_today += fee
            future_qty += diff
            future_code = code if future_qty else None
            trade('FUTURES', code, -diff, price, fee, reason, signal)
        margin_action = False
        if margin_pending and margin_pending['date'] == date:
            if future_qty and future_code == margin_pending['code']:
                change_future(min(future_qty, margin_pending['target']), future_code, 'MARGIN_CALL_REDUCTION', margin_pending['signal'])
                margin_action = True
            margin_pending = None
        if pending and pending['execution_date'] == date:
            p = pending
            if put_qty and put_code != p['put_code']:
                change_put(0, put_code, 'ROLL_CLOSE', p['signal_date'])
            if future_qty and (future_code != p['future_code'] or future_qty > p['future_target']):
                change_future(p['future_target'] if future_code == p['future_code'] else 0, future_code, 'ROLL_CLOSE_OR_REDUCE', p['signal_date'])
            if not put_qty or put_code == p['put_code']:
                change_put(p['put_target'], p['put_code'], 'ENTER_RESIZE_OR_ROLL', p['signal_date'])
            if mode == 'PUT_FUTURES' and (not margin_action) and (not future_qty or future_code == p['future_code']):
                change_future(p['future_target'], p['future_code'], 'ENTER_RESIZE_OR_ROLL', p['signal_date'])
            pending = None
        for kind, qty, code in [('PUT', put_qty, put_code), ('FUTURES', future_qty, future_code)]:
            if qty and quote(kind, code, date).expiry_date <= date:
                if executable(kind, code):
                    (change_put if kind == 'PUT' else change_future)(0, code, 'EXPIRY_CLOSE', date)
                else:
                    raise ValueError(f'{date.date()} {code}: 만기 청산 불가. 실제 최종결제 자료가 필요합니다.')
        if available() < -1e-05 or float(b.base_cash) + cash_delta < -1e-05:
            raise ValueError(f'현금 부족 {strategy} {date.date()}: {available():,.0f}원. 앞 모듈의 현금 경로를 수정해야 합니다. 자동 주식 매도·차입은 하지 않습니다.')
        nav = value()
        if nav <= 0:
            raise ValueError('Portfolio insolvent')
        stock_pnl = float(b.base_nav) - prev_base
        error = nav - (prev_nav + stock_pnl + put_pnl + future_pnl - cost_today)
        if abs(error) > max(0.0001, nav * 1e-10):
            raise RuntimeError(f'NAV identity failed {date}: {error}')
        peak = max(peak, float(b.base_nav))
        dd = 1 - float(b.base_nav) / peak
        state, reason = update_risk_level(state, dd, cfg)
        S = float(market.index.loc[date, 'close'])
        macro = market.macro.loc[date]
        q = float(macro.q)
        r = float(macro.r_continuous)
        if mode != 'NONE' and (not np.isfinite([q, r]).all()):
            raise ValueError('Missing observed dividend/rate')
        delta = 0.0
        if put_qty:
            po = quote('PUT', put_code, date)
            if not np.isfinite(po.iv) or po.iv <= 0:
                raise ValueError('Missing observed IV')
            delta = put_delta(S, float(po.strike), float(po.iv), (po.expiry_date - date).days / 365, r, q)
        net_beta = (float(b.base_nav) * float(b.beta_current) + put_qty * mult * S * delta - (future_qty * mult * float(quote('FUTURES', future_code, date).close) if future_qty else 0)) / nav
        cap = cfg['beta_caps'][state.level - 1] if state.level else np.nan
        initial, maintenance = (margin(), margin(maintenance=True))
        breach = bool(future_qty and available() < maintenance - 1e-05)
        next_day = f.index[i + 1] if i + 1 < len(f) else None
        if breach:
            price = float(quote('FUTURES', future_code, date).close)
            fee1 = costs.fee('FUTURES', 1, price, mult, date)
            target = max(0, min(future_qty, floor((available() - future_qty * fee1) / max(margin(1) - fee1, 1e-12))))
            event('MARGIN_CALL', initial_required=initial, maintenance_required=maintenance, available_cash=available(), target_short=target)
            if next_day is not None:
                margin_pending = {'date': next_day, 'signal': date, 'target': target, 'code': future_code}
        navs.append({'date': date, 'strategy_id': strategy, 'run_id': run_id, 'scenario': scenario, 'nav': nav, 'base_nav': float(b.base_nav), 'base_cash': float(b.base_cash), 'overlay_cash_delta': cash_delta, 'total_cash': float(b.base_cash) + cash_delta, 'hedge_cash_available': available(), 'stock_value': float(b.base_nav) - float(b.base_cash), 'put_value': put_value(), 'margin_initial_required': initial, 'margin_maintenance_required': maintenance, 'margin_locked': min(max(available(), 0), initial), 'free_hedge_cash': max(0, available() - initial), 'put_code': put_code, 'put_quantity': put_qty, 'put_delta': delta, 'future_code': future_code, 'futures_short_quantity': future_qty, 'net_stock_pnl': stock_pnl, 'put_pnl': put_pnl, 'futures_pnl': future_pnl, 'base_cost': float(b.base_cost), 'overlay_cost': cost_today, 'base_drawdown': dd, 'risk_level': state.level, 'risk_level_reason': reason, 'net_beta': net_beta, 'beta_cap': cap, 'risk_limit_exceeded': bool(state.level and net_beta > cap + 1e-10), 'negative_beta': net_beta < 0, 'margin_call': breach, 'blocked_orders': blocked, 'q': q, 'q_status': macro.q_status, 'self_financing_error': error, 'margin_date_status': rate_row().eod_date_status if future_qty else 'NO_FUTURES_POSITION'})
        last_put = 0.0 if not put_qty else float(quote('PUT', put_code, date).close)
        last_future = 0.0 if not future_qty else float(quote('FUTURES', future_code, date).close)
        prev_nav, prev_base = (nav, float(b.base_nav))
        if date in schedule:
            candidate_put = schedule[date].get('PUT', candidate_put)
            candidate_future = schedule[date].get('FUTURES', candidate_future)
        if next_day is not None and mode != 'NONE' and candidate_put and (mode == 'PUT' or candidate_future):
            po = quote('PUT', candidate_put, date)
            if po.expiry_date <= next_day or po.close <= 0 or po.volume <= 0 or (po.open_interest <= 0) or (not np.isfinite(po.iv)) or (po.iv <= 0):
                event('SIGNAL_OPTION_UNUSABLE', contract_code=candidate_put)
                continue
            resize = candidate_put != put_code or (cfg['put_resize_on_equity_rebalance'] and bool(b.equity_rebalance_signal))
            notional = max(0.0, float(b.base_nav) * float(b.beta_forecast))
            target_put = floor(cfg['put_coverage'] * notional / (S * mult)) if resize else put_qty
            pdelt = put_delta(S, float(po.strike), float(po.iv), (po.expiry_date - date).days / 365, r, q)
            target_future = 0
            if mode == 'PUT_FUTURES' and state.level:
                fo = quote('FUTURES', candidate_future, date)
                if fo.close > 0 and fo.volume > 0 and (fo.open_interest > 0):
                    remaining = max(0.0, notional + target_put * S * mult * pdelt)
                    target_future = min(ceil(max(0.0, remaining - cap * nav) / (float(fo.close) * mult)), floor(remaining / (float(fo.close) * mult)))
                else:
                    event('SIGNAL_FUTURE_UNUSABLE', contract_code=candidate_future)
            pending = {'signal_date': date, 'execution_date': next_day, 'put_code': candidate_put, 'put_target': target_put, 'future_code': candidate_future, 'future_target': target_future}
    daily = pd.DataFrame(navs)
    rets = daily.nav.pct_change(fill_method=None)
    daily['volatility_60d'] = rets.rolling(cfg['vol_window']).std(ddof=1) * np.sqrt(252)

    def tail(a):
        losses = -a
        n = max(1, ceil((1 - cfg['es_confidence']) * len(a)))
        return np.sort(losses)[-n:].mean()
    daily['es_95_252d'] = rets.rolling(cfg['es_window']).apply(tail, raw=True)
    stresses = []
    for shock in [-0.1, -0.2, -0.3]:
        for iv_change in [0.1, 0.2]:
            if put_qty:
                po = quote('PUT', put_code, date)
                pc = {'underlying_price': S, 'iv': float(po.iv), 'days_to_expiry': (po.expiry_date - date).days, 'strike': float(po.strike), 'multiplier': mult}
                v = stress_test(float(b.base_nav) * float(b.beta_current), put_qty, pc, future_qty, last_future, mult, r, q, shock, iv_change)
            else:
                sp = float(b.base_nav) * float(b.beta_current) * shock
                fp = -future_qty * mult * last_future * shock
                v = {'stock_pnl': sp, 'option_pnl': 0.0, 'futures_pnl': fp, 'total_pnl': sp + fp}
            stresses.append({'run_id': run_id, 'date': date, 'index_shock': shock, 'iv_change_decimal': iv_change, **v, 'nav_shock': v['total_pnl'] / nav})
    tc = ['date', 'signal_date', 'run_id', 'instrument', 'contract_code', 'quantity_change', 'price', 'multiplier', 'cost', 'reason', 'cost_source']
    return OverlayResult(daily, pd.DataFrame(trades, columns=tc), pd.DataFrame(events) if events else pd.DataFrame(columns=['date', 'run_id', 'reason']), pd.DataFrame(stresses), {'funding': 'cash_already_in_upstream_NAV', 'risk_config': cfg, 'margin': 'standalone underlying notional times published rate; no COMS offsets', 'futures_pnl': 'observed_close_to_close; not official settlement cash ledger'})
