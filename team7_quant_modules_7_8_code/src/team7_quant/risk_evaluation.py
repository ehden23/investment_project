"""Module 8: input readiness and performance reports, without assumed costs."""
from pathlib import Path
import base64
import json
from math import ceil
import numpy as np
import pandas as pd
from .risk_data import RiskMarket
from .risk_engine import DEFAULT_RISK, run_overlay
from .risk_inputs import CostBook, attach_cash_limits

def performance(nav, index, cd_rate):
    nav = nav.sort_index().astype(float)
    if len(nav) < 2 or nav.index.has_duplicates or nav.isna().any() or (nav <= 0).any():
        raise ValueError('Invalid NAV series')
    r = nav.pct_change(fill_method=None).iloc[1:]
    rf = (cd_rate.reindex(nav.index).shift(1) * nav.index.to_series().diff().dt.days / 365).iloc[1:]
    b = index.reindex(nav.index)
    if rf.isna().any() or b.isna().any() or (b <= 0).any():
        raise ValueError('Missing RF/benchmark')
    br = b.pct_change(fill_method=None).iloc[1:]
    active = r - br
    excess = r - rf
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    if years <= 0:
        raise ValueError('Invalid evaluation dates')
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    bcagr = (b.iloc[-1] / b.iloc[0]) ** (1 / years) - 1
    beta = r.cov(br) / br.var(ddof=1) if br.var(ddof=1) > 0 else np.nan
    mex = br - rf
    cbeta = excess.cov(mex) / mex.var(ddof=1) if mex.var(ddof=1) > 0 else np.nan
    return {'total_return': nav.iloc[-1] / nav.iloc[0] - 1, 'cagr': cagr, 'ann_vol': r.std(ddof=1) * np.sqrt(252), 'sharpe': excess.mean() / excess.std(ddof=1) * np.sqrt(252) if excess.std(ddof=1) > 0 else np.nan, 'mdd': (nav / nav.cummax() - 1).min(), 'benchmark_cagr': bcagr, 'cagr_minus_benchmark': cagr - bcagr, 'active_total_return': nav.iloc[-1] / nav.iloc[0] - b.iloc[-1] / b.iloc[0], 'tracking_error': active.std(ddof=1) * np.sqrt(252), 'information_ratio': active.mean() / active.std(ddof=1) * np.sqrt(252) if active.std(ddof=1) > 0 else np.nan, 'market_beta_realized': beta, 'capm_beta_excess_returns': cbeta, 'capm_alpha_annual_arithmetic': (excess.mean() - cbeta * mex.mean()) * 252}

def check_risk_inputs(prepared, cfg, baseline=None):
    folder = Path(prepared)
    issues = []
    facts = {}
    market = RiskMarket.load(folder)
    f = pd.read_csv(folder / 'baseline_features.csv', parse_dates=['date'])
    risk = {**DEFAULT_RISK, **cfg.get('risk', {})}
    try:
        costs = CostBook.from_config(risk)
        costs.at(sorted(f.date.unique())[min(1, f.date.nunique() - 1)])
    except (ValueError, FileNotFoundError) as e:
        issues.append({'code': 'DERIVATIVE_COSTS_REQUIRED', 'detail': str(e)})
    try:
        f = attach_cash_limits(f, risk.get('cash_file'))
        for strategy, g in f.groupby('strategy_id'):
            live = g[g.date > g.date.min()]
            first_execution = market.schedule.execution_date.min()
            initial = live[live.date == first_execution]
            if live.empty or live.hedge_cash_available.max() <= 1e-05 or (not initial.empty and initial.hedge_cash_available.max() <= 1e-05):
                issues.append({'code': 'UPSTREAM_HEDGE_CASH_REQUIRED', 'strategy': strategy, 'detail': '최초 헤지 체결일의 가용 현금이 0입니다. 일부 거래불가 잔액과 별개로, 팀원이 헤지 현금을 남긴 NAV·보유·목표비중으로 갱신해야 합니다.'})
        facts['cash_policy'] = 'already included in upstream NAV; no external contribution or equity liquidation'
    except (ValueError, FileNotFoundError) as e:
        issues.append({'code': 'INVALID_CASH_INPUT', 'detail': str(e)})
    if baseline is not None:
        p = Path(baseline) / 'daily_nav.csv'
        if p.exists():
            b = pd.read_csv(p, parse_dates=['date'])
            z = f.merge(b[['date', 'strategy_id', 'nav', 'cash']], on=['date', 'strategy_id'], how='left', validate='one_to_one')
            if z[['nav', 'cash']].isna().any().any() or not np.allclose(z.base_nav, z.nav, atol=0.0001, rtol=1e-10) or (not np.allclose(z.base_cash, z.cash, atol=0.0001, rtol=1e-10)):
                issues.append({'code': 'STALE_BASELINE_FEATURES', 'detail': 'baseline_features와 기초 NAV/현금이 다릅니다. prepare-risk-data를 다시 실행하세요.'})
    dates = pd.DatetimeIndex(sorted(f.date.unique()))
    for label, df, cols in [('macro', market.macro, ['q', 'r_continuous', 'cd_rate']), ('index', market.index, ['close'])]:
        d = df.reindex(dates)
        if d[cols].isna().any().any() or not np.isfinite(d[cols].to_numpy(float)).all():
            issues.append({'code': 'MISSING_' + label.upper(), 'detail': '일별 필수 자료 누락'})
    live_dates = dates[1:]
    if market.margin is None or market.margin.reindex(live_dates)[['eod_initial_rate', 'eod_maintenance_rate']].isna().any().any():
        issues.append({'code': 'MARGIN_HISTORY_REQUIRED', 'detail': '정리한 margin_rates.csv를 연결하세요.'})
    if market.metadata.get('dividend_policy') != 'observed_index_yield':
        issues.append({'code': 'OBSERVED_DIVIDENDS_REQUIRED', 'detail': '연간 배당 대용치를 일별 KRX 배당 자료로 교체하세요.'})
    if f.beta_fallback_count.gt(0).any():
        issues.append({'code': 'BETA_HISTORY_REQUIRED', 'detail': 'beta=1 대체가 포함돼 있습니다.'})
    facts.update({'strategies': sorted(f.strategy_id.unique()), 'feature_rows': len(f), 'put_contracts': int(market.options.contract_code.nunique()), 'futures_contracts': int(market.futures.contract_code.nunique()), 'margin_days': 0 if market.margin is None else len(market.margin), 'dividend_policy': market.metadata.get('dividend_policy')})
    return {'ready_for_hedge_backtest': not issues, 'issues': issues, 'facts': facts}

def run_risk_evaluation(prepared, out, cfg, baseline=None, *, baseline_only=False):
    prepared, out = (Path(prepared), Path(out))
    check = check_risk_inputs(prepared, cfg, baseline)
    if not baseline_only and (not check['ready_for_hedge_backtest']):
        raise ValueError('헤지 백테스트 입력 미완료:\n' + '\n'.join((x['code'] + ': ' + x['detail'] for x in check['issues'])))
    market = RiskMarket.load(prepared)
    features = pd.read_csv(prepared / 'baseline_features.csv', parse_dates=['date'])
    risk = {**DEFAULT_RISK, **cfg.get('risk', {})}
    scenarios = [('NO_HEDGE', 'NONE')] if baseline_only else [('NO_HEDGE', 'NONE'), ('PUT_ONLY', 'PUT'), ('PUT_FUTURES', 'PUT_FUTURES')]
    all_nav = []
    all_trades = []
    all_events = []
    all_stress = []
    summary = []
    for strategy, f in features.groupby('strategy_id', sort=True):
        for name, mode in scenarios:
            print(f'{strategy}: {name}', flush=True)
            result = run_overlay(f, market, cfg=risk, mode=mode, scenario=name)
            d = result.nav
            all_nav.append(d)
            all_trades.append(result.trades)
            all_events.append(result.events)
            all_stress.append(result.stress)
            n = d.set_index('date').nav
            summary.append({'strategy_id': strategy, 'scenario': name, 'run_id': d.run_id.iloc[0], **performance(n, market.index.close, market.macro.cd_rate), 'base_cost_sum': d.base_cost.sum(), 'overlay_cost_sum': d.overlay_cost.sum(), 'put_pnl_sum': d.put_pnl.sum(), 'futures_pnl_sum': d.futures_pnl.sum(), 'blocked_order_days': int(d.blocked_orders.gt(0).sum()), 'margin_call_days': int(d.margin_call.sum()), 'risk_limit_exceeded_days': int(d.risk_limit_exceeded.sum()), 'negative_beta_days': int(d.negative_beta.sum()), 'max_margin_nav_ratio': float((d.margin_initial_required / d.nav).max()), 'max_self_financing_error': float(d.self_financing_error.abs().max())})
    out.mkdir(parents=True, exist_ok=True)
    daily = pd.concat(all_nav, ignore_index=True)
    summary = pd.DataFrame(summary)
    for name, items in [('risk_trades', all_trades), ('risk_events', all_events), ('stress_scenarios', all_stress)]:
        nonempty = [x for x in items if not x.empty]
        (pd.concat(nonempty, ignore_index=True) if nonempty else items[0]).to_csv(out / f'{name}.csv', index=False)
    daily.to_csv(out / 'risk_daily_nav.csv', index=False)
    summary.to_csv(out / 'performance_summary.csv', index=False)
    annual = []
    for run_id, g in daily.groupby('run_id'):
        x = g.set_index('date').nav
        previous = x.iloc[0]
        for year, y in x.groupby(x.index.year):
            if year < int(cfg.get('backtest', {}).get('evaluation_start', '2016')[:4]):
                continue
            annual.append({'run_id': run_id, 'year': year, 'return': y.iloc[-1] / previous - 1})
            previous = y.iloc[-1]
    pd.DataFrame(annual).to_csv(out / 'annual_returns.csv', index=False)
    dates = pd.DatetimeIndex(sorted(daily.date.unique()))
    benchmark = market.index.close.reindex(dates)
    pd.DataFrame({'date': dates, 'kospi200_close': benchmark.values, 'nav_100': 100 * benchmark.values / benchmark.iloc[0]}).to_csv(out / 'kospi200_benchmark.csv', index=False)
    a = daily.groupby(['strategy_id', 'scenario'])[['net_stock_pnl', 'put_pnl', 'futures_pnl', 'base_cost', 'overlay_cost']].sum().reset_index()
    a['nav_change_reconciled'] = a.net_stock_pnl + a.put_pnl + a.futures_pnl - a.overlay_cost
    a.to_csv(out / 'pnl_attribution.csv', index=False)
    regime_path = Path(baseline) / 'monthly_regimes.csv' if baseline is not None else None
    if regime_path is not None and regime_path.exists():
        reg = pd.read_csv(regime_path, parse_dates=['date'])
        rows = []
        for run_id, g in daily.groupby('run_id'):
            z = g.sort_values('date').copy()
            z['daily_return'] = z.nav.pct_change(fill_method=None)
            z = pd.merge_asof(z, reg[['date', 'confirmed_regime']].sort_values('date'), on='date', direction='backward', allow_exact_matches=False)
            for label, s in z.groupby('confirmed_regime', dropna=False):
                v = s.daily_return.dropna()
                rows.append({'run_id': run_id, 'regime': label, 'sessions': len(v), 'mean_daily_return': v.mean(), 'ann_vol': v.std(ddof=1) * np.sqrt(252), 'linked_selected_day_return': (1 + v).prod() - 1})
        pd.DataFrame(rows).to_csv(out / 'regime_analysis.csv', index=False)
    notes = {'status': 'BASELINE_ONLY_NO_HEDGE_EVALUATION' if baseline_only else 'CASH_CONSTRAINED_OVERLAY_RESEARCH', 'input_check': check, 'risk_config': risk, 'market_metadata': market.metadata, 'limitations': ['Historical margin rates include user-authorized inferred effective dates, labelled in the input.', 'Margin is standalone futures underlying-notional times published rate, without COMS offsets or intraday calls.', 'Futures P&L and fills use observed closes, not independently verified official settlement prices.', 'BSM delta uses observed historical KRX index dividend yield and CD91 proxy; forward dividends and an interest-rate curve are not claimed.', 'Current-constituent upstream equity data omit delisted history; survivorship bias remains.', 'Upstream revised financial/macro inputs and their timing conventions are inherited.', 'Only provided put strike candidates are available; optimality over a complete historical chain is not established.', 'Margin call detected at a close queues reduction at the next close; unmet calls remain reported.', 'Initial and terminal values are marked NAV; no forced final liquidation.']}
    (out / 'methodology_and_limits.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding='utf-8')
    make_report(daily, summary, benchmark, out, baseline_only)
    return summary

def make_report(daily, summary, benchmark, out, baseline_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = list(daily.groupby('strategy_id'))
    cols = min(2, len(groups))
    rows = ceil(len(groups) / cols)
    for name, dd in [('nav_comparison', False), ('drawdown_comparison', True)]:
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.7 * rows), squeeze=False, constrained_layout=True)
        for ax, (strategy, g) in zip(axes.flat, groups):
            for scenario, z in g.groupby('scenario'):
                x = z.set_index('date').nav
                y = 100 * (x / x.cummax() - 1) if dd else x / x.iloc[0]
                ax.plot(x.index, y, label=scenario, lw=1)
            y = 100 * (benchmark / benchmark.cummax() - 1) if dd else benchmark / benchmark.iloc[0]
            ax.plot(benchmark.index, y, 'k--', lw=0.8, label='KOSPI200 PR')
            ax.set_title(strategy)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        fig.savefig(out / f'{name}.png', dpi=140)
        plt.close(fig)
    t = summary[['strategy_id', 'scenario', 'cagr', 'ann_vol', 'sharpe', 'mdd', 'blocked_order_days', 'margin_call_days']].copy()
    for col in ['cagr', 'ann_vol', 'mdd']:
        t[col] = t[col].map(lambda x: f'{x:.2%}')
    t.sharpe = t.sharpe.map(lambda x: f'{x:.3f}')
    title = '기존 무헤지 전략 성과' if baseline_only else '모듈 7·8 현금 제약 헤지 비교'
    notice = '기존 NAV 평가만 수행했습니다. 이 문서에는 새 헤지 성과가 없습니다.' if baseline_only else '기초 NAV에 이미 포함된 현금 안에서만 헤지합니다. 비용은 명시 입력값이며, 증거금은 게시 요율에 의한 단독 선물 한도입니다.'
    doc = f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>{title}</title><style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:35px auto;padding:0 20px;line-height:1.7;color:#172d45}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}img{{width:100%}}.note{{padding:16px;background:#fff1d8}}</style><h1>{title}</h1><p class="note">{notice}</p>' + t.to_html(index=False, escape=True)
    for name in ['nav_comparison', 'drawdown_comparison']:
        data = base64.b64encode((out / f'{name}.png').read_bytes()).decode()
        doc += f'<img alt="{name}" src="data:image/png;base64,{data}">'
    doc += '<p>일별 현금·보험료·선물손익·증거금·주문 실패·한도 초과는 CSV에서 확인할 수 있습니다. 다른 월의 추정 시행일은 입력 파일에 표시돼 있습니다. 배당은 관측 지수 배당수익률을 사용한 모형 입력이며 만기별 예상 배당률은 아닙니다. 상장폐지 종목 누락에 따른 생존편향과 종가 기준 체결·선물 손익의 한계가 남습니다. 세부 정의는 methodology_and_limits.json을 참고하세요.</p></html>'
    (out / 'report.html').write_text(doc, encoding='utf-8')
