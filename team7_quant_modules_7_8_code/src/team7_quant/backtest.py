from __future__ import annotations

"""모듈 2(선정)·모듈 4(가중치) 결과를 모듈 5(목표비중)·모듈 6(체결·비용)에 연결하는 일별 백테스트 엔진.

시점 규칙 (미래 정보 차단의 핵심)
- 신호일 s = 달력상 월말. 신호에 쓰는 정보는 s 이하의 마지막 거래일(cutoff) 종가까지다.
- 체결일 = s 다음 첫 거래일. 체결가격은 그날 종가(시가 데이터가 없어서 module6 주석의 '시가' 대신 종가).
- 체결일에는 먼저 그날 종가로 기존 보유를 평가한 뒤, 같은 종가로 리밸런싱한다.
- 거래량이 0인 종목은 그날 체결 불가(module6의 executable=False). 재시도 없이 다음 리밸런싱까지 둔다.
- 현금 이자는 0(무위험수익률 자료 없음). 무헤지 NAV이며 배당은 수정주가에 포함된 범위만 반영된다.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .daily_panel import DailyPanel
from .module2 import run_rebalance_dates
from .module34 import adapt_weights_for_module2
from .module5 import build_targets
from .module6 import full_rebalance
from .pipeline24 import _module2_prices, _sector_history


# 전략 정의: (선정 계열, 배분 방식). 선정 계열이 같으면 같은 30종목을 쓴다.
STRATEGIES: dict[str, tuple[str, str]] = {
    "STATIC_EW": ("STATIC", "EW"),
    "STATIC_BL": ("STATIC", "BL_ALPHA_003"),
    "ROTATION_EW": ("ROTATION", "EW"),
    "ROTATION_BL": ("ROTATION", "BL_ALPHA_003"),
}


# ---------------------------------------------------------------------------
# 1. 거래일 달력 보조함수
# ---------------------------------------------------------------------------

def signal_cutoff(sessions: pd.DatetimeIndex, signal_date: pd.Timestamp) -> pd.Timestamp:
    """신호일 s 이하의 마지막 거래일. 신호 계산은 이 날 종가까지만 본다."""
    pos = sessions.searchsorted(pd.Timestamp(signal_date), side="right") - 1
    if pos < 0:
        raise ValueError(f"{signal_date} 이전 거래일이 없습니다.")
    return sessions[pos]


def execution_session(sessions: pd.DatetimeIndex, signal_date: pd.Timestamp) -> pd.Timestamp | None:
    """신호일 s 다음 첫 거래일(체결일). 자료 마지막 날 이후 신호면 None."""
    pos = sessions.searchsorted(pd.Timestamp(signal_date), side="right")
    return sessions[pos] if pos < len(sessions) else None


# ---------------------------------------------------------------------------
# 2. 모듈 2 선정 결과 준비 (정적 / 국면 로테이션)
# ---------------------------------------------------------------------------

def _read_csv(path: Path, dates: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_code": str})
    for col in dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


@dataclass
class ModuleInputs:
    """pipeline24 출력 폴더에서 다시 읽은 모듈 2 입력과 모듈 3·4 결과."""

    universe: pd.DataFrame
    data: dict[str, pd.DataFrame]
    monthly_weights: pd.DataFrame
    annual_weights: pd.DataFrame
    regimes: pd.DataFrame
    pipeline_selections: pd.DataFrame


def load_module_outputs(modules_out: str | Path) -> ModuleInputs:
    """pipeline24(run-modules-2-4) 출력물을 읽는다. 모듈 1~4를 다시 계산하지 않는다."""
    d = Path(modules_out)
    market = _read_csv(d / "normalized" / "monthly_market.csv", ["date"])
    master = _read_csv(d / "normalized" / "stock_master.csv", ["listing_date", "delisting_date"])
    financials = _read_csv(d / "normalized" / "financials.csv", ["fiscal_month_end", "available_at"])
    data = {"prices": _module2_prices(market), "financials": financials, "sectors": _sector_history(master)}
    return ModuleInputs(
        universe=_read_csv(d / "approximate_universe.csv", ["date"]),
        data=data,
        monthly_weights=_read_csv(d / "monthly_factor_weights.csv", ["date"]),
        annual_weights=_read_csv(d / "annual_regime_weights.csv", []),
        regimes=_read_csv(d / "monthly_regimes.csv", ["date"]),
        pipeline_selections=_read_csv(d / "module2_selections.csv", ["date"]),
    )


def initial_rotation_weights(first_signal: pd.Timestamp, inputs: ModuleInputs, regime_cfg: dict[str, Any]) -> dict[str, Any]:
    """첫 신호(2015-12-31)의 로테이션 가중치.

    모듈 4의 월별 가중치는 2016년부터만 있으므로, 첫 신호일에 확정된 국면에
    '다음 해(2016) 가중치표'를 적용한다. 2016년 표의 학습 구간은 2009~2015년이고
    학습 표본은 2015-12-31까지 공개된 수익률만 쓰므로 신호일 기준 미래 정보가 아니다.
    국면이 미확정이면 모듈 4 규칙과 같게 0.5:0.5 중립을 쓴다.
    """
    row = inputs.regimes[inputs.regimes["date"] == first_signal]
    if row.empty:
        raise ValueError(f"{first_signal.date()} 국면 판단 결과가 없습니다.")
    regime = row.iloc[0]["confirmed_regime"]
    year = first_signal.year + 1
    if pd.isna(regime):
        relative = {"w_V": 0.5, "w_Q": 0.5, "weight_source": "preparation_neutral"}
    else:
        table = inputs.annual_weights
        hit = table[(table["application_year"] == year) & (table["regime"] == regime)]
        if hit.empty:
            raise ValueError(f"{year}년 {regime} 가중치표가 없습니다.")
        relative = {"w_V": float(hit.iloc[0]["w_V"]), "w_Q": float(hit.iloc[0]["w_Q"]),
                    "weight_source": f"annual_regime_table_{year}"}
    out = adapt_weights_for_module2({"date": first_signal, "confirmed_regime": regime, **relative}, regime_cfg)
    return out


def build_selection_schedule(inputs: ModuleInputs, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """선정 계열별 신호일·팩터 가중치·선정 종목을 만든다.

    - STATIC: 정적 가중치(config), 첫 신호 + 분기말(3·6·9·12월).
    - ROTATION: 모듈 4 월별 가중치, 첫 신호 + (분기말 ∪ 국면 전환 달). pipeline24와 같은 규칙.
    """
    bt = cfg["backtest"]
    first = pd.Timestamp(bt["first_signal_date"])
    end = pd.Timestamp(bt["evaluation_end"])
    months = set(bt.get("static_rebalance_months", [3, 6, 9, 12]))

    # 정적: 매 신호일에 같은 가중치를 명시적으로 넘긴다(module2 내부 기본값에 의존하지 않음).
    static_dates = [first] + [d for d in pd.date_range(first + pd.offsets.MonthEnd(1), end, freq="ME") if d.month in months]
    sw = bt["static_weights"]
    static_weights = pd.DataFrame({"date": static_dates, "w_V": sw["value"], "w_Q": sw["quality"], "w_M": sw["momentum"]})

    # 로테이션: pipeline24와 같은 마스크(분기말 또는 국면 전환)로 신호일을 고른다.
    mw = inputs.monthly_weights.copy()
    mw = mw[(mw["date"] > first) & (mw["date"] <= end)]
    regime_changed = mw["regime_changed"].astype(str).str.lower().eq("true")
    mask = mw["date"].dt.month.isin(cfg["factor"].get("rebalance_months", [3, 6, 9, 12])) | regime_changed
    first_row = initial_rotation_weights(first, inputs, cfg["regime"])
    rotation_weights = pd.concat([
        pd.DataFrame([{"date": first, "w_V": first_row["w_V"], "w_Q": first_row["w_Q"], "w_M": first_row["w_M"],
                       "confirmed_regime": first_row["confirmed_regime"]}]),
        mw.loc[mask, ["date", "w_V", "w_Q", "w_M", "confirmed_regime"]],
    ], ignore_index=True)

    schedule: dict[str, dict[str, Any]] = {}
    for family, weights in [("STATIC", static_weights), ("ROTATION", rotation_weights)]:
        result = run_rebalance_dates(list(weights["date"]), inputs.universe, inputs.data, weights, cfg["factor"])
        schedule[family] = {"weights": weights, "selections": result["selections"], "logs": result["logs"]}
    return schedule


def compare_with_pipeline(rotation_selections: pd.DataFrame, pipeline_selections: pd.DataFrame) -> dict[str, Any]:
    """엔진이 다시 계산한 로테이션 선정이 pipeline24 결과와 같은지 검사(재현성 확인)."""
    common = sorted(set(rotation_selections["date"]) & set(pipeline_selections["date"]))
    mismatched = []
    for d in common:
        a = set(rotation_selections.loc[rotation_selections["date"] == d, "stock_code"])
        b = set(pipeline_selections.loc[pipeline_selections["date"] == d, "stock_code"])
        if a != b:
            mismatched.append(str(pd.Timestamp(d).date()))
    return {"common_dates": len(common), "mismatched_dates": mismatched}


# ---------------------------------------------------------------------------
# 3. 모듈 5 연결: 신호일별 목표비중
# ---------------------------------------------------------------------------

def common_history_length(returns: pd.DataFrame, codes: list[str], cutoff: pd.Timestamp, max_len: int) -> int:
    """cutoff까지 30종목 모두 수익률이 있는 '끝에서부터 연속된' 거래일 수(최대 max_len)."""
    window = returns.loc[:cutoff, codes].tail(max_len)
    complete = window.notna().all(axis=1).to_numpy()
    # 끝에서부터 처음으로 결측이 나오기 전까지의 길이
    broken = np.flatnonzero(~complete[::-1])
    return int(broken[0]) if len(broken) else int(len(complete))


def build_signal_targets(
    signal_date: pd.Timestamp, selections: pd.DataFrame, panel: DailyPanel, returns: pd.DataFrame,
    alloc_cfg: dict[str, Any],
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    """한 신호일의 EW·BL 목표비중을 만든다. cutoff(신호일 이하 마지막 거래일) 이후 자료는 쓰지 않는다.

    module5.build_targets는 결측 없는 252×30 수익률을 요구하므로 엔진에서 미리 처리한다.
    - 공통 이력 L ≥ 252: 그대로 252일
    - min_lookback ≤ L < 252: lookback을 L로 줄여 호출 (SHORTENED)
    - L < min_lookback 또는 module5 오류: BL을 동일가중으로 대체 (EW_FALLBACK)
    값을 만들어 채우지 않으며, 모든 처리 내역을 log로 돌려준다.
    """
    signal_date = pd.Timestamp(signal_date)
    cutoff = signal_cutoff(panel.adj_close.index, signal_date)
    sel = selections[selections["date"] == signal_date].sort_values("rank")
    codes = sorted(sel["stock_code"].astype(str))
    n_expected = int(alloc_cfg["selected_count"])
    full_len = int(alloc_cfg["lookback_sessions"])
    min_len = int(alloc_cfg["min_lookback_sessions"])
    log: dict[str, Any] = {"signal_date": signal_date, "cutoff_session": cutoff, "n_selected": len(codes)}

    if not codes:
        raise ValueError(f"{signal_date.date()} 선정 종목이 없습니다.")
    ew = pd.Series(1.0 / len(codes), index=codes)
    missing_px = [c for c in codes if c not in panel.adj_close.columns]
    if missing_px:
        raise ValueError(f"{signal_date.date()} 일별 주가가 없는 선정 종목: {missing_px}")

    available = common_history_length(returns, codes, cutoff, full_len)
    log["available_sessions"] = available
    if len(codes) != n_expected:
        log.update({"bl_status": "EW_FALLBACK", "lookback_used": 0, "reason": f"selected_{len(codes)}_not_{n_expected}"})
        return {"EW": ew, "BL_ALPHA_003": ew.copy()}, log
    if available < min_len:
        log.update({"bl_status": "EW_FALLBACK", "lookback_used": 0,
                    "reason": f"common_history_{available}_lt_{min_len}"})
        return {"EW": ew, "BL_ALPHA_003": ew.copy()}, log

    lookback = min(available, full_len)
    # module5 입력 형식(long)으로 변환: 결측 없는 마지막 lookback 거래일만 넘긴다.
    block = returns.loc[:cutoff, codes].tail(lookback)
    risk_returns = block.stack().rename("return").reset_index()
    risk_returns.columns = ["date", "stock_code", "return"]
    # module5는 시가총액을 date == signal_date로 찾으므로 cutoff 값에 신호일 날짜를 붙인다.
    caps = panel.market_cap.loc[cutoff, codes]
    market_caps = pd.DataFrame({"date": signal_date, "stock_code": codes, "market_cap": caps.to_numpy()})
    m5_cfg = {**alloc_cfg, "lookback_sessions": lookback}
    try:
        result = build_targets(signal_date, sel, risk_returns, market_caps, m5_cfg)
    except (ValueError, RuntimeError) as exc:
        log.update({"bl_status": "EW_FALLBACK", "lookback_used": lookback, "reason": f"module5_error: {exc}"})
        return {"EW": ew, "BL_ALPHA_003": ew.copy()}, log

    t = result.targets
    targets = {aid: g.set_index("stock_code")["target_weight"].astype(float) for aid, g in t.groupby("allocation_id")}
    diag = result.diagnostics
    log.update({
        "bl_status": "OK" if lookback == full_len else "SHORTENED",
        "lookback_used": lookback,
        "reason": "" if lookback == full_len else f"common_history_{available}_lt_{full_len}",
        "rbar": diag["rbar"], "shrinkage": diag["shrinkage"],
        "optimizer_backend": diag["optimizer"]["backend"], "bl_max_weight": diag["optimizer"]["max_weight"],
    })
    return targets, log


# ---------------------------------------------------------------------------
# 4. 모듈 6 연결: 일별 NAV 시뮬레이션
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    daily_nav: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    untradable: pd.DataFrame
    rebalance_count: int = 0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


def run_strategy(
    strategy_id: str, targets_by_signal: dict[pd.Timestamp, pd.Series], panel: DailyPanel,
    exec_cfg: dict[str, Any], initial_capital: float, end_date: pd.Timestamp,
) -> StrategyResult:
    """한 전략의 일별 NAV를 계산한다.

    상태는 '종목별 보유 주식수(수정주가 기준)'와 현금이다.
    매 거래일: (1) 그날 종가로 평가 → (2) 체결일이면 module6.full_rebalance로 목표비중까지 거래
    → (3) 거래 후 NAV·현금·회전율·비용·보유비중 기록.
    """
    px = panel.adj_close
    sessions = px.index
    vol = panel.volume
    # 신호일 → 체결일 매핑 (체결일이 자료 범위를 벗어난 신호는 버린다)
    orders: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Series]] = {}
    for s, w in sorted(targets_by_signal.items()):
        exec_day = execution_session(sessions, s)
        if exec_day is not None and exec_day <= end_date:
            orders[exec_day] = (s, w)
    if not orders:
        raise ValueError("체결 가능한 신호가 없습니다.")
    first_exec = min(orders)
    base_day = sessions[sessions.get_loc(first_exec) - 1]
    days = sessions[(sessions >= first_exec) & (sessions <= end_date)]

    shares = pd.Series(dtype=float)   # 종목코드 → 보유 수량(수정주가 단위)
    cash = float(initial_capital)
    last_px: dict[str, float] = {}
    nav_rows = [{"date": base_day, "strategy_id": strategy_id, "nav": cash, "cash": cash, "turnover": 0.0, "cost": 0.0}]
    hold_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    untradable_rows: list[dict[str, Any]] = []
    diags: list[dict[str, Any]] = []

    for day in days:
        # (1) 평가: 보유 종목 종가. 종가가 없으면(이 자료에선 없어야 정상) 직전 종가로 평가하고 기록.
        prices_today = px.loc[day, shares.index] if len(shares) else pd.Series(dtype=float)
        for code in prices_today.index[prices_today.isna()]:
            prices_today[code] = last_px[code]
            untradable_rows.append({"date": day, "strategy_id": strategy_id, "stock_code": code,
                                    "side": "VALUATION", "intended_value": np.nan, "reason": "missing_price_carried"})
        last_px.update(prices_today.to_dict())
        values = shares * prices_today
        turnover = cost = 0.0

        # (2) 체결일이면 리밸런싱
        if day in orders:
            signal_date, target = orders[day]
            ids = values.index.union(target.index)
            px_ids = px.loc[day, ids]
            executable = (vol.loc[day, ids].fillna(0) > 0) & px_ids.notna()
            pre_nav = float(values.sum() + cash)
            batch = full_rebalance(values, cash, target, executable, exec_cfg)
            post_nav = float(batch.final_values.sum() + batch.final_cash)
            cost = float(batch.costs["total"])
            # 비용 차감 후 NAV 보존: 거래 전 NAV − 비용 = 거래 후 NAV
            if abs(pre_nav - cost - post_nav) > 1e-6 * pre_nav:
                raise RuntimeError(f"{strategy_id} {day.date()} NAV 보존 위반: {pre_nav - cost - post_nav}")
            if not batch.trades.empty:
                traded = float(batch.trades["reference_notional"].sum())
                turnover = traded / (2.0 * pre_nav)
                tr = batch.trades
                trade_rows.append(pd.DataFrame({
                    "date": day, "strategy_id": strategy_id, "stock_code": tr["stock_code"].astype(str),
                    "side": tr["side"], "value": tr["reference_notional"],
                    "cost": tr["commission"] + tr["tax"] + tr["slippage"],
                }))
            # 체결 불가 로그: 목표와 현재 평가액 차이(=하려던 거래 금액)
            x = values.reindex(ids, fill_value=0.0)
            w = target.reindex(ids, fill_value=0.0)
            intended = w * pre_nav - x
            for code in ids[~executable.to_numpy()]:
                if abs(intended[code]) > 1.0:  # 1원 미만 차이는 무시
                    untradable_rows.append({
                        "date": day, "strategy_id": strategy_id, "stock_code": code,
                        "side": "BUY" if intended[code] > 0 else "SELL", "intended_value": float(abs(intended[code])),
                        "reason": "zero_volume" if px_ids.notna()[code] else "no_price",
                    })
            # 거래 후 수량으로 환산 (가격이 없는 종목은 거래되지 않았으므로 기존 수량 유지)
            fv = batch.final_values[batch.final_values.abs() > 1e-9]
            new_px = px_ids.reindex(fv.index).fillna(prices_today.reindex(fv.index))
            shares = fv / new_px
            values = fv
            cash = float(batch.final_cash)
            diags.append({"strategy_id": strategy_id, "signal_date": signal_date, "execution_date": day,
                          "pre_nav": pre_nav, "post_nav": post_nav, **batch.diagnostics})

        nav = float(values.sum() + cash)
        nav_rows.append({"date": day, "strategy_id": strategy_id, "nav": nav, "cash": cash,
                         "turnover": turnover, "cost": cost})
        if len(values):
            hold_rows.append(pd.DataFrame({"date": day, "strategy_id": strategy_id,
                                           "stock_code": values.index.astype(str), "weight": (values / nav).to_numpy()}))

    empty_trades = pd.DataFrame(columns=["date", "strategy_id", "stock_code", "side", "value", "cost"])
    return StrategyResult(
        daily_nav=pd.DataFrame(nav_rows),
        holdings=pd.concat(hold_rows, ignore_index=True) if hold_rows else pd.DataFrame(),
        trades=pd.concat(trade_rows, ignore_index=True) if trade_rows else empty_trades,
        untradable=pd.DataFrame(untradable_rows, columns=["date", "strategy_id", "stock_code", "side",
                                                          "intended_value", "reason"]),
        rebalance_count=len(diags),
        diagnostics=diags,
    )


# ---------------------------------------------------------------------------
# 5. 성과 요약
# ---------------------------------------------------------------------------

BENCHMARK_ID = "BENCHMARK_PROXY_CAPW"


def build_benchmark_proxy(panel: DailyPanel, base_day: pd.Timestamp, end_day: pd.Timestamp,
                          initial_level: float) -> pd.DataFrame:
    """주가 패널 전 종목(933개)의 전일 시가총액 가중 지수. 공식 KOSPI 지수가 아닌 대용치다.

    r_t = Σ_i cap_{i,t-1} · ret_{i,t} / Σ_i cap_{i,t-1}  (전일 시총과 당일 수익률이 모두 있는 종목만)
    - 전일 시총을 쓰므로 당일 정보로 가중치를 정하지 않는다.
    - 수익률은 수정주가 기준 가격수익률(현금배당 미반영)이며, 상장폐지 종목이 없는 생존편향이 그대로 있다.
    """
    ret = panel.returns()
    w = panel.market_cap.shift(1)
    valid = ret.notna() & w.notna() & (w > 0)
    num = (w.where(valid, 0.0) * ret.where(valid, 0.0)).sum(axis=1)
    den = w.where(valid, 0.0).sum(axis=1)
    daily = (num / den).where(den > 0)
    days = panel.adj_close.index[(panel.adj_close.index >= base_day) & (panel.adj_close.index <= end_day)]
    r = daily.reindex(days)
    r.iloc[0] = 0.0                      # 기준일(첫 체결 전 거래일)은 수익률 0, 지수 = 초기값
    level = initial_level * (1 + r).cumprod()
    return pd.DataFrame({"date": days, "benchmark_id": BENCHMARK_ID, "daily_return": r.to_numpy(),
                         "level": level.to_numpy(), "n_constituents": valid.reindex(days).sum(axis=1).to_numpy()})


def performance_stats(nav: pd.Series, annual_sessions: int = 252) -> dict[str, Any]:
    """NAV(또는 지수 수준) 시계열의 총수익률·CAGR·연변동성·MDD."""
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return {
        "start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
        "final_nav": float(nav.iloc[-1]),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1),
        "ann_vol": float(ret.std(ddof=1) * np.sqrt(annual_sessions)),
        "mdd": float((nav / nav.cummax() - 1).min()),
    }


def annual_returns(series_by_id: dict[str, pd.Series]) -> pd.DataFrame:
    """연도별 수익률 = 해당 연도 마지막 값 / 직전 연도 마지막 값 − 1 (첫 해는 기준일 값 대비)."""
    cols = {}
    for sid, s in series_by_id.items():
        year_end = s.groupby(s.index.year).last()
        prev = year_end.shift(1)
        prev.iloc[0] = s.iloc[0]
        cols[sid] = year_end / prev - 1
    out = pd.DataFrame(cols)
    out = out[out.index > series_by_id[next(iter(series_by_id))].index[0].year]  # 기준일만 있는 해(2015) 제외
    out.index.name = "year"
    return out


def summarize_strategy(result: StrategyResult, initial_capital: float, annual_sessions: int = 252) -> dict[str, Any]:
    """CAGR, 연변동성, MDD, 평균 회전율, 누적 비용. 무위험수익률이 없어 샤프비율은 계산하지 않는다."""
    nav = result.daily_nav.set_index("date")["nav"]
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    rebal = result.daily_nav[result.daily_nav["turnover"] > 0]
    return {
        "strategy_id": result.daily_nav["strategy_id"].iloc[0],
        **performance_stats(nav, annual_sessions),
        "rebalance_count": result.rebalance_count,
        "avg_turnover_per_rebalance": float(rebal["turnover"].mean()) if len(rebal) else 0.0,
        "annual_turnover": float(result.daily_nav["turnover"].sum() / years),
        "cumulative_cost": float(result.daily_nav["cost"].sum()),
        "cumulative_cost_pct_initial": float(result.daily_nav["cost"].sum() / initial_capital),
        "untradable_events": int((result.untradable["side"] != "VALUATION").sum()),
        "avg_cash_weight": float((result.daily_nav["cash"] / result.daily_nav["nav"]).mean()),
    }


# ---------------------------------------------------------------------------
# 6. 전체 실행
# ---------------------------------------------------------------------------

def run_backtest(modules_out: str | Path, panel: DailyPanel, cfg: dict[str, Any], out_dir: str | Path) -> pd.DataFrame:
    """전략 4개를 같은 패널·비용·체결 규칙으로 실행하고 약속된 형식의 CSV를 저장한다."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    bt, alloc_cfg, exec_cfg = cfg["backtest"], cfg["allocation"], cfg["execution"]
    initial = float(bt["initial_capital"])
    end = signal_cutoff(panel.adj_close.index, pd.Timestamp(bt["evaluation_end"]))

    inputs = load_module_outputs(modules_out)
    schedule = build_selection_schedule(inputs, cfg)
    check = compare_with_pipeline(schedule["ROTATION"]["selections"], inputs.pipeline_selections)

    returns = panel.returns()
    targets: dict[str, dict[str, dict[pd.Timestamp, pd.Series]]] = {}
    alloc_logs, target_rows = [], []
    for family, item in schedule.items():
        targets[family] = {"EW": {}, "BL_ALPHA_003": {}}
        item["selections"].assign(family=family).to_csv(out / f"selections_{family.lower()}.csv", index=False)
        for s in sorted(item["selections"]["date"].unique()):
            s = pd.Timestamp(s)
            tg, log = build_signal_targets(s, item["selections"], panel, returns, alloc_cfg)
            alloc_logs.append({"family": family, **log})
            for aid, w in tg.items():
                targets[family][aid][s] = w
                target_rows.append(pd.DataFrame({"signal_date": s, "family": family, "allocation_id": aid,
                                                 "stock_code": w.index, "target_weight": w.to_numpy()}))
    pd.DataFrame(alloc_logs).to_csv(out / "allocation_log.csv", index=False)
    pd.concat(target_rows, ignore_index=True).to_csv(out / "target_weights.csv", index=False)

    results = {}
    for sid, (family, aid) in STRATEGIES.items():
        results[sid] = run_strategy(sid, targets[family][aid], panel, exec_cfg, initial, end)

    pd.concat([r.daily_nav for r in results.values()]).to_csv(out / "daily_nav.csv", index=False)
    pd.concat([r.holdings for r in results.values()]).to_csv(out / "holdings.csv", index=False)
    pd.concat([r.trades for r in results.values()]).to_csv(out / "trades.csv", index=False)
    pd.concat([r.untradable for r in results.values()]).to_csv(out / "untradable_log.csv", index=False)
    pd.DataFrame([d for r in results.values() for d in r.diagnostics]).to_csv(out / "rebalance_diagnostics.csv", index=False)

    # 벤치마크 대용치: 전략 NAV와 같은 기준일(첫 체결 전 거래일)·같은 초기값에서 시작
    navs = {sid: r.daily_nav.set_index("date")["nav"] for sid, r in results.items()}
    base_day = next(iter(navs.values())).index[0]
    bench = build_benchmark_proxy(panel, base_day, end, initial)
    bench.to_csv(out / "benchmark_proxy.csv", index=False)
    bench_level = bench.set_index("date")["level"]

    rows = [summarize_strategy(r, initial) for r in results.values()]
    rows.append({"strategy_id": BENCHMARK_ID, **performance_stats(bench_level)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    yearly = annual_returns({**navs, BENCHMARK_ID: bench_level})
    yearly.to_csv(out / "annual_returns.csv")
    notes = {
        "benchmark_note": "BENCHMARK_PROXY_CAPW는 공식 KOSPI 지수가 아니다. 주가 패널 933종목(현재 KOSPI 구성종목)의 "
                          "전일 시가총액 가중 가격수익률 지수로 만든 대용치이며, 전략과 같은 생존편향·배당 미반영 조건이다.",
        "comparison_caveat": "정적 전략은 분기말에만, 로테이션 전략은 분기말과 국면 전환 달에 리밸런싱하므로 "
                             "두 계열의 성과 차이에는 가중치 효과와 리밸런싱 횟수 차이가 섞여 있다.",
        "rebalance_counts": {sid: r.rebalance_count for sid, r in results.items()},
        "execution": "신호=월말(마지막 거래일 종가까지), 체결=다음 거래일 종가, 거래량 0 종목 체결 불가(재시도 없음)",
        "round_trip_cost": round_trip_cost(exec_cfg),
        "cash_rate": 0.0,
        "pipeline_reproduction_check": check,
        "allocation_status_counts": pd.DataFrame(alloc_logs).groupby(["family", "bl_status"]).size()
                                     .rename("n").reset_index().to_dict("records"),
        "limitations": [
            "현재 KOSPI 구성종목만 포함되어 상장폐지 종목이 빠진 생존편향이 있다(수정하지 않음).",
            "시가 자료가 없어 다음 거래일 종가로 체결한다.",
            "벤치마크·무위험수익률 자료가 없어 초과수익·샤프비율을 계산하지 않았고 현금 이자는 0이다.",
            "거래비용은 고정 비율이며 연도별 증권거래세 변화는 반영하지 않았다.",
            "수정주가는 현금배당 미반영(가격수익률)이므로 전략·벤치마크 대용치 모두 배당만큼 총수익률보다 낮다.",
        ],
    }
    (out / "backtest_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def round_trip_cost(exec_cfg: dict[str, Any]) -> float:
    """왕복 비용 = 매수(수수료+슬리피지) + 매도(수수료+세금+슬리피지)."""
    return (float(exec_cfg["buy_commission"]) + float(exec_cfg["sell_commission"])
            + float(exec_cfg["sell_tax"]) + 2 * float(exec_cfg["slippage_each_side"]))
