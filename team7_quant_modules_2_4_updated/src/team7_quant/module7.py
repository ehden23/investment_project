from __future__ import annotations

from dataclasses import dataclass, asdict
from math import exp, floor, log, sqrt, ceil
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class RiskState:
    level: int = 0
    release_count: int = 0
    put_contract_code: str | None = None
    put_quantity: int = 0
    futures_contract_code: str | None = None
    futures_short_quantity: int = 0


def annualized_volatility(returns: pd.Series | np.ndarray, window: int = 60, annual_sessions: int = 252) -> float:
    x = pd.Series(returns, dtype=float).dropna().tail(window)
    return float(x.std(ddof=1) * np.sqrt(annual_sessions)) if len(x) >= window else np.nan


def historical_es(returns: pd.Series | np.ndarray, window: int = 252, confidence: float = 0.95) -> tuple[float, int]:
    """손실을 양수로 표시한 역사적 ES. 가장 나쁜 ceil((1-c)N)개 평균."""
    x = pd.Series(returns, dtype=float).dropna().tail(window)
    if len(x) < window:
        return np.nan, 0
    losses = -x.to_numpy(float); n_tail = max(1, int(np.ceil((1 - confidence) * len(losses))))
    return float(np.sort(losses)[-n_tail:].mean()), n_tail


def drawdown(nav: float, historical_peak: float) -> float:
    if nav <= 0 or historical_peak <= 0:
        raise ValueError("NAV와 과거 최고 NAV는 양수여야 합니다.")
    return float(max(0.0, 1.0 - nav / historical_peak))


def put_delta(S: float, K: float, sigma: float, T: float, r: float, q: float) -> float:
    if min(S, K, sigma, T) <= 0:
        raise ValueError("S, K, sigma, T는 양수여야 합니다.")
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return float(exp(-q * T) * (norm.cdf(d1) - 1.0))


def validate_option_data(options: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "contract_code", "option_type", "expiry_date", "strike", "close", "volume",
                "open_interest", "underlying_price", "iv", "days_to_expiry", "multiplier"}
    missing = required - set(options.columns)
    if missing:
        raise ValueError(f"옵션 필수 열 누락: {sorted(missing)}")
    o = options.copy()
    o["date"] = pd.to_datetime(o["date"]); o["expiry_date"] = pd.to_datetime(o["expiry_date"])
    if (o["multiplier"] <= 0).any() or (o["strike"] <= 0).any():
        raise ValueError("행사가격과 거래승수는 양수여야 합니다.")
    if (o["iv"] > 3).any():
        raise ValueError("IV가 3보다 큽니다. 20이 20%를 의미하는 원자료라면 0.20으로 정규화하세요.")
    if ((o["iv"] <= 0) & o["iv"].notna()).any():
        raise ValueError("관측된 IV는 양수여야 합니다.")
    return o


def update_risk_level(previous: RiskState, current_dd: float, config: dict[str, Any]) -> tuple[RiskState, str]:
    thresholds = list(map(float, config["drawdown_thresholds"]))
    release = list(map(float, config["release_thresholds"]))
    desired = 3 if current_dd >= thresholds[2] else 2 if current_dd >= thresholds[1] else 1 if current_dd >= thresholds[0] else 0
    state = RiskState(**asdict(previous))
    if desired > state.level:
        state.level, state.release_count = desired, 0
        return state, "risk_level_strengthened"
    if state.level == 0:
        state.release_count = 0
        return state, "risk_level_zero"
    # 단계별 완화 기준: level 3->13%, 2->10%, 1->6%
    release_limit = release[state.level - 1]
    if current_dd < release_limit:
        state.release_count += 1
        if state.release_count >= int(config["release_days"]):
            state.level -= 1; state.release_count = 0
            return state, "risk_level_released_one_step"
    else:
        state.release_count = 0
    return state, "risk_level_maintained"


def choose_put_contract(asof_date: pd.Timestamp, option_chain: pd.DataFrame, underlying: float) -> pd.Series:
    chain = validate_option_data(option_chain)
    c = chain[(chain["date"] == asof_date) & chain["option_type"].astype(str).str.upper().isin(["P", "PUT"])]
    c = c[(c["expiry_date"] > asof_date) & (c["volume"] > 0) & (c["open_interest"] > 0) & c["close"].notna()]
    if c.empty:
        raise ValueError("해당 일자에 체결 가능한 일반월물 풋옵션 후보가 없습니다.")
    nearest_expiry = c["expiry_date"].min(); c = c[c["expiry_date"] == nearest_expiry].copy()
    c["strike_distance"] = (c["strike"] - 0.95 * underlying).abs()
    return c.sort_values(["strike_distance", "open_interest", "contract_code"], ascending=[True, False, True]).iloc[0]


def decide_risk_orders(
    asof_date: str | pd.Timestamp,
    portfolio_snapshot: dict[str, Any],
    risk_state: RiskState,
    historical_data: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, RiskState, dict[str, Any]]:
    """t일 종가 정보로 t+1 위험 주문 목표를 만든다.

    portfolio_snapshot 필수 키: base_nav, base_peak, actual_nav, stock_beta_notional,
    cash, current_put_qty, current_futures_short_qty. historical_data에는 base_returns,
    option_chain, futures_row, r, q가 필요하다.
    """
    asof = pd.Timestamp(asof_date)
    dd = drawdown(float(portfolio_snapshot["base_nav"]), float(portfolio_snapshot["base_peak"]))
    new_state, level_reason = update_risk_level(risk_state, dd, config)
    base_returns = pd.Series(historical_data["base_returns"], dtype=float)
    vol = annualized_volatility(base_returns, int(config["vol_window"]), int(config["annual_sessions"]))
    es, es_tail_n = historical_es(base_returns, int(config["es_window"]), float(config["es_confidence"]))
    actual_nav = float(portfolio_snapshot["actual_nav"])
    e_beta = max(float(portfolio_snapshot["stock_beta_notional"]), 0.0)
    option = choose_put_contract(asof, historical_data["option_chain"], float(historical_data["underlying_price"]))
    S, mult_put = float(option["underlying_price"]), float(option["multiplier"])
    target_put = floor(float(config["put_coverage"]) * e_beta / (S * mult_put))
    T = float(option["days_to_expiry"]) / 365.0
    delta = put_delta(S, float(option["strike"]), float(option["iv"]), T,
                      float(historical_data["r"]), float(historical_data["q"]))
    b_before = (e_beta + target_put * mult_put * S * delta) / actual_nav
    futures = historical_data["futures_row"]
    target_short = 0
    beta_cap = None
    if new_state.level > 0:
        beta_cap = float(config["beta_caps"][new_state.level - 1])
        required = max(0.0, b_before - beta_cap) * actual_nav
        per_contract = float(futures["price"]) * float(futures["multiplier"])
        target_short = int(ceil(required / per_contract)) if required > 0 else 0
        # 계약 단위 때문에 순베타가 음수가 되는 과도한 매도는 제한한다.
        max_nonnegative = int(floor(max(0.0, b_before * actual_nav) / per_contract))
        target_short = min(target_short, max_nonnegative)
    final_beta = b_before - target_short * float(futures["price"]) * float(futures["multiplier"]) / actual_nav
    orders = pd.DataFrame([
        {"asof_date": asof, "execute_on": historical_data.get("next_session"), "instrument": "PUT",
         "contract_code": option["contract_code"], "target_quantity": target_put,
         "current_quantity": int(portfolio_snapshot.get("current_put_qty", 0)),
         "order_quantity": target_put - int(portfolio_snapshot.get("current_put_qty", 0))},
        {"asof_date": asof, "execute_on": historical_data.get("next_session"), "instrument": "FUTURES_SHORT",
         "contract_code": futures["contract_code"], "target_quantity": target_short,
         "current_quantity": int(portfolio_snapshot.get("current_futures_short_qty", 0)),
         "order_quantity": target_short - int(portfolio_snapshot.get("current_futures_short_qty", 0))},
    ])
    new_state.put_contract_code = str(option["contract_code"]); new_state.put_quantity = target_put
    new_state.futures_contract_code = str(futures["contract_code"]); new_state.futures_short_quantity = target_short
    min_cash = float(config.get("minimum_cash_ratio", 0.05)) * actual_nav
    evidence = {"drawdown": dd, "risk_level": new_state.level, "risk_level_reason": level_reason,
                "volatility_60d": vol, "es_95": es, "es_tail_count": es_tail_n,
                "put_delta": delta, "put_nominal_coverage": target_put * S * mult_put / e_beta if e_beta else 0,
                "beta_before_futures": b_before, "beta_cap": beta_cap, "final_beta": final_beta,
                "minimum_cash_required": min_cash, "cash_shortfall": max(0.0, min_cash - float(portfolio_snapshot["cash"])),
                "execution_timing": "signal_at_t_close_execute_t_plus_1_close_default"}
    return orders, new_state, evidence


def stress_test(
    stock_beta_notional: float, put_qty: int, put_contract: dict[str, float], futures_short_qty: int,
    futures_price: float, futures_multiplier: float, r: float, q: float,
    index_shock: float, iv_point_change: float,
) -> dict[str, float]:
    S0 = put_contract["underlying_price"]; S1 = S0 * (1 + index_shock)
    sigma1 = put_contract["iv"] + iv_point_change
    T = put_contract["days_to_expiry"] / 365
    # 스트레스 목적의 간단한 Black-Scholes 풋 재평가.
    def put_value(S: float, sigma: float) -> float:
        d1 = (log(S / put_contract["strike"]) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return put_contract["strike"] * exp(-r*T) * norm.cdf(-d2) - S * exp(-q*T) * norm.cdf(-d1)
    option_pnl = put_qty * put_contract["multiplier"] * (put_value(S1, sigma1) - put_value(S0, put_contract["iv"]))
    stock_pnl = stock_beta_notional * index_shock
    futures_pnl = -futures_short_qty * futures_multiplier * futures_price * index_shock
    return {"stock_pnl": stock_pnl, "option_pnl": option_pnl, "futures_pnl": futures_pnl,
            "total_pnl": stock_pnl + option_pnl + futures_pnl}
