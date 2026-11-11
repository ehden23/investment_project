from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq


@dataclass
class TradeBatch:
    final_values: pd.Series
    final_cash: float
    trades: pd.DataFrame
    costs: dict[str, float]
    diagnostics: dict[str, Any]


def cost_rates(config: dict[str, Any]) -> tuple[float, float]:
    buy = float(config["buy_commission"]) + float(config["slippage_each_side"])
    sell = float(config["sell_commission"]) + float(config["sell_tax"]) + float(config["slippage_each_side"])
    return buy, sell


def portfolio_drift(current_values: pd.Series, cash: float, target_weights: pd.Series) -> float:
    """현금과 목표 0 종목까지 포함한 half-L1 괴리도."""
    ids = current_values.index.union(target_weights.index)
    x = current_values.reindex(ids, fill_value=0.0)
    w = target_weights.reindex(ids, fill_value=0.0)
    capital = float(x.sum() + cash)
    if capital <= 0:
        return np.nan
    return float(0.5 * ((x / capital - w).abs().sum() + abs(cash / capital)))


def _cost_table(sells: pd.Series, buys: pd.Series, cfg: dict[str, Any], batch: str) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for code, value in sells[sells > 0].items():
        rows.append({"stock_code": code, "side": "SELL", "reference_notional": value, "batch": batch,
                     "commission": value * cfg["sell_commission"], "tax": value * cfg["sell_tax"],
                     "slippage": value * cfg["slippage_each_side"]})
    for code, value in buys[buys > 0].items():
        rows.append({"stock_code": code, "side": "BUY", "reference_notional": value, "batch": batch,
                     "commission": value * cfg["buy_commission"], "tax": 0.0,
                     "slippage": value * cfg["slippage_each_side"]})
    table = pd.DataFrame(rows)
    if table.empty:
        return table, {"commission": 0.0, "tax": 0.0, "slippage": 0.0, "total": 0.0}
    costs = {k: float(table[k].sum()) for k in ["commission", "tax", "slippage"]}
    costs["total"] = sum(costs.values())
    return table, costs


def full_rebalance(
    current_values: pd.Series, cash: float, target_weights: pd.Series,
    executable: pd.Series | None, config: dict[str, Any],
) -> TradeBatch:
    """비용을 포함한 self-financing 전면 리밸런싱."""
    ids = current_values.index.union(target_weights.index)
    x = current_values.reindex(ids, fill_value=0.0).astype(float)
    w = target_weights.reindex(ids, fill_value=0.0).astype(float)
    if abs(w.sum() - 1) > 1e-8 or (w < 0).any():
        raise ValueError("목표비중은 비음수이고 합이 1이어야 합니다.")
    can = pd.Series(True, index=ids) if executable is None else executable.reindex(ids, fill_value=False).astype(bool)
    buy_rate, sell_rate = cost_rates(config)
    capital = float(x.sum() + cash)

    def components(n: float) -> tuple[pd.Series, pd.Series, float, float]:
        sells = ((x - w * n).clip(lower=0) * can).astype(float)
        req = ((w * n - x).clip(lower=0) * can).astype(float)
        available = cash + (1 - sell_rate) * sells.sum()
        buy_total = min(float(req.sum()), max(0.0, available / (1 + buy_rate)))
        buys = req * (buy_total / req.sum()) if req.sum() > 0 else req
        return sells, buys, float(sells.sum()), float(buys.sum())

    def f(n: float) -> float:
        _, _, s, b = components(n)
        return n + sell_rate * s + buy_rate * b - capital

    if abs(f(0)) < 1e-14: n_star = 0.0
    elif abs(f(capital)) < 1e-14: n_star = capital
    else: n_star = brentq(f, 0.0, capital, xtol=max(1e-14, 1e-12 * max(capital, 1)), rtol=1e-12, maxiter=1000)
    sells, buys, s, b = components(n_star)
    final_values = x - sells + buys
    final_cash = float(cash + (1 - sell_rate) * s - (1 + buy_rate) * b)
    if final_cash < -max(1e-12, 1e-10 * max(capital, 1)):
        raise RuntimeError("리밸런싱 후 현금이 음수입니다.")
    final_cash = max(final_cash, 0.0)
    trades, costs = _cost_table(sells, buys, config, "FULL")
    diag = {"pretrade_capital": capital, "posttrade_nav": float(final_values.sum() + final_cash),
            "target_capital_root": n_star, "root_residual": f(n_star),
            "unavailable_count": int((~can).sum()), "exact_target": bool(can.all())}
    return TradeBatch(final_values, final_cash, trades, costs, diag)


def mandatory_replacement(
    current_values: pd.Series, cash: float, selected_ids: set[str], target_weights: pd.Series,
    executable: pd.Series, config: dict[str, Any],
) -> TradeBatch:
    """비용인식형 정책의 완전 편출·순매각대금 재투자 배치."""
    ids = current_values.index.union(target_weights.index)
    x = current_values.reindex(ids, fill_value=0.0).astype(float)
    can = executable.reindex(ids, fill_value=False).astype(bool)
    w = target_weights.reindex(ids, fill_value=0.0).astype(float)
    buy_rate, sell_rate = cost_rates(config)
    departure = pd.Series([i not in selected_ids for i in ids], index=ids) & (x > 0)
    sells = x.where(departure & can, 0.0)
    after = x - sells
    net_proceeds = float((1 - sell_rate) * sells.sum())
    capital_after_exit = float(after.sum() + cash + net_proceeds)
    bmax = net_proceeds / (1 + buy_rate)

    def gaps(b: float) -> pd.Series:
        n = capital_after_exit - buy_rate * b
        return (w * n - after).clip(lower=0).where(can & pd.Series([i in selected_ids for i in ids], index=ids), 0.0)
    g0 = gaps(bmax); gsum = float(g0.sum())
    if gsum >= bmax:
        buy_total, g = bmax, g0
    elif gaps(0).sum() <= 0:
        buy_total, g = 0.0, gaps(0)
    else:
        root = brentq(lambda b: b - float(gaps(b).sum()), 0.0, bmax)
        buy_total, g = root, gaps(root)
    buys = g * (buy_total / g.sum()) if g.sum() > 0 else g
    final_values = after + buys
    final_cash = float(cash + net_proceeds - (1 + buy_rate) * buys.sum())
    trades, costs = _cost_table(sells, buys, config, "MANDATORY_REPLACEMENT")
    return TradeBatch(final_values, max(final_cash, 0.0), trades, costs,
                      {"departures": int(departure.sum()), "pending_exits": list(ids[departure & ~can]),
                       "net_exit_proceeds": net_proceeds, "replacement_buy": float(buys.sum())})


def half_gap_adjustment(
    current_values: pd.Series, cash: float, target_weights: pd.Series,
    executable: pd.Series, config: dict[str, Any],
) -> TradeBatch:
    ids = current_values.index.union(target_weights.index)
    x = current_values.reindex(ids, fill_value=0.0).astype(float)
    w = target_weights.reindex(ids, fill_value=0.0).astype(float)
    can = executable.reindex(ids, fill_value=False).astype(bool)
    buy_rate, sell_rate = cost_rates(config); lam = float(config["partial_fraction"])
    capital = float(x.sum() + cash); gap = w * capital - x
    sells = (lam * (-gap).clip(lower=0)).where(can, 0.0)
    intended_buys = (lam * gap.clip(lower=0)).where(can, 0.0)
    available = cash + (1 - sell_rate) * sells.sum()
    kappa = min(1.0, available / ((1 + buy_rate) * intended_buys.sum())) if intended_buys.sum() > 0 else 0.0
    buys = intended_buys * max(kappa, 0.0)
    final_values = x - sells + buys
    final_cash = float(cash + (1 - sell_rate) * sells.sum() - (1 + buy_rate) * buys.sum())
    trades, costs = _cost_table(sells, buys, config, "HALF_GAP")
    return TradeBatch(final_values, max(final_cash, 0.0), trades, costs,
                      {"kappa": kappa, "pretrade_drift": portfolio_drift(x, cash, w)})


def apply_policy_review(
    policy: str, month: int, initialization: bool, regime_transition: bool,
    current_values: pd.Series, cash: float, target_weights: pd.Series,
    selected_ids: set[str], executable: pd.Series, config: dict[str, Any],
) -> TradeBatch:
    """월말에 동결한 의도를 다음 거래일 시가에 적용하는 핵심 정책 함수."""
    if initialization or policy == "MONTHLY_FULL" or (policy == "QUARTERLY_FULL" and month in {3, 6, 9, 12}):
        return full_rebalance(current_values, cash, target_weights, executable, config)
    if policy == "QUARTERLY_FULL":
        return TradeBatch(current_values.copy(), cash, pd.DataFrame(), {"total": 0.0}, {"review": "not_scheduled"})
    if policy != "MONTHLY_COST_AWARE":
        raise ValueError(f"알 수 없는 정책: {policy}")
    frozen_d = portfolio_drift(current_values, cash, target_weights)
    first = mandatory_replacement(current_values, cash, selected_ids, target_weights, executable, config)
    if frozen_d > float(config["drift_threshold"]) or regime_transition:
        second = half_gap_adjustment(first.final_values, first.final_cash, target_weights, executable, config)
        trades = pd.concat([first.trades, second.trades], ignore_index=True)
        costs = {k: first.costs.get(k, 0.0) + second.costs.get(k, 0.0) for k in set(first.costs) | set(second.costs)}
        return TradeBatch(second.final_values, second.final_cash, trades, costs,
                          {"frozen_drift": frozen_d, "ordinary_trigger": True, **first.diagnostics, **second.diagnostics})
    first.diagnostics.update({"frozen_drift": frozen_d, "ordinary_trigger": False})
    return first
