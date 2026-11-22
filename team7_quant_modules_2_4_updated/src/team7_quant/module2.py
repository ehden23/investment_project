from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import validate_module_interface
from .io import require_columns


PRICE_COLS = {"date", "stock_code", "adj_close", "market_cap"}
FIN_COLS = {
    "stock_code", "fiscal_year", "fiscal_month_end", "ni_parent", "equity_parent",
    "cfo", "gross_profit", "total_assets", "total_liabilities", "statement_type",
}
UNIVERSE_COLS = {"date", "stock_code"}
SECTOR_COLS = {"stock_code", "date", "sector_code", "is_financial"}


@dataclass
class SelectionResult:
    selected: pd.DataFrame
    scores: pd.DataFrame
    evidence: dict[str, Any]
    logs: pd.DataFrame


def percentile_rank(values: pd.Series) -> pd.Series:
    """(평균순위-1)/(유효개수-1). 유효값이 하나면 중립값 0.5."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()
    n = int(valid.sum())
    if n == 1:
        out.loc[valid] = 0.5
    elif n >= 2:
        out.loc[valid] = (values.loc[valid].rank(method="average") - 1.0) / (n - 1.0)
    return out


def _month_end_price(prices: pd.DataFrame, target: pd.Timestamp, codes: pd.Index) -> pd.Series:
    p = prices[(prices["date"] <= target) & prices["stock_code"].isin(codes)]
    if p.empty:
        return pd.Series(dtype=float)
    p = p.sort_values(["stock_code", "date"]).groupby("stock_code", as_index=False).tail(1)
    return p.set_index("stock_code")["adj_close"]


def _financial_snapshot(financials: pd.DataFrame, asof: pd.Timestamp, lag_months: int) -> pd.DataFrame:
    f = financials.copy()
    f["fiscal_month_end"] = pd.to_datetime(f["fiscal_month_end"])
    assumed = f["fiscal_month_end"] + pd.DateOffset(months=lag_months) + pd.offsets.MonthEnd(0)
    if "available_at" in f.columns:
        reported = pd.to_datetime(f["available_at"], errors="coerce")
        f["available_at"] = reported.fillna(assumed)
        f["availability_source"] = np.where(reported.notna(), "actual", "assumed_lag")
    else:
        f["available_at"] = assumed
        f["availability_source"] = "assumed_lag"
    f = f[f["available_at"] <= asof].copy()
    if f.empty:
        return f
    # 같은 결산기에 연결과 별도가 모두 있으면 연결을 우선한다.
    f["statement_priority"] = f["statement_type"].astype(str).str.contains("연결|consol", case=False, regex=True).map({True: 0, False: 1})
    f = f.sort_values(["stock_code", "fiscal_month_end", "statement_priority"])
    f = f.drop_duplicates(["stock_code", "fiscal_month_end"], keep="first")
    f["equity_begin"] = f.groupby("stock_code")["equity_parent"].shift(1)
    f = f.sort_values(["stock_code", "fiscal_month_end"]).groupby("stock_code", as_index=False).tail(1)
    f["standalone_fallback"] = ~f["statement_type"].astype(str).str.contains("연결|consol", case=False, regex=True)
    return f


def _validate_weights(weights: dict[str, float] | None, profile: str) -> tuple[dict[str, float], str]:
    if weights is None:
        weights = {"value": 0.4, "quality": 0.4, "momentum": 0.2}
        source = "default"
    else:
        weights = {k: float(v) for k, v in weights.items()}
        source = "module4"
    required = {"value", "quality", "momentum"}
    if set(weights) != required or abs(sum(weights.values()) - 1.0) > 1e-12:
        raise ValueError("가중치는 value, quality, momentum을 포함하고 합이 1이어야 합니다.")
    validate_module_interface(profile, weights["momentum"])
    if any(v < 0 or v > 1 for v in weights.values()):
        raise ValueError("가중치는 0과 1 사이여야 합니다.")
    return weights, source


def select_stocks(
    asof_date: str | pd.Timestamp,
    universe: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    factor_weights: dict[str, float] | None,
    config: dict[str, Any],
) -> SelectionResult:
    """한 시점의 팩터 스코어와 선정 종목을 계산한다.

    data에는 prices, financials, sectors가 필요하다. 유니버스 필터는 다시 수행하지 않는다.
    """
    asof = pd.Timestamp(asof_date).normalize()
    prices, financials, sectors = data["prices"].copy(), data["financials"].copy(), data["sectors"].copy()
    require_columns(universe, UNIVERSE_COLS, "universe")
    require_columns(prices, PRICE_COLS, "prices")
    require_columns(financials, FIN_COLS, "financials")
    require_columns(sectors, SECTOR_COLS, "sectors")
    for frame in (universe, prices, sectors):
        frame["date"] = pd.to_datetime(frame["date"])
    profile = config.get("profile", "weighted_momentum")
    weights, weight_source = _validate_weights(factor_weights, profile)
    u = universe[universe["date"] == asof][["stock_code"]].drop_duplicates().copy()
    if u.empty:
        raise ValueError(f"{asof.date()} 유니버스가 없습니다.")
    latest_price = prices[(prices["date"] <= asof) & prices["stock_code"].isin(u["stock_code"])]
    latest_price = latest_price.sort_values(["stock_code", "date"]).groupby("stock_code", as_index=False).tail(1)
    base = u.merge(latest_price[["stock_code", "date", "adj_close", "market_cap"]], on="stock_code", how="left")
    fin = _financial_snapshot(financials, asof, int(config.get("financial_lag_months", 4)))
    base = base.merge(fin, on="stock_code", how="left", suffixes=("", "_fin"))
    sec = sectors[(sectors["date"] <= asof) & sectors["stock_code"].isin(u["stock_code"])]
    sec = sec.sort_values(["stock_code", "date"]).groupby("stock_code", as_index=False).tail(1)
    base = base.merge(sec[["stock_code", "sector_code", "is_financial"]], on="stock_code", how="left")

    # 원시 지표: 금액 단위는 사전에 동일해야 한다.
    base["EP"] = base["ni_parent"] / base["market_cap"]
    base["BP"] = base["equity_parent"] / base["market_cap"]
    base["CFP"] = base["cfo"] / base["market_cap"]
    base["GPA"] = base["gross_profit"] / base["total_assets"]
    base.loc[base["is_financial"].fillna(False).astype(bool), "GPA"] = np.nan
    avg_equity = (base["equity_begin"] + base["equity_parent"]) / 2.0
    base["ROE"] = base["ni_parent"] / avg_equity
    base["roe_invalid_equity"] = avg_equity <= 0
    base.loc[base["roe_invalid_equity"], "ROE"] = np.nan
    base["LEV_neg"] = -(base["total_liabilities"] / base["total_assets"])

    p1 = _month_end_price(prices, asof - pd.offsets.MonthEnd(1), pd.Index(u["stock_code"]))
    p12 = _month_end_price(prices, asof - pd.offsets.MonthEnd(12), pd.Index(u["stock_code"]))
    base["price_t_minus_1"] = base["stock_code"].map(p1)
    base["price_t_minus_12"] = base["stock_code"].map(p12)
    base["MOM"] = base["price_t_minus_1"] / base["price_t_minus_12"] - 1.0
    base.loc[(base["price_t_minus_1"] <= 0) | (base["price_t_minus_12"] <= 0), "MOM"] = np.nan

    raw_groups = {"value": ["EP", "BP", "CFP"], "quality": ["GPA", "ROE", "LEV_neg"], "momentum": ["MOM"]}
    for cols in raw_groups.values():
        for col in cols:
            base[f"p_{col}"] = percentile_rank(base[col])
    base["value_mean"] = base[["p_EP", "p_BP", "p_CFP"]].mean(axis=1, skipna=True)
    base["quality_mean"] = base[["p_GPA", "p_ROE", "p_LEV_neg"]].mean(axis=1, skipna=True)
    base["value_all_missing"] = base[["EP", "BP", "CFP"]].isna().all(axis=1)
    base["quality_all_missing"] = base[["GPA", "ROE", "LEV_neg"]].isna().all(axis=1)
    base["momentum_all_missing"] = base["MOM"].isna()
    base["S_V"] = percentile_rank(base["value_mean"])
    base["S_Q"] = percentile_rank(base["quality_mean"])
    base["S_M"] = base["p_MOM"]
    missing_score = float(config.get("missing_factor_score", 0.5))
    base.loc[base["value_all_missing"], "S_V"] = missing_score
    base.loc[base["quality_all_missing"], "S_Q"] = missing_score
    base.loc[base["momentum_all_missing"], "S_M"] = missing_score
    base["all_factors_missing"] = base[["value_all_missing", "quality_all_missing", "momentum_all_missing"]].all(axis=1)
    base["unit_error_candidate"] = (base[["EP", "BP"]].abs() >= float(config.get("unit_error_abs_ratio", 100))).any(axis=1)

    if profile == "weighted_momentum":
        base["CoreScore"] = weights["value"] * base["S_V"] + weights["quality"] * base["S_Q"] + weights["momentum"] * base["S_M"]
        candidates = base[~base["all_factors_missing"]].copy()
    else:
        # 모듈 3·4 명세와 연결하는 별도 프로필: 1차 40개 후 모멘텀 하위 20% 제외.
        base["CoreScore"] = weights["value"] * base["S_V"] + weights["quality"] * base["S_Q"]
        cutoff = base["S_M"].quantile(0.20)
        candidates = base[(~base["all_factors_missing"]) & (base["S_M"] > cutoff)].copy()
        candidates = candidates.sort_values(["CoreScore", "S_V", "market_cap", "stock_code"], ascending=[False, False, False, True]).head(40)
    candidates = candidates.sort_values(["CoreScore", "S_V", "market_cap", "stock_code"], ascending=[False, False, False, True])
    n_select = int(config.get("selected_count", 30))
    selected = candidates.head(n_select).copy()
    selected["rank"] = np.arange(1, len(selected) + 1)
    for key, val in [("w_V", weights["value"]), ("w_Q", weights["quality"]), ("w_M", weights["momentum"])]:
        base[key] = val
        selected[key] = val
    selected["date"] = asof

    log_rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        reasons = []
        if row["value_all_missing"]: reasons.append("value_all_missing_neutral_0.5")
        if row["quality_all_missing"]: reasons.append("quality_all_missing_neutral_0.5")
        if row["momentum_all_missing"]: reasons.append("momentum_missing_neutral_0.5")
        if row["all_factors_missing"]: reasons.append("excluded_all_factors_missing")
        if row["roe_invalid_equity"]: reasons.append("roe_denominator_nonpositive")
        if row["unit_error_candidate"]: reasons.append("unit_error_candidate")
        if reasons:
            log_rows.append({"date": asof, "stock_code": row["stock_code"], "reason": ";".join(reasons)})
    if len(selected) < n_select:
        log_rows.append({"date": asof, "stock_code": "", "reason": f"selected_less_than_{n_select}:{len(selected)}"})
    evidence = {
        "asof_date": str(asof.date()), "profile": profile, "weights": weights,
        "weight_source": weight_source, "universe_count": len(u), "selected_count": len(selected),
        "financial_cutoff_rule": f"fiscal_month_end+{config.get('financial_lag_months', 4)}months_month_end",
    }
    output_cols = ["date", "stock_code", "S_V", "S_Q", "S_M", "w_V", "w_Q", "w_M", "CoreScore", "rank"]
    return SelectionResult(selected[output_cols], base, evidence, pd.DataFrame(log_rows))


def run_rebalance_dates(
    dates: list[pd.Timestamp], universe: pd.DataFrame, data: dict[str, pd.DataFrame],
    weights_by_date: pd.DataFrame | None, config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    selections, scores, logs, turnover = [], [], [], []
    previous: set[str] = set()
    for date in sorted(pd.to_datetime(dates)):
        weights = None
        if weights_by_date is not None:
            row = weights_by_date[pd.to_datetime(weights_by_date["date"]) == date]
            if not row.empty:
                weights = {"value": row.iloc[0]["w_V"], "quality": row.iloc[0]["w_Q"], "momentum": row.iloc[0].get("w_M", 0.2)}
        result = select_stocks(date, universe, data, weights, config)
        selections.append(result.selected)
        s = result.scores.copy(); s["asof_date"] = date; scores.append(s)
        logs.append(result.logs)
        current = set(result.selected["stock_code"])
        turnover.append({"date": date, "entries": len(current - previous) if previous else len(current), "exits": len(previous - current)})
        previous = current
    return {
        "selections": pd.concat(selections, ignore_index=True) if selections else pd.DataFrame(),
        "scores": pd.concat(scores, ignore_index=True) if scores else pd.DataFrame(),
        "logs": pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(),
        "replacement_counts": pd.DataFrame(turnover),
    }


def compute_ic(scores: pd.DataFrame, prices: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """월말 점수와 다음 달 수정주가 수익률의 Spearman IC를 계산한다."""
    s = scores.copy()
    s["asof_date"] = pd.to_datetime(s["asof_date"])
    prices = prices.copy(); prices["date"] = pd.to_datetime(prices["date"])
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for date, group in s[(s["asof_date"] >= start_ts) & (s["asof_date"] <= end_ts)].groupby("asof_date"):
        next_end = date + pd.offsets.MonthEnd(1)
        if next_end > end_ts:
            continue
        p0 = _month_end_price(prices, date, pd.Index(group["stock_code"]))
        p1 = _month_end_price(prices, next_end, pd.Index(group["stock_code"]))
        ret = p1 / p0 - 1
        for factor in ["S_V", "S_Q", "S_M"]:
            x = group.set_index("stock_code")[factor]
            z = pd.concat([x.rename("score"), ret.rename("forward_return")], axis=1).dropna()
            ic = spearmanr(z["score"], z["forward_return"]).statistic if len(z) >= 3 else np.nan
            rows.append({"date": date, "factor": factor, "ic": ic, "n_stocks": len(z)})
    monthly = pd.DataFrame(rows)
    summary_rows = []
    for factor, g in monthly.groupby("factor"):
        vals = g["ic"].dropna(); n = len(vals); mean = vals.mean(); std = vals.std(ddof=1)
        summary_rows.append({"factor": factor, "mean_ic": mean, "std_ic": std,
                             "icir": mean / std if std > 0 else np.nan,
                             "t_value": mean / (std / np.sqrt(n)) if n > 1 and std > 0 else np.nan,
                             "n_months": n})
    return monthly, pd.DataFrame(summary_rows)
