from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .io import normalize_stock_code, require_columns


MASTER_COLS = {
    "stock_code", "listing_date", "delisting_date", "market_type",
    "security_type", "share_class",
}
PRICE_COLS = {"date", "stock_code", "trading_value", "market_cap"}
FIN_COLS = {
    "stock_code", "fiscal_month_end", "equity_parent", "ni_parent", "cfo",
    "gross_profit", "total_assets", "total_liabilities", "statement_type",
}


@dataclass
class UniverseResult:
    eligible: pd.DataFrame
    audit: pd.DataFrame
    summary: dict[str, Any]


def _as_bool_text(series: pd.Series, accepted: set[str]) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin({x.lower() for x in accepted})


def _financial_snapshot(financials: pd.DataFrame, asof: pd.Timestamp, lag_months: int) -> pd.DataFrame:
    """공시일이 있으면 실제 사용 가능일, 없으면 결산일+시차를 적용한다."""
    f = financials.copy()
    f["stock_code"] = normalize_stock_code(f["stock_code"])
    f["fiscal_month_end"] = pd.to_datetime(f["fiscal_month_end"])
    assumed = f["fiscal_month_end"] + pd.DateOffset(months=lag_months) + pd.offsets.MonthEnd(0)
    if "available_at" in f.columns:
        actual = pd.to_datetime(f["available_at"], errors="coerce")
        f["usable_at"] = actual.fillna(assumed)
        f["availability_source"] = np.where(actual.notna(), "actual", "assumed_lag")
    else:
        f["usable_at"] = assumed
        f["availability_source"] = "assumed_lag"
    f = f[f["usable_at"] <= asof].copy()
    if f.empty:
        return f
    is_consolidated = f["statement_type"].astype("string").str.contains(
        "연결|consol", case=False, regex=True, na=False
    )
    f["statement_priority"] = np.where(is_consolidated, 0, 1)
    f = f.sort_values(["stock_code", "fiscal_month_end", "statement_priority"])
    f = f.drop_duplicates(["stock_code", "fiscal_month_end"], keep="first")
    return f.groupby("stock_code", as_index=False).tail(1)


def _point_in_time_security(master: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """종목-적용기간 자료가 있으면 당시 행을 선택하고, 없으면 기본정보를 사용한다."""
    m = master.copy()
    m["stock_code"] = normalize_stock_code(m["stock_code"])
    for col in ["listing_date", "delisting_date", "effective_from", "effective_to"]:
        if col in m:
            m[col] = pd.to_datetime(m[col], errors="coerce")
    if {"effective_from", "effective_to"}.issubset(m.columns):
        active_version = (m["effective_from"].isna() | (m["effective_from"] <= asof)) & (
            m["effective_to"].isna() | (asof <= m["effective_to"])
        )
        m = m[active_version].copy()
    return m.sort_values("stock_code").drop_duplicates("stock_code", keep="last")


def build_universe(
    asof_date: str | pd.Timestamp,
    security_master: pd.DataFrame,
    prices: pd.DataFrame,
    financials: pd.DataFrame,
    config: dict[str, Any],
) -> UniverseResult:
    """한 리밸런싱 시점의 KOSPI 보통주 투자 유니버스를 만든다."""
    asof = pd.Timestamp(asof_date).normalize()
    require_columns(security_master, MASTER_COLS, "security_master")
    require_columns(prices, PRICE_COLS, "prices")
    require_columns(financials, FIN_COLS, "financials")

    m = _point_in_time_security(security_master, asof)
    m["listing_date"] = pd.to_datetime(m["listing_date"], errors="coerce")
    m["delisting_date"] = pd.to_datetime(m["delisting_date"], errors="coerce")
    market_values = set(config.get("kospi_labels", ["KOSPI", "유가증권시장"]))
    common_values = set(config.get("common_share_labels", ["보통주", "COMMON", "common"]))
    allowed_security = set(config.get("allowed_security_types", ["주권", "주식", "STOCK", "stock"]))
    m["is_kospi"] = _as_bool_text(m["market_type"], market_values)
    m["is_common"] = _as_bool_text(m["share_class"], common_values)
    m["is_stock_security"] = _as_bool_text(m["security_type"], allowed_security)
    m["listed_at_asof"] = (m["listing_date"] <= asof) & (
        m["delisting_date"].isna() | (asof < m["delisting_date"])
    )
    seasoning_months = int(config.get("minimum_listing_months", 12))
    m["seasoned"] = m["listing_date"] + pd.DateOffset(months=seasoning_months) <= asof

    p = prices.copy()
    p["stock_code"] = normalize_stock_code(p["stock_code"])
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"] <= asof]
    lookback_sessions = int(config.get("liquidity_lookback_sessions", 63))
    trailing = p.sort_values(["stock_code", "date"]).groupby("stock_code", group_keys=False).tail(lookback_sessions)
    liquidity = trailing.groupby("stock_code", as_index=False).agg(
        avg_trading_value_3m=("trading_value", "mean"),
        liquidity_observations=("trading_value", "count"),
    )
    latest = p.sort_values(["stock_code", "date"]).groupby("stock_code", as_index=False).tail(1)
    latest = latest[["stock_code", "date", "market_cap"]].rename(columns={"date": "market_data_date"})
    audit = m.merge(liquidity, on="stock_code", how="left").merge(latest, on="stock_code", how="left")

    fin = _financial_snapshot(financials, asof, int(config.get("financial_lag_months", 4)))
    keep = ["stock_code", "fiscal_month_end", "equity_parent", "ni_parent", "cfo", "gross_profit",
            "total_assets", "total_liabilities", "statement_type", "usable_at", "availability_source"]
    audit = audit.merge(fin[keep] if not fin.empty else pd.DataFrame(columns=keep), on="stock_code", how="left")

    # 기본 자격을 먼저 정한 뒤 그 집단 안에서 횡단면 하위 20%를 계산한다.
    audit["base_eligible"] = (
        audit["is_kospi"] & audit["is_common"] & audit["is_stock_security"]
        & audit["listed_at_asof"] & audit["seasoned"]
    )
    cross = audit["base_eligible"]
    audit["liquidity_percentile"] = np.nan
    audit["market_cap_percentile"] = np.nan
    audit.loc[cross, "liquidity_percentile"] = audit.loc[cross, "avg_trading_value_3m"].rank(pct=True, method="average")
    audit.loc[cross, "market_cap_percentile"] = audit.loc[cross, "market_cap"].rank(pct=True, method="average")
    cutoff = float(config.get("bottom_percentile_cutoff", 0.20))
    required_fin = config.get(
        "required_financial_columns",
        ["ni_parent", "equity_parent", "cfo", "gross_profit", "total_assets", "total_liabilities"],
    )
    audit["has_required_financials"] = audit[required_fin].notna().all(axis=1)
    audit["positive_equity"] = audit["equity_parent"] > 0
    audit["liquid_enough"] = audit["liquidity_percentile"] > cutoff
    audit["large_enough"] = audit["market_cap_percentile"] > cutoff
    audit["is_eligible"] = (
        audit["base_eligible"] & audit["liquid_enough"] & audit["large_enough"]
        & audit["positive_equity"] & audit["has_required_financials"]
    )

    reason_rules = [
        (~audit["is_kospi"], "not_kospi"), (~audit["is_common"], "not_common_share"),
        (~audit["is_stock_security"], "excluded_security_type"),
        (~audit["listed_at_asof"], "not_listed_at_asof"), (~audit["seasoned"], "listed_under_12_months"),
        (audit["avg_trading_value_3m"].isna(), "missing_trading_value"),
        (audit["liquidity_percentile"].le(cutoff), "bottom_20pct_liquidity"),
        (audit["market_cap"].isna(), "missing_market_cap"),
        (audit["market_cap_percentile"].le(cutoff), "bottom_20pct_market_cap"),
        (~audit["positive_equity"], "nonpositive_or_missing_equity"),
        (~audit["has_required_financials"], "missing_required_financials"),
    ]
    audit["exclusion_reason"] = ""
    for mask, label in reason_rules:
        audit.loc[mask, "exclusion_reason"] = audit.loc[mask, "exclusion_reason"].map(
            lambda x: f"{x};{label}" if x else label
        )
    audit["date"] = asof
    eligible = audit.loc[audit["is_eligible"], ["date", "stock_code"]].sort_values("stock_code").reset_index(drop=True)
    summary = {
        "date": str(asof.date()), "input_security_count": int(len(audit)),
        "eligible_count": int(len(eligible)), "liquidity_lookback_sessions": lookback_sessions,
        "bottom_percentile_cutoff": cutoff, "financial_lag_months": int(config.get("financial_lag_months", 4)),
    }
    return UniverseResult(eligible=eligible, audit=audit, summary=summary)


def build_universe_history(
    dates: list[str | pd.Timestamp], security_master: pd.DataFrame, prices: pd.DataFrame,
    financials: pd.DataFrame, config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    universes, audits, summaries = [], [], []
    for date in sorted(pd.to_datetime(dates)):
        result = build_universe(date, security_master, prices, financials, config)
        universes.append(result.eligible); audits.append(result.audit); summaries.append(result.summary)
    return {
        "universe": pd.concat(universes, ignore_index=True) if universes else pd.DataFrame(),
        "audit": pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(),
        "summary": pd.DataFrame(summaries),
    }
