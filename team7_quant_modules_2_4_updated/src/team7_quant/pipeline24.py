from __future__ import annotations

"""업로드된 주가·마스터·재무·선행지수·배당 원본으로 모듈 2~4를 실행한다."""

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .dataguide import load_four_inputs, save_metadata
from .module2 import compute_ic, run_rebalance_dates
from .module34 import build_monthly_regime_history, build_weight_outputs, regime_summary


def _financial_snapshot(financials: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    usable = financials[pd.to_datetime(financials["available_at"]) <= asof].copy()
    if usable.empty:
        return usable
    return usable.sort_values(["stock_code", "fiscal_month_end"]).groupby("stock_code", as_index=False).tail(1)


def build_approximate_universe(
    month_ends: list[pd.Timestamp],
    market: pd.DataFrame,
    master: pd.DataFrame,
    financials: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시점별 KOSPI 투자 가능 종목을 만든다.

    상폐 종목도 해당 시점에 가격·재무가 존재하고 상폐일 전이면 포함한다.
    마스터의 과거 시장구분은 완전한 이력이 아니므로 KS 또는 업종명에
    '코스피'가 남아 있는 종목을 KOSPI 대용 분류로 사용한다.
    """
    cutoff = float(config.get("bottom_percentile_cutoff", 0.20))
    minimum_months = int(config.get("minimum_listing_months", 12))
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    required = ["ni_parent", "equity_parent", "cfo", "total_assets", "total_liabilities"]

    for asof in month_ends:
        p = market[market["date"] <= asof].sort_values(["stock_code", "date"]).groupby("stock_code", as_index=False).tail(1)
        p = p[p["date"].dt.to_period("M") == asof.to_period("M")].copy()
        p = p.merge(master[["stock_code", "listing_date", "delisting_date", "market_type", "sector_name"]], on="stock_code", how="left")
        p["kospi_history_proxy"] = p["market_type"].eq("KS") | p["sector_name"].fillna("").str.contains("코스피")
        p["listed"] = p["listing_date"].notna() & (p["listing_date"] + pd.DateOffset(months=minimum_months) <= asof)
        p["not_delisted"] = p["delisting_date"].isna() | (asof <= p["delisting_date"])
        base = p["kospi_history_proxy"] & p["listed"] & p["not_delisted"] & p["market_cap"].gt(0) & p["avg_trading_value_63d"].notna()
        p["liquidity_percentile"] = np.nan
        p["market_cap_percentile"] = np.nan
        p.loc[base, "liquidity_percentile"] = p.loc[base, "avg_trading_value_63d"].rank(pct=True, method="average")
        p.loc[base, "market_cap_percentile"] = p.loc[base, "market_cap"].rank(pct=True, method="average")
        fin = _financial_snapshot(financials, asof)
        p = p.merge(fin[["stock_code", *required]] if not fin.empty else pd.DataFrame(columns=["stock_code", *required]), on="stock_code", how="left")
        p["has_financials"] = p[required].notna().all(axis=1)
        p["eligible"] = (
            base
            & p["liquidity_percentile"].gt(cutoff)
            & p["market_cap_percentile"].gt(cutoff)
            & p["equity_parent"].gt(0)
            & p["has_financials"]
        )
        selected = p.loc[p["eligible"], ["stock_code"]].copy()
        selected["date"] = asof
        rows.append(selected[["date", "stock_code"]])
        summaries.append({
            "date": asof,
            "market_observations": len(p),
            "eligible_count": int(p["eligible"].sum()),
            "bottom_liquidity_excluded": int((base & p["liquidity_percentile"].le(cutoff)).sum()),
            "bottom_market_cap_excluded": int((base & p["market_cap_percentile"].le(cutoff)).sum()),
            "missing_financials": int((base & ~p["has_financials"]).sum()),
            "delisted_firms_eligible_at_that_time": int((p["eligible"] & p["delisting_date"].notna()).sum()),
            "market_history_ambiguous": int((~p["kospi_history_proxy"] & p["market_type"].eq("EX")).sum()),
        })
    universe = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "stock_code"])
    return universe, pd.DataFrame(summaries)


def _sector_history(master: pd.DataFrame) -> pd.DataFrame:
    name = master["sector_name"].fillna("").astype(str)
    return pd.DataFrame({
        "stock_code": master["stock_code"],
        "date": pd.Timestamp("1900-01-31"),
        "sector_code": master["sector_code"].fillna("UNKNOWN"),
        "is_financial": name.str.contains("금융|은행|보험|증권", regex=True),
    })


def _module2_prices(monthly_market: pd.DataFrame) -> pd.DataFrame:
    return monthly_market[["date", "stock_code", "adj_close", "market_cap"]].copy()


def add_dividend_reinvestment_proxy(
    prices: pd.DataFrame, dividends: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """주당현금배당금(우선) 또는 연간 지급액(대체)을 이용해 총수익지수를 만든다.

    지급액은 공시 사용 가능 월의 시가총액으로 나누고 그 월에 재투자한다.
    실제 배당락일·주당배당금이 없으므로 이는 정확한 총수익이 아닌 대용치다.
    100%를 넘는 비정상 대용 배당수익률은 적용하지 않고 로그에 남긴다.
    """
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["month"] = p["date"].dt.to_period("M")
    p = p.sort_values(["stock_code", "date"]).drop_duplicates(["stock_code", "month"], keep="last")
    p["dividend_yield_proxy"] = 0.0
    p["dividend_per_share_applied"] = 0.0
    logs: list[dict[str, Any]] = []
    if not dividends.empty:
        d = dividends.copy()
        d["month"] = pd.to_datetime(d["available_at"]).dt.to_period("M")
        if "dividend_per_share" in d.columns:
            d = d.groupby(["stock_code", "month"], as_index=False)["dividend_per_share"].sum(min_count=1)
            p = p.merge(d, on=["stock_code", "month"], how="left")
            previous_price = p.groupby("stock_code")["adj_close"].shift(1)
            raw_yield = p["dividend_per_share"] / previous_price
            p.loc[p["dividend_per_share"].notna(), "dividend_per_share_applied"] = p.loc[
                p["dividend_per_share"].notna(), "dividend_per_share"
            ]
        else:
            d = d.groupby(["stock_code", "month"], as_index=False)["dividend_cash_paid"].sum(min_count=1)
            p = p.merge(d, on=["stock_code", "month"], how="left")
            raw_yield = p["dividend_cash_paid"] / p["market_cap"]
        valid = raw_yield.between(0, 1, inclusive="both")
        p.loc[valid, "dividend_yield_proxy"] = raw_yield[valid]
        bad = p[raw_yield.notna() & ~valid]
        logs.extend({"date": r.date, "stock_code": r.stock_code,
                     "reason": "dividend_proxy_yield_out_of_range", "raw_yield": float(y)}
                    for r, y in zip(bad.itertuples(), raw_yield.loc[bad.index]))
    p["price_return"] = p.groupby("stock_code")["adj_close"].pct_change(fill_method=None)
    previous_month = p.groupby("stock_code")["month"].shift(1)
    consecutive = (p["month"].astype("int64") - previous_month.astype("int64")) == 1
    p.loc[~consecutive, "price_return"] = np.nan
    # DPS 수익률은 기간 초 가격을 분모로 하므로 가격수익률에 더한다.
    # 현금흐름 대용치는 기존 방식대로 복리 결합한다.
    if "dividend_per_share" in p.columns:
        p["total_return_proxy"] = p["price_return"] + p["dividend_yield_proxy"]
    else:
        p["total_return_proxy"] = (1 + p["price_return"]) * (1 + p["dividend_yield_proxy"]) - 1

    def compound(group: pd.DataFrame) -> pd.Series:
        values = np.full(len(group), np.nan)
        level = 100.0
        if len(group):
            values[0] = level
        for i, ret in enumerate(group["total_return_proxy"].iloc[1:], start=1):
            if pd.notna(ret):
                level *= 1.0 + float(ret)
                values[i] = level
        return pd.Series(values, index=group.index)

    p["total_return_index_proxy"] = np.nan
    for _, index in p.groupby("stock_code", sort=False).groups.items():
        p.loc[index, "total_return_index_proxy"] = compound(p.loc[index]).to_numpy()
    p = p.drop(columns=["month"], errors="ignore")
    return p, pd.DataFrame(logs)


def build_factor_returns(scores: pd.DataFrame, prices: pd.DataFrame, count: int = 30, price_col: str = "adj_close") -> pd.DataFrame:
    """동일 유니버스에서 가치 단독·퀄리티 단독 동일가중 월수익률을 만든다."""
    prices = prices.copy()
    prices["month"] = pd.to_datetime(prices["date"]).dt.to_period("M")
    px = prices.sort_values("date").groupby(["month", "stock_code"], as_index=False).tail(1)
    px = px.set_index(["month", "stock_code"])[price_col]
    rows: list[dict[str, Any]] = []
    for date, group in scores.groupby("asof_date"):
        date = pd.Timestamp(date)
        formation_month = date.to_period("M")
        return_month = formation_month + 1
        if return_month > prices["month"].max():
            continue
        g = group[~group["all_factors_missing"]].copy()
        g = g.sort_values(["stock_code"]).drop_duplicates("stock_code")

        def portfolio_return(score: str) -> tuple[float, int]:
            names = g.sort_values([score, "market_cap", "stock_code"], ascending=[False, False, True]).head(count)["stock_code"]
            start = px.reindex(pd.MultiIndex.from_product([[formation_month], names])).droplevel(0)
            end = px.reindex(pd.MultiIndex.from_product([[return_month], names])).droplevel(0)
            ret = (end / start - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
            return (float(ret.mean()) if len(ret) else np.nan, int(len(ret)))

        value_return, value_n = portfolio_return("S_V")
        quality_return, quality_n = portfolio_return("S_Q")
        rows.append({
            "formation_date": date,
            "return_month": return_month.to_timestamp("M"),
            "return_available_at": return_month.to_timestamp("M"),
            "value_return": value_return,
            "quality_return": quality_return,
            "value_n": value_n,
            "quality_n": quality_n,
            "return_basis": "equal_weight_dividend_reinvestment_proxy_before_cost" if price_col != "adj_close" else "equal_weight_adjusted_price_before_cost",
            "portfolio_rule_id": f"standalone_top_{count}_same_universe",
        })
    return pd.DataFrame(rows)


def run_modules_2_to_4(input_folder: str | Path, output_folder: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    input_folder, output_folder = Path(input_folder), Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    normalized = output_folder / "normalized"
    normalized.mkdir(exist_ok=True)

    inputs = load_four_inputs(input_folder)
    inputs.monthly_market.to_csv(normalized / "monthly_market.csv", index=False)
    inputs.master.to_csv(normalized / "stock_master.csv", index=False)
    inputs.financials.to_csv(normalized / "financials.csv", index=False)
    inputs.macro.to_csv(normalized / "macro.csv", index=False)
    inputs.dividends.to_csv(normalized / "dividends.csv", index=False)
    save_metadata(inputs.metadata, output_folder / "input_metadata.json")

    calendar_month_ends = list(pd.date_range("2008-12-31", "2025-12-31", freq="ME"))
    universe, universe_summary = build_approximate_universe(
        calendar_month_ends, inputs.monthly_market, inputs.master, inputs.financials, config["universe"]
    )
    universe.to_csv(output_folder / "approximate_universe.csv", index=False)
    universe_summary.to_csv(output_folder / "universe_summary.csv", index=False)

    prices = _module2_prices(inputs.monthly_market)
    return_prices, dividend_logs = add_dividend_reinvestment_proxy(prices, inputs.dividends)
    return_prices.to_csv(normalized / "monthly_total_return_proxy.csv", index=False)
    dividend_logs.to_csv(output_folder / "dividend_proxy_logs.csv", index=False)
    sectors = _sector_history(inputs.master)
    all_scores = run_rebalance_dates(
        calendar_month_ends,
        universe,
        {"prices": prices, "financials": inputs.financials, "sectors": sectors},
        None,
        config["factor"],
    )
    scores = all_scores["scores"]
    factor_returns = build_factor_returns(
        scores, return_prices, int(config["factor"].get("selected_count", 30)), "total_return_index_proxy"
    )
    factor_returns.to_csv(output_folder / "factor_returns.csv", index=False)

    macro_month_ends = list(pd.date_range("2008-01-31", "2025-12-31", freq="ME"))
    regimes = build_monthly_regime_history(macro_month_ends, inputs.macro, config["regime"])
    weights = build_weight_outputs(regimes, factor_returns, config["regime"])
    regimes.to_csv(output_folder / "monthly_regimes.csv", index=False)
    regime_summary(regimes).to_csv(output_folder / "regime_summary.csv", index=False)
    for name, frame in weights.items():
        frame.to_csv(output_folder / f"{name}.csv", index=False)

    monthly_weights = weights["monthly_factor_weights"].copy()
    rebalance_mask = monthly_weights["date"].dt.month.isin(config["factor"].get("rebalance_months", [3, 6, 9, 12]))
    rebalance_mask |= monthly_weights["regime_changed"].fillna(False)
    rebalance_dates = list(pd.to_datetime(monthly_weights.loc[rebalance_mask, "date"]))
    final_module2 = run_rebalance_dates(
        rebalance_dates,
        universe,
        {"prices": prices, "financials": inputs.financials, "sectors": sectors},
        monthly_weights,
        config["factor"],
    )
    for name, frame in final_module2.items():
        frame.to_csv(output_folder / f"module2_{name}.csv", index=False)

    scores.to_csv(output_folder / "module2_monthly_score_history.csv", index=False)
    ic_monthly, ic_summary = compute_ic(
        scores,
        return_prices.assign(adj_close=return_prices["total_return_index_proxy"]),
        config["factor"].get("ic_start", "2009-01-01"),
        config["factor"].get("ic_end", "2015-12-31"),
    )
    ic_monthly.to_csv(output_folder / "module2_ic_monthly.csv", index=False)
    ic_summary.to_csv(output_folder / "module2_ic_summary.csv", index=False)

    run_summary = {
        "input_files": inputs.metadata["input_files"],
        "macro_indicator": "선행지수 순환변동치(2020=100)",
        "dividend_return_basis": (
            "보통주 주당현금배당금/직전 월말 수정주가; 결산 후 4개월 시점 재투자 대용"
            if "dividend_per_share" in inputs.dividends.columns
            else "연간 현금흐름표 배당금 지급액/시가총액 대용치; 사용 가능 월 재투자"
        ),
        "delisted_securities_with_price_history": inputs.metadata["delisted_securities_with_price_history"],
        "universe_months": len(calendar_month_ends),
        "score_rows": len(scores),
        "factor_return_months": len(factor_returns),
        "regime_months": len(regimes),
        "final_rebalance_dates": len(rebalance_dates),
        "final_selected_rows": len(final_module2["selections"]),
        "limitations": inputs.metadata["limitations"],
    }
    (output_folder / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_summary
