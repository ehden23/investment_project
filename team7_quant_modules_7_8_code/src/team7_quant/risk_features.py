"""As-of market beta and upstream portfolio features for the overlay engine."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .risk_data import RiskMarket


def rolling_stock_betas(prices: pd.DataFrame, market: pd.Series, window: int = 252,
                       minimum: int = 126) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing covariance/variance using only paired returns through date t."""
    prices = prices.reindex(market.index)
    stock_returns = prices.pct_change(fill_method=None)
    market_returns = market.pct_change(fill_method=None)
    valid = stock_returns.notna().mul(market_returns.notna(), axis=0)
    # Each stock gets its own paired market variance (avoid mismatched samples).
    paired_market = pd.DataFrame(np.broadcast_to(market_returns.to_numpy()[:, None], stock_returns.shape),
                                 index=stock_returns.index, columns=stock_returns.columns).where(valid)
    covariance = stock_returns.rolling(window, min_periods=minimum).cov(market_returns)
    variance = paired_market.rolling(window, min_periods=minimum).var(ddof=1)
    betas = covariance / variance.where(variance > 0)
    counts = valid.astype(float).rolling(window, min_periods=1).sum()
    return betas, counts


def load_stock_cache(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with np.load(path, allow_pickle=False) as z:
        dates, codes = pd.DatetimeIndex(z["dates"]), z["codes"].astype(str)
        return (pd.DataFrame(z["adj_close"], index=dates, columns=codes),
                pd.DataFrame(z["volume"], index=dates, columns=codes))


def build_stock_cache(source: str | Path, out: str | Path) -> None:
    """Accept either the original wide Excel or module-1 normalized long CSV."""
    source, out = Path(source), Path(out)
    if source.suffix.lower() == ".csv":
        d = pd.read_csv(source, dtype={"stock_code": str}, parse_dates=["date"])
        d.stock_code = d.stock_code.str.zfill(6)
        if d.duplicated(["date", "stock_code"]).any():
            raise ValueError("Duplicate normalized stock prices")
        vol_col = "volume" if "volume" in d else "trading_volume"
        prices = d.pivot(index="date", columns="stock_code", values="adj_close").sort_index()
        volume = d.pivot(index="date", columns="stock_code", values=vol_col).reindex_like(prices)
    else:
        from .daily_panel import parse_daily_panel
        p = parse_daily_panel(source)
        prices, volume = p.adj_close, p.volume
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, dates=prices.index.to_numpy(dtype="datetime64[ns]"),
                        codes=prices.columns.to_numpy(dtype=str),
                        adj_close=prices.to_numpy(), volume=volume.to_numpy())


def prepare_baseline_features(baseline: str | Path, market: RiskMarket,
                              stock_cache: str | Path, out: str | Path,
                              window: int = 252, minimum: int = 126) -> pd.DataFrame:
    baseline, out = Path(baseline), Path(out)
    nav = pd.read_csv(baseline / "daily_nav.csv", parse_dates=["date"])
    holdings = pd.read_csv(baseline / "holdings.csv", dtype={"stock_code": str}, parse_dates=["date"])
    targets = pd.read_csv(baseline / "target_weights.csv", dtype={"stock_code": str}, parse_dates=["signal_date"])
    for d in [holdings, targets]:
        d.stock_code = d.stock_code.str.zfill(6)
    prices, volumes = load_stock_cache(stock_cache)
    used_codes = sorted(set(holdings.stock_code) | set(targets.stock_code))
    missing_codes = set(used_codes) - set(prices.columns)
    if missing_codes:
        raise ValueError(f"Stock history missing for {sorted(missing_codes)}")
    betas, counts = rolling_stock_betas(prices[used_codes], market.index.close, window, minimum)
    sessions = market.index.index
    nav_dates = pd.DatetimeIndex(sorted(nav.date.unique()))
    if not nav_dates.difference(sessions).empty:
        raise ValueError("Baseline NAV dates not present in KOSPI200 calendar")
    rows = []
    for strategy, g in nav.groupby("strategy_id"):
        family, aid = strategy.split("_", 1)
        aid = "EW" if aid == "EW" else "BL_ALPHA_003"
        g = g.sort_values("date").set_index("date")
        h = holdings[holdings.strategy_id == strategy].pivot(index="date", columns="stock_code", values="weight")
        h = h.reindex(index=g.index, columns=used_codes).fillna(0)
        # At inception base holdings are all cash; scheduled next-day targets
        # are known from the frozen upstream signal and used for the hedge plan.
        scheduled = targets[(targets.family == family) & (targets.allocation_id == aid)]
        by_signal = {}
        for signal, s in scheduled.groupby("signal_date"):
            cutoff = sessions[sessions <= signal][-1]
            later = sessions[sessions > signal]
            if len(later) and later[0] <= g.index.max():
                w = s.set_index("stock_code").target_weight
                if not 0 < w.sum() <= 1 + 1e-8 or (w < 0).any():
                    raise ValueError(f"Invalid upstream targets: {strategy} {signal}")
                by_signal[cutoff] = (later[0], w.reindex(used_codes).fillna(0))
        for date, n in g.iterrows():
            weights = h.loc[date]
            predicted_weights = by_signal[date][1] if date in by_signal else weights
            b = betas.loc[date]
            count = counts.loc[date]
            unavailable = b.isna()
            needed = (weights > 0) | (predicted_weights > 0)
            if (unavailable & needed).any():
                raise ValueError(f"{strategy} {date.date()}: 베타 이력 부족 {list(b.index[unavailable & needed])}; beta=1로 대체하지 않습니다.")
            b = b.fillna(0.0)  # only unused securities; positive weights checked above
            positive = weights > 1e-12
            funding_tradable = bool((volumes.reindex(index=[date], columns=used_codes).iloc[0][positive].fillna(0) > 0).all())
            rows.append({
                "date": date, "strategy_id": strategy, "base_nav": float(n.nav),
                "base_cash": float(n.cash), "base_cost": float(n.cost),
                "base_equity_weight": float(weights.sum()),
                "beta_current": float(weights @ b), "beta_forecast": float(predicted_weights @ b),
                "beta_fallback_count": int((unavailable & (predicted_weights > 0)).sum()),
                "beta_min_pairs": float(count[predicted_weights > 0].min()) if (predicted_weights > 0).any() else 0,
                "equity_rebalance_signal": date in by_signal,
                "funding_tradable": funding_tradable,
            })
    features = pd.DataFrame(rows).sort_values(["strategy_id", "date"])
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "baseline_features.csv", index=False)
    metadata = {"source": "supplied backtest outputs; unchanged upstream strategy NAV and target weights",
                "beta_window": window, "beta_minimum_pairs": minimum,
                "beta_fallback": "none; missing beta for a held/target security stops preparation",
                "stock_codes_used": len(used_codes), "baseline_strategies": sorted(features.strategy_id.unique()),
                "survivorship_bias": "UNRESOLVED: source equity panel contains current KOSPI constituents"}
    (out / "baseline_feature_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return features
