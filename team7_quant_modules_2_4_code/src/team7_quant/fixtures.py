from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_fixture(out: str | Path, seed: int = 7) -> Path:
    """기능 검증 전용 결정적 합성 입력을 만든다. 실제 성과로 해석하면 안 된다."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}" for i in range(1, 36)]
    dates = pd.bdate_range("2014-11-03", "2016-12-30")
    common = rng.normal(0.0002, 0.008, len(dates))
    ret = np.vstack([common + rng.normal(0, 0.012, len(dates)) for _ in codes]).T
    prices = 10000 * np.cumprod(1 + ret, axis=0)
    price_long = pd.DataFrame(prices, index=dates, columns=codes).stack().rename("adj_close").reset_index()
    price_long.columns = ["date", "stock_code", "adj_close"]
    price_long["close"] = price_long["adj_close"]
    price_long["trading_volume"] = 100_000 + price_long["stock_code"].astype(int) * 1_000
    price_long["trading_value"] = price_long["close"] * price_long["trading_volume"]
    price_long["market_cap"] = price_long["adj_close"] * price_long["stock_code"].astype(int).rsub(100).abs() * 1_000_000
    price_long.to_csv(out / "prices.csv", index=False)

    rets = pd.DataFrame(ret, index=dates, columns=codes).stack().rename("return").reset_index()
    rets.columns = ["date", "stock_code", "return"]
    rets.to_csv(out / "risk_returns.csv", index=False)

    fin_rows = []
    for code in codes:
        k = int(code)
        for year in [2013, 2014, 2015]:
            assets = 100_000 + 1000*k + 2000*(year-2013)
            equity = 0.55 * assets
            fin_rows.append({"stock_code": code, "fiscal_year": year, "fiscal_month_end": f"{year}-12-31",
                             "ni_parent": 0.08*equity*(1+rng.normal(0,0.1)), "equity_parent": equity,
                             "cfo": 0.07*assets, "gross_profit": 0.20*assets, "total_assets": assets,
                             "total_liabilities": assets-equity, "statement_type": "연결"})
    pd.DataFrame(fin_rows).to_csv(out / "financials.csv", index=False)
    pd.DataFrame({"stock_code": codes, "listing_date": "2005-01-01", "delisting_date": pd.NaT,
                  "market_type": "KOSPI", "security_type": "주권", "share_class": "보통주"}).to_csv(out / "security_master.csv", index=False)
    pd.DataFrame({"stock_code": codes, "date": "2010-01-01", "sector_code": ["FIN" if i%10==0 else "IND" for i in range(35)],
                  "is_financial": [i%10==0 for i in range(35)]}).to_csv(out / "sectors.csv", index=False)
    # 실제 거래일 달력의 각 분기 마지막 영업일을 신호일로 사용한다.
    rebalance = pd.to_datetime(["2015-12-31", "2016-03-31", "2016-06-30", "2016-09-30", "2016-12-30"])

    # 발표시차와 빈티지가 포함된 합성 경기자료
    months = pd.date_range("2008-01-31", "2016-12-31", freq="ME")
    vals = 100 + 1.5*np.sin(np.arange(len(months))/8)
    macro = pd.DataFrame({"observation_month": months, "release_date": months + pd.Timedelta(days=30),
                          "vintage_date": months + pd.Timedelta(days=30), "available_at": months + pd.Timedelta(days=30),
                          "indicator_code": "COINCIDENT_CYCLE", "value": vals,
                          "source": "SYNTHETIC", "vintage_status": "SYNTHETIC"})
    macro.to_csv(out / "macro.csv", index=False)
    ret_months = pd.date_range("2009-01-31", "2016-12-31", freq="ME")
    factor_ret = pd.DataFrame({"formation_date": ret_months-pd.offsets.MonthEnd(1), "return_month": ret_months,
                               "return_available_at": ret_months+pd.Timedelta(days=1),
                               "value_return": rng.normal(0.007,0.03,len(ret_months)),
                               "quality_return": rng.normal(0.006,0.025,len(ret_months)),
                               "return_basis":"SYNTHETIC_GROSS", "portfolio_rule_id":"SYNTH_V1"})
    factor_ret.to_csv(out / "factor_returns.csv", index=False)
    (out / "SYNTHETIC_ONLY.txt").write_text("기능 검증용 합성자료입니다. 실제 투자성과가 아닙니다.\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"synthetic": True, "seed": seed, "codes": len(codes),
                                                     "date_start": str(dates.min().date()), "date_end": str(dates.max().date())},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    return out

