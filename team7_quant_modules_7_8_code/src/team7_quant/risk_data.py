"""Read the supplied DataGuide workbooks without changing the upstream modules.

All caches written here are derived from local, user supplied files.  The loader
handles multiple sheets and multiple export blocks on a single sheet, including
the offset futures table and the overlapping dividend exports in this project.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import re
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd


ITEMS = {
    "DF10000400": "close", "DF10000900": "volume",
    "DF10001200": "open_interest", "DF10001400": "iv",
    "DF10000500": "theoretical_settlement",
    "I31000010F": "open", "I31000040F": "close",
    "E11.02.003.005": "cd_percent",
    "P6000706033": "dividend_percent",
}
MULTIPLIER_CHANGE = pd.Timestamp("2017-03-27")


def contract_multiplier(date: str | pd.Timestamp) -> float:
    """Standard KOSPI200 only; outstanding positions doubled on 2017-03-27."""
    return 500_000.0 if pd.Timestamp(date) < MULTIPLIER_CHANGE else 250_000.0


def read_export_blocks(path: str | Path, item_map: dict[str, str]) -> pd.DataFrame:
    """Return date/code/field/value; preserve missing data and block provenance."""
    blocks: list[pd.DataFrame] = []
    for sheet, raw in pd.read_excel(path, sheet_name=None, header=None, dtype=object).items():
        for row in range(len(raw)):
            for col in range(raw.shape[1]):
                if raw.iat[row, col] != "아이템코드":
                    continue
                if row < 3 or row + 3 >= len(raw):
                    continue
                # Standard DataGuide block: code/name/type/item/name/frequency/data.
                date_values = raw.iloc[row + 3:, col]
                # Parse datetimes only: metadata or numerical Excel serials must
                # not silently become nanoseconds since the Unix epoch.
                dates = pd.to_datetime(date_values.where(date_values.map(
                    lambda v: isinstance(v, (pd.Timestamp, __import__('datetime').datetime)))), errors="coerce")
                for j in range(col + 1, raw.shape[1]):
                    item = str(raw.iat[row, j])
                    if item not in item_map:
                        continue
                    values = pd.to_numeric(raw.iloc[row + 3:, j], errors="coerce")
                    b = pd.DataFrame({"date": dates.to_numpy(), "value": values.to_numpy()})
                    b = b[b.date.notna() & b.value.notna()].copy()
                    if b.empty:
                        continue
                    b["contract_code"] = str(raw.iat[row - 3, j])
                    b["contract_name"] = str(raw.iat[row - 2, j])
                    b["field"] = item_map[item]
                    b["source_item"] = item
                    b["source_block"] = f"{sheet}:row{row + 1}:col{col + 1}"
                    blocks.append(b)
    if not blocks:
        raise ValueError(f"No requested numeric series in {path}")
    result = pd.concat(blocks, ignore_index=True)
    keys = ["date", "contract_code", "field"]
    duplicates = result[result.duplicated(keys, keep=False)]
    if not duplicates.empty and (duplicates.groupby(keys).value.nunique() > 1).any():
        raise ValueError(f"Conflicting duplicate observations in {path}")
    return result.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def wide_series(path: str | Path, item_map: dict[str, str]) -> pd.DataFrame:
    d = read_export_blocks(path, item_map)
    if d.contract_code.nunique() != 1:
        raise ValueError(f"Expected one underlying series in {path}")
    return d.pivot(index="date", columns="field", values="value").sort_index().rename_axis(columns=None)


def monthly_expiry(month: str, sessions: pd.DatetimeIndex) -> tuple[pd.Timestamp, str]:
    """Second Thursday, moved to the preceding observed session on a holiday.

    Dates outside the supplied trading calendar are explicitly flagged as a
    rule extrapolation, never described as verified contract master data.
    """
    start = pd.Timestamp(f"{month[:4]}-{month[4:]}-01")
    expiry = start + pd.Timedelta(days=(3 - start.weekday()) % 7 + 7)
    if expiry <= sessions.max():
        choices = sessions[sessions <= expiry]
        if not len(choices):
            raise ValueError(f"Trading calendar does not cover {month}")
        return choices[-1], "second_thursday_observed_calendar"
    return expiry, "second_thursday_extrapolated_right_censored"


def derivatives(path: str | Path, kind: str, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    d = read_export_blocks(path, {k: v for k, v in ITEMS.items() if k.startswith("DF")})
    names = d[["contract_code", "contract_name"]].drop_duplicates().set_index("contract_code")
    frames = []
    for code, g in d.groupby("contract_code", sort=False):
        name = names.loc[code, "contract_name"]
        pattern = r"P\s+(\d{6})\s+([\d.]+)" if kind == "PUT" else r"F\s+(\d{6})"
        m = re.search(pattern, name)
        if m is None:
            raise ValueError(f"Unexpected {kind} contract: {name}")
        expiry, source = monthly_expiry(m.group(1), sessions)
        a = g.pivot(index="date", columns="field", values="value").sort_index().rename_axis(columns=None)
        required = {"close", "volume", "open_interest"} | ({"iv"} if kind == "PUT" else set())
        if not required.issubset(a):
            raise ValueError(f"Missing fields for {name}: {required - set(a.columns)}")
        # A last observation is a cross-check, not the rule selecting an expiry.
        # Missing final records never move a contract's maturity backwards.
        last_quote = a.loc[a.close.notna()].index.max()
        if expiry <= sessions.max() and last_quote != expiry:
            raise ValueError(f"Verify expiry of {name}: rule={expiry.date()}, last quote={last_quote.date()}")
        a["contract_code"], a["contract_name"] = code, name
        a["month"], a["kind"], a["expiry_date"], a["expiry_source"] = m.group(1), kind, expiry, source
        a["multiplier"] = [contract_multiplier(t) for t in a.index]
        if kind == "PUT":
            a["strike"] = float(m.group(2))
            a["iv"] = a.iv.where(a.iv > 0)
            if a.iv.max() > 3:
                raise ValueError(f"IV must be in decimal units: {name}")
        frames.append(a.reset_index())
    return pd.concat(frames, ignore_index=True).sort_values(["date", "contract_code"]).reset_index(drop=True)


@dataclass
class RiskMarket:
    index: pd.DataFrame
    options: pd.DataFrame
    futures: pd.DataFrame
    macro: pd.DataFrame
    schedule: pd.DataFrame
    metadata: dict[str, Any]
    margin: pd.DataFrame | None = None

    @classmethod
    def load(cls, folder: str | Path) -> "RiskMarket":
        folder = Path(folder)
        meta = json.loads((folder / "risk_input_metadata.json").read_text(encoding="utf-8"))
        def read(name: str, dates: list[str]) -> pd.DataFrame:
            return pd.read_csv(folder / name, parse_dates=dates)
        margin_path = folder / "margin_rates.csv"
        margin = read("margin_rates.csv", ["date"]).set_index("date") if margin_path.exists() else None
        if margin is not None and margin.index.has_duplicates:
            raise ValueError("Duplicate margin dates")
        return cls(read("kospi200.csv", ["date"]).set_index("date"),
                   read("puts.csv", ["date", "expiry_date"]), read("futures.csv", ["date", "expiry_date"]),
                   read("risk_macro.csv", ["date", "available_at"]).set_index("date"),
                   read("hedge_schedule.csv", ["signal_date", "execution_date", "expiry_date"]), meta, margin)


def observed_dividends(path: str | Path) -> pd.DataFrame:
    """KRX CSV/ZIP. Actual historical observations, no annual lag or zero fill."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as z:
            inputs = [(n, z.read(n)) for n in z.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/')]
    else:
        inputs = [(path.name, path.read_bytes())]
    rows = []
    for name, data in inputs:
        try: d = pd.read_csv(BytesIO(data), encoding="utf-8-sig")
        except UnicodeDecodeError: d = pd.read_csv(BytesIO(data), encoding="cp949")
        date = "date" if "date" in d else "일자"
        q = pd.to_numeric(d["dividend_yield_decimal"], errors="raise") if "dividend_yield_decimal" in d else pd.to_numeric(d["배당수익률"], errors="raise") / 100
        a = pd.DataFrame({"date": pd.to_datetime(d[date]), "q": q, "q_source_file": name})
        if "종가" in d: a["source_index_close"] = pd.to_numeric(d["종가"], errors="raise")
        elif "kospi200_close" in d: a["source_index_close"] = pd.to_numeric(d["kospi200_close"], errors="raise")
        rows.append(a)
    if not rows: raise ValueError("배당 ZIP에 CSV가 없습니다.")
    d = pd.concat(rows, ignore_index=True)
    if d.date.isna().any() or not np.isfinite(d.q).all() or ((d.q < 0) | (d.q >= 1)).any():
        raise ValueError("배당 날짜/값/단위를 확인하세요.")
    columns = [c for c in ['q', 'source_index_close'] if c in d]
    if d.groupby('date')[columns].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Conflicting daily dividend observations")
    d = d.drop_duplicates('date').sort_values('date').set_index('date')
    d['available_at'] = d.index
    d['q_status'] = 'OBSERVED_KRX_INDEX_YIELD_NOT_FORWARD_DIVIDEND_FORECAST'
    return d


def connect_observed_inputs(prepared: str | Path, dividend_source: str | Path,
                            margin_source: str | Path) -> None:
    """Update audited market inputs without re-running or changing modules 1–6."""
    folder = Path(prepared)
    index = pd.read_csv(folder/'kospi200.csv', parse_dates=['date']).set_index('date')
    macro = pd.read_csv(folder/'risk_macro.csv', parse_dates=['date']).set_index('date')
    dividend = observed_dividends(dividend_source).reindex(index.index)
    if dividend.q.isna().any(): raise ValueError("일별 배당이 지수 달력을 모두 덮지 않습니다.")
    if 'source_index_close' in dividend and (dividend.source_index_close-index.close).abs().gt(.005).any():
        raise ValueError("배당 파일의 지수 종가가 KOSPI200과 다릅니다.")
    updated = dividend[['q','q_status','available_at','q_source_file']].join(macro[['cd_rate','r_continuous']])
    margin = pd.read_csv(margin_source, parse_dates=['date'])
    required = ['eod_initial_rate','eod_maintenance_rate','session_initial_rate','session_maintenance_rate']
    if margin.date.isna().any() or margin.date.duplicated().any() or not set(required).issubset(margin):
        raise ValueError("Invalid margin date/columns")
    if not np.isfinite(margin[required].to_numpy(float)).all() or (margin[required]<=0).any().any() or (margin[required]>1).any().any():
        raise ValueError("증거금 요율 결측/단위를 확인하세요.")
    if (margin.eod_maintenance_rate > margin.eod_initial_rate).any(): raise ValueError("유지증거금률 > 위탁증거금률")
    if margin.date.min() not in index.index: raise ValueError("증거금 날짜와 지수 달력을 확인하세요.")
    meta = json.loads((folder/'risk_input_metadata.json').read_text(encoding='utf-8'))
    meta.pop('dividend_item', None)
    meta['inputs'] = [x for x in meta.get('inputs', []) if not x['name'].startswith('현금배당수익률')]
    meta.update({'dividend_policy':'observed_index_yield',
                 'dividend_definition':'Daily historical KRX index yield; BSM input proxy, not maturity-specific expected dividends.',
                 'margin_policy':'Published standalone futures rate history; date inference flags retained; no COMS offsets.',
                 'margin_missing_rates':0})
    meta['observed_input_sources'] = [{'name':Path(p).name,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in [dividend_source,margin_source]]
    updated.to_csv(folder/'risk_macro.csv',index_label='date')
    margin.to_csv(folder/'margin_rates.csv',index=False)
    (folder/'risk_input_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')


def find_input(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {pattern!r} in {folder}; found {[p.name for p in matches]}")
    return matches[0]


def prepare_risk_market(source: str | Path, out: str | Path, cfg: dict[str, Any]) -> RiskMarket:
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index_file = find_input(source, "KOSPI200 *지수.xlsx")
    cd_file = find_input(source, "CD금리*.xlsx")
    dividend_file = find_input(source, "KOSPI200배당수익률.zip")
    put_file = source / "2016~2025 풋옵션 data.xlsx"
    future_file = source / "2016~2025 선물 data.xlsx"
    index = wide_series(index_file, {"I31000010F": "open", "I31000040F": "close"})
    cd = wide_series(cd_file, {"E11.02.003.005": "cd_percent"})
    div = observed_dividends(dividend_file)
    if index.index.duplicated().any() or (index.close <= 0).any():
        raise ValueError("Invalid KOSPI200 calendar/prices")
    if not index.index.difference(cd.index).empty:
        raise ValueError("CD series does not cover the full index calendar")
    macro = div.reindex(index.index).copy()
    if macro.q.isna().any(): raise ValueError("Missing observed daily dividend yield")
    macro["cd_rate"] = cd.cd_percent.reindex(index.index) / 100
    # Convert the quoted 91-day simple rate to a continuous equivalent.
    tenor = 91 / 365
    macro["r_continuous"] = np.log1p(macro.cd_rate * tenor) / tenor
    puts = derivatives(put_file, "PUT", index.index)
    futures = derivatives(future_file, "FUTURES", index.index)
    start = pd.Timestamp(cfg.get("first_signal_date", "2015-12-31"))
    end = pd.Timestamp(cfg.get("evaluation_end", "2025-12-31"))
    roll_days = int(cfg.get("roll_sessions", 5))
    end_session = index.index[index.index <= end][-1]
    rows = []
    for kind, data in [("PUT", puts), ("FUTURES", futures)]:
        contracts = data[["contract_code", "contract_name", "month", "expiry_date", "expiry_source"]].drop_duplicates().sort_values("expiry_date")
        if kind == "PUT" and contracts.month.duplicated().any():
            raise ValueError("This input mode requires one preselected put per monthly maturity")
        for i, (_, contract) in enumerate(contracts.iterrows()):
            if i == 0:
                signal = index.index[index.index <= start][-1]
                choices = index.index[index.index > start]
                if not len(choices):
                    continue
                execution = choices[0]
            else:
                expiry = contracts.iloc[i - 1].expiry_date
                if expiry > end_session:
                    continue
                pos = index.index.get_loc(expiry)
                execution, signal = index.index[pos - roll_days], index.index[pos - roll_days - 1]
            if execution > end_session:
                continue
            g = data[(data.contract_code == contract.contract_code) & (data.date == signal)]
            if g.empty or g[["close", "volume", "open_interest"]].isna().any().any() or (g[["close", "volume", "open_interest"]] <= 0).any().any():
                raise ValueError(f"Unusable formation quote: {contract.contract_name}, {signal}")
            row = {**contract.to_dict(), "kind": kind, "signal_date": signal, "execution_date": execution}
            if kind == "PUT":
                row["strike"] = float(g.iloc[0].strike)
                row["target_strike"] = float(index.loc[signal, "close"] * 0.95)
                row["strike_gap"] = row["strike"] - row["target_strike"]
                if row['strike'] <= 0: raise ValueError('Invalid strike')
            rows.append(row)
    schedule = pd.DataFrame(rows).sort_values(["signal_date", "kind"])
    files = [index_file, cd_file, dividend_file, put_file, future_file]
    meta = {
        "inputs": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
        "put_contracts": int(puts.contract_code.nunique()), "futures_contracts": int(futures.contract_code.nunique()),
        "dividend_units": "percent converted to decimal",
        "dividend_policy": "observed_index_yield",
        "dividend_definition": "Historical KRX index yield, not maturity-specific expected dividends",
        "execution": "signal at t close; execution at next session close",
        "multiplier_source": "Daishin/Hanwha 2017-03-27 notice: 500000 -> 250000; outstanding quantities doubled",
        "multiplier_source_url": "https://money2.daishin.com/html/Notice/2017/grch0321.html",
        "futures_mark": "observed close, NOT DF10000500 theoretical settlement",
        "expiry_policy": "second Thursday adjusted by supplied index calendar; last quote cross-check; 2026 maturities right censored",
        "option_selection": "user-collected monthly candidates; distance from 95% target reported; full-chain optimality not claimed",
    }
    index.to_csv(out / "kospi200.csv", index_label="date")
    macro.to_csv(out / "risk_macro.csv", index_label="date")
    puts.to_csv(out / "puts.csv", index=False)
    futures.to_csv(out / "futures.csv", index=False)
    schedule.to_csv(out / "hedge_schedule.csv", index=False)
    (out / "risk_input_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    margin_path = Path(cfg.get('margin_file') or out/'margin_rates.csv')
    if margin_path.exists(): connect_observed_inputs(out, dividend_file, margin_path)
    return RiskMarket.load(out)
