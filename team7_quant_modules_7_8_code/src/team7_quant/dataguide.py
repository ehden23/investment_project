from __future__ import annotations

"""사용자가 내려받은 DataGuide/KOSIS Excel 원본 전용 로더.

주가 파일은 5천 개가 넘는 열을 가진 넓은 표라서 일반적인 read_excel로는
정상적으로 읽히지 않는다. XML을 한 행씩 읽으면서 월말 자료만 남기고,
직전 63거래일 거래대금 평균도 동시에 계산한다.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .io import normalize_stock_code


_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


@dataclass
class LoadedInputs:
    monthly_market: pd.DataFrame
    master: pd.DataFrame
    financials: pd.DataFrame
    macro: pd.DataFrame
    metadata: dict[str, Any]


def _excel_col_number(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def _shared_strings(book: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in book.namelist():
        return []
    root = ET.parse(book.open("xl/sharedStrings.xml")).getroot()
    return ["".join((t.text or "") for t in si.iter(_NS + "t")) for si in root.findall(_NS + "si")]


def _cell_value(cell: ET.Element, strings: list[str]) -> Any:
    typ = cell.attrib.get("t")
    value = cell.find(_NS + "v")
    if typ == "inlineStr":
        text = cell.find(".//" + _NS + "t")
        return None if text is None else text.text
    if value is None:
        return None
    raw = value.text
    if typ == "s":
        return strings[int(raw)]
    if typ in {"str", "b"}:
        return raw
    try:
        return float(raw)
    except (TypeError, ValueError):
        return raw


def load_dataguide_monthly_market(path: str | Path, liquidity_sessions: int = 63) -> tuple[pd.DataFrame, dict[str, Any]]:
    """일별 DataGuide 원본을 월말 시장자료로 축약한다.

    시가총액은 원본의 백만원을 원으로 변환한다. 수정주가와 거래대금은
    원본 단위가 이미 원이므로 그대로 보존한다.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as book:
        strings = _shared_strings(book)
        stream = book.open("xl/worksheets/sheet1.xml")
        codes_by_col: dict[int, str] = {}
        names_by_col: dict[int, str] = {}
        fields_by_col: dict[int, str] = {}
        dates: list[pd.Timestamp] = []
        monthly_blocks: list[pd.DataFrame] = []
        rolling: deque[np.ndarray] = deque()
        rolling_sum: np.ndarray | None = None
        rolling_count: np.ndarray | None = None
        last_date: pd.Timestamp | None = None
        last_adj: np.ndarray | None = None
        last_cap: np.ndarray | None = None
        last_liq: np.ndarray | None = None
        current_period = None
        codes: list[str] = []
        names: list[str] = []
        col_map: dict[int, tuple[int, str]] = {}

        def finalize_month() -> None:
            if last_date is None or last_adj is None or last_cap is None or last_liq is None:
                return
            valid = np.isfinite(last_adj) | np.isfinite(last_cap) | np.isfinite(last_liq)
            block = pd.DataFrame({
                "date": pd.Timestamp(last_date),
                "stock_code": np.asarray(codes, dtype=object),
                "stock_name": np.asarray(names, dtype=object),
                "adj_close": last_adj,
                "market_cap": last_cap,
                "avg_trading_value_63d": last_liq,
            })
            monthly_blocks.append(block.loc[valid].copy())

        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag != _NS + "row":
                continue
            row_number = int(element.attrib.get("r", "0"))
            values: dict[int, Any] = {}
            for cell in element.findall(_NS + "c"):
                values[_excel_col_number(cell.attrib["r"])] = _cell_value(cell, strings)

            if row_number in {26, 27, 29}:
                target = codes_by_col if row_number == 26 else names_by_col if row_number == 27 else fields_by_col
                for col, value in values.items():
                    if col >= 21 and value is not None:
                        target[col] = str(value)
                if row_number == 29:
                    ordered_codes: list[str] = []
                    ordered_names: list[str] = []
                    index_by_code: dict[str, int] = {}
                    item_map = {
                        "S410000700": "adj_close",
                        "S410001250": "market_cap",
                        "S410000900": "trading_value",
                    }
                    for col in sorted(fields_by_col):
                        field = item_map.get(fields_by_col[col])
                        code_raw = codes_by_col.get(col)
                        if field is None or code_raw is None:
                            continue
                        code = str(normalize_stock_code(pd.Series([code_raw])).iloc[0])
                        if code not in index_by_code:
                            index_by_code[code] = len(ordered_codes)
                            ordered_codes.append(code)
                            ordered_names.append(names_by_col.get(col, ""))
                        col_map[col] = (index_by_code[code], field)
                    codes, names = ordered_codes, ordered_names
                    rolling_sum = np.zeros(len(codes), dtype=float)
                    rolling_count = np.zeros(len(codes), dtype=np.int32)
                element.clear()
                continue

            if row_number >= 32 and 20 in values and codes:
                serial = values[20]
                try:
                    date = pd.Timestamp("1899-12-30") + pd.to_timedelta(float(serial), unit="D")
                except (TypeError, ValueError):
                    element.clear()
                    continue
                period = date.to_period("M")
                if current_period is not None and period != current_period:
                    finalize_month()
                n = len(codes)
                adj = np.full(n, np.nan)
                cap = np.full(n, np.nan)
                trading = np.full(n, np.nan)
                for col, raw in values.items():
                    mapping = col_map.get(col)
                    if mapping is None or raw is None:
                        continue
                    idx, field = mapping
                    try:
                        number = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if field == "adj_close":
                        adj[idx] = number
                    elif field == "market_cap":
                        cap[idx] = number * 1_000_000.0
                    elif field == "trading_value":
                        trading[idx] = number
                observed = np.isfinite(trading)
                rolling_sum[observed] += trading[observed]
                rolling_count[observed] += 1
                rolling.append(trading)
                if len(rolling) > liquidity_sessions:
                    old = rolling.popleft()
                    old_observed = np.isfinite(old)
                    rolling_sum[old_observed] -= old[old_observed]
                    rolling_count[old_observed] -= 1
                average = np.divide(
                    rolling_sum,
                    rolling_count,
                    out=np.full(n, np.nan),
                    where=rolling_count > 0,
                )
                last_date, last_adj, last_cap, last_liq = date, adj, cap, average
                current_period = period
                dates.append(date)
            element.clear()
        finalize_month()

    result = pd.concat(monthly_blocks, ignore_index=True)
    result = result.sort_values(["date", "stock_code"]).reset_index(drop=True)
    metadata = {
        "source_file": path.name,
        "source_format": "DataGuide wide daily calendar",
        "daily_rows_read": len(dates),
        "security_count": len(codes),
        "monthly_rows": len(result),
        "first_date": str(result["date"].min().date()),
        "last_date": str(result["date"].max().date()),
        "units": {"adj_close": "KRW", "market_cap": "KRW", "avg_trading_value_63d": "KRW"},
    }
    return result, metadata


def _parse_yyyymmdd(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def load_dataguide_master(path: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=5, dtype={"주식코드": "string", "코드": "string"})
    out = pd.DataFrame({
        "stock_code": normalize_stock_code(raw["주식코드"].fillna(raw["코드"])),
        "stock_name": raw["한글풀네임"].fillna(raw["코드명"]).astype("string"),
        "listing_date": _parse_yyyymmdd(raw["최초상장일"]),
        "delisting_date": _parse_yyyymmdd(raw["상장폐지일"]),
        "market_type": raw["시장구분"].astype("string"),
        "sector_code": raw["거래소 업종코드"].astype("string"),
        "sector_name": raw["거래소 업종"].astype("string"),
        "fiscal_month": pd.to_numeric(raw["결산월"], errors="coerce"),
        "market_transfer_date": _parse_yyyymmdd(raw["시장이전일"]),
    })
    return out.dropna(subset=["stock_code"]).drop_duplicates("stock_code", keep="last").reset_index(drop=True)


def load_dataguide_financials(path: str | Path) -> pd.DataFrame:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows: list[tuple[Any, ...]] = []
    for row in ws.iter_rows(min_row=12, values_only=True):
        code, _, month, year, quarter, ni, gp, equity, assets, liabilities, cfo = row[:11]
        if code is None or year is None or quarter != "결산":
            continue
        metrics = (ni, gp, equity, assets, liabilities, cfo)
        if not any(value is not None for value in metrics):
            continue
        rows.append((code, month, year, ni, equity, cfo, gp, assets, liabilities))
    wb.close()
    out = pd.DataFrame(rows, columns=[
        "stock_code", "fiscal_month", "fiscal_year", "ni_parent", "equity_parent",
        "cfo", "gross_profit", "total_assets", "total_liabilities",
    ])
    out["stock_code"] = normalize_stock_code(out["stock_code"])
    out["fiscal_month"] = pd.to_numeric(out["fiscal_month"], errors="coerce").fillna(12).astype(int)
    out["fiscal_year"] = pd.to_numeric(out["fiscal_year"], errors="coerce").astype("Int64")
    out["fiscal_month_end"] = pd.to_datetime(
        out["fiscal_year"].astype("string") + "-" + out["fiscal_month"].astype("string").str.zfill(2) + "-01",
        errors="coerce",
    ) + pd.offsets.MonthEnd(0)
    metric_cols = ["ni_parent", "equity_parent", "cfo", "gross_profit", "total_assets", "total_liabilities"]
    for col in metric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce") * 1_000.0  # 천원 → 원
    out["statement_type"] = "DataGuide_NFS_IFRS_M"
    out["available_at"] = out["fiscal_month_end"] + pd.DateOffset(months=4) + pd.offsets.MonthEnd(0)
    return out.sort_values(["stock_code", "fiscal_month_end"]).reset_index(drop=True)


def load_coincident_index(path: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    item_code = str(raw.iloc[11, 1])
    item_name = str(raw.iloc[12, 1])
    data = raw.iloc[14:, :2].copy()
    data.columns = ["observation_month", "value"]
    data["observation_month"] = pd.to_datetime(data["observation_month"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["observation_month", "value"])
    data["release_date"] = data["observation_month"] + pd.offsets.MonthEnd(1)
    data["available_at"] = data["release_date"]
    data["vintage_date"] = data["release_date"]
    data["indicator_code"] = item_code
    data["indicator_name"] = item_name
    data["source"] = "DataGuide/KOSIS latest revised series"
    data["vintage_status"] = "LATEST_REVISED_PROXY"
    return data.reset_index(drop=True)


def load_four_inputs(folder: str | Path) -> LoadedInputs:
    folder = Path(folder)
    market, market_meta = load_dataguide_monthly_market(folder / "stock_daily_prices_current_kospi.xlsx")
    master = load_dataguide_master(folder / "stock_master_history.xlsx")
    financials = load_dataguide_financials(folder / "financials_annual.xlsx")
    macro = load_coincident_index(folder / "macro_coincident_index.xlsx")
    metadata = {
        "market": market_meta,
        "master_rows": len(master),
        "financial_rows": len(financials),
        "macro_rows": len(macro),
        "limitations": [
            "일별 주가 파일은 다운로드 시점의 KOSPI 구성종목만 포함하므로 과거 상장폐지 종목 주가가 없다.",
            "종목종류를 구분하지 않아 우선주 등이 포함될 수 있다.",
            "재무자료의 연결·별도 구분을 추가로 재분류하지 않고 DataGuide 선택 항목을 그대로 사용한다.",
            "거시자료는 최신 수정계열이며 관측월 다음 달 말 공개를 가정한다.",
        ],
    }
    return LoadedInputs(market, master, financials, macro, metadata)


def save_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
