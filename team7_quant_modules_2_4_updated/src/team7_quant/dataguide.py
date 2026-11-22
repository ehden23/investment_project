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
    dividends: pd.DataFrame
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
        stream.close()

    if not monthly_blocks:
        raise ValueError(
            "월별 주가를 한 건도 읽지 못했습니다. 종목·항목이 행이고 날짜가 열인 파일이면 "
            "전치형 로더가 선택되어야 합니다. 헤더에서 '코드'와 '아이템코드'를 확인하세요."
        )
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


def load_dataguide_transposed_market(
    path: str | Path, liquidity_sessions: int = 63
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """종목·항목이 행이고 거래일이 열인 DataGuide 파일을 월말 자료로 변환한다.

    열 위치는 고정하지 않는다. 실제 캡처 파일처럼 앞쪽에 빈 열이 있고
    J:O가 코드/코드명/유형/아이템코드/아이템명/집계주기, P열 이후가
    일별 날짜인 경우도 헤더 이름으로 자동 탐지한다. 월말 가격·시총·
    상장주식수와 해당 월말까지 최근 63거래일 평균 거래대금을 산출한다.
    """
    path = Path(path)
    item_map = {
        "S41000060F": "close",
        "S410000700": "adj_close",
        "S41000080F": "volume",
        "S410000900": "trading_value",
        "S41000100F": "listed_shares",
        "S410001200": "market_cap",
        "S410001250": "market_cap",  # 다른 DataGuide 화면에서 쓰이는 시총 코드
    }
    records: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
    codes_seen: set[str] = set()
    daily_columns = 0
    with zipfile.ZipFile(path) as book:
        strings = _shared_strings(book)
        stream = book.open("xl/worksheets/sheet1.xml")
        date_by_col: dict[int, pd.Timestamp] = {}
        month_end_cols: dict[int, pd.Timestamp] = {}
        header_row_number: int | None = None
        header_cols: dict[str, int] = {}
        first_date_col: int | None = None
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag != _NS + "row":
                continue
            row_number = int(element.attrib.get("r", "0"))
            values = {_excel_col_number(c.attrib["r"]): _cell_value(c, strings)
                      for c in element.findall(_NS + "c")}
            normalized_by_col = {
                col: re.sub(r"\s+", "", str(value))
                for col, value in values.items() if value is not None
            }
            row_labels = set(normalized_by_col.values())
            # 일부 DataGuide 파일은 헤더 문구를 수식/서식 형태로 저장해 문자열
            # 탐지가 되지 않는다. 이 경우 G열 이후의 연속된 거래일로 헤더를 찾는다.
            date_candidates: dict[int, pd.Timestamp] = {}
            if header_row_number is None:
                for col, raw in values.items():
                    if raw is None:
                        continue
                    parsed = pd.NaT
                    if isinstance(raw, (int, float)) and 30_000 <= float(raw) <= 60_000:
                        parsed = pd.Timestamp("1899-12-30") + pd.to_timedelta(float(raw), unit="D")
                    elif isinstance(raw, str) and re.search(r"\d{4}[-/.]?\d{1,2}[-/.]?\d{1,2}", raw):
                        parsed = pd.to_datetime(raw, errors="coerce")
                    if pd.notna(parsed):
                        date_candidates[col] = pd.Timestamp(parsed).normalize()
            candidate_cols = sorted(date_candidates)
            # 조회조건 영역에도 시작일·종료일 두 개가 있으므로 그것을 데이터
            # 헤더로 오인하지 않는다. 날짜만으로 탐지하는 예외 모드는 날짜가
            # 3개 이상 연속 열에 있고 오름차순일 때만 허용한다.
            dates_are_ordered = (
                len(date_candidates) >= 3
                and candidate_cols == list(range(candidate_cols[0], candidate_cols[-1] + 1))
                and [date_candidates[c] for c in candidate_cols]
                == sorted(date_candidates.values())
            )
            has_named_header = {
                "코드", "코드명", "아이템코드", "아이템명", "집계주기"
            }.issubset(row_labels)
            if header_row_number is None and (has_named_header or dates_are_ordered):
                header_row_number = row_number
                aliases = {
                    "stock_code": {"코드", "주식코드"},
                    "stock_name": {"코드명", "종목명", "한글풀네임"},
                    "security_type": {"유형", "유형코드"},
                    "item_code": {"아이템코드", "항목코드"},
                    "item_name": {"아이템명", "항목명"},
                    "frequency": {"집계주기", "주기"},
                }
                for key, names in aliases.items():
                    matched = [col for col, label in normalized_by_col.items() if label in names]
                    if matched:
                        header_cols[key] = matched[0]
                metadata_end_col = header_cols.get("frequency", 0)
                date_candidates = {
                    col: date for col, date in date_candidates.items()
                    if col > metadata_end_col
                }
                source_dates = date_candidates if date_candidates else values
                for col, raw in source_dates.items():
                    if col <= metadata_end_col or raw is None:
                        continue
                    if isinstance(raw, pd.Timestamp):
                        parsed = raw
                    elif isinstance(raw, (int, float)):
                        try:
                            parsed = pd.Timestamp("1899-12-30") + pd.to_timedelta(float(raw), unit="D")
                        except (TypeError, ValueError):
                            continue
                    else:
                        parsed = pd.to_datetime(raw, errors="coerce")
                        if pd.isna(parsed):
                            continue
                    date_by_col[col] = pd.Timestamp(parsed).normalize()
                if date_by_col:
                    first_date_col = min(date_by_col)
                ordered = sorted(date_by_col)
                for i, col in enumerate(ordered):
                    date = date_by_col[col]
                    next_month = i == len(ordered) - 1 or date_by_col[ordered[i + 1]].to_period("M") != date.to_period("M")
                    if next_month:
                        month_end_cols[col] = date
                daily_columns = len(date_by_col)
                element.clear()
                continue
            if header_row_number is None or row_number <= header_row_number or not date_by_col:
                element.clear()
                continue
            code_col = header_cols.get("stock_code")
            name_col = header_cols.get("stock_name")
            item_col = header_cols.get("item_code")
            if code_col is None or item_col is None or first_date_col is None:
                element.clear()
                continue
            raw_code = values.get(code_col)
            raw_name = values.get(name_col) if name_col is not None else ""
            item_code = values.get(item_col)
            field = item_map.get(str(item_code))
            if raw_code is None or field is None:
                element.clear()
                continue
            code = str(normalize_stock_code(pd.Series([raw_code])).iloc[0])
            codes_seen.add(code)
            numeric_by_col: dict[int, float] = {}
            for col, raw in values.items():
                if col < first_date_col or raw is None:
                    continue
                try:
                    numeric_by_col[col] = float(raw)
                except (TypeError, ValueError):
                    continue
            if field == "trading_value":
                ordered_cols = sorted(date_by_col)
                daily = pd.Series([numeric_by_col.get(c, np.nan) for c in ordered_cols], dtype=float)
                rolling_mean = daily.rolling(liquidity_sessions, min_periods=1).mean()
                rolling_by_col = dict(zip(ordered_cols, rolling_mean))
            else:
                rolling_by_col = {}
            for col, date in month_end_cols.items():
                key = (date, code)
                rec = records.setdefault(key, {"date": date, "stock_code": code, "stock_name": str(raw_name or "")})
                if field == "trading_value":
                    rec["avg_trading_value_63d"] = rolling_by_col.get(col, np.nan)
                else:
                    value = numeric_by_col.get(col, np.nan)
                    rec[field] = value * 1_000_000.0 if field == "market_cap" and pd.notna(value) else value
            element.clear()
        stream.close()
    result = pd.DataFrame(records.values())
    if result.empty:
        detected = ", ".join(f"{key}={value}" for key, value in sorted(header_cols.items())) or "없음"
        raise ValueError(
            "전치형 주가 파일의 헤더는 찾았지만 데이터 행을 읽지 못했습니다. "
            f"탐지된 헤더 행={header_row_number}, 열={detected}, "
            f"첫 날짜 열={first_date_col}. DataGuide 파일에서 13행의 코드·아이템코드와 "
            "P열 이후 날짜가 실제 값으로 저장되어 있는지 확인하세요."
        )
    required = ["close", "adj_close", "volume", "avg_trading_value_63d", "listed_shares", "market_cap"]
    for col in required:
        if col not in result:
            result[col] = np.nan
    result = result.sort_values(["date", "stock_code"]).reset_index(drop=True)
    metadata = {
        "source_file": path.name,
        "source_format": "DataGuide transposed daily (security-item rows, date columns)",
        "daily_columns_read": daily_columns,
        "header_row": header_row_number,
        "detected_columns": header_cols,
        "first_date_column_number": first_date_col,
        "security_count": len(codes_seen),
        "monthly_rows": len(result),
        "first_date": str(result["date"].min().date()) if len(result) else None,
        "last_date": str(result["date"].max().date()) if len(result) else None,
        "units": {"close": "KRW", "adj_close": "KRW", "market_cap": "KRW",
                  "avg_trading_value_63d": "KRW", "listed_shares": "shares"},
    }
    return result, metadata


def load_market_auto(path: str | Path, liquidity_sessions: int = 63) -> tuple[pd.DataFrame, dict[str, Any]]:
    """두 DataGuide 출력 방향을 헤더로 판별해 알맞은 로더를 호출한다."""
    # 파일명·한글 헤더가 달라도 항목코드 위치로 출력 방향을 판별한다.
    # 전치형: S410... 항목코드가 D열 등 왼쪽 메타데이터 영역에 있음.
    # 기존형: S410... 항목코드가 20열 이후의 넓은 헤더에 있음.
    transposed = False
    with zipfile.ZipFile(path) as book:
        strings = _shared_strings(book)
        with book.open("xl/worksheets/sheet1.xml") as stream:
            parser = ET.iterparse(stream, events=("end",))
            for _, element in parser:
                if element.tag != _NS + "row":
                    continue
                row_number = int(element.attrib.get("r", "0"))
                labels = set()
                left_item_code = False
                for cell in element.findall(_NS + "c"):
                    col = _excel_col_number(cell.attrib["r"])
                    value = _cell_value(cell, strings)
                    if value is None:
                        continue
                    compact = re.sub(r"\s+", "", str(value))
                    if col <= 30:
                        labels.add(compact)
                    if col <= 30 and compact.upper().startswith("S410"):
                        left_item_code = True
                element.clear()
                if left_item_code or ("코드" in labels and "아이템코드" in labels):
                    transposed = True
                    break
                if row_number >= 100:
                    break
            del parser
    if transposed:
        return load_dataguide_transposed_market(path, liquidity_sessions)
    return load_dataguide_monthly_market(path, liquidity_sessions)


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


def load_kosis_leading_cycle(path: str | Path) -> pd.DataFrame:
    """KOSIS의 넓은 표에서 '선행지수순환변동치'만 정규화한다.

    업로드 파일은 발표일/빈티지를 제공하지 않으므로 관측월 다음 달 말에
    이용 가능했다고 보수적으로 가정한다. 최신 수정계열이라는 한계는
    ``vintage_status``에 남긴다.
    """
    raw = pd.read_excel(path, header=None)
    labels = raw.iloc[:, 0].astype("string").str.replace(" ", "", regex=False)
    matched = raw.index[labels.eq("선행지수순환변동치")]
    if len(matched) != 1:
        raise ValueError("선행지수순환변동치 행을 정확히 하나 찾을 수 없습니다.")
    month_row = raw.iloc[2, 1:]
    value_row = raw.loc[matched[0], raw.columns[1:]]
    month_text = month_row.astype("string").str.extract(r"(\d{6})", expand=False)
    out = pd.DataFrame({
        "observation_month": pd.to_datetime(month_text, format="%Y%m", errors="coerce"),
        "value": pd.to_numeric(value_row, errors="coerce"),
    }).dropna()
    out["observation_month"] = out["observation_month"] + pd.offsets.MonthEnd(0)
    out["release_date"] = out["observation_month"] + pd.offsets.MonthEnd(1)
    out["available_at"] = out["release_date"]
    out["vintage_date"] = out["release_date"]
    out["indicator_code"] = "KOSIS_LEADING_CYCLE_2020_100"
    out["indicator_name"] = "선행지수 순환변동치(2020=100)"
    out["source"] = "KOSIS 경기종합지수 latest revised series"
    out["vintage_status"] = "LATEST_REVISED_PROXY"
    return out.reset_index(drop=True)


def load_dataguide_dividends(path: str | Path) -> pd.DataFrame:
    """연간 현금흐름표의 배당금 지급액을 읽는다.

    이 항목은 주당배당금이나 배당락 자료가 아니다. 따라서 산출된 값은
    정확한 배당수익률이 아니라 '연간 현금배당 지급액 대용치'로 명명한다.
    """
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows: list[tuple[Any, ...]] = []
    item_code = str(ws.cell(10, ws.max_column).value or "")
    item_name = str(ws.cell(11, ws.max_column).value or "")
    for row in ws.iter_rows(min_row=13, values_only=True):
        # DataGuide 회계기준 파일은 앞쪽 빈 열 수가 파일마다 다를 수 있다.
        populated = [(i, v) for i, v in enumerate(row) if v is not None]
        if len(populated) < 5:
            continue
        start = next((i for i, v in populated if str(v).startswith("A") and str(v)[1:].isdigit()), None)
        if start is None or start + 5 > len(row):
            continue
        code, _, fiscal_month, fiscal_year, quarter = row[start:start + 5]
        value = row[-1]
        if code is None or fiscal_year is None or quarter != "결산" or value is None:
            continue
        rows.append((code, fiscal_month, fiscal_year, value))
    wb.close()
    value_col = "dividend_per_share" if item_code == "M000602008" or "DPS" in item_name else "dividend_cashflow_raw"
    out = pd.DataFrame(rows, columns=["stock_code", "fiscal_month", "fiscal_year", value_col])
    out["stock_code"] = normalize_stock_code(out["stock_code"])
    out["fiscal_month"] = pd.to_numeric(out["fiscal_month"], errors="coerce").fillna(12).astype(int)
    out["fiscal_year"] = pd.to_numeric(out["fiscal_year"], errors="coerce").astype("Int64")
    out["fiscal_month_end"] = pd.to_datetime(
        out["fiscal_year"].astype("string") + "-" + out["fiscal_month"].astype("string").str.zfill(2) + "-01",
        errors="coerce",
    ) + pd.offsets.MonthEnd(0)
    if value_col == "dividend_per_share":
        out["dividend_per_share"] = pd.to_numeric(out["dividend_per_share"], errors="coerce")
        out.loc[out["dividend_per_share"] < 0, "dividend_per_share"] = np.nan
        out["dividend_data_type"] = "ANNUAL_DPS_COMMON_CASH"
    else:
        out["dividend_cash_paid"] = -pd.to_numeric(out["dividend_cashflow_raw"], errors="coerce") * 1_000.0
        out.loc[out["dividend_cash_paid"] < 0, "dividend_cash_paid"] = np.nan
        out["dividend_data_type"] = "ANNUAL_CORPORATE_CASHFLOW_PROXY"
    out["available_at"] = out["fiscal_month_end"] + pd.DateOffset(months=4) + pd.offsets.MonthEnd(0)
    return out.sort_values(["stock_code", "fiscal_month_end"]).reset_index(drop=True)


def _find_input(folder: Path, candidates: list[str], required: bool = True) -> Path | None:
    for name in candidates:
        path = folder / name
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(f"입력 파일을 찾지 못했습니다: {candidates}")
    return None


def _looks_like_market_file(path: Path) -> bool:
    """파일명과 무관하게 DataGuide 일별 주가 항목코드가 있는지 확인한다."""
    if path.name.startswith("~$") or path.suffix.lower() != ".xlsx":
        return False
    try:
        with zipfile.ZipFile(path) as book:
            strings = _shared_strings(book)
            with book.open("xl/worksheets/sheet1.xml") as stream:
                parser = ET.iterparse(stream, events=("end",))
                for _, element in parser:
                    if element.tag != _NS + "row":
                        continue
                    row_number = int(element.attrib.get("r", "0"))
                    for cell in element.findall(_NS + "c"):
                        value = _cell_value(cell, strings)
                        if value is not None and str(value).strip().upper().startswith("S410"):
                            del parser
                            return True
                    element.clear()
                    if row_number >= 100:
                        break
                del parser
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError):
        return False
    return False


def _find_market_input(folder: Path) -> Path:
    """원래 파일명을 유지한 채 내용으로 일별 주가 파일을 찾는다."""
    preferred_names = [
        "stock daily prices with delisted.xlsx",
        "stockdailypriceswithdelisted.xlsx",
        "stock_daily_prices_with_delisted.xlsx",
        "stock daily prices.xlsx",
        "주가데이터.xlsx",
        "주가 데이터.xlsx",
        "stock_daily_prices(2).xlsx",
        "stock_daily_prices_current_kospi.xlsx",
    ]
    for name in preferred_names:
        path = folder / name
        if path.exists():
            return path
    matches = [p for p in folder.glob("*.xlsx") if _looks_like_market_file(p)]
    if not matches:
        raise FileNotFoundError(
            "upload 폴더에서 DataGuide 일별 주가 항목코드(S410...)가 있는 Excel을 찾지 못했습니다."
        )
    # 같은 자료의 사본이 여러 개면 가장 최근에 수정된 파일을 사용한다.
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_four_inputs(folder: str | Path) -> LoadedInputs:
    folder = Path(folder)
    market_path = _find_market_input(folder)
    master_path = _find_input(folder, ["stock_master_history(2).xlsx", "stock_master_history.xlsx"])
    financial_path = _find_input(folder, ["financials_annual(2).xlsx", "financials_annual.xlsx"])
    macro_path = _find_input(folder, ["선행지수 순환변동치.xlsx"])
    dividend_path = _find_input(folder, ["dividends(4).xlsx", "dividends(2).xlsx", "dividends.xlsx"], required=False)
    market, market_meta = load_market_auto(market_path)
    master = load_dataguide_master(master_path)
    financials = load_dataguide_financials(financial_path)
    macro = load_kosis_leading_cycle(macro_path)
    dividends = load_dataguide_dividends(dividend_path) if dividend_path else pd.DataFrame()
    market_codes = set(market["stock_code"])
    delisted_codes = set(master.loc[master["delisting_date"].notna(), "stock_code"])
    delisted_with_prices = len(market_codes & delisted_codes)
    metadata = {
        "market": market_meta,
        "master_rows": len(master),
        "financial_rows": len(financials),
        "macro_rows": len(macro),
        "dividend_rows": len(dividends),
        "delisted_master_count": len(delisted_codes),
        "delisted_securities_with_price_history": delisted_with_prices,
        "input_files": {
            "market": market_path.name, "master": master_path.name,
            "financials": financial_path.name, "macro": macro_path.name,
            "dividends": dividend_path.name if dividend_path else None,
        },
        "limitations": [
            "상장폐지 종목은 마스터·재무에서 제거하지 않는다. 다만 실제 편입은 주가 파일에 해당 종목의 과거 가격이 있을 때만 가능하다.",
            "종목종류를 구분하지 않아 우선주 등이 포함될 수 있다.",
            "재무자료의 연결·별도 구분을 추가로 재분류하지 않고 DataGuide 선택 항목을 그대로 사용한다.",
            "경기국면은 선행지수 순환변동치 최신 수정계열이며 관측월 다음 달 말 공개를 가정한다.",
            (
                "주당현금배당금은 확보했지만 배당락일이 없어 결산 후 4개월 사용 가능 월에 재투자하는 대용치다."
                if "dividend_per_share" in dividends.columns
                else "배당 파일은 주당배당금·배당락일이 아닌 연간 현금흐름표 배당금 지급액이므로 재투자 수익은 대용치다."
            ),
        ],
    }
    return LoadedInputs(market, master, financials, macro, dividends, metadata)


def save_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
