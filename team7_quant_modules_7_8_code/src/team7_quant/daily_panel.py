from __future__ import annotations

"""DataGuide 일별 주가 원본을 날짜×종목 패널로 읽고 캐시하는 로더 (백테스트 엔진용).

dataguide.py의 월말 로더는 수정하지 않고, 그 XML 파싱 보조함수만 재사용한다.
원본 시트는 가로형(종목마다 6개 항목 열)이며 openpyxl로는 빈 값으로 읽히므로
시트 XML을 한 행(=한 거래일)씩 스트리밍 파싱한다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .dataguide import _NS, _cell_value, _excel_col_number, _shared_strings
from .io import normalize_stock_code


# DataGuide 아이템코드 → 패널 이름. 시가총액만 단위 변환(백만원→원)이 필요하다.
DAILY_ITEMS: dict[str, str] = {
    "S410000700": "adj_close",   # 수정주가(원)
    "S410001250": "market_cap",  # 시가총액(백만원) → 원으로 변환
    "S41000080F": "volume",      # 거래량(주)
}
_UNIT_MULTIPLIER = {"adj_close": 1.0, "market_cap": 1_000_000.0, "volume": 1.0}

# 원본 레이아웃(dataguide.py와 동일): 26행 코드, 27행 종목명, 29행 아이템코드, 32행부터 자료, 20열(T) 날짜.
_ROW_CODE, _ROW_NAME, _ROW_ITEM, _FIRST_DATA_ROW, _DATE_COL, _FIRST_DATA_COL = 26, 27, 29, 32, 20, 21

CACHE_VERSION = 1


@dataclass
class DailyPanel:
    """일별 패널 묶음. 각 DataFrame은 index=거래일(DatetimeIndex), columns=6자리 종목코드."""

    adj_close: pd.DataFrame
    market_cap: pd.DataFrame
    volume: pd.DataFrame
    metadata: dict[str, Any]

    def returns(self) -> pd.DataFrame:
        """수정주가 일간 단순수익률. 상장 전(NaN) 구간은 NaN으로 남긴다."""
        return self.adj_close.pct_change(fill_method=None)


def parse_daily_panel(path: str | Path) -> DailyPanel:
    """원본 xlsx를 스트리밍 파싱해 수정주가·시가총액·거래량 패널을 만든다.

    - 헤더 3개 행(코드/종목명/아이템코드)으로 '열 번호 → (종목 인덱스, 항목)' 지도를 만든다.
    - 이후 각 날짜 행에서 필요한 열만 숫자로 꺼내 numpy 배열에 채운다.
    - 결측은 NaN 그대로 둔다(0이나 직전값으로 채우지 않는다).
    """
    path = Path(path)
    codes_by_col: dict[int, str] = {}
    names_by_col: dict[int, str] = {}
    items_by_col: dict[int, str] = {}
    col_map: dict[int, tuple[int, str]] = {}
    codes: list[str] = []
    names: list[str] = []
    dates: list[pd.Timestamp] = []
    rows: dict[str, list[np.ndarray]] = {name: [] for name in DAILY_ITEMS.values()}

    with zipfile.ZipFile(path) as book:
        strings = _shared_strings(book)
        stream = book.open("xl/worksheets/sheet1.xml")
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag != _NS + "row":
                continue
            row_number = int(element.attrib.get("r", "0"))

            if row_number in (_ROW_CODE, _ROW_NAME, _ROW_ITEM):
                target = {_ROW_CODE: codes_by_col, _ROW_NAME: names_by_col, _ROW_ITEM: items_by_col}[row_number]
                for cell in element.findall(_NS + "c"):
                    col = _excel_col_number(cell.attrib["r"])
                    value = _cell_value(cell, strings)
                    if col >= _FIRST_DATA_COL and value is not None:
                        target[col] = str(value)
                if row_number == _ROW_ITEM:
                    # 헤더가 모두 모였으면 종목 순서와 열 지도를 확정한다.
                    index_by_code: dict[str, int] = {}
                    for col in sorted(items_by_col):
                        field = DAILY_ITEMS.get(items_by_col[col])
                        raw_code = codes_by_col.get(col)
                        if field is None or raw_code is None:
                            continue
                        code = str(normalize_stock_code(pd.Series([raw_code])).iloc[0])
                        if code not in index_by_code:
                            index_by_code[code] = len(codes)
                            codes.append(code)
                            names.append(names_by_col.get(col, ""))
                        col_map[col] = (index_by_code[code], field)
                element.clear()
                continue

            if row_number >= _FIRST_DATA_ROW and codes:
                cells = {_excel_col_number(c.attrib["r"]): c for c in element.findall(_NS + "c")}
                date_cell = cells.get(_DATE_COL)
                serial = None if date_cell is None else _cell_value(date_cell, strings)
                try:
                    # 엑셀 일련번호(1900 체계) → 날짜
                    date = pd.Timestamp("1899-12-30") + pd.to_timedelta(float(serial), unit="D")
                except (TypeError, ValueError):
                    element.clear()
                    continue
                arrays = {name: np.full(len(codes), np.nan) for name in rows}
                for col, cell in cells.items():
                    mapping = col_map.get(col)
                    if mapping is None:
                        continue
                    raw = _cell_value(cell, strings)
                    if raw is None:
                        continue
                    try:
                        arrays[mapping[1]][mapping[0]] = float(raw)
                    except (TypeError, ValueError):
                        continue
                for name, arr in arrays.items():
                    rows[name].append(arr * _UNIT_MULTIPLIER[name])
                dates.append(date.normalize())
            element.clear()

    index = pd.DatetimeIndex(dates, name="date")
    frames = {
        name: pd.DataFrame(np.vstack(arrs), index=index, columns=pd.Index(codes, name="stock_code"))
        for name, arrs in rows.items()
    }
    metadata = {
        "source_file": path.name,
        "cache_version": CACHE_VERSION,
        "items": DAILY_ITEMS,
        "units": {"adj_close": "KRW", "market_cap": "KRW (원본 백만원×1e6)", "volume": "shares"},
        "first_date": str(index.min().date()),
        "last_date": str(index.max().date()),
        "sessions": int(len(index)),
        "security_count": int(len(codes)),
        "stock_names": dict(zip(codes, names)),
    }
    return DailyPanel(frames["adj_close"], frames["market_cap"], frames["volume"], metadata)


def panel_quality_report(panel: DailyPanel) -> dict[str, Any]:
    """패널 품질 점검: 중복 날짜, 상장 후 중간 결측, 거래량 0 일수 등을 숫자로 요약한다."""
    px = panel.adj_close
    first_valid = px.notna().cummax()                 # 첫 관측 이후 True
    last_valid = px[::-1].notna().cummax()[::-1]      # 마지막 관측 이전 True
    inner_missing = (first_valid & last_valid & px.isna()).sum()
    vol = panel.volume
    zero_volume = ((vol == 0) & px.notna()).sum()
    return {
        "duplicate_dates": int(px.index.duplicated().sum()),
        "dates_sorted": bool(px.index.is_monotonic_increasing),
        "stocks_with_inner_missing_price": int((inner_missing > 0).sum()),
        "inner_missing_price_cells": int(inner_missing.sum()),
        "stocks_never_priced": int(px.notna().sum().eq(0).sum()),
        "stocks_listed_after_start": int(px.iloc[0].isna().sum()),
        "zero_volume_cells": int(zero_volume.sum()),
        "stocks_with_zero_volume_days": int((zero_volume > 0).sum()),
        "nonpositive_price_cells": int((px <= 0).sum().sum()),
        "missing_volume_where_priced": int((vol.isna() & px.notna()).sum().sum()),
    }


def _cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {name: cache_dir / f"daily_{name}.parquet" for name in DAILY_ITEMS.values()} | {
        "meta": cache_dir / "daily_panel_meta.json"
    }


def load_daily_panel(source_xlsx: str | Path, cache_dir: str | Path, rebuild: bool = False) -> DailyPanel:
    """캐시가 있고 원본 파일 크기·수정시각이 같으면 캐시를, 아니면 원본을 파싱해 캐시를 만든다."""
    source_xlsx, cache_dir = Path(source_xlsx), Path(cache_dir)
    paths = _cache_paths(cache_dir)
    stat = source_xlsx.stat()
    fingerprint = {"size": stat.st_size, "mtime": int(stat.st_mtime), "cache_version": CACHE_VERSION}
    if not rebuild and all(p.exists() for p in paths.values()):
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            frames = {name: pd.read_parquet(paths[name]) for name in DAILY_ITEMS.values()}
            for f in frames.values():
                f.columns = f.columns.astype(str)
                f.columns.name = "stock_code"
            return DailyPanel(frames["adj_close"], frames["market_cap"], frames["volume"], meta)
    panel = parse_daily_panel(source_xlsx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in DAILY_ITEMS.values():
        getattr(panel, name).to_parquet(paths[name])
    panel.metadata["fingerprint"] = fingerprint
    panel.metadata["quality"] = panel_quality_report(panel)
    paths["meta"].write_text(json.dumps(panel.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return panel
