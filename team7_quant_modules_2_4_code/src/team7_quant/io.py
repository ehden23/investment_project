from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class FileInspection:
    path: str
    kind: str
    sheets: list[str]
    columns: dict[str, list[str]]
    rows: dict[str, int]
    warning: str = ""


def inspect_input_file(path: str | Path) -> FileInspection:
    """실제 Excel/CSV의 시트·헤더·행 수를 추측 없이 확인한다."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        book = pd.ExcelFile(path)
        columns, rows = {}, {}
        for sheet in book.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            columns[sheet] = [str(c) for c in df.columns]
            rows[sheet] = len(df)
        return FileInspection(str(path), "excel", book.sheet_names, columns, rows)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str)
        return FileInspection(str(path), "csv", ["csv"], {"csv": list(df.columns)}, {"csv": len(df)})
    return FileInspection(str(path), "unsupported", [], {}, {}, "지원하지 않는 확장자")


def inspect_folder(folder: str | Path) -> list[FileInspection]:
    supported = {".xlsx", ".xlsm", ".xls", ".csv"}
    return [inspect_input_file(p) for p in sorted(Path(folder).glob("*")) if p.suffix.lower() in supported]


def save_inspection(report: Iterable[FileInspection], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(x) for x in report], ensure_ascii=False, indent=2), encoding="utf-8")


def require_columns(df: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{table} 필수 열 누락: {missing}")


def normalize_stock_code(series: pd.Series) -> pd.Series:
    """앞자리 0을 보존한다. 원본 코드가 숫자로 훼손된 경우 6자리로 보정한다."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"^[Aa]", "", regex=True)
    )
    return cleaned.str.zfill(6)


def read_normalized_csv(path: str | Path, required: Iterable[str], table: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_code": "string"})
    require_columns(df, required, table)
    if "stock_code" in df:
        df["stock_code"] = normalize_stock_code(df["stock_code"])
    for col in [c for c in df.columns if c.endswith("date") or c.endswith("_at")]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
