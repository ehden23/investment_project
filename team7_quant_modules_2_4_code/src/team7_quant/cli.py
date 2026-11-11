from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .fixtures import make_synthetic_fixture
from .io import inspect_folder, save_inspection
from .module1 import build_universe_history
from .module2 import run_rebalance_dates
from .module34 import build_monthly_regime_history, build_weight_outputs, regime_summary
from .pipeline24 import run_modules_2_to_4


def _read(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    if not path.exists():
        raise FileNotFoundError(f"필수 입력 없음: {path}")
    return pd.read_csv(path, dtype={"stock_code": "string"})


def cmd_inspect(args: argparse.Namespace) -> None:
    report = inspect_folder(args.input)
    save_inspection(report, args.out)
    print(f"검사 파일 {len(report)}개 -> {args.out}")


def cmd_fixture(args: argparse.Namespace) -> None:
    print(make_synthetic_fixture(args.out))


def cmd_demo(args: argparse.Namespace) -> None:
    folder, out = Path(args.input), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    prices = _read(folder,"prices.csv"); prices["date"] = pd.to_datetime(prices["date"])
    financials = _read(folder,"financials.csv")
    sectors = _read(folder,"sectors.csv"); sectors["date"] = pd.to_datetime(sectors["date"])
    security_master = _read(folder, "security_master.csv")
    dates = list(pd.to_datetime(["2015-12-31", "2016-03-31", "2016-06-30", "2016-09-30", "2016-12-30"]))
    m1 = build_universe_history(dates, security_master, prices, financials, cfg["universe"])
    universe = m1["universe"]
    for name, df in m1.items(): df.to_csv(out/f"module1_{name}.csv",index=False)
    m2 = run_rebalance_dates(dates, universe, {"prices":prices,"financials":financials,"sectors":sectors}, None, cfg["factor"])
    for name, df in m2.items(): df.to_csv(out/f"module2_{name}.csv",index=False)

    macro = _read(folder,"macro.csv"); factor_returns = _read(folder,"factor_returns.csv")
    month_ends = list(pd.date_range("2008-01-31","2016-12-31",freq="ME"))
    regimes = build_monthly_regime_history(month_ends, macro, cfg["regime"])
    weights = build_weight_outputs(regimes, factor_returns, cfg["regime"])
    regimes.to_csv(out/"monthly_regimes.csv",index=False); regime_summary(regimes).to_csv(out/"regime_summary.csv",index=False)
    for name, df in weights.items(): df.to_csv(out/f"{name}.csv",index=False)

    (out/"RUN_NOTICE.txt").write_text("합성자료 기능 검증 결과이며 실제 백테스트 성과가 아닙니다.\n",encoding="utf-8")
    print(f"합성 파이프라인 결과 -> {out}")


def cmd_run_modules_2_4(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    summary = run_modules_2_to_4(args.input, args.out, cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="팀7 모듈 2~4 연구 엔진")
    sub = p.add_subparsers(dest="cmd", required=True)
    x=sub.add_parser("inspect"); x.add_argument("--input",required=True); x.add_argument("--out",required=True); x.set_defaults(func=cmd_inspect)
    x=sub.add_parser("make-fixture"); x.add_argument("--out",required=True); x.set_defaults(func=cmd_fixture)
    x=sub.add_parser("demo"); x.add_argument("--input",required=True); x.add_argument("--config",default="config/base.json"); x.add_argument("--out",required=True); x.set_defaults(func=cmd_demo)
    x=sub.add_parser("run-modules-2-4", help="네 개 DataGuide/KOSIS Excel 원본으로 실제 모듈 2~4 실행")
    x.add_argument("--input", required=True)
    x.add_argument("--config", default="config/base.json")
    x.add_argument("--out", required=True)
    x.set_defaults(func=cmd_run_modules_2_4)
    args=p.parse_args(argv); args.func(args)

