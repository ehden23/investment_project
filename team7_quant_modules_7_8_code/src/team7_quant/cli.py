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


def cmd_build_daily_panel(args: argparse.Namespace) -> None:
    """일별 주가 원본을 파싱해 parquet 캐시를 만들고 품질 점검 결과를 출력한다."""
    from .daily_panel import load_daily_panel
    cfg = load_config(args.config)
    panel = load_daily_panel(Path(args.input) / cfg["backtest"]["price_file"],
                             args.cache or cfg["backtest"]["cache_dir"], rebuild=args.rebuild)
    meta = {k: v for k, v in panel.metadata.items() if k != "stock_names"}
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def cmd_run_backtest(args: argparse.Namespace) -> None:
    """모듈 2~4 출력 + 일별 패널로 전략 4개의 무헤지 일별 NAV를 계산한다."""
    from .backtest import run_backtest
    from .daily_panel import load_daily_panel
    cfg = load_config(args.config)
    panel = load_daily_panel(Path(args.input) / cfg["backtest"]["price_file"], cfg["backtest"]["cache_dir"])
    summary = run_backtest(args.modules_out, panel, cfg, args.out)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.to_string(index=False))
        yearly = pd.read_csv(Path(args.out) / "annual_returns.csv", index_col="year")
        print("\n연도별 수익률 (BENCHMARK_PROXY_CAPW는 공식 KOSPI가 아닌 대용치)")
        print((yearly * 100).round(2).to_string())
    print(f"결과 -> {args.out}")


def cmd_prepare_risk(args: argparse.Namespace) -> None:
    from .risk_data import prepare_risk_market
    from .risk_features import prepare_baseline_features
    from .risk_overlay import DEFAULT_RISK
    cfg = load_config(args.config)
    risk = {**DEFAULT_RISK, **cfg.get("risk", {}),
            "first_signal_date": cfg["backtest"]["first_signal_date"],
            "evaluation_end": cfg["backtest"]["evaluation_end"]}
    if args.margin_file: risk['margin_file'] = args.margin_file
    market = prepare_risk_market(args.input, args.out, risk)
    if args.stock_cache:
        prepare_baseline_features(args.baseline, market, args.stock_cache, args.out,
                                  risk["beta_window"], risk["beta_minimum"])
    print(f"정규화 자료 -> {args.out}; puts={market.options.contract_code.nunique()}, futures={market.futures.contract_code.nunique()}")


def cmd_build_risk_cache(args: argparse.Namespace) -> None:
    from .risk_features import build_stock_cache
    build_stock_cache(args.input, args.out)
    print(f"베타 추정용 주가 캐시 -> {args.out}")


def cmd_run_risk(args: argparse.Namespace) -> None:
    from .module8 import run_risk_evaluation
    cfg = load_config(args.config)
    result = run_risk_evaluation(args.prepared, args.out, cfg, baseline=args.baseline, baseline_only=args.baseline_only)
    print(result[["strategy_id", "scenario", "cagr", "ann_vol", "sharpe", "mdd"]].to_string(index=False))
    print(f"결과 및 report.html -> {args.out}")


def cmd_check_risk(args: argparse.Namespace) -> None:
    from .module8 import check_risk_inputs
    report = check_risk_inputs(args.prepared, load_config(args.config), args.baseline)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding='utf-8')
    print(text)


def cmd_connect_risk(args: argparse.Namespace) -> None:
    from .risk_data import connect_observed_inputs
    connect_observed_inputs(args.prepared, args.dividend, args.margin)
    print(f'일별 관측 배당·증거금 연결 완료: {args.prepared}')


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
    x = sub.add_parser("build-daily-panel", help="일별 주가 원본 → 수정주가·시가총액·거래량 parquet 캐시")
    x.add_argument("--input", required=True, help="원본 xlsx가 있는 폴더 (예: upload)")
    x.add_argument("--config", default="config/base.json")
    x.add_argument("--cache", default=None, help="캐시 폴더 (기본: config의 backtest.cache_dir)")
    x.add_argument("--rebuild", action="store_true", help="캐시가 있어도 원본을 다시 파싱")
    x.set_defaults(func=cmd_build_daily_panel)
    x = sub.add_parser("run-backtest", help="모듈 5·6 연결 일별 NAV 백테스트 (전략 4개)")
    x.add_argument("--input", required=True, help="원본 xlsx가 있는 폴더 (예: upload)")
    x.add_argument("--modules-out", required=True, help="run-modules-2-4 출력 폴더")
    x.add_argument("--config", default="config/base.json")
    x.add_argument("--out", required=True)
    x.set_defaults(func=cmd_run_backtest)
    x = sub.add_parser("prepare-risk-data", help="제공된 옵션·선물·지수·금리·배당 원본 정규화")
    x.add_argument("--input", required=True)
    x.add_argument("--baseline", default="results/backtest")
    x.add_argument("--stock-cache", default=None, help="베타 추정용 risk_stock_panel.npz (없으면 이미 생성한 baseline_features.csv 유지)")
    x.add_argument("--margin-file", default=None, help="검토·날짜 연결을 마친 일별 증거금 CSV")
    x.add_argument("--config", default="config/base.json")
    x.add_argument("--out", default="data/risk_prepared")
    x.set_defaults(func=cmd_prepare_risk)
    x = sub.add_parser("build-risk-stock-cache", help="주가 Excel 또는 표준 long CSV → 베타 추정용 npz")
    x.add_argument("--input", required=True)
    x.add_argument("--out", default="data/cache/risk_stock_panel.npz")
    x.set_defaults(func=cmd_build_risk_cache)
    x = sub.add_parser("run-risk-backtest", help="모듈 7·8: 기존 현금 내 헤지·일별 증거금·성과평가")
    x.add_argument("--prepared", default="data/risk_prepared")
    x.add_argument("--baseline", default="results/backtest")
    x.add_argument("--config", default="config/base.json")
    x.add_argument("--out", default="results/module7_8")
    x.add_argument("--baseline-only", action="store_true", help="기존 무헤지 NAV만 평가; 헤지 성과를 만들지 않음")
    x.set_defaults(func=cmd_run_risk)
    x = sub.add_parser('check-risk-inputs', help='헤지 실행 전 미확정 비용·가용 현금·자료 누락 확인')
    x.add_argument('--prepared', default='data/risk_prepared')
    x.add_argument('--baseline', default='results/backtest')
    x.add_argument('--config', default='config/base.json')
    x.add_argument('--out', default='results/input_check.json')
    x.set_defaults(func=cmd_check_risk)
    x = sub.add_parser('connect-risk-inputs', help='검토한 관측 배당·증거금 CSV/ZIP 연결')
    x.add_argument('--prepared', default='data/risk_prepared')
    x.add_argument('--dividend', required=True)
    x.add_argument('--margin', required=True)
    x.set_defaults(func=cmd_connect_risk)
    args=p.parse_args(argv); args.func(args)

