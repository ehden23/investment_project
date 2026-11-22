from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/actual")
    args = parser.parse_args()
    folder = Path(args.results)

    selections = pd.read_csv(folder / "module2_selections.csv")
    scores = pd.read_csv(folder / "module2_monthly_score_history.csv")
    weights = pd.read_csv(folder / "monthly_factor_weights.csv")
    annual = pd.read_csv(folder / "annual_regime_weights.csv")
    factor_returns = pd.read_csv(folder / "factor_returns.csv")

    sizes = selections.groupby("date").size()
    assert not sizes.empty and sizes.eq(30).all(), "리밸런싱 시점별 30종목 선정 실패"
    assert selections.duplicated(["date", "stock_code"]).sum() == 0, "동일 시점 중복 종목"
    for column in ["S_V", "S_Q", "S_M", "CoreScore"]:
        assert scores[column].between(0, 1).all(), f"{column} 범위 오류"
    assert ((weights["w_V"] + weights["w_Q"] + weights["w_M"] - 1).abs() < 1e-12).all(), "월별 가중치 합 오류"
    assert set(map(tuple, annual[["w_V", "w_Q"]].drop_duplicates().round(10).to_numpy())) <= {
        (0.6, 0.4), (0.5, 0.5), (0.4, 0.6)
    }, "허용되지 않은 가치·퀄리티 가중치"
    assert factor_returns[["value_return", "quality_return"]].notna().all().all(), "단독 팩터 수익률 결측"
    scores["asof_date"] = pd.to_datetime(scores["asof_date"])
    scores["available_at"] = pd.to_datetime(scores["available_at"])
    assert not (scores["available_at"] > scores["asof_date"]).any(), "미래 재무자료 사용"
    print("PASS: actual modules 2-4 output validation")
    print(f"rebalance_dates={len(sizes)}, selected_rows={len(selections)}, score_rows={len(scores)}")


if __name__ == "__main__":
    main()
