from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """JSON 설정을 읽고 기본적인 범위를 검증한다."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    u = cfg["universe"]
    if not 0 <= u["bottom_percentile_cutoff"] < 1:
        raise ValueError("bottom_percentile_cutoff는 0 이상 1 미만이어야 합니다.")
    f = cfg["factor"]
    weights = f["default_weights"]
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise ValueError("기본 팩터 가중치의 합은 1이어야 합니다.")
    if f["selected_count"] <= 0:
        raise ValueError("selected_count는 양수여야 합니다.")
    if cfg["regime"].get("factor_profile") != f["profile"]:
        raise ValueError("factor.profile과 regime.factor_profile이 일치해야 합니다.")
    if "execution" in cfg:
        validate_backtest_config(cfg)
    if "risk" in cfg:
        r = cfg["risk"]
        enter, release, caps = r["drawdown_thresholds"], r["release_thresholds"], r["beta_caps"]
        if len(enter) != 3 or len(release) != 3 or len(caps) != 3:
            raise ValueError("위험 진입·완화·베타 상한은 각각 세 단계여야 합니다.")
        if not (0 < enter[0] < enter[1] < enter[2] < 1):
            raise ValueError("낙폭 진입 기준은 0~1의 오름차순이어야 합니다.")
        if any(not 0 <= release[i] < enter[i] for i in range(3)) or not caps[0] >= caps[1] >= caps[2] >= 0:
            raise ValueError("완화 기준은 진입 기준보다 작고 베타 상한은 내림차순이어야 합니다.")
        if not 0 <= r["put_coverage"] <= 1 or r["release_days"] < 1:
            raise ValueError("헤지비율·완화일수 설정을 확인하세요.")
        if "margin_rate" in r or "force_q_zero" in r or "dividend_lag_months" in r:
            raise ValueError("고정 증거금/임의 배당 대체 설정을 제거하고 새 risk 설정을 사용하세요.")
    return cfg


def validate_backtest_config(cfg: dict[str, Any]) -> None:
    """모듈 5·6·백테스트 엔진 설정 검증. 왕복 거래비용이 하한(기본 0.23%) 미만이면 실행을 막는다."""
    e, a, b = cfg["execution"], cfg["allocation"], cfg["backtest"]
    rates = ["buy_commission", "sell_commission", "sell_tax", "slippage_each_side"]
    if any(float(e[k]) < 0 for k in rates):
        raise ValueError("거래비용 비율은 음수일 수 없습니다.")
    round_trip = float(e["buy_commission"]) + float(e["sell_commission"]) + float(e["sell_tax"]) + 2 * float(e["slippage_each_side"])
    if round_trip < float(e.get("min_round_trip_cost", 0.0023)) - 1e-12:
        raise ValueError(f"왕복 거래비용 {round_trip:.4%}가 하한 {float(e['min_round_trip_cost']):.4%}보다 작습니다.")
    if e.get("execution_price") != "next_session_close":
        raise ValueError("시가 자료가 없으므로 execution_price는 next_session_close만 지원합니다.")
    if a["selected_count"] != cfg["factor"]["selected_count"]:
        raise ValueError("allocation.selected_count와 factor.selected_count가 같아야 합니다.")
    if not 0 < a["min_lookback_sessions"] <= a["lookback_sessions"]:
        raise ValueError("0 < min_lookback_sessions <= lookback_sessions 이어야 합니다.")
    if a["selected_count"] * a["max_weight"] < 1:
        raise ValueError("max_weight × 종목 수가 1 미만이면 합계 1 포트폴리오를 만들 수 없습니다.")
    if abs(sum(b["static_weights"].values()) - 1.0) > 1e-12:
        raise ValueError("정적 팩터 가중치의 합은 1이어야 합니다.")
    if float(b["initial_capital"]) <= 0:
        raise ValueError("initial_capital은 양수여야 합니다.")



def validate_module_interface(factor_profile: str, momentum_weight: float | None) -> None:
    """모듈 2와 모듈 3·4의 상충하는 투자규칙을 조용히 혼합하지 않는다."""
    if factor_profile == "weighted_momentum" and momentum_weight != 0.2:
        raise ValueError("weighted_momentum 프로필은 모멘텀 가중치가 정확히 0.2여야 합니다.")
    if factor_profile == "momentum_filter" and momentum_weight not in (None, 0.0):
        raise ValueError("momentum_filter 프로필은 모멘텀을 CoreScore에 합산할 수 없습니다.")
    if factor_profile not in {"weighted_momentum", "momentum_filter"}:
        raise ValueError(f"알 수 없는 factor profile: {factor_profile}")

