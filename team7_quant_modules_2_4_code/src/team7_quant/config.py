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
    return cfg


def validate_module_interface(factor_profile: str, momentum_weight: float | None) -> None:
    """모듈 2와 모듈 3·4의 상충하는 투자규칙을 조용히 혼합하지 않는다."""
    if factor_profile == "weighted_momentum" and momentum_weight != 0.2:
        raise ValueError("weighted_momentum 프로필은 모멘텀 가중치가 정확히 0.2여야 합니다.")
    if factor_profile == "momentum_filter" and momentum_weight not in (None, 0.0):
        raise ValueError("momentum_filter 프로필은 모멘텀을 CoreScore에 합산할 수 없습니다.")
    if factor_profile not in {"weighted_momentum", "momentum_filter"}:
        raise ValueError(f"알 수 없는 factor profile: {factor_profile}")

