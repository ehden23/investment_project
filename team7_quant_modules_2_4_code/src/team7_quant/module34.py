from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .io import require_columns


MACRO_COLS = {
    "observation_month", "release_date", "vintage_date", "available_at",
    "indicator_code", "value", "source", "vintage_status",
}
RETURN_COLS = {
    "formation_date", "return_month", "return_available_at", "value_return",
    "quality_return", "return_basis", "portfolio_rule_id",
}
REGIMES = ("확장", "둔화", "수축", "회복")


@dataclass
class RegimeState:
    confirmed_regime: str | None = None
    pending_regime: str | None = None
    confirmation_count: int = 0
    last_processed_observation_month: pd.Timestamp | None = None


def classify_observed_regime(current_value: float, lagged_value: float, baseline: float = 100.0) -> str | None:
    if pd.isna(current_value) or pd.isna(lagged_value):
        return None
    change = current_value - lagged_value
    if change == 0:
        return None
    if current_value >= baseline:
        return "확장" if change > 0 else "둔화"
    return "회복" if change > 0 else "수축"


def build_asof_macro_snapshot(asof_date: str | pd.Timestamp, macro_data: pd.DataFrame) -> pd.DataFrame:
    """판단시점에 이용 가능했던 각 관측월의 최신 버전만 반환한다."""
    require_columns(macro_data, MACRO_COLS, "macro_data")
    m = macro_data.copy()
    for c in ["observation_month", "release_date", "vintage_date", "available_at"]:
        m[c] = pd.to_datetime(m[c])
    m = m[m["available_at"] <= pd.Timestamp(asof_date)]
    if m.empty:
        return m
    return m.sort_values(["observation_month", "available_at", "vintage_date"]).groupby("observation_month", as_index=False).tail(1)


def _observe(asof_date: pd.Timestamp, macro_data: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    snap = build_asof_macro_snapshot(asof_date, macro_data)
    if snap.empty:
        return {"status_reason": "no_available_macro", "observed_regime": None}
    current = snap.sort_values("observation_month").iloc[-1]
    obs = pd.Timestamp(current["observation_month"]).to_period("M").to_timestamp("M")
    lag_month = (obs.to_period("M") - int(config.get("change_months", 3))).to_timestamp("M")
    lag = snap[pd.to_datetime(snap["observation_month"]).dt.to_period("M") == lag_month.to_period("M")]
    if lag.empty:
        return {"observation_month": obs, "available_at": current["available_at"], "current_value": current["value"],
                "lagged_value": np.nan, "change_3m": np.nan, "observed_regime": None,
                "vintage_status": current["vintage_status"], "status_reason": "missing_exact_m_minus_3"}
    lagged = lag.iloc[-1]
    regime = classify_observed_regime(float(current["value"]), float(lagged["value"]), float(config.get("baseline", 100)))
    return {"observation_month": obs, "available_at": current["available_at"], "current_value": float(current["value"]),
            "lagged_value": float(lagged["value"]), "change_3m": float(current["value"] - lagged["value"]),
            "observed_regime": regime, "vintage_status": current["vintage_status"],
            "status_reason": "ok" if regime else "zero_change_or_missing"}


def update_regime_state(previous: RegimeState, observation: dict[str, Any], config: dict[str, Any]) -> tuple[RegimeState, bool, str]:
    obs_month = observation.get("observation_month")
    observed = observation.get("observed_regime")
    if obs_month is None or (previous.last_processed_observation_month is not None and obs_month <= previous.last_processed_observation_month):
        return previous, False, "no_new_observation"
    state = RegimeState(previous.confirmed_regime, previous.pending_regime, previous.confirmation_count, obs_month)
    if observed is None:
        state.pending_regime, state.confirmation_count = None, 0
        return state, False, "unclassifiable_reset_pending"
    # 관측월이 연속되지 않으면 후보 연속 확인을 초기화한다.
    if previous.last_processed_observation_month is not None:
        expected = (previous.last_processed_observation_month.to_period("M") + 1).to_timestamp("M")
        if obs_month.to_period("M") != expected.to_period("M"):
            state.pending_regime, state.confirmation_count = None, 0
    if state.confirmed_regime == observed:
        state.pending_regime, state.confirmation_count = None, 0
        return state, False, "same_as_confirmed"
    if state.pending_regime == observed:
        state.confirmation_count += 1
    else:
        state.pending_regime, state.confirmation_count = observed, 1
    if state.confirmation_count >= int(config.get("confirmations", 2)):
        initial = state.confirmed_regime is None
        state.confirmed_regime = observed
        state.pending_regime, state.confirmation_count = None, 0
        return state, not initial, "initial_confirmation" if initial else "regime_transition"
    return state, False, "pending_confirmation"


def build_monthly_regime_history(month_ends: list[pd.Timestamp], macro_data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    state = RegimeState()
    rows = []
    for date in sorted(pd.to_datetime(month_ends)):
        obs = _observe(date, macro_data, config)
        old_confirmed = state.confirmed_regime
        state, changed, reason = update_regime_state(state, obs, config)
        rows.append({"date": date, **obs, "confirmed_regime": state.confirmed_regime,
                     "pending_regime": state.pending_regime, "confirmation_count": state.confirmation_count,
                     "new_observation": reason != "no_new_observation", "regime_changed": changed,
                     "initial_confirmation": old_confirmed is None and state.confirmed_regime is not None,
                     "status_reason": reason})
    return pd.DataFrame(rows)


def align_regimes_and_forward_returns(regime_history: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
    require_columns(factor_returns, RETURN_COLS, "factor_returns")
    r = factor_returns.copy()
    r["return_month"] = pd.to_datetime(r["return_month"]).dt.to_period("M")
    r["return_available_at"] = pd.to_datetime(r["return_available_at"])
    h = regime_history.copy()
    h["return_month"] = (pd.to_datetime(h["date"]).dt.to_period("M") + 1)
    cols = ["date", "return_month", "confirmed_regime"]
    out = h[cols].merge(r, on="return_month", how="left")
    out["spread"] = out["value_return"] - out["quality_return"]
    out["included_pair"] = out[["value_return", "quality_return"]].notna().all(axis=1)
    return out.rename(columns={"date": "regime_judgment_date"})


def _count_runs(series: pd.Series, regime: str) -> int:
    x = series.eq(regime)
    return int((x & ~x.shift(fill_value=False)).sum())


def fit_annual_weight_table(
    application_year: int,
    regime_history: pd.DataFrame,
    factor_returns: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = align_regimes_and_forward_returns(regime_history, factor_returns)
    training_start = pd.Timestamp(application_year - int(config.get("training_years", 7)), 1, 1)
    training_end = pd.Timestamp(application_year - 1, 12, 31)
    realized = aligned["return_month"].dt.to_timestamp("M")
    usable = aligned[(realized >= training_start) & (realized <= training_end)].copy()
    usable["available_in_time"] = usable["return_available_at"] <= training_end
    usable["include"] = usable["included_pair"] & usable["available_in_time"] & usable["confirmed_regime"].isin(REGIMES)
    usable["exclusion_reason"] = np.select(
        [~usable["included_pair"], ~usable["available_in_time"], ~usable["confirmed_regime"].isin(REGIMES)],
        ["missing_return_pair", "return_not_available", "unconfirmed_regime"], default="included")
    rows = []
    min_n, band = int(config.get("min_samples", 12)), float(config.get("neutral_band", 0.001))
    for regime in REGIMES:
        g = usable[(usable["confirmed_regime"] == regime) & usable["include"]]
        n = len(g); spread = g["spread"].mean() if n else np.nan
        if n < min_n:
            wv, wq, reason = 0.5, 0.5, "insufficient_samples"
        elif spread > band:
            wv, wq, reason = 0.6, 0.4, "value_outperformance"
        elif spread < -band:
            wv, wq, reason = 0.4, 0.6, "quality_outperformance"
        else:
            wv, wq, reason = 0.5, 0.5, "neutral_band"
        rows.append({"application_year": application_year, "training_start": training_start,
                     "training_end": training_end, "regime": regime, "n_obs": n,
                     "mean_value_return": g["value_return"].mean(), "mean_quality_return": g["quality_return"].mean(),
                     "mean_spread": spread, "n_runs": _count_runs(usable["confirmed_regime"], regime),
                     "w_V": wv, "w_Q": wq, "selection_reason": reason,
                     "neutral_band": band, "min_samples": min_n,
                     "weight_table_id": f"W{application_year}_{application_year-int(config.get('training_years', 7))}_{application_year-1}"})
    return pd.DataFrame(rows), usable


def get_monthly_factor_weights(asof_date: pd.Timestamp, confirmed_regime: str | None, annual_table: pd.DataFrame) -> dict[str, Any]:
    if confirmed_regime is None:
        return {"date": asof_date, "confirmed_regime": None, "w_V": 0.5, "w_Q": 0.5,
                "weight_source": "preparation_neutral", "weight_table_id": None}
    row = annual_table[(annual_table["application_year"] == asof_date.year) & (annual_table["regime"] == confirmed_regime)]
    if row.empty:
        raise ValueError(f"{asof_date.year}년 {confirmed_regime} 가중치표가 없습니다.")
    r = row.iloc[0]
    return {"date": asof_date, "confirmed_regime": confirmed_regime, "w_V": r["w_V"], "w_Q": r["w_Q"],
            "weight_source": "annual_regime_table", "weight_table_id": r["weight_table_id"]}


def adapt_weights_for_module2(relative: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """모듈 4의 가치:퀄리티 상대비중을 모듈 2의 명목 가중치로 변환한다."""
    profile = config.get("factor_profile", "weighted_momentum")
    if profile == "weighted_momentum":
        wm = float(config.get("fixed_momentum_weight", 0.2))
        scale = 1.0 - wm
        return {**relative, "w_V": scale * relative["w_V"], "w_Q": scale * relative["w_Q"], "w_M": wm}
    if profile == "momentum_filter":
        return {**relative, "w_M": 0.0}
    raise ValueError(f"알 수 없는 factor_profile: {profile}")


def build_weight_outputs(regime_history: pd.DataFrame, factor_returns: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    tables, samples = [], []
    for year in range(int(config["evaluation_start_year"]), int(config["evaluation_end_year"]) + 1):
        table, sample = fit_annual_weight_table(year, regime_history, factor_returns, config)
        tables.append(table); sample["application_year"] = year; samples.append(sample)
    annual = pd.concat(tables, ignore_index=True)
    monthly_rows = []
    prev = None
    for _, row in regime_history.iterrows():
        date = pd.Timestamp(row["date"])
        if date.year < int(config["evaluation_start_year"]) or date.year > int(config["evaluation_end_year"]):
            continue
        item = adapt_weights_for_module2(get_monthly_factor_weights(date, row["confirmed_regime"], annual), config)
        item["regime_changed"] = bool(row["regime_changed"])
        item["weight_changed"] = prev is not None and (item["w_V"], item["w_Q"]) != prev
        prev = (item["w_V"], item["w_Q"])
        monthly_rows.append(item)
    return {"annual_regime_weights": annual, "monthly_factor_weights": pd.DataFrame(monthly_rows),
            "training_samples": pd.concat(samples, ignore_index=True)}


def regime_summary(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    h = history.copy()
    for regime, g in h.groupby("confirmed_regime", dropna=False):
        label = regime if pd.notna(regime) else "미확정"
        mask = h["confirmed_regime"].fillna("미확정").eq(label)
        run_id = (mask != mask.shift()).cumsum()
        durations = h[mask].groupby(run_id[mask]).size()
        rows.append({"regime": label, "months": len(g), "share": len(g) / len(h), "runs": len(durations),
                     "mean_duration": durations.mean(), "median_duration": durations.median(), "max_duration": durations.max()})
    return pd.DataFrame(rows)
