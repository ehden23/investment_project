from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class CovarianceResult:
    covariance_daily: np.ndarray
    covariance_annual: np.ndarray
    sample_covariance: np.ndarray
    target_covariance: np.ndarray
    rbar: float
    shrinkage: float
    degenerate: bool


@dataclass
class AllocationResult:
    targets: pd.DataFrame
    diagnostics: dict[str, Any]


def lw_constant_correlation(returns: np.ndarray, annual_factor: float = 252.0) -> CovarianceResult:
    """Ledoit–Wolf constant-correlation shrinkage의 명세식 구현."""
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] < 2 or not np.isfinite(r).all():
        raise ValueError("returns는 유한한 값의 2차원 배열이며 관측치·자산이 각각 2개 이상이어야 합니다.")
    n = r.shape[0] - 1
    x = r - r.mean(axis=0)
    s = x.T @ x / n
    v = np.diag(s)
    if np.any(v <= 0):
        raise ValueError("표본분산이 0 이하인 자산은 공분산 추정에 사용할 수 없습니다.")
    std = np.sqrt(v); p = r.shape[1]
    corr = s / np.outer(std, std)
    rbar = float((corr.sum() - p) / (p * (p - 1)))
    f = rbar * np.outer(std, std)
    np.fill_diagonal(f, v)
    a = (x**2).T @ (x**2) / n - s**2
    phi_hat = float(a.sum())
    theta = (x**3).T @ x / n - v[:, None] * s
    np.fill_diagonal(theta, 0.0)
    rho_hat = float(np.diag(a).sum() + rbar * ((std[None, :] / std[:, None]) * theta).sum())
    gamma_hat = float(((s - f) ** 2).sum())
    scale = max(float((s**2).sum()), float((f**2).sum()))
    degenerate = gamma_hat <= 64 * np.finfo(float).eps * scale
    eta = 0.0 if degenerate else float(np.clip((phi_hat - rho_hat) / (n * gamma_hat), 0.0, 1.0))
    sigma_d = (1 - eta) * s + eta * f
    sigma_d = (sigma_d + sigma_d.T) / 2
    eig = np.linalg.eigvalsh(sigma_d)
    if eig.min() < -1e-10 * max(abs(eig).max(), 1e-30):
        raise ValueError("축소공분산이 실질적으로 PSD가 아닙니다.")
    return CovarianceResult(sigma_d, sigma_d * annual_factor, s, f, rbar, eta, degenerate)


def black_litterman_posterior(
    sigma: np.ndarray, market_weights: np.ndarray, top_indices: list[int],
    delta: float = 2.5, alpha: float = 0.03, tau: float = 0.025,
) -> tuple[np.ndarray, dict[str, Any]]:
    sigma = np.asarray(sigma, float); b = np.asarray(market_weights, float)
    n = len(b)
    if sigma.shape != (n, n) or np.any(b <= 0) or not np.isclose(b.sum(), 1):
        raise ValueError("sigma와 시장가중치의 차원·양수·합계 조건을 확인하세요.")
    p = np.zeros(n); p[top_indices] = 1.0 / len(top_indices)
    pi = delta * sigma @ b
    v = float(p @ sigma @ p)
    if v <= 0:
        raise ValueError("뷰 바스켓 분산은 양수여야 합니다.")
    # Omega=tau*P Sigma P'이면 tau가 평균식에서 소거된다.
    mu = pi + alpha / (2 * v) * (sigma @ p)
    return mu, {"pi": pi, "p_view": p, "view_variance": v, "prior_basket": float(p @ pi),
                "posterior_basket": float(p @ mu), "tau": tau, "alpha": alpha}


def optimize_long_only_capped(
    mu: np.ndarray, sigma: np.ndarray, delta: float, upper: float = 0.10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """제약 평균-분산 문제를 푼다.

    CVXPY/Clarabel이 설치되어 있으면 이를 사용하고, 현재 환경처럼 없으면 SciPy
    SLSQP로 같은 경제적 목적함수를 푼 뒤 독립 제약 검사를 수행한다.
    """
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    n = len(mu)
    if n * upper < 1 - 1e-12:
        raise ValueError("상한 제약으로 합계 1 포트폴리오를 만들 수 없습니다.")
    backend = "SCIPY_SLSQP_FALLBACK"
    try:
        import cvxpy as cp  # type: ignore
        wv = cp.Variable(n)
        problem = cp.Problem(cp.Minimize(delta / 2 * cp.quad_form(wv, cp.psd_wrap(sigma)) - mu @ wv),
                             [cp.sum(wv) == 1, wv >= 0, wv <= upper])
        problem.solve(solver="CLARABEL", max_iter=1000, tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                      tol_feas=1e-10, warm_start=False)
        if problem.status not in {"optimal", "optimal_inaccurate"} or wv.value is None:
            raise RuntimeError(problem.status)
        w = np.asarray(wv.value).reshape(-1); backend = "CVXPY_CLARABEL"
        success, message, nit = True, problem.status, problem.solver_stats.num_iters
    except (ImportError, ModuleNotFoundError):
        def objective(w: np.ndarray) -> float:
            return float(delta / 2 * w @ sigma @ w - mu @ w)
        res = minimize(objective, np.full(n, 1 / n), method="SLSQP",
                       bounds=[(0.0, upper)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"ftol": 1e-12, "maxiter": 3000})
        if not res.success:
            raise RuntimeError(f"최적화 실패: {res.message}")
        w, success, message, nit = res.x, res.success, res.message, res.nit
    if abs(w.sum() - 1) > 1e-8 or w.min() < -1e-8 or w.max() > upper + 1e-8:
        raise RuntimeError("최적화 결과가 금융 제약 검사를 통과하지 못했습니다.")
    objective_value = float(delta / 2 * w @ sigma @ w - mu @ w)
    return w, {"backend": backend, "success": success, "message": str(message), "iterations": nit,
               "sum_error": float(abs(w.sum() - 1)), "min_weight": float(w.min()),
               "max_weight": float(w.max()), "objective_min": objective_value}


def build_targets(
    signal_date: str | pd.Timestamp,
    selections: pd.DataFrame,
    risk_returns: pd.DataFrame,
    market_caps: pd.DataFrame,
    config: dict[str, Any],
) -> AllocationResult:
    """같은 30종목에 대해 EW와 BL 목표비중을 생성한다."""
    date = pd.Timestamp(signal_date)
    sel = selections[pd.to_datetime(selections["date"]) == date].sort_values("rank")
    n_expected = int(config.get("selected_count", 30))
    if len(sel) != n_expected or sel["stock_code"].nunique() != n_expected:
        raise ValueError(f"{date.date()} 선정 종목은 정확히 {n_expected}개여야 합니다.")
    codes = sorted(sel["stock_code"].astype(str))
    rr = risk_returns.copy(); rr["date"] = pd.to_datetime(rr["date"])
    window = rr[(rr["date"] <= date) & rr["stock_code"].astype(str).isin(codes)]
    pivot = window.pivot(index="date", columns="stock_code", values="return").sort_index().tail(int(config.get("lookback_sessions", 252)))
    pivot = pivot.reindex(columns=codes)
    if pivot.shape != (int(config.get("lookback_sessions", 252)), n_expected) or pivot.isna().any().any():
        raise ValueError(f"공분산 입력은 결측 없는 {config.get('lookback_sessions',252)}x{n_expected} 행렬이어야 합니다.")
    cov = lw_constant_correlation(pivot.to_numpy(), float(config.get("annual_factor", 252)))
    caps = market_caps[pd.to_datetime(market_caps["date"]) == date].set_index("stock_code")["market_cap"].reindex(codes)
    if caps.isna().any() or (caps <= 0).any():
        raise ValueError("선정 종목의 양수 시가총액이 모두 필요합니다.")
    b = caps.to_numpy(float) / caps.sum()
    rank_map = sel.set_index(sel["stock_code"].astype(str))["rank"]
    top_codes = list(rank_map.sort_values().head(int(config.get("top_count", 5))).index)
    top_indices = [codes.index(c) for c in top_codes]
    mu, bl_diag = black_litterman_posterior(cov.covariance_annual, b, top_indices,
                                            float(config.get("delta", 2.5)), float(config.get("alpha", 0.03)),
                                            float(config.get("tau", 0.025)))
    bl_w, opt_diag = optimize_long_only_capped(mu, cov.covariance_annual, float(config.get("delta", 2.5)),
                                               float(config.get("max_weight", 0.10)))
    ew = np.full(n_expected, 1.0 / n_expected)
    targets = pd.concat([
        pd.DataFrame({"signal_date": date, "allocation_id": "EW", "stock_code": codes, "target_weight": ew}),
        pd.DataFrame({"signal_date": date, "allocation_id": "BL_ALPHA_003", "stock_code": codes, "target_weight": bl_w}),
    ], ignore_index=True)
    return AllocationResult(targets, {"rbar": cov.rbar, "shrinkage": cov.shrinkage,
                                      "covariance_degenerate": cov.degenerate, "bl": bl_diag, "optimizer": opt_diag})
