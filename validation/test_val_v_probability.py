"""
Phase 10 — Probability of Collision (Pc) and Monte Carlo Cross-Validation.

Validates:
1. Exact analytical reference case: Isotropic covariance with zero miss distance.
   Pc_exact = 1 - exp(-HBR² / (2 σ²))
2. Small-HBR isotropic analytical approximation:
   Pc ≈ (HBR² / (2 σ²)) * exp(-d² / (2 σ²)) for HBR << σ
3. Probability Dilution Peak phenomenon:
   For fixed non-zero miss distance d, as σ increases from 0, Pc increases to a maximum
   at σ_opt ≈ d / √2, and then decreases as σ → ∞.
4. Physical monotonicity:
   - Increasing miss distance d strictly decreases Pc (for fixed σ).
   - Increasing HBR strictly increases Pc.
5. Monte Carlo Statistical Cross-Validation:
   Deterministic 2D Gaussian polar quadrature compared against 200,000-sample empirical
   hit-count Monte Carlo simulation.
6. Extreme edge cases:
   - Overwhelmingly large HBR: Pc → 1.0
   - Far separation: Pc → 0.0
   - Singular covariance deterministic collision vs miss.
"""

import math
import pytest
import numpy as np

from theseus.uncertainty.b_plane import BPlaneUncertainty
from theseus.uncertainty.collision_probability import (
    compute_collision_probability,
    monte_carlo_pc_validation,
)
from theseus.conjunction.b_plane import BPlaneResult


def make_b_plane_unc(b_t: float, b_r: float, sigma_t: float, sigma_r: float, cov_tr: float = 0.0) -> BPlaneUncertainty:
    """Helper to build a BPlaneUncertainty container directly."""
    p_b = np.array([
        [sigma_t**2, cov_tr],
        [cov_tr, sigma_r**2],
    ], dtype=np.float64)

    eigvals, eigvecs = np.linalg.eigh(p_b)
    eigvals = np.maximum(0.0, eigvals)
    idx_sort = np.argsort(eigvals)
    lam_min = float(eigvals[idx_sort[0]])
    lam_max = float(eigvals[idx_sort[1]])
    v_maj = eigvecs[:, idx_sort[1]]

    denom = sigma_t * sigma_r
    corr = (cov_tr / denom) if denom > 1e-12 else 0.0

    b_res = BPlaneResult(
        applicable=True,
        reason="Test fixture",
        b_magnitude=math.sqrt(b_t**2 + b_r**2),
        b_dot_t=b_t,
        b_dot_r=b_r,
        s_hat=np.array([0.0, 0.0, 1.0]),
        t_hat=np.array([1.0, 0.0, 0.0]),
        r_hat=np.array([0.0, 1.0, 0.0]),
    )

    return BPlaneUncertainty(
        b_plane_covariance=p_b,
        b_dot_t=b_t,
        b_dot_r=b_r,
        sigma_t=sigma_t,
        sigma_r=sigma_r,
        cov_tr=cov_tr,
        correlation=corr,
        sigma_major=math.sqrt(lam_max),
        sigma_minor=math.sqrt(lam_min),
        ellipse_angle_deg=math.degrees(math.atan2(v_maj[1], v_maj[0])),
        ellipse_angle_rad=math.atan2(v_maj[1], v_maj[0]),
        eigenvalues=np.array([lam_min, lam_max]),
        eigenvectors=eigvecs,
        b_plane_result=b_res,
    )


def test_val_exact_isotropic_zero_miss():
    """
    Validation Test 1: Exact Analytical Solution for Zero Miss Distance
    For d = 0 and isotropic covariance σ_T = σ_R = σ:
    The 2D Gaussian integral over disk r ≤ HBR evaluates analytically to:
        Pc_exact = 1 - exp(-HBR² / (2 σ²))

    Reference: Akella & Alfriend, "Probability of Collision Between Space Objects", JSR 2000.
    """
    sigma = 50.0  # m
    hbr_values = [2.0, 5.0, 15.0, 30.0, 50.0, 100.0]

    b_unc = make_b_plane_unc(b_t=0.0, b_r=0.0, sigma_t=sigma, sigma_r=sigma)

    for hbr in hbr_values:
        res = compute_collision_probability(b_unc, hbr_m=hbr)
        pc_exact = 1.0 - math.exp(-(hbr**2) / (2.0 * sigma**2))

        # Must match exact analytical solution to high precision (< 1e-6)
        assert res.probability == pytest.approx(pc_exact, rel=1e-5, abs=1e-7)


def test_val_small_hbr_analytical_approximation():
    """
    Validation Test 2: Small-HBR Analytical Series
    When HBR << σ:
        Pc ≈ (HBR² / (2 σ²)) * exp(-d² / (2 σ²))
    """
    sigma = 200.0  # m
    d = 100.0      # m (miss distance)
    hbr = 2.0      # m (HBR << sigma)

    b_unc = make_b_plane_unc(b_t=d, b_r=0.0, sigma_t=sigma, sigma_r=sigma)
    res = compute_collision_probability(b_unc, hbr_m=hbr)

    # First-order approximation
    pc_approx = (hbr**2 / (2.0 * sigma**2)) * math.exp(-(d**2) / (2.0 * sigma**2))

    # Should match to within 1% relative error for HBR/sigma = 0.01
    assert res.probability == pytest.approx(pc_approx, rel=0.01)


def test_val_probability_dilution_peak():
    """
    Validation Test 3: Probability Dilution Phenomenon
    For a fixed miss distance d = 100 m and HBR = 10 m:
    - For tiny σ (e.g. 5 m), Pc is near zero (miss distance is 20-sigma away).
    - As σ increases, Pc rises to a peak near σ ≈ d / √2 ≈ 70.7 m.
    - As σ increases further (e.g. 5000 m), Pc decays toward zero (dilution).
    """
    d = 100.0
    hbr = 10.0
    sigmas = [5.0, 20.0, 40.0, 70.7, 120.0, 300.0, 1000.0, 5000.0]

    pc_curve = []
    for s in sigmas:
        b_unc = make_b_plane_unc(b_t=d, b_r=0.0, sigma_t=s, sigma_r=s)
        res = compute_collision_probability(b_unc, hbr_m=hbr)
        pc_curve.append(res.probability)

    # 1. Pc at small sigma is negligible
    assert pc_curve[0] < 1e-15

    # 2. Maximum Pc occurs near the theoretical peak (around index 3: 70.7m)
    max_idx = int(np.argmax(pc_curve))
    assert max_idx in [2, 3, 4]

    # 3. Dilution decay: Pc at huge sigma is much smaller than the peak
    assert pc_curve[-1] < pc_curve[3] / 10.0


def test_val_monotonicity_miss_and_hbr():
    """
    Validation Test 4: Physical Monotonicity
    """
    sigma_t = 150.0
    sigma_r = 80.0
    cov_tr = 4000.0  # Correlated covariance

    # Miss distance sweep
    pc_miss = []
    for d in [10.0, 30.0, 70.0, 150.0, 300.0]:
        b_unc = make_b_plane_unc(b_t=d, b_r=d/2.0, sigma_t=sigma_t, sigma_r=sigma_r, cov_tr=cov_tr)
        res = compute_collision_probability(b_unc, hbr_m=12.0)
        pc_miss.append(res.probability)

    for i in range(len(pc_miss) - 1):
        assert pc_miss[i] > pc_miss[i + 1]

    # HBR sweep
    pc_hbr = []
    b_fixed = make_b_plane_unc(b_t=50.0, b_r=30.0, sigma_t=sigma_t, sigma_r=sigma_r, cov_tr=cov_tr)
    for h in [1.0, 5.0, 10.0, 20.0, 40.0]:
        res = compute_collision_probability(b_fixed, hbr_m=h)
        pc_hbr.append(res.probability)

    for i in range(len(pc_hbr) - 1):
        assert pc_hbr[i] < pc_hbr[i + 1]


def test_val_monte_carlo_cross_validation():
    """
    Validation Test 5: Monte Carlo Statistical Cross-Validation
    Runs 200,000 Monte Carlo samples and validates that the deterministic Pc
    lies within the 99% statistical confidence interval.
    """
    b_unc = make_b_plane_unc(b_t=35.0, b_r=-20.0, sigma_t=60.0, sigma_r=40.0, cov_tr=800.0)
    hbr = 15.0

    mc_res = monte_carlo_pc_validation(b_unc, hbr_m=hbr, sample_count=200_000, seed=12345)

    assert mc_res.sample_count == 200_000
    assert mc_res.is_consistent is True
    # Deterministic Pc must be inside or right at the 99.7% margin
    assert abs(mc_res.empirical_pc - mc_res.deterministic_pc) < 3.5 * mc_res.standard_error


def test_val_extreme_cases():
    """
    Validation Test 6: Extreme Limits
    - Enormous HBR encompassing all probability mass: Pc -> 1.0
    - Enormous miss distance: Pc -> 0.0
    """
    # 1. Huge HBR
    b_unc = make_b_plane_unc(b_t=10.0, b_r=10.0, sigma_t=20.0, sigma_r=20.0)
    res_huge_hbr = compute_collision_probability(b_unc, hbr_m=500.0)  # 25-sigma radius
    assert res_huge_hbr.probability == pytest.approx(1.0, abs=1e-5)

    # 2. Huge miss distance
    b_unc_far = make_b_plane_unc(b_t=100_000.0, b_r=0.0, sigma_t=20.0, sigma_r=20.0)
    res_far = compute_collision_probability(b_unc_far, hbr_m=10.0)
    assert res_far.probability == 0.0
