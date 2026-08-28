"""
P10-09 — trace step 2 must measure what it claims.
P10-10 — branch selection must not depend on an arbitrary physical scale.

P10-09
------
Step 2 named three tests as its equation -- ``P = P^T``, ``lambda_i(P) >= 0``,
``diag(P) >= 0`` -- and then reported::

    "symmetry_verified": True,
    "psd_verified": True,
    "non_negative_variances": True,
    result: "COVARIANCES MATHEMATICALLY VALIDATED"

as literals, whatever the matrices contained.  ``StateCovariance.validate()``
is real and does reject bad matrices at construction, but it cannot be the
source of this claim: it raises rather than reporting, and it *repairs* --
zeroing slightly negative diagonals, symmetrising within tolerance, clipping
small negative eigenvalues -- so it never has a negative verdict to hand back.

Two reachable demonstrations, measured before the fix:

* two inputs each asymmetric by 4.000e+03 (relative 4.000e-01 against a
  1.000e-07 tolerance) whose asymmetries cancel in their sum, so the
  downstream relative-covariance construction does not fire, and step 2
  reported all three claims verified;
* a direct call with a covariance whose minimum eigenvalue and minimum
  variance are both -9.000e+03 -- step 2 again reported success.

P10-10
------
Pc is dimensionless.  Scaling sigma, the miss distance and the hard-body
radius by a common factor leaves every dimensionless ratio unchanged, so Pc
must be unchanged.  Measured before the fix, for
sigma_minor/sigma_major = 0.2, |b| = 1.5 sigma_major, R = 0.3 sigma_major:

    scale      Pc                 branch
    1e+06 m    5.80078271e-02     polar quadrature
    1e+00 m    5.80078271e-02     polar quadrature
    1e-02 m    5.80078271e-02     polar quadrature
    1e-04 m    0.00000000e+00     deterministic_limit   <-
    1e-09 m    0.00000000e+00     deterministic_limit   <-

The determinant test ``det_p < 1e-12`` has units of metres to the fourth, so
it scales as the fourth power of the length unit: a 5.8 % probability was
reported as exactly impossible, and as converged, purely because the same
encounter was expressed at a smaller scale.  The far-separation branch's
``max(sigma_major, 1.0)`` floor measured separation in metres once sigma fell
below one metre -- conservative rather than wrong, since the quadrature then
returns the same 0.0, but scale-dependent all the same.

Independent references
----------------------
For P10-09, the three claims are recomputed directly from the matrices with
numpy -- no production validator involved.  For P10-10, Monte Carlo sampling
and a dense two-dimensional polar grid (both from the P10-08 module, neither
being production's one-dimensional verification integral).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from theseus.uncertainty.b_plane import project_covariance_to_b_plane
from theseus.uncertainty.collision_probability import (
    DETERMINISTIC_LIMIT_SIGMA_RATIO,
    FAR_SEPARATION_DISK_SIGMA,
    FAR_SEPARATION_SIGMA,
    FLOAT_SAFE_SIGMA_M,
    compute_collision_probability,
)
from theseus.uncertainty.covariance import StateCovariance, CovarianceValidationError
from theseus.uncertainty.hard_body import compute_hard_body_radius
from theseus.uncertainty.relative import compute_relative_covariance
from theseus.uncertainty.results import (
    build_phase10_calculation_trace,
    measure_covariance_validity,
)
from theseus.uncertainty.risk import PROFILE_STANDARD, classify_risk

from tests.test_phase10_pc_convergence import dense_polar_pc, monte_carlo_pc


# ---------------------------------------------------------------------------
# Shared construction
# ---------------------------------------------------------------------------

def encounter(sigma_major_m, sigma_minor_m, miss_t_m=0.0, miss_r_m=0.0):
    return project_covariance_to_b_plane(
        rel_pos_cov=np.diag([sigma_major_m ** 2, sigma_minor_m ** 2, 1.0e-30]),
        r_rel=np.array([miss_t_m, miss_r_m, 0.0]),
        v_rel=np.array([0.0, 0.0, 1.0e4]),
    )


def trace_step(index, cov_a, cov_b, rel_cov=None):
    bpu = encounter(500.0, 50.0, 300.0)
    pc_res = compute_collision_probability(bpu, 10.0)
    steps = build_phase10_calculation_trace(
        initial_cov_a=cov_a, initial_cov_b=cov_b,
        cov_a_tca=cov_a, cov_b_tca=cov_b,
        rel_cov=rel_cov if rel_cov is not None
        else compute_relative_covariance(cov_a, cov_b),
        tca_s=100.0,
        r_rel_tca=np.array([300.0, 0.0, 0.0]),
        v_rel_tca=np.array([0.0, 0.0, 1.0e4]),
        b_plane_unc=bpu,
        hbr_res=compute_hard_body_radius(custom_hbr_m=10.0),
        pc_res=pc_res,
        risk=classify_risk(pc_res.probability, PROFILE_STANDARD),
    )
    return next(s for s in steps if s["stepIndex"] == index)


def independent_claims(matrix):
    """
    The three properties step 2 asserts, computed directly.  Uses no
    production validator, so it can disagree with one.

    ``min_eigenvalue`` is taken on the correlation form D⁻¹PD⁻¹, which is the
    basis the validator has judged positive semi-definiteness in since P10-11:
    PSD is congruence-invariant, so the sign is the same, but the raw
    eigenvalue is not resolvable when the position and velocity blocks span
    many orders of magnitude.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    variances = np.diag(matrix)
    positive = variances > 0.0
    if np.any(positive):
        deviations = np.sqrt(variances[positive])
        correlation = matrix[np.ix_(positive, positive)] / np.outer(deviations, deviations)
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(
            0.5 * (correlation + correlation.T))))
    else:
        min_eigenvalue = 0.0
    return {
        "max_asymmetry": float(np.max(np.abs(matrix - matrix.T))),
        "min_eigenvalue": min_eigenvalue,
        "min_variance": float(np.min(variances)),
    }


def valid_covariance(name="OK"):
    return StateCovariance.from_diagonal([100.0] * 3, [0.1] * 3, name=name)


def cancelling_asymmetric_pair():
    """
    Two covariances each asymmetric by 4.0e+03 whose asymmetries cancel in
    their sum, so the downstream relative-covariance construction accepts the
    pair and the corruption reaches the trace.
    """
    a = valid_covariance("A")
    b = valid_covariance("B")
    a.matrix[0, 1] += 4.0e3
    b.matrix[1, 0] += 4.0e3
    return a, b


# ===========================================================================
# P10-09
# ===========================================================================

def test_valid_covariances_report_all_three_claims_verified():
    step = trace_step(2, valid_covariance("A"), valid_covariance("B"))

    assert step["substitutions"]["symmetry_verified"] is True
    assert step["substitutions"]["psd_verified"] is True
    assert step["substitutions"]["non_negative_variances"] is True
    assert step["status"] == "completed"
    assert step["result"] == "COVARIANCES MATHEMATICALLY VALIDATED"


def test_cancelling_asymmetries_reach_the_trace_and_are_reported():
    """
    The reachable case.  Each input is asymmetric far beyond tolerance, but
    their sum is exactly symmetric, so nothing downstream objects.
    """
    a, b = cancelling_asymmetric_pair()

    # The corruption is real: each matrix would be rejected on its own.
    for cov in (a, b):
        assert independent_claims(cov.matrix)["max_asymmetry"] == pytest.approx(4.0e3)
        with pytest.raises(CovarianceValidationError):
            cov.validate()

    # And the barrier genuinely does not fire on the pair.
    rel = compute_relative_covariance(a, b)
    assert float(np.max(np.abs(rel.relative_covariance.matrix
                               - rel.relative_covariance.matrix.T))) == 0.0

    step = trace_step(2, a, b, rel_cov=rel)
    assert step["substitutions"]["symmetry_verified"] is False
    assert step["status"] == "warning"
    assert "VALIDATION FAILED" in step["result"]
    assert "asymmetric" in step["result"]


def test_each_claim_is_reported_independently():
    """
    A negative variance must not be reported as an asymmetry, and vice versa.
    A single pass/fail verdict would hide which invariant broke.
    """
    bad = valid_covariance("C")
    bad.matrix[2, 2] = -9.0e3
    measured = independent_claims(bad.matrix)
    assert measured["min_variance"] == pytest.approx(-9.0e3)
    assert measured["max_asymmetry"] == 0.0

    step = trace_step(2, bad, valid_covariance("D"),
                      rel_cov=compute_relative_covariance(valid_covariance("D"),
                                                          valid_covariance("D")))
    substitutions = step["substitutions"]
    assert substitutions["symmetry_verified"] is True
    # A negative variance is itself a proof of non-PSD (e_i^T P e_i < 0), so
    # both claims must fail -- the correlation form alone cannot see it,
    # because that row is excluded from the normalisation.
    assert substitutions["psd_verified"] is False
    assert substitutions["non_negative_variances"] is False
    assert "negative variance" in step["result"]


@pytest.mark.parametrize("corruption,expected_failure", [
    ("asymmetry", "symmetry_verified"),
    ("negative_variance", "non_negative_variances"),
    ("indefinite", "psd_verified"),
])
def test_the_verdict_depends_on_the_measured_property(corruption, expected_failure):
    """
    Each verdict must move when, and only when, its own quantity moves.
    """
    cov = valid_covariance("X")
    if corruption == "asymmetry":
        cov.matrix[0, 1] += 4.0e3
    elif corruption == "negative_variance":
        cov.matrix[3, 3] = -1.0
    else:
        # symmetric, non-negative diagonal, still indefinite
        cov.matrix[0, 1] = cov.matrix[1, 0] = 1.0e5

    measured = measure_covariance_validity(cov)
    assert measured["valid"] is False

    key_map = {
        "symmetry_verified": "symmetric",
        "non_negative_variances": "non_negative_variances",
        "psd_verified": "positive_semidefinite",
    }
    assert measured[key_map[expected_failure]] is False


def test_measurement_matches_an_independent_computation():
    """
    ``measure_covariance_validity`` against direct numpy, over random matrices
    including borderline ones.
    """
    rng = np.random.default_rng(20260826)
    for _ in range(60):
        base = rng.normal(0.0, 100.0, (6, 6))
        matrix = base @ base.T
        if rng.random() < 0.4:
            matrix[0, 1] += rng.choice([0.0, 1e-9, 1e3])
        if rng.random() < 0.3:
            matrix[2, 2] = -abs(rng.normal(0.0, 10.0))

        cov = StateCovariance.from_diagonal([1.0] * 3, [1.0] * 3, name="R")
        cov.matrix = matrix

        measured = measure_covariance_validity(cov)
        reference = independent_claims(matrix)

        assert measured["max_asymmetry"] == pytest.approx(reference["max_asymmetry"])
        assert measured["min_eigenvalue"] == pytest.approx(reference["min_eigenvalue"])
        assert measured["min_variance"] == pytest.approx(reference["min_variance"])
        assert measured["non_negative_variances"] is (reference["min_variance"] >= 0.0)


def test_measurement_does_not_repair_the_matrix():
    """
    Unlike ``validate()``, measuring must leave the matrix exactly as it was --
    otherwise the trace would be describing something the analysis did not use.
    """
    cov = valid_covariance("Y")
    cov.matrix[0, 1] += 4.0e3
    before = cov.matrix.copy()

    measure_covariance_validity(cov)

    np.testing.assert_array_equal(cov.matrix, before)


def test_step_2_schema_is_preserved():
    """The three original keys survive, as booleans, for existing consumers."""
    step = trace_step(2, valid_covariance("A"), valid_covariance("B"))
    for key in ("symmetry_verified", "psd_verified", "non_negative_variances"):
        assert key in step["substitutions"]
        assert isinstance(step["substitutions"][key], bool)
    assert step["equation"] == "P = P^T,  λ_i(P) ≥ 0,  diag(P) ≥ 0"


# ===========================================================================
# P10-10
# ===========================================================================

#: sigma_minor/sigma_major = 0.2, |b| = 1.5 sigma_major, R = 0.3 sigma_major.
#: Independently computed Pc for this geometry, scale-free.
SCALE_FREE_PC = 5.8007827098e-02
SCALES_M = (1e6, 1e3, 1e0, 1e-2, 1e-4, 1e-6, 1e-9)


@pytest.mark.parametrize("scale", SCALES_M)
def test_probability_is_invariant_under_a_common_rescaling(scale):
    """
    The headline P10-10 assertion.  Below 1e-2 m this used to return exactly
    0.0 through the deterministic branch.
    """
    result = compute_collision_probability(
        encounter(scale, 0.2 * scale, 1.5 * scale, 0.0), 0.3 * scale)

    assert result.probability == pytest.approx(SCALE_FREE_PC, rel=1e-6)
    assert result.method == "2D_Gaussian_Polar_Quadrature_Principal_Axes"


def test_the_scale_free_probability_matches_an_independent_reference():
    """
    ``SCALE_FREE_PC`` is not taken from the implementation: Monte Carlo and a
    dense polar grid agree on it.
    """
    bpu = encounter(1.0e3, 2.0e2, 1.5e3, 0.0)
    hbr = 3.0e2

    grid_coarse = dense_polar_pc(bpu, hbr, nodes=600)
    grid_fine = dense_polar_pc(bpu, hbr, nodes=1200)
    assert abs(grid_fine - grid_coarse) / grid_fine < 1e-8
    assert grid_fine == pytest.approx(SCALE_FREE_PC, rel=1e-6)

    estimate, standard_error = monte_carlo_pc(bpu, hbr)
    assert abs(estimate - SCALE_FREE_PC) < 5.0 * standard_error


def test_deterministic_limit_fires_on_a_ratio_not_on_a_length():
    """
    The branch must be selected by sigma relative to the distance from the
    hit/miss boundary, and that selection must survive rescaling.
    """
    for scale in (1e6, 1e0, 1e-6):
        # sigma is 1e-10 of the boundary clearance: deterministically inside.
        miss = 1.0 * scale
        hbr = 3.0 * scale
        sigma = 1e-10 * abs(miss - hbr)
        result = compute_collision_probability(
            encounter(sigma, 0.5 * sigma, miss, 0.0), hbr)

        assert result.method == "deterministic_limit"
        assert result.probability == 1.0          # miss < hbr -> certain overlap
        assert result.diagnostics["result_kind"] == "deterministic_limit"
        assert (result.diagnostics["sigma_major_over_boundary_clearance"]
                <= DETERMINISTIC_LIMIT_SIGMA_RATIO)


def test_deterministic_limit_agrees_with_direct_integration_where_it_fires():
    """
    Where the shortcut is taken, the full integral must give the same answer --
    otherwise the shortcut is not a limit, it is a guess.
    """
    miss, hbr = 1.0e3, 3.0e3
    sigma = 1e-10 * abs(miss - hbr)
    bpu = encounter(sigma, 0.5 * sigma, miss, 0.0)

    shortcut = compute_collision_probability(bpu, hbr)
    assert shortcut.method == "deterministic_limit"

    estimate, _ = monte_carlo_pc(bpu, hbr, samples=200_000)
    assert estimate == pytest.approx(shortcut.probability, abs=1e-12)


def test_deterministic_limit_does_not_fire_near_the_boundary():
    """
    When the uncertainty can reach the hit/miss boundary the 0/1 answer is not
    a limit of anything, so the quadrature must run.
    """
    miss, hbr = 1000.0, 1000.5
    result = compute_collision_probability(encounter(50.0, 10.0, miss, 0.0), hbr)
    assert result.method == "2D_Gaussian_Polar_Quadrature_Principal_Axes"
    assert 0.0 < result.probability < 1.0


@pytest.mark.parametrize("sigma_major", (1e4, 1e2, 1e0, 1e-1, 1e-2))
def test_far_separation_branch_is_scale_free(sigma_major):
    """
    Below one metre the old ``max(sigma_major, 1.0)`` floor blocked this
    shortcut.  The answer was the same 0.0 either way, so this is about the
    criterion, not the value -- which the assertion on the probability pins.
    """
    miss = 200.0 * sigma_major
    hbr = 0.3 * sigma_major
    result = compute_collision_probability(
        encounter(sigma_major, 0.2 * sigma_major, miss, 0.0), hbr)

    assert result.method == "analytic_far_separation"
    assert result.probability == 0.0
    assert result.diagnostics["result_kind"] == "analytic_exact"
    assert result.diagnostics["miss_over_sigma_major"] > FAR_SEPARATION_SIGMA


def test_far_separation_shortcut_matches_the_full_integral():
    """The density really has underflowed; the shortcut is exact, not a cutoff."""
    sigma_major = 100.0
    miss = 200.0 * sigma_major
    hbr = 0.3 * sigma_major
    bpu = encounter(sigma_major, 0.2 * sigma_major, miss, 0.0)

    shortcut = compute_collision_probability(bpu, hbr)
    assert shortcut.method == "analytic_far_separation"

    # The Mahalanobis distance to the nearest point of the disk, whose square
    # is the exponent the density is evaluated at.  exp(-d^2/2) underflows to
    # exactly zero once d^2/2 exceeds about 745, i.e. beyond roughly 38.7
    # sigma, so past that the shortcut's 0.0 is the exact answer and not a
    # cutoff.  The branch only fires at 50 sigma, well clear of the boundary.
    covariance = np.asarray(bpu.b_plane_covariance, dtype=float)
    mean = np.array([bpu.b_dot_t, bpu.b_dot_r], dtype=float)
    mahalanobis = math.sqrt(float(mean @ np.linalg.inv(covariance) @ mean))
    nearest = mahalanobis - hbr / math.sqrt(float(np.min(np.linalg.eigvalsh(covariance))))
    assert nearest > 39.0
    assert math.exp(-0.5 * nearest ** 2) == 0.0
    # And the branch's own trigger point is already past underflow.
    assert math.exp(-0.5 * FAR_SEPARATION_SIGMA ** 2) == 0.0

    # The dense grid agrees to within its own exponent floor: its integrand is
    # clipped at exp(-700), so it cannot return a true zero, only something
    # below the smallest normal double.
    assert dense_polar_pc(bpu, hbr, nodes=400) < 1e-300


def test_no_false_early_returns_across_ordinary_geometries():
    """
    A sweep of unremarkable encounters must all reach the quadrature: an early
    return that fires when it should not is a silent wrong answer.
    """
    rng = np.random.default_rng(11)
    for _ in range(40):
        sigma_major = 10.0 ** rng.uniform(0.5, 4.0)
        sigma_minor = sigma_major * rng.uniform(0.1, 1.0)
        miss = sigma_major * rng.uniform(0.0, 3.0)
        hbr = sigma_major * rng.uniform(0.01, 0.5)
        result = compute_collision_probability(
            encounter(sigma_major, sigma_minor, miss, 0.0), hbr)
        assert result.method == "2D_Gaussian_Polar_Quadrature_Principal_Axes"
        assert 0.0 < result.probability <= 1.0


def test_float_safety_guard_is_dimensional_for_a_stated_reason():
    """
    The one surviving dimensional threshold.  Below it ``1/(2 sigma^2)``
    overflows, which is a property of IEEE-754 and not of any physical scale.
    """
    assert FLOAT_SAFE_SIGMA_M < 1e-140

    # Below the bound the square underflows to zero outright, so forming the
    # density is not merely inaccurate -- it is impossible.
    assert (1e-200) ** 2 == 0.0
    with pytest.raises(ZeroDivisionError):
        1.0 / (2.0 * (1e-200) ** 2)

    result = compute_collision_probability(encounter(1e-200, 1e-200, 0.0, 0.0), 1e-200)
    assert result.method == "deterministic_limit"
    assert result.diagnostics["density_is_representable"] is False
    assert math.isfinite(result.probability)


@pytest.mark.parametrize("hbr", (0.05, 0.12, 0.6, 1.9))
def test_small_hard_body_radii_still_reach_the_quadrature(hbr):
    """P10-04 interaction: no early return may swallow the small-HBR range."""
    result = compute_collision_probability(encounter(500.0, 100.0, 300.0, 0.0), hbr)
    assert result.method == "2D_Gaussian_Polar_Quadrature_Principal_Axes"
    assert result.probability > 0.0
    assert result.converged is True


# ===========================================================================
# Cross-finding
# ===========================================================================

def test_analytic_branches_are_distinguished_from_converged_quadrature():
    """
    An exact analytic result and a numerically converged one both report
    ``converged=True``; the trace must be able to tell them apart.
    """
    exact = compute_collision_probability(encounter(500.0, 50.0), 0.0)
    assert exact.converged is True
    assert exact.diagnostics["result_kind"] == "analytic_exact"
    assert "verification_disagreement" not in exact.diagnostics

    numerical = compute_collision_probability(encounter(500.0, 50.0, 300.0), 10.0)
    assert numerical.converged is True
    assert numerical.diagnostics["result_kind"] == "numerical_quadrature"
    assert "verification_disagreement" in numerical.diagnostics


def test_an_early_return_does_not_suppress_a_validation_failure():
    """
    Cross-finding requirement: a far-separation shortcut must not stop step 2
    from reporting that an input covariance is invalid.
    """
    a, b = cancelling_asymmetric_pair()
    rel = compute_relative_covariance(a, b)

    bpu = encounter(100.0, 20.0, 20000.0, 0.0)
    pc_res = compute_collision_probability(bpu, 10.0)
    assert pc_res.method == "analytic_far_separation"

    steps = build_phase10_calculation_trace(
        initial_cov_a=a, initial_cov_b=b, cov_a_tca=a, cov_b_tca=b, rel_cov=rel,
        tca_s=100.0, r_rel_tca=np.array([20000.0, 0.0, 0.0]),
        v_rel_tca=np.array([0.0, 0.0, 1.0e4]), b_plane_unc=bpu,
        hbr_res=compute_hard_body_radius(custom_hbr_m=10.0), pc_res=pc_res,
        risk=classify_risk(pc_res.probability, PROFILE_STANDARD),
    )
    step_2 = next(s for s in steps if s["stepIndex"] == 2)
    assert step_2["status"] == "warning"
    assert step_2["substitutions"]["symmetry_verified"] is False


def test_p10_08_convergence_reporting_is_unchanged():
    """
    P10-08 stays authoritative for the quadrature: a high-anisotropy case is
    still flagged, a healthy one still is not.
    """
    # P10-12 note: this geometry is still detected as one the polar quadrature
    # gets wrong -- that is P10-08's contribution and it is unchanged.  What
    # changed is that the detection is now acted on, so the reported value comes
    # from the exact reduction and the result converges.
    broken = compute_collision_probability(encounter(1.0e5, 1.0e0), 2000.0)
    assert broken.diagnostics["polar_quadrature_agrees"] is False
    assert broken.diagnostics["polar_quadrature_superseded"] is True
    assert broken.diagnostics["result_kind"] == "exact_reduction"

    # And a geometry that still cannot be corroborated at all is still flagged.
    uncertifiable = compute_collision_probability(encounter(1.0e2, 1.0e-3), 1000.0)
    assert uncertifiable.converged is False
    assert uncertifiable.diagnostics["certified"] is False

    healthy = compute_collision_probability(encounter(5.0e2, 5.0e1, 300.0), 10.0)
    assert healthy.converged is True


def test_risk_api_is_unaffected():
    """End to end: an ordinary request still produces the same shape of answer."""
    from fastapi.testclient import TestClient
    from theseus.server.app import app

    client = TestClient(app)
    response = client.post("/api/simulate/conjunction/risk", json={
        "object_a_alt_km": 400.0, "object_a_inc_deg": 51.6, "object_a_phase_deg": 0.0,
        "object_b_alt_km": 400.05, "object_b_inc_deg": 55.0, "object_b_phase_deg": 0.02,
        "analysis_duration_hours": 2.0, "screening_threshold_km": 100.0,
        "coarse_dt_s": 30.0, "hard_body_radius_m": 15.0,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_status"] == "COMPLETE"
    assert data["collision_probability"]["converged"] is True

    step_2 = next(s for s in data["calculation_steps"] if s["stepIndex"] == 2)
    assert step_2["status"] == "completed"
    assert step_2["substitutions"]["symmetry_verified"] is True
    assert step_2["substitutions"]["psd_verified"] is True
