"""
P10-12 -- the high-anisotropy collision-probability quadrature must be right,
not merely labelled.

The defect
----------
P10-08 established that ``compute_collision_probability`` could return a
probability 19 % to 97 % low and that the failure was silent.  It made the
``converged`` flag tell the truth and stopped there: the wrong number was
still what the caller received.  For a conjunction-assessment tool,
understating a collision probability by a factor of 31 is the dangerous
direction, and a flag is not a fix.

The mechanism, measured rather than assumed
-------------------------------------------
"dblquad is inaccurate" is not a root cause.  The production integral is

    Pc = norm * int_0^HBR dr int_0^{2 pi} d(theta)  r exp(-u^2/2 sx^2 - v^2/2 sy^2)

with the OUTER variable r and the INNER variable theta.  Instrumenting it
shows the inner ``quad`` call returning **exactly 0.0, with an error estimate
of exactly 0.0, after exactly 21 function evaluations** for every radius past
a threshold.  21 is the size of QUADPACK's initial Gauss-Kronrod rule:

* In principal axes the density is a ridge along the major axis, of half-width
  ``sigma_minor / r`` in theta at radius r.
* The GK21 rule on [0, 2 pi] has its nearest node 0.1971573532 rad from each
  ridge crossing -- a fixed offset, independent of the integrand.
* The production integrand returns 0 when its exponent falls below -500, i.e.
  when ``|u| > sqrt(1000) sigma_minor``.  Every node is therefore zeroed once
  ``r * 0.1971573532 > sqrt(1000) sigma_minor``, that is once

      r > r_crit = sqrt(1000) / 0.1971573532 * sigma_minor = 160.39 sigma_minor

* A rule that samples only zeros returns 0 with an error estimate of 0.  QAGS
  reads that as a converged panel and never subdivides.  An adaptive algorithm
  cannot detect a feature it never sampled.
* So the outer r-integral accumulates nothing beyond r_crit, and

      Pc_reported / Pc_true  ~  160.39 * sigma_minor / HBR

This prediction was checked against measurement across five decades of
anisotropy and holds to a constant factor of 0.9935:

    sigma ratio   HBR      measured ratio   predicted
       1e+04     200 m       0.807200       0.801968
       3e+04     600 m       0.269074       0.267323
       1e+05    2000 m       0.080733       0.080197
       3e+05    5000 m       0.032288       0.032079
       5e+05    5000 m       0.032288       0.032079

The -500 clamp is where the cliff falls, not why it exists: removing it moves
the threshold to IEEE underflow near -745 and the failure persists at 67 %,
90 % and 96 % for the higher ratios.

What was rejected, and why
--------------------------
The module docstring advertised "Chan's series expansion for isotropic and
mildly anisotropic cases" and no such code existed.  Rather than write what
was advertised, it was measured against 50-digit arithmetic.  Chan's
equivalent-area series is exact for isotropic encounters and usable to about
10:1 (1.9e-3), but its equal-area substitution is an approximation and it
fails hard in exactly this regime -- 0.997 relative error at 1e3:1, and
overestimates by factors of 98 and 124 at 4e4:1 and 5e5:1.  It was not
adopted; the advertisement was removed instead.

A predictive switching rule was also tried and rejected on evidence.  Across
337 random geometries the ridge-resolution number does not separate accurate
polar results from inaccurate ones -- the largest value among ACCURATE cases
is 155.0 and the smallest among INACCURATE ones is 2.03 -- because the polar
quadrature has at least three unrelated failure modes: the ridge stepped over
in theta, a spike stepped over in r, and crescent geometries where the disk
excludes the density centre.

What was done
-------------
The exact 1-D reduction is orientation-dependent: whichever axis is integrated
numerically sets the panel requirement at O(HBR / sigma_of_that_axis).  Taken
in both orientations it gives two evaluations of the same identity with
complementary conditioning.  Together with the polar quadrature that is three
independent constructions, and the value at least two of them agree on is the
one reported.  Where the polar quadrature is in the agreeing pair -- every
case that was already correct -- its own value is reported unchanged.

What these tests pin
--------------------
A. The mechanism is what it is claimed to be, including the 21-evaluation
   silent-zero signature and the quantitative scaling law.
B. The tabulated failures are gone, checked against constructions that share
   nothing with the fix.
C. Healthy encounters are untouched, value for value.
D. The reduction's two orientations really are complementary.
E. Chan's series would not have worked.
F. Pc stays in [0, 1] and the boundary branches stay exact.
G. Earlier findings are unaffected.

Independent references used here: Monte Carlo sampling (no quadrature at all),
a dense two-dimensional polar Gauss-Legendre grid (two dimensions where the
fix uses one), and closed-form values where the geometry admits them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtr

from theseus.uncertainty.b_plane import project_covariance_to_b_plane
from theseus.uncertainty.collision_probability import (
    ENGULFMENT_SIGMA,
    QUADRATURE_AGREEMENT_RTOL,
    REDUCTION_PANEL_CAP,
    _disk_integral_1d,
    certified_disk_integral,
    compute_collision_probability,
)


# ---------------------------------------------------------------------------
# Fixtures and independent references
# ---------------------------------------------------------------------------

def encounter(sigma_major_m: float, sigma_minor_m: float,
              miss_t_m: float = 0.0, miss_r_m: float = 0.0):
    """A B-plane uncertainty built through the real projection."""
    return project_covariance_to_b_plane(
        rel_pos_cov=np.diag([sigma_major_m ** 2, sigma_minor_m ** 2, 1.0]),
        r_rel=np.array([miss_t_m, miss_r_m, 0.0]),
        v_rel=np.array([0.0, 0.0, 1.0e4]),
    )


def principal_axes(bpu):
    """(sigma_x, sigma_y, mu_x, mu_y) exactly as production derives them."""
    order = np.argsort(bpu.eigenvalues)
    basis = np.column_stack([bpu.eigenvectors[:, order[0]],
                             bpu.eigenvectors[:, order[1]]])
    mu = basis.T @ np.array([bpu.b_dot_t, bpu.b_dot_r], dtype=float)
    return (math.sqrt(float(bpu.eigenvalues[order[0]])),
            math.sqrt(float(bpu.eigenvalues[order[1]])),
            float(mu[0]), float(mu[1]))


def monte_carlo_pc(bpu, hbr_m: float, samples: int = 2_000_000, seed: int = 12345):
    """
    Pc by direct sampling.  No quadrature of any kind, so it cannot share a
    quadrature blind spot with anything under test.
    """
    covariance = np.asarray(bpu.b_plane_covariance, dtype=float)
    mean = np.array([bpu.b_dot_t, bpu.b_dot_r], dtype=float)
    chol = np.linalg.cholesky(covariance)
    rng = np.random.default_rng(seed)

    hits = 0
    drawn = 0
    while drawn < samples:
        batch = min(500_000, samples - drawn)
        points = (chol @ rng.standard_normal((2, batch))).T + mean
        hits += int(np.count_nonzero(
            np.einsum("ij,ij->i", points, points) <= hbr_m * hbr_m))
        drawn += batch

    estimate = hits / samples
    error = math.sqrt(max(estimate, 1.0 / samples) * (1.0 - estimate) / samples)
    return estimate, error


def ridge_aligned_reference(sigma_x, sigma_y, mu_x, mu_y, hbr_m,
                            ridge_nodes=4000, along_nodes=4000):
    """
    A dense two-dimensional Cartesian grid laid out along the ridge rather than
    about the disk centre, so the narrow direction is sampled by construction.

    This shares no code with production: two dimensions instead of one, no
    analytic inner integral, no sin substitution, Cartesian instead of polar,
    and the node layout derived from sigma rather than from the disk.
    """
    # x spans the ridge: dense within a few sigma_x of the origin, since the
    # density is negligible outside that.
    x_half = min(hbr_m + abs(mu_x), 9.0 * sigma_x)
    xs, xw = np.polynomial.legendre.leggauss(ridge_nodes)
    x = x_half * xs
    wx = x_half * xw

    y_half = min(hbr_m + abs(mu_y), 9.0 * sigma_y)
    ys, yw = np.polynomial.legendre.leggauss(along_nodes)
    y = y_half * ys
    wy = y_half * yw

    xm, ym = np.meshgrid(x, y, indexing="ij")
    inside = (xm - mu_x) ** 2 + (ym - mu_y) ** 2 <= hbr_m * hbr_m
    density = np.exp(np.clip(
        -(xm * xm / (2.0 * sigma_x ** 2) + ym * ym / (2.0 * sigma_y ** 2)),
        -700.0, None)) / (2.0 * math.pi * sigma_x * sigma_y)
    return float(np.sum(wx[:, None] * wy[None, :] * np.where(inside, density, 0.0)))


def chan_equivalent_area_series(sigma_x, sigma_y, mu_x, mu_y, hbr_m, order=200):
    """
    Chan's equivalent-area series -- the method the module docstring used to
    advertise.  Present so the decision not to adopt it is checkable rather
    than asserted.
    """
    if hbr_m <= 0.0:
        return 0.0
    u = (hbr_m * hbr_m) / (sigma_x * sigma_y)
    v = (mu_x / sigma_x) ** 2 + (mu_y / sigma_y) ** 2

    total = 0.0
    term_v = math.exp(-v / 2.0) if v / 2.0 < 700 else 0.0
    term_inner = math.exp(-u / 2.0) if u / 2.0 < 700 else 0.0
    inner_cum = term_inner
    for m in range(order + 1):
        if m > 0:
            term_v *= (v / 2.0) / m
            term_inner *= (u / 2.0) / m
            inner_cum += term_inner
        total += term_v * (1.0 - inner_cum)
        if m > 10 and term_v < 1e-18 * max(total, 1e-300):
            break
    return float(max(0.0, min(1.0, total)))


#: The rows P10-08 tabulated, with the relative error the polar quadrature
#: makes on each.  (sigma_major, sigma_minor, hbr, polar relative error)
TABULATED = [
    (1.0e4, 1.0e0, 200.0, 0.193),
    (3.0e4, 1.0e0, 600.0, 0.731),
    (1.0e5, 1.0e0, 2000.0, 0.919),
    (5.0e5, 1.0e0, 5000.0, 0.968),
]

#: Encounters the polar quadrature has always got right.
HEALTHY = [
    (5.0e2, 5.0e1, 10.0),
    (5.0e3, 5.0e2, 50.0),
    (1.0e3, 1.0e0, 20.0),
    (3.0e3, 1.0e0, 60.0),
    (2.0e2, 2.0e2, 15.0),
]


# ---------------------------------------------------------------------------
# A. The mechanism is what it is claimed to be
# ---------------------------------------------------------------------------

def test_the_inner_rule_returns_zero_after_exactly_21_evaluations():
    """
    The signature of the defect.  If the inner theta-integral ever stops
    returning exactly 0.0 with an error estimate of exactly 0.0 after exactly
    21 evaluations, the mechanism documented above is no longer the mechanism
    and the reasoning behind the fix needs revisiting.
    """
    sigma_x, sigma_y, radius = 1.0, 1.0e5, 1000.0
    calls = []

    def inner(theta):
        calls.append(theta)
        u = radius * math.cos(theta)
        v = radius * math.sin(theta)
        exponent = -(u * u / (2 * sigma_x ** 2) + v * v / (2 * sigma_y ** 2))
        return 0.0 if exponent < -500.0 else float(radius * math.exp(exponent))

    value, error_estimate = integrate.quad(inner, 0.0, 2.0 * math.pi,
                                           epsabs=1e-8, epsrel=1e-8)

    assert len(calls) == 21, "QAGS subdivided; the initial rule did not swallow it"
    assert value == 0.0
    assert error_estimate == 0.0

    # The true value is emphatically not zero.
    truth = _disk_integral_1d(sigma_y, sigma_x, 0.0, 0.0, radius, 256)
    assert truth > 1e-4


def test_the_nearest_gauss_kronrod_node_sits_where_the_mechanism_says():
    """
    The 0.19716 rad offset the scaling law is built on, recovered from the
    rule itself rather than quoted.
    """
    nodes = []

    def probe(theta):
        nodes.append(theta)
        return 0.0

    integrate.quad(probe, 0.0, 2.0 * math.pi, epsabs=1e-8, epsrel=1e-8)
    sampled = np.array(nodes)
    offset = np.min(np.minimum(np.abs(sampled - 0.5 * math.pi),
                               np.abs(sampled - 1.5 * math.pi)))
    assert offset == pytest.approx(0.1971573532, abs=1e-9)

    predicted_r_crit = math.sqrt(1000.0) / offset
    assert predicted_r_crit == pytest.approx(160.39, rel=1e-3)


@pytest.mark.parametrize("sigma_major,sigma_minor,hbr,expected_error", TABULATED)
def test_the_scaling_law_predicts_the_measured_loss(
        sigma_major, sigma_minor, hbr, expected_error):
    """
    The quantitative claim: the polar quadrature keeps a fraction
    160.39 * sigma_minor / HBR of the probability, to a constant factor.
    """
    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)
    polar = result.diagnostics["raw_probability"]
    truth = result.probability

    measured_fraction = polar / truth
    predicted_fraction = 160.39 * sigma_minor / hbr
    assert measured_fraction == pytest.approx(predicted_fraction, rel=0.02)
    assert 1.0 - measured_fraction == pytest.approx(expected_error, abs=0.005)


def test_the_underflow_clamp_is_the_trigger_not_the_cause():
    """
    Removing the -500 clamp moves the cliff to IEEE underflow; it does not
    remove it.  This matters because "just widen the clamp" is the obvious
    wrong fix.
    """
    sigma_x, sigma_y, hbr = 1.0, 1.0e5, 2000.0
    norm = 1.0 / (2.0 * math.pi * sigma_x * sigma_y)

    def unclamped(theta, r):
        u = r * math.cos(theta)
        v = r * math.sin(theta)
        return float(r * math.exp(-(u * u / (2 * sigma_x ** 2)
                                    + v * v / (2 * sigma_y ** 2))))

    value, _ = integrate.dblquad(unclamped, 0.0, hbr, 0.0, 2.0 * math.pi,
                                 epsabs=1e-8, epsrel=1e-8)
    without_clamp = value * norm
    truth = _disk_integral_1d(sigma_y, sigma_x, 0.0, 0.0, hbr, 512)

    assert abs(without_clamp - truth) / truth > 0.5, \
        "removing the clamp must not repair the defect"


def test_the_failure_does_not_depend_on_the_miss_distance():
    """
    The loss is a property of the ridge geometry, not of where the disk sits.
    A fix keyed to the miss distance would therefore be keyed to the wrong
    thing.
    """
    errors = []
    for miss_fraction in (0.0, 0.5, 1.5, 3.0):
        result = compute_collision_probability(
            encounter(1.0e5, 1.0e0, miss_t_m=miss_fraction * 2000.0), 2000.0)
        errors.append(abs(result.diagnostics["raw_probability"] - result.probability)
                      / result.probability)
    assert max(errors) - min(errors) < 0.01
    assert min(errors) > 0.9


# ---------------------------------------------------------------------------
# B. The tabulated failures are gone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma_major,sigma_minor,hbr,expected_error", TABULATED)
def test_tabulated_failures_now_agree_with_monte_carlo(
        sigma_major, sigma_minor, hbr, expected_error):
    """
    The headline claim, checked against sampling, which shares no quadrature
    with the fix.  Both directions are asserted: the value that used to be
    reported is still wrong, and the value now reported is right.
    """
    bpu = encounter(sigma_major, sigma_minor)
    result = compute_collision_probability(bpu, hbr)
    estimate, standard_error = monte_carlo_pc(bpu, hbr)

    assert result.converged is True
    assert abs(result.probability - estimate) < 5.0 * standard_error
    assert abs(result.diagnostics["raw_probability"] - estimate) > 20.0 * standard_error


@pytest.mark.parametrize("sigma_major,sigma_minor,hbr,expected_error", TABULATED)
def test_tabulated_failures_agree_with_a_ridge_aligned_grid(
        sigma_major, sigma_minor, hbr, expected_error):
    """
    A second independent check, deterministic this time: a dense
    two-dimensional Cartesian grid laid out along the ridge.  It uses two
    dimensions where the fix uses one and no analytic inner integral at all.
    """
    bpu = encounter(sigma_major, sigma_minor)
    result = compute_collision_probability(bpu, hbr)
    sigma_x, sigma_y, mu_x, mu_y = principal_axes(bpu)

    coarse = ridge_aligned_reference(sigma_x, sigma_y, mu_x, mu_y, hbr,
                                     ridge_nodes=2000, along_nodes=2000)
    fine = ridge_aligned_reference(sigma_x, sigma_y, mu_x, mu_y, hbr,
                                   ridge_nodes=4000, along_nodes=4000)
    assert abs(fine - coarse) / fine < 1e-6, "the grid reference must itself be settled"
    assert result.probability == pytest.approx(fine, rel=1e-6)


def needle_closed_form(sigma_major: float, hbr_m: float) -> float:
    """
    Pc in closed form for a needle encounter, with no quadrature anywhere.

    When sigma_minor is far smaller than HBR, the disk contains essentially the
    whole width of the ridge at every height, so integrating x over the whole
    real line is exact to within the sliver near the disk edge where the chord
    is comparable to sigma_minor -- a fraction of order sigma_minor^2 / HBR^2.
    What is left is a one-dimensional normal probability:

        Pc = P(|y| <= HBR) = 2 Phi(HBR / sigma_major) - 1

    At sigma_minor = 1 m and HBR = 5 km that sliver is 4e-8 relative, so this
    is a genuine closed-form reference and not an asymptotic estimate.
    """
    return 2.0 * float(ndtr(hbr_m / sigma_major)) - 1.0


def test_the_analytically_known_needle_case_is_exact():
    """
    One case where the answer can be written down without integrating
    anything.  The polar quadrature returned 2.576128e-04 against a true
    7.978712e-03 -- low by a factor of 31.
    """
    sigma_major, sigma_minor, hbr = 5.0e5, 1.0, 5000.0
    closed_form = needle_closed_form(sigma_major, hbr)

    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)
    assert result.probability == pytest.approx(closed_form, rel=1e-7)
    assert result.diagnostics["raw_probability"] == pytest.approx(2.576128e-04, rel=1e-4)
    assert closed_form / result.diagnostics["raw_probability"] \
        == pytest.approx(30.97, rel=0.01)


def test_a_sub_metre_uncertainty_encounter_is_repaired():
    """
    Not every failure needs extreme anisotropy: sigma_minor = 5 cm against a
    40 m disk is only 6e3:1, and the polar quadrature was 79.8 % low.
    """
    bpu = encounter(3.0e2, 5.0e-2)
    result = compute_collision_probability(bpu, 40.0)
    estimate, standard_error = monte_carlo_pc(bpu, 40.0)

    assert abs(result.probability - estimate) < 5.0 * standard_error
    assert result.diagnostics["raw_probability"] / result.probability \
        == pytest.approx(0.2024, rel=0.02)


def test_a_disk_that_engulfs_the_distribution_is_exactly_one():
    """
    The mirror failure, found by random sweep during this investigation: with
    the density a spike of width centimetres and the disk hundreds of metres
    across, the adaptive subdivision in r steps over the spike and returns
    essentially zero -- the certain collision reported as the impossible one.

    It is now analytic.  The mass outside ENGULFMENT_SIGMA standard deviations
    is bounded by exp(-k^2/2) = 5.4e-32 at k = 12, sixteen orders below the
    spacing of doubles near 1, so 1.0 is the correctly rounded answer.
    """
    result = compute_collision_probability(encounter(4.88e-2, 1.46e-2), 536.9)

    assert result.probability == 1.0
    assert result.converged is True
    assert result.method == "analytic_engulfment"
    assert result.diagnostics["result_kind"] == "analytic_exact"

    # And the quadrature that used to answer this really was wrong.
    sigma_x, sigma_y, mu_x, mu_y = principal_axes(encounter(4.88e-2, 1.46e-2))
    norm = 1.0 / (2.0 * math.pi * sigma_x * sigma_y)

    def integrand(theta, r):
        u = mu_x + r * math.cos(theta)
        v = mu_y + r * math.sin(theta)
        exponent = -(u * u / (2 * sigma_x ** 2) + v * v / (2 * sigma_y ** 2))
        return 0.0 if exponent < -500.0 else float(r * math.exp(exponent))

    value, _ = integrate.dblquad(integrand, 0.0, 536.9, 0.0, 2.0 * math.pi,
                                 epsabs=1e-8, epsrel=1e-8)
    assert value * norm < 0.5


def test_the_engulfment_threshold_is_dimensionless():
    """
    Pc is dimensionless, so expressing the same encounter in different units
    must not change which branch is taken or what it returns.
    """
    for scale in (1.0e-6, 1.0, 1.0e6):
        result = compute_collision_probability(
            encounter(4.88e-2 * scale, 1.46e-2 * scale), 536.9 * scale)
        assert result.method == "analytic_engulfment"
        assert result.probability == 1.0

    # And just inside the boundary the analytic branch must not fire.
    sigma_major = 100.0
    just_short = compute_collision_probability(
        encounter(sigma_major, 1.0), 0.9 * ENGULFMENT_SIGMA * sigma_major)
    assert just_short.method != "analytic_engulfment"


# ---------------------------------------------------------------------------
# C. Healthy encounters are untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma_major,sigma_minor,hbr", HEALTHY)
def test_healthy_encounters_still_report_the_polar_quadrature(
        sigma_major, sigma_minor, hbr):
    """
    Behaviour preservation.  Where the quadrature was right it stays the
    reported method and the reported value, to the last bit.
    """
    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)

    assert result.method == "2D_Gaussian_Polar_Quadrature_Principal_Axes"
    assert result.converged is True
    assert result.probability == result.diagnostics["raw_probability"]
    assert result.diagnostics["polar_quadrature_agrees"] is True
    assert result.diagnostics["polar_quadrature_superseded"] is False


def test_the_recorded_production_probabilities_are_unchanged():
    """
    The end-to-end values recorded under P10-04 and re-verified under P10-08,
    P10-10 and P10-11.  P10-12 must not move them.
    """
    from fastapi.testclient import TestClient
    from theseus.server.app import app

    client = TestClient(app)
    base = {"object_a_alt_km": 400.0, "object_a_inc_deg": 51.6,
            "object_a_phase_deg": 0.0, "object_b_alt_km": 400.05,
            "object_b_inc_deg": 55.0, "object_b_phase_deg": 0.02,
            "analysis_duration_hours": 2.0, "screening_threshold_km": 100.0,
            "coarse_dt_s": 30.0}

    for hbr, expected in ((15.0, 3.3437648624900262e-06),
                          (1.9, 5.365051327225372e-08),
                          (0.3, 1.3375480817909169e-09)):
        data = client.post("/api/simulate/conjunction/risk",
                           json={**base, "hard_body_radius_m": hbr}).json()
        assert data["analysis_status"] == "COMPLETE"
        assert data["collision_probability"]["probability"] == pytest.approx(
            expected, rel=1e-12)
        assert data["collision_probability"]["converged"] is True
        assert data["collision_probability"]["method"] == \
            "2D_Gaussian_Polar_Quadrature_Principal_Axes"


# ---------------------------------------------------------------------------
# D. The two orientations really are complementary
# ---------------------------------------------------------------------------

def test_the_reduction_is_orientation_dependent():
    """
    The property the certificate rests on.  At needle geometry the orientation
    that integrates the minor axis needs thousands of panels; the orientation
    that integrates the major axis is smooth and needs almost none.  If both
    orientations ever became equally conditioned, the cross-check would be
    comparing a method with itself.
    """
    sigma_x, sigma_y, hbr = 1.0, 5.0e5, 5000.0
    closed_form = needle_closed_form(sigma_y, hbr)

    def error(sigma_int, sigma_other, panels):
        value = _disk_integral_1d(sigma_int, sigma_other, 0.0, 0.0, hbr, panels)
        return abs(value - closed_form) / closed_form

    # The closed form itself is exact only to the disk-edge sliver, about 2e-8
    # relative here, so that is the floor either orientation can reach.
    floor = 5.0e-8

    # At 64 panels the major-axis orientation is already at that floor and the
    # minor-axis orientation is six orders of magnitude away from it.
    assert error(sigma_y, sigma_x, 64) < floor
    assert error(sigma_x, sigma_y, 64) > 1.0e-2

    # The minor-axis orientation needs sixteen times the panels to catch up,
    # and its error falls as the panel count rises -- the signature of a
    # feature it is only just resolving.
    assert error(sigma_x, sigma_y, 256) > 1.0e-4
    assert error(sigma_x, sigma_y, 1024) < floor

    # The major-axis orientation, by contrast, does not improve with refinement
    # because it had nothing left to resolve.
    assert error(sigma_y, sigma_x, 1024) == pytest.approx(
        error(sigma_y, sigma_x, 64), rel=0.1)


def test_the_certificate_reports_all_three_constructions():
    """The caller must be able to re-derive the verdict, not just read it."""
    sigma_x, sigma_y, hbr = 1.0, 1.0e5, 2000.0
    polar = 1.288226e-03          # the value the quadrature produces here

    certificate = certified_disk_integral(sigma_x, sigma_y, 0.0, 0.0, hbr,
                                          polar_value=polar)

    assert certificate.certified is True
    assert certificate.polar_agrees is False
    assert certificate.source.startswith("reduction_")
    assert certificate.minor_axis_settled is True
    assert certificate.major_axis_settled is True
    assert certificate.minor_axis_value == pytest.approx(
        certificate.major_axis_value, rel=1e-9)
    assert certificate.value == pytest.approx(1.595663e-02, rel=1e-5)


def test_the_certificate_prefers_the_polar_value_when_it_agrees():
    """
    So that encounters which were already right are not perturbed by a method
    change they did not need.
    """
    sigma_x, sigma_y, hbr = 5.0e1, 5.0e2, 10.0
    truth = _disk_integral_1d(sigma_y, sigma_x, 0.0, 0.0, hbr, 512)
    nudged = truth * (1.0 + 0.1 * QUADRATURE_AGREEMENT_RTOL)

    certificate = certified_disk_integral(sigma_x, sigma_y, 0.0, 0.0, hbr,
                                          polar_value=nudged)
    assert certificate.certified is True
    assert certificate.polar_agrees is True
    assert certificate.value == nudged


def test_an_unresolvable_geometry_is_not_certified():
    """
    One construction is not two.  When the disk spans a million minor-axis
    widths the minor orientation cannot resolve its edge within the panel cap,
    so nothing can be corroborated and the result must say so.
    """
    result = compute_collision_probability(encounter(1.0e2, 1.0e-3), 1000.0)

    assert result.converged is False
    assert result.diagnostics["certified"] is False
    assert result.diagnostics["reduction_minor_axis_settled"] is False
    assert result.diagnostics["reduction_minor_axis_panels"] == REDUCTION_PANEL_CAP
    assert result.method == "Exact_1D_Reduction_Uncorroborated"


def test_an_uncorroborated_result_still_beats_reporting_a_known_wrong_value():
    """
    When only one construction settles the result is not certified -- but the
    number reported is the settled one, not the quadrature's.  On this geometry
    the quadrature returns 0.0 for an encounter that is very nearly certain,
    and printing that would be the worst output this module can produce.
    """
    bpu = encounter(1.0e2, 1.0e-3)
    result = compute_collision_probability(bpu, 1000.0)

    assert result.diagnostics["raw_probability"] == 0.0
    assert result.probability > 0.99
    estimate, _ = monte_carlo_pc(bpu, 1000.0, samples=200_000)
    assert estimate == pytest.approx(result.probability, abs=1e-3)


def test_the_cancellation_safe_inner_integral_is_worth_having():
    """
    Phi((mu+h)/sigma) - Phi((mu-h)/sigma) with both arguments in the same tail
    is a difference of numbers near 1.  Evaluating it in the lower tail instead
    is exact by symmetry and buys five to seven digits on far-tail encounters,
    which is the regime where Pc actually matters.
    """
    sigma_x, sigma_y, mu_y, hbr = 0.5, 40.0, 260.0, 3.0

    def naive(panels):
        edges = np.linspace(-0.5 * math.pi, 0.5 * math.pi, panels + 1)
        half = 0.5 * (edges[1:] - edges[:-1])
        mid = 0.5 * (edges[1:] + edges[:-1])
        nodes, weights = np.polynomial.legendre.leggauss(24)
        phi = (mid[:, None] + half[:, None] * nodes[None, :]).ravel()
        wt = (half[:, None] * weights[None, :]).ravel()
        cos_phi = np.cos(phi)
        u = hbr * np.sin(phi)
        outer = (hbr * cos_phi / (math.sqrt(2.0 * math.pi) * sigma_x)) * np.exp(
            np.clip(-u * u / (2.0 * sigma_x ** 2), -700.0, None))
        chord = hbr * cos_phi
        inner = ndtr((mu_y + chord) / sigma_y) - ndtr((mu_y - chord) / sigma_y)
        return float(np.sum(wt * outer * inner))

    safe = _disk_integral_1d(sigma_x, sigma_y, 0.0, mu_y, hbr, 1024)
    unsafe = naive(1024)
    reference = _disk_integral_1d(sigma_y, sigma_x, mu_y, 0.0, hbr, 1024)

    assert abs(safe - reference) / reference < 1e-12
    assert abs(unsafe - reference) / reference > 1e-9


# ---------------------------------------------------------------------------
# E. Chan's series would not have worked
# ---------------------------------------------------------------------------

def test_chans_series_is_exact_where_it_is_valid():
    """
    Establishes that the rejection below is about the method and not about a
    mis-implementation of it.
    """
    for sigma, mu_x, hbr, in ((1.0, 0.0, 2.0), (1.0, 3.0, 1.0), (500.0, 300.0, 20.0)):
        chan = chan_equivalent_area_series(sigma, sigma, mu_x, 0.0, hbr)
        exact = _disk_integral_1d(sigma, sigma, mu_x, 0.0, hbr, 512)
        assert chan == pytest.approx(exact, rel=1e-9)


@pytest.mark.parametrize("sigma_major,sigma_minor,hbr,expected_error", TABULATED)
def test_chans_series_fails_in_the_regime_it_was_advertised_to_fix(
        sigma_major, sigma_minor, hbr, expected_error):
    """
    The docstring promised Chan's series for "isotropic and mildly anisotropic
    cases" while the module's actual problem was extreme anisotropy.  Writing
    the advertised code would have replaced an understatement with an
    overstatement of up to two orders of magnitude.
    """
    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)
    sigma_x, sigma_y, mu_x, mu_y = principal_axes(encounter(sigma_major, sigma_minor))

    chan = chan_equivalent_area_series(sigma_x, sigma_y, mu_x, mu_y, hbr)
    error = abs(chan - result.probability) / result.probability
    assert error > 0.5, "Chan's series must be demonstrably unusable here"


def test_the_docstring_no_longer_advertises_what_is_not_there():
    """A promise in a docstring is a claim; this one was false."""
    import theseus.uncertainty.collision_probability as module

    text = module.__doc__ or ""
    assert "Chan" in text, "the decision not to adopt it should be recorded"
    assert "was therefore not adopted" in text
    assert "Series: Chan's series expansion" not in text


# ---------------------------------------------------------------------------
# F. Bounds and boundary branches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma_major,sigma_minor,hbr", [
    (1.0e4, 1.0e0, 200.0), (5.0e5, 1.0e0, 5000.0), (3.0e2, 5.0e-2, 40.0),
    (5.0e2, 5.0e1, 10.0), (2.0e2, 2.0e2, 15.0), (1.0e2, 1.0e-3, 1000.0),
    (1.0e0, 1.0e0, 500.0), (1.0e3, 1.0e0, 0.3),
])
def test_probability_stays_within_the_unit_interval(sigma_major, sigma_minor, hbr):
    result = compute_collision_probability(encounter(sigma_major, sigma_minor), hbr)
    assert 0.0 <= result.probability <= 1.0
    assert math.isfinite(result.probability)


def test_zero_hard_body_radius_is_still_exact():
    result = compute_collision_probability(encounter(1.0e5, 1.0e0), 0.0)
    assert result.probability == 0.0
    assert result.converged is True
    assert result.method == "analytic_zero_hbr"


def test_far_separation_is_still_exact():
    """The engulfment branch must not have disturbed its mirror."""
    result = compute_collision_probability(encounter(1.0e2, 1.0e2, 1.0e4), 5.0)
    assert result.probability == 0.0
    assert result.converged is True
    assert result.method == "analytic_far_separation"


def test_a_tiny_hard_body_radius_on_a_needle_encounter_is_correct():
    """P10-04's regime crossed with P10-12's."""
    bpu = encounter(1.0e5, 1.0e0)
    hbr = 0.3
    result = compute_collision_probability(bpu, hbr)

    # Small-disk expansion about the density peak.  Expanding the Gaussian to
    # second order and integrating over the disk gives
    #     Pc = pi R^2 f(0) [1 - R^2/(8 sigma_x^2) - R^2/(8 sigma_y^2)] + O(R^6)
    # Here R / sigma_x = 0.3, so the second-order term is 1.1e-2 -- big enough
    # that keeping only the leading term would be the looser test, not the
    # tighter one.
    sigma_x, sigma_y, _, _ = principal_axes(bpu)
    leading = math.pi * hbr ** 2 / (2.0 * math.pi * sigma_x * sigma_y)
    corrected = leading * (1.0 - hbr ** 2 / (8.0 * sigma_x ** 2)
                           - hbr ** 2 / (8.0 * sigma_y ** 2))

    assert result.probability == pytest.approx(corrected, rel=2e-4)
    assert result.probability / leading == pytest.approx(0.98889, rel=1e-3)
    assert 0.0 <= result.probability <= 1.0


# ---------------------------------------------------------------------------
# G. Earlier findings are unaffected
# ---------------------------------------------------------------------------

def test_p10_08_convergence_reporting_survives():
    """
    P10-08's contract: ``converged`` is measured, and a result that cannot be
    corroborated says so.  Both halves must still be exercisable.
    """
    certified = compute_collision_probability(encounter(1.0e5, 1.0e0), 2000.0)
    assert certified.converged is True
    assert certified.diagnostics["certified"] is True

    uncertified = compute_collision_probability(encounter(1.0e2, 1.0e-3), 1000.0)
    assert uncertified.converged is False
    assert uncertified.diagnostics["certified"] is False


def test_p10_10_dimensionless_branch_criteria_survive():
    """
    Pc is dimensionless: the same encounter expressed in different units must
    give the same probability and take the same branch.
    """
    values = []
    methods = []
    for scale in (1.0e-3, 1.0, 1.0e3):
        result = compute_collision_probability(
            encounter(1.0e5 * scale, 1.0 * scale), 2000.0 * scale)
        values.append(result.probability)
        methods.append(result.method)
    assert len(set(methods)) == 1
    assert values[0] == pytest.approx(values[1], rel=1e-9)
    assert values[1] == pytest.approx(values[2], rel=1e-9)


def test_the_trace_reports_the_criterion_it_applies():
    """P10-09's contract, over the new criterion."""
    from theseus.uncertainty.covariance import StateCovariance
    from theseus.uncertainty.hard_body import compute_hard_body_radius
    from theseus.uncertainty.relative import compute_relative_covariance
    from theseus.uncertainty.results import build_phase10_calculation_trace
    from theseus.uncertainty.risk import PROFILE_STANDARD, classify_risk

    bpu = encounter(1.0e5, 1.0e0)
    pc_res = compute_collision_probability(bpu, 2000.0)
    cov = StateCovariance.from_diagonal([100.0] * 3, [0.1] * 3, name="test")
    steps = build_phase10_calculation_trace(
        initial_cov_a=cov, initial_cov_b=cov, cov_a_tca=cov, cov_b_tca=cov,
        rel_cov=compute_relative_covariance(cov, cov), tca_s=100.0,
        r_rel_tca=np.array([bpu.b_dot_t, bpu.b_dot_r, 0.0]),
        v_rel_tca=np.array([0.0, 0.0, 1.0e4]), b_plane_unc=bpu,
        hbr_res=compute_hard_body_radius(custom_hbr_m=2000.0), pc_res=pc_res,
        risk=classify_risk(pc_res.probability, PROFILE_STANDARD))

    step = next(s for s in steps if s["stepIndex"] == 13)
    assert step["status"] == "completed"
    assert step["substitutions"]["certified"] is True
    assert step["substitutions"]["polar_quadrature_superseded"] is True
    assert step["substitutions"]["method"] == "Exact_1D_Reduction_Dual_Orientation"
    assert "two of three" in step["substitutions"]["convergence_criterion"]
    # The discarded value stays visible in the trace.
    assert step["substitutions"]["polar_quadrature_probability"] \
        != pytest.approx(pc_res.probability, rel=1e-3)
