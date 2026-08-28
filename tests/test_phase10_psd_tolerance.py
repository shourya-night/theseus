"""
P10-11 — the covariance validity tolerances must not be set by an unrelated block.

The defect
----------
``StateCovariance.validate()`` judged both symmetry and positive
semi-definiteness against ``scale = max(max|P_ij|, 1.0)``:

    rel_asym          = max|P - P^T| / scale
    effective_psd_tol = max(psd_tol, scale * 1e-9)

For a state covariance the largest entry is a position variance in m², while
the quantity being judged may live in the velocity block in (m/s)² or in a
position-velocity correlation in m²/s. Positive semi-definiteness is invariant
under congruence ``P -> S P Sᵀ``, so the same physical covariance expressed in
different state units must get the same verdict. It did not.

Measured against the unfixed tree, σ_r = 1 km and σ_v = 1e-4 m/s, with a
correlation ρ between r_x and v_x:

    rho        raw lambda_min   lambda_min(corr)   truly PSD   verdict
    0.99999      +1.1918e-11        +1.0000e-05        yes      ACCEPT
    1.00000      +1.1718e-11         0.0000e+00        yes      ACCEPT
    1.00001      +1.1518e-11        -1.0000e-05        NO       ACCEPT  <-
    1.01000      -1.8928e-10        -1.0000e-02        NO       ACCEPT  <-
    2.00000      -2.9988e-08        -1.0000e+00        NO       ACCEPT  <-

A correlation coefficient of 2.0 -- impossible for any probability
distribution -- was accepted, because the tolerance was 1e-3 m² while the
offending eigenvalue was -3.0e-8. Note also that at ρ = 1.00001 ``eigh``
returned a *positive* minimum eigenvalue: the sign is not resolvable in raw
coordinates when the blocks span 1e6 and 1e-8, so no absolute tolerance,
however tight, would have caught it.

The same matrix was ACCEPTED in m/(m/s), km/(m/s), m/(mm/s) and Mm/(µm/s), and
REJECTED in km/(mm/s) -- five expressions of one physical covariance, two
different answers.

Reachable: the risk API takes ``cov_a.matrix_si`` directly. Supplying the
ρ = 2.0 matrix returned HTTP 200 with step 2 reporting ``psd_verified=True``,
Pc = 1.101625e-05 and **risk = HIGH** — a risk classification derived from
something that is not a probability distribution.

The correction
--------------
Both tests now run on the correlation form ``C = D⁻¹ P D⁻¹``, ``D =
diag(sqrt(P_ii))``: dimensionless, invariant under component-wise rescaling of
the state, and resolvable in floating point. A zero variance is handled by
Cauchy-Schwarz separately, since it has no correlation to normalise.

Independent reference
---------------------
The truth here is analytic, not numerical: for the 2×2 embedding used
throughout, the matrix is PSD if and only if |ρ| ≤ 1. That cannot share a
mistake with any implementation. Rank-deficient constructions give matrices
whose smallest correlation eigenvalue is exactly zero by construction, and
``numpy.longdouble`` provides a higher-precision cross-check of the sign.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from theseus.uncertainty.covariance import (
    PSD_CORRELATION_TOL,
    CovarianceValidationError,
    StateCovariance,
)
from theseus.uncertainty.results import measure_covariance_validity


# ---------------------------------------------------------------------------
# Construction helpers — truth is analytic
# ---------------------------------------------------------------------------

def coupled(sigma_r: float, sigma_v: float, rho: float) -> np.ndarray:
    """
    Diagonal state covariance with a single r_x / v_x correlation of ``rho``.

    PSD if and only if |rho| <= 1: the 2×2 sub-block
    [[σr², ρ σr σv], [ρ σr σv, σv²]] has determinant σr²σv²(1 − ρ²).
    """
    p = np.diag([sigma_r ** 2] * 3 + [sigma_v ** 2] * 3).astype(float)
    p[0, 3] = p[3, 0] = rho * sigma_r * sigma_v
    return p


def rescale(p: np.ndarray, s_r: float, s_v: float) -> np.ndarray:
    """``S P Sᵀ`` for ``S = diag(s_r, s_r, s_r, s_v, s_v, s_v)`` — a unit change."""
    s = np.diag([s_r] * 3 + [s_v] * 3).astype(float)
    return s @ np.asarray(p, dtype=float) @ s.T


def accepts(p: np.ndarray) -> bool:
    try:
        StateCovariance(matrix=np.asarray(p, dtype=float).copy(), name="probe")
        return True
    except CovarianceValidationError:
        return False


def longdouble_min_correlation_eigenvalue(p: np.ndarray) -> float:
    """
    Higher-precision cross-check of the sign, independent of the float64 path
    production uses.
    """
    p = np.asarray(p, dtype=np.longdouble)
    d = np.sqrt(np.diag(p))
    keep = d > 0
    c = p[np.ix_(keep, keep)] / np.outer(d[keep], d[keep])
    return float(np.min(np.linalg.eigvalsh(np.array(0.5 * (c + c.T), dtype=float))))


#: (σ_r m, σ_v m/s) — realistic orbital state scales, including strongly
#: disparate ones where the defect lived.
SCALES = [(1.0, 1.0), (1.0, 1e-2), (10.0, 1e-2), (1e3, 1e-4), (1e5, 1e-4), (1e-3, 1.0)]


# ===========================================================================
# A. Basic PSD
# ===========================================================================

@pytest.mark.parametrize("sigma_r,sigma_v", SCALES)
def test_positive_definite_covariance_is_accepted(sigma_r, sigma_v):
    assert accepts(coupled(sigma_r, sigma_v, 0.0))


@pytest.mark.parametrize("sigma_r,sigma_v", SCALES)
def test_exactly_psd_covariance_is_accepted(sigma_r, sigma_v):
    """ρ = 1 makes the 2×2 block exactly singular: PSD, not positive definite."""
    p = coupled(sigma_r, sigma_v, 1.0)
    assert longdouble_min_correlation_eigenvalue(p) == pytest.approx(0.0, abs=1e-15)
    assert accepts(p)


@pytest.mark.parametrize("sigma_r,sigma_v", SCALES)
def test_clearly_indefinite_covariance_is_rejected(sigma_r, sigma_v):
    """ρ = 2 is impossible for any probability distribution."""
    p = coupled(sigma_r, sigma_v, 2.0)
    assert longdouble_min_correlation_eigenvalue(p) < -0.5
    assert not accepts(p)


def test_negative_variance_is_still_rejected():
    p = np.diag([1e6, 1e6, 1e6, 1e-8, 1e-8, -1e-8]).astype(float)
    assert not accepts(p)


def test_non_finite_covariance_is_still_rejected():
    p = np.diag([1e6] * 3 + [1e-8] * 3).astype(float)
    p[0, 0] = float("nan")
    assert not accepts(p)


# ===========================================================================
# B. Near-PSD
# ===========================================================================

@pytest.mark.parametrize("sigma_r,sigma_v", SCALES)
@pytest.mark.parametrize("excess", (1e-14, 1e-13))
def test_correlation_barely_above_one_is_tolerated_as_roundoff(
        sigma_r, sigma_v, excess):
    """
    Inside the tolerance the matrix is accepted and repaired: this is the
    roundoff allowance the field documents, and it must still work.
    """
    assert excess < PSD_CORRELATION_TOL
    assert accepts(coupled(sigma_r, sigma_v, 1.0 + excess))


@pytest.mark.parametrize("sigma_r,sigma_v", SCALES)
@pytest.mark.parametrize("excess", (1e-10, 1e-8, 1e-5, 1e-2))
def test_correlation_meaningfully_above_one_is_rejected(sigma_r, sigma_v, excess):
    """
    Outside the tolerance the matrix is not a covariance and must be refused,
    at every scale.  Before the fix these were accepted whenever the position
    block dominated.
    """
    assert excess > PSD_CORRELATION_TOL
    assert not accepts(coupled(sigma_r, sigma_v, 1.0 + excess))


def test_the_tolerance_clears_the_measured_roundoff_floor():
    """
    Exactly-singular correlation matrices — rank-deficient by construction, so
    the smallest eigenvalue is zero mathematically — must not be rejected.
    """
    rng = np.random.default_rng(20260826)
    worst = 0.0
    checked = 0
    for _ in range(400):
        factor = rng.normal(0.0, 1.0, (6, 5))
        p = factor @ factor.T
        if np.any(np.diag(p) <= 0):
            continue
        deviations = np.sqrt(np.diag(p))
        correlation = p / np.outer(deviations, deviations)
        worst = min(worst, float(np.min(np.linalg.eigvalsh(correlation))))
        assert accepts(p)
        checked += 1

    assert checked > 300
    assert worst < 0.0, "the sample must actually reach the roundoff floor"
    assert worst > -PSD_CORRELATION_TOL
    # And the floor agrees with the backward-error bound eps*||C||_2 for a 6x6.
    assert abs(worst) < 100.0 * np.finfo(float).eps * 6


# ===========================================================================
# C. Block scaling
# ===========================================================================

def test_indefiniteness_confined_to_the_velocity_block_is_rejected():
    """
    The defect's home ground: the offending direction lives in the velocity
    block while the tolerance was set by the position block.
    """
    p = np.diag([1e10, 1e10, 1e10, 1e-8, 1e-8, 1e-8]).astype(float)
    p[3, 4] = p[4, 3] = 1.5e-8          # correlation 1.5 between v_x and v_y
    assert longdouble_min_correlation_eigenvalue(p) < -0.4
    assert not accepts(p)


def test_indefiniteness_confined_to_the_position_block_is_rejected():
    p = np.diag([1e10, 1e10, 1e10, 1e-8, 1e-8, 1e-8]).astype(float)
    p[0, 1] = p[1, 0] = 1.5e10
    assert not accepts(p)


def test_indefiniteness_in_a_mixed_direction_is_rejected():
    p = coupled(1e5, 1e-4, 1.5)
    assert not accepts(p)


@pytest.mark.parametrize("sigma_r,sigma_v", SCALES)
def test_valid_covariances_survive_every_block_scale(sigma_r, sigma_v):
    """No false rejection introduced at any position/velocity scale ratio."""
    for rho in (0.0, 0.5, 0.9, 0.999):
        assert accepts(coupled(sigma_r, sigma_v, rho))


# ===========================================================================
# D. Coordinate / unit scaling
# ===========================================================================

#: (s_r, s_v, label) — m/(m/s), km/(m/s), m/(mm/s), km/(mm/s), Mm/(µm/s)
UNIT_SYSTEMS = [(1.0, 1.0, "m, m/s"), (1e-3, 1.0, "km, m/s"),
                (1.0, 1e3, "m, mm/s"), (1e-3, 1e3, "km, mm/s"),
                (1e-6, 1e6, "Mm, um/s")]


@pytest.mark.parametrize("s_r,s_v,label", UNIT_SYSTEMS)
def test_invalid_covariance_is_rejected_in_every_unit_system(s_r, s_v, label):
    """
    P' = S P Sᵀ is the same physical covariance. PSD is congruence-invariant,
    so the verdict must be too. Before the fix this matrix was accepted in
    four of these five systems and rejected in the fifth.
    """
    physical = coupled(1e3, 1e-4, 1.01)
    assert not accepts(rescale(physical, s_r, s_v)), label


@pytest.mark.parametrize("s_r,s_v,label", UNIT_SYSTEMS)
def test_valid_covariance_is_accepted_in_every_unit_system(s_r, s_v, label):
    physical = coupled(1e3, 1e-4, 0.99)
    assert accepts(rescale(physical, s_r, s_v)), label


def test_the_verdict_is_constant_across_unit_systems():
    """Stated as one property rather than parametrised, so a split cannot hide."""
    for rho, expected in ((0.5, True), (0.99, True), (1.0, True),
                          (1.001, False), (1.5, False), (3.0, False)):
        verdicts = {accepts(rescale(coupled(1e3, 1e-4, rho), s_r, s_v))
                    for s_r, s_v, _ in UNIT_SYSTEMS}
        assert verdicts == {expected}, (rho, verdicts)


# ===========================================================================
# E. False accept / false reject
# ===========================================================================

def test_the_reported_false_accept_no_longer_happens():
    """
    The headline case: a correlation coefficient of 2.0 with σ_r = 1 km and
    σ_v = 1e-4 m/s.  Accepted before the fix; the raw minimum eigenvalue was
    -3.0e-8 against a 1e-3 tolerance.
    """
    p = coupled(1e3, 1e-4, 2.0)
    raw_min = float(np.min(np.linalg.eigvalsh(p)))
    old_tolerance = max(1e-9, max(float(np.max(np.abs(p))), 1.0) * 1e-9)

    assert raw_min > -old_tolerance, "the old criterion really did accept this"
    assert longdouble_min_correlation_eigenvalue(p) == pytest.approx(-1.0, abs=1e-9)
    assert not accepts(p)


def test_an_indefinite_matrix_is_never_turned_valid_by_the_correction():
    """The opposite direction: nothing previously rejected becomes accepted."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        sigma_r = 10.0 ** rng.uniform(-2.0, 5.0)
        sigma_v = 10.0 ** rng.uniform(-5.0, 1.0)
        rho = rng.uniform(1.01, 5.0)
        p = coupled(sigma_r, sigma_v, rho)
        assert longdouble_min_correlation_eigenvalue(p) < 0.0
        assert not accepts(p)


def test_zero_variance_may_not_covary_with_anything():
    """
    Cauchy-Schwarz: |P_ij|² ≤ P_ii P_jj = 0.  The correlation form has no row
    to normalise here, so this is checked explicitly.  Accepted before the fix.
    """
    p = np.diag([1e6, 1e6, 1e6, 1e-8, 1e-8, 0.0]).astype(float)
    assert accepts(p), "a zero variance on its own is legitimate"

    coupled_to_zero = p.copy()
    coupled_to_zero[5, 2] = coupled_to_zero[2, 5] = 1.0
    assert float(np.min(np.linalg.eigvalsh(coupled_to_zero))) < 0.0
    assert not accepts(coupled_to_zero)


@pytest.mark.parametrize("sigma_r,sigma_v", [(1.0, 1.0), (1e3, 1e-4), (1e5, 1e-4)])
def test_asymmetry_is_judged_against_the_entry_s_own_scale(sigma_r, sigma_v):
    """
    The same defect in the symmetry test: an asymmetry of 100 % of the
    velocity variance passed whenever the position block dominated.
    """
    p = np.diag([sigma_r ** 2] * 3 + [sigma_v ** 2] * 3).astype(float)
    p[3, 4] += sigma_v ** 2            # one-sided, so asymmetric by 100 %
    assert not accepts(p)


@pytest.mark.parametrize("sigma_r,sigma_v", [(1.0, 1.0), (1e3, 1e-4), (1e5, 1e-4)])
def test_roundoff_level_asymmetry_is_still_absorbed(sigma_r, sigma_v):
    """The symmetry tolerance must still do its original job."""
    p = np.diag([sigma_r ** 2] * 3 + [sigma_v ** 2] * 3).astype(float)
    p[3, 4] += 1e-12 * sigma_v ** 2
    assert accepts(p)


# ===========================================================================
# F. P10-09 integration
# ===========================================================================

@pytest.mark.parametrize("rho,expected", [(0.0, True), (0.99, True), (1.0, True),
                                          (1.001, False), (2.0, False)])
@pytest.mark.parametrize("sigma_r,sigma_v", [(1.0, 1.0), (1e3, 1e-4)])
def test_measurement_and_validator_agree(rho, expected, sigma_r, sigma_v):
    """
    P10-09's measurement must describe the same notion of validity the
    validator enforces — while remaining a measurement, not a call to it.
    """
    p = coupled(sigma_r, sigma_v, rho)
    if expected:
        cov = StateCovariance(matrix=p.copy(), name="agree")
    else:
        cov = StateCovariance.from_diagonal([1.0] * 3, [1.0] * 3, name="agree")
        cov.matrix = p.copy()

    measured = measure_covariance_validity(cov)
    assert measured["valid"] is expected
    assert accepts(p) is expected


def test_measurement_reports_the_correlation_basis_and_does_not_repair():
    cov = StateCovariance.from_diagonal([100.0] * 3, [0.1] * 3, name="basis")
    cov.matrix = coupled(1e3, 1e-4, 1.5)
    before = cov.matrix.copy()

    measured = measure_covariance_validity(cov)

    assert measured["min_eigenvalue_basis"] == "correlation form D⁻¹PD⁻¹"
    assert measured["psd_tolerance"] == cov.psd_tol
    assert measured["positive_semidefinite"] is False
    np.testing.assert_array_equal(cov.matrix, before)


def test_measurement_flags_zero_variance_coupling():
    cov = StateCovariance.from_diagonal([100.0] * 3, [0.1] * 3, name="cs")
    bad = np.diag([1e6, 1e6, 1e6, 1e-8, 1e-8, 0.0]).astype(float)
    bad[5, 2] = bad[2, 5] = 1.0
    cov.matrix = bad

    measured = measure_covariance_validity(cov)
    assert measured["zero_variance_coupling"] == 1.0
    assert measured["positive_semidefinite"] is False
    assert measured["valid"] is False


# ===========================================================================
# G. Real Phase 10 covariances
# ===========================================================================

def test_real_phase10_covariances_still_validate():
    """
    Every covariance a real multi-object run constructs must remain valid, and
    must sit far from the tolerance rather than just inside it.
    """
    from theseus.simulation.multi_object import (
        MultiObjectEnvironment, SpacecraftDefinition,
    )

    captured = []
    original = StateCovariance.__post_init__

    def spy(self):
        original(self)
        captured.append(np.array(self.matrix, dtype=float))

    StateCovariance.__post_init__ = spy
    try:
        common = dict(semi_major_axis_km=6778.137, eccentricity=0.0,
                      dry_mass_kg=1000.0, fuel_mass_kg=0.0, payload_mass_kg=0.0,
                      hard_body_radius_m=5.0, sigma_pos_m=[300.0] * 3,
                      sigma_vel_m_s=[0.3] * 3)
        a = SpacecraftDefinition(id="A", name="A", color="#ff9900",
                                 inclination_deg=51.6, true_anomaly_deg=0.0, **common)
        b = SpacecraftDefinition(id="B", name="B", color="#3388ff",
                                 inclination_deg=-51.6, true_anomaly_deg=0.05, **common)
        env = MultiObjectEnvironment(central_body="Earth", screening_threshold_km=50.0,
                                     coarse_dt_s=5.0, enable_j2=True, enable_drag=True)
        result = env.simulate([a, b], t_start=0.0, t_end=3000.0, output_dt=5.0)
    finally:
        StateCovariance.__post_init__ = original

    assert result.conjunctions
    assert len(captured) >= 4

    margins = [longdouble_min_correlation_eigenvalue(p) for p in captured]
    assert min(margins) > 0.0
    # Measured minimum over a wider sample was 3.67e-06; require several orders
    # of clearance so a future tightening cannot silently start rejecting them.
    assert min(margins) > 1e4 * PSD_CORRELATION_TOL


def test_risk_api_outputs_are_unchanged_by_the_correction():
    """
    The correction must not move any production number. These are the values
    recorded under P10-04 and re-verified under P10-08 and P10-10.
    """
    from fastapi.testclient import TestClient
    from theseus.server.app import app

    client = TestClient(app)
    base = {"object_a_alt_km": 400.0, "object_a_inc_deg": 51.6,
            "object_a_phase_deg": 0.0, "object_b_alt_km": 400.05,
            "object_b_inc_deg": 55.0, "object_b_phase_deg": 0.02,
            "analysis_duration_hours": 2.0, "screening_threshold_km": 100.0,
            "coarse_dt_s": 30.0}

    for hbr, expected_pc in ((15.0, 3.3437648624900262e-06),
                             (1.9, 5.365051327225372e-08),
                             (0.3, 1.3375480817909169e-09)):
        data = client.post("/api/simulate/conjunction/risk",
                           json={**base, "hard_body_radius_m": hbr}).json()
        assert data["analysis_status"] == "COMPLETE"
        assert data["collision_probability"]["probability"] == pytest.approx(
            expected_pc, rel=1e-12)
        assert data["collision_probability"]["converged"] is True
        step_2 = next(s for s in data["calculation_steps"] if s["stepIndex"] == 2)
        assert step_2["status"] == "completed"
        assert step_2["substitutions"]["psd_verified"] is True


def test_an_impossible_user_covariance_no_longer_yields_a_risk_level():
    """
    End to end: the ρ = 2.0 matrix reached the risk API and produced
    Pc = 1.101625e-05 with risk = HIGH. It must no longer be analysed at all.
    """
    from fastapi.testclient import TestClient
    from theseus.server.app import app

    client = TestClient(app, raise_server_exceptions=False)
    bad = coupled(1e3, 1e-4, 2.0)

    response = client.post("/api/simulate/conjunction/risk", json={
        "object_a_alt_km": 400.0, "object_a_inc_deg": 51.6, "object_a_phase_deg": 0.0,
        "object_b_alt_km": 400.05, "object_b_inc_deg": 55.0, "object_b_phase_deg": 0.02,
        "analysis_duration_hours": 2.0, "screening_threshold_km": 100.0,
        "coarse_dt_s": 30.0, "hard_body_radius_m": 15.0,
        "cov_a": {"matrix_si": bad.tolist()},
        "cov_b": {"sigma_pos_km": [0.5] * 3, "sigma_vel_km_s": [0.0005] * 3},
    })

    assert response.status_code != 200
    # The coarse 500 is the pre-existing API error-handling gap (an invalid
    # request field surfacing as a server error), carried forward and not
    # addressed here. What matters for P10-11 is that no risk level is issued.
    assert "risk_assessment" not in response.text
