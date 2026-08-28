"""
Unit tests for THESEUS Phase 10: Uncertainty, Covariance, STM, and Pc.
"""

import math
import numpy as np
import pytest

from theseus.uncertainty.covariance import (
    StateCovariance,
    CovarianceValidationError,
)
from theseus.uncertainty.state_transition import (
    gravity_jacobian,
    j2_jacobian,
    numerical_jacobian,
    propagate_stm,
    build_dynamics_jacobian,
)
from theseus.uncertainty.propagation import (
    ProcessNoiseModel,
    CovariancePropagator,
    propagate_covariance,
)
from theseus.uncertainty.relative import (
    compute_relative_covariance,
)
from theseus.uncertainty.b_plane import (
    project_covariance_to_b_plane,
)
from theseus.uncertainty.hard_body import (
    CollisionGeometry,
    compute_hard_body_radius,
    PRESET_ISS,
    PRESET_CUBESAT,
)
from theseus.uncertainty.collision_probability import (
    compute_collision_probability,
    monte_carlo_pc_validation,
)
from theseus.uncertainty.risk import (
    RiskLevel,
    RiskThresholds,
    classify_risk,
    PROFILE_STANDARD,
    PROFILE_CONSERVATIVE,
)
from theseus.uncertainty.results import (
    run_uncertainty_conjunction_analysis,
)


# ===========================================================================
# 1. StateCovariance & Validation Tests
# ===========================================================================

def test_covariance_diagonal_construction():
    """Verify diagonal covariance creation, units, and sub-blocks."""
    sigma_pos = [100.0, 200.0, 300.0]  # m
    sigma_vel = [0.1, 0.2, 0.3]        # m/s
    cov = StateCovariance.from_diagonal(sigma_pos, sigma_vel, epoch_s=100.0, frame="ICRF")

    assert cov.matrix.shape == (6, 6)
    np.testing.assert_allclose(cov.sigma_position, sigma_pos)
    np.testing.assert_allclose(cov.sigma_velocity, sigma_vel)
    assert cov.sigma_pos_3d == pytest.approx(math.sqrt(100**2 + 200**2 + 300**2))
    assert cov.sigma_vel_3d == pytest.approx(math.sqrt(0.1**2 + 0.2**2 + 0.3**2))
    assert cov.epoch_s == 100.0
    assert cov.frame == "ICRF"

    # Sub-blocks
    prr = cov.position_covariance
    pvv = cov.velocity_covariance
    prv = cov.pos_vel_covariance
    assert prr.shape == (3, 3)
    assert pvv.shape == (3, 3)
    assert prv.shape == (3, 3)
    np.testing.assert_allclose(np.diag(prr), [10000.0, 40000.0, 90000.0])
    np.testing.assert_allclose(np.diag(pvv), [0.01, 0.04, 0.09])
    np.testing.assert_allclose(prv, 0.0)


def test_covariance_validation_rejections():
    """Verify that non-finite, wrong shape, asymmetric, and non-PSD matrices are rejected."""
    # 1. Wrong shape
    with pytest.raises(CovarianceValidationError, match="Expected 6×6"):
        StateCovariance(matrix=np.eye(5))

    # 2. NaN
    bad_nan = np.eye(6)
    bad_nan[0, 0] = np.nan
    with pytest.raises(CovarianceValidationError, match="non-finite"):
        StateCovariance(matrix=bad_nan)

    # 3. Inf
    bad_inf = np.eye(6)
    bad_inf[1, 1] = np.inf
    with pytest.raises(CovarianceValidationError, match="non-finite"):
        StateCovariance(matrix=bad_inf)

    # 4. Negative diagonal variance
    bad_neg_diag = np.eye(6)
    bad_neg_diag[2, 2] = -10.0
    with pytest.raises(CovarianceValidationError, match="Negative variance"):
        StateCovariance(matrix=bad_neg_diag)

    # 5. Gross asymmetry
    bad_asym = np.eye(6)
    bad_asym[0, 1] = 50.0
    bad_asym[1, 0] = 1.0
    with pytest.raises(CovarianceValidationError, match="Gross asymmetry"):
        StateCovariance(matrix=bad_asym)

    # 6. Negative eigenvalue (not positive semi-definite)
    bad_non_psd = np.array([
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Det = 1 - 4 = -3 -> negative eigenvalue
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    with pytest.raises(CovarianceValidationError, match="Negative eigenvalue"):
        StateCovariance(matrix=bad_non_psd)


def test_covariance_serialization():
    """Verify serialization to and from JSON dictionary."""
    cov = StateCovariance.from_isotropic(sigma_pos_m=500.0, sigma_vel_m_s=0.5, epoch_s=42.0)
    d = cov.to_dict()
    assert "matrix_si" in d
    assert d["sigma_pos_3d_m"] == pytest.approx(500.0 * math.sqrt(3))
    assert d["sigma_pos_3d_km"] == pytest.approx(0.5 * math.sqrt(3))

    rebuilt = StateCovariance.from_dict(d)
    np.testing.assert_allclose(rebuilt.matrix, cov.matrix)
    assert rebuilt.epoch_s == 42.0


# ===========================================================================
# 2. State Transition Matrix Tests
# ===========================================================================

def test_stm_initial_identity():
    """Verify Φ(t0, t0) = I₆."""
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])
    mu = 3.986004418e14

    def simple_acc(t, r, v):
        return -mu / (np.linalg.norm(r)**3) * r

    res = propagate_stm(simple_acc, r0, v0, (100.0, 100.0), mu=mu)
    np.testing.assert_allclose(res.stm, np.eye(6))
    assert res.t0 == 100.0
    assert res.tf == 100.0


def test_analytic_vs_numerical_gravity_jacobian():
    """Verify that analytic two-body gravity Jacobian matches numerical finite differences."""
    mu = 3.986004418e14
    r = np.array([6800e3, 1200e3, -3400e3])
    v = np.array([-1.2e3, 6.8e3, 2.4e3])

    def acc_fn(t, pos, vel):
        r_mag = np.linalg.norm(pos)
        return -mu / (r_mag**3) * pos

    da_dr_analytic = gravity_jacobian(r, mu)
    da_dr_num, da_dv_num = numerical_jacobian(acc_fn, 0.0, r, v, dr=0.1, dv=0.01)

    np.testing.assert_allclose(da_dr_analytic, da_dr_num, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(da_dv_num, 0.0, atol=1e-10)


def test_analytic_vs_numerical_j2_jacobian():
    """Verify that analytic J₂ Jacobian matches finite differences."""
    mu = 3.986004418e14
    j2 = 1.08262668e-3
    radius = 6378137.0
    r = np.array([5500e3, -3000e3, 4000e3])
    v = np.array([1.5e3, 5.0e3, -4.0e3])

    def j2_acc_fn(t, pos, vel):
        r_mag = np.linalg.norm(pos)
        k = 1.5 * mu * j2 * (radius**2) / (r_mag**5)
        z2_over_r2 = (pos[2]**2) / (r_mag**2)
        ax = k * pos[0] * (5.0 * z2_over_r2 - 1.0)
        ay = k * pos[1] * (5.0 * z2_over_r2 - 1.0)
        az = k * pos[2] * (5.0 * z2_over_r2 - 3.0)
        return np.array([ax, ay, az])

    da_dr_analytic = j2_jacobian(r, mu, j2, radius)
    da_dr_num, _ = numerical_jacobian(j2_acc_fn, 0.0, r, v, dr=0.5)

    np.testing.assert_allclose(da_dr_analytic, da_dr_num, rtol=1e-4, atol=1e-8)


# ===========================================================================
# 3. Covariance Propagation Tests
# ===========================================================================

def test_covariance_propagation_static():
    """Static state with identity STM must preserve covariance."""
    p0 = StateCovariance.from_diagonal([10.0, 20.0, 30.0], [0.1, 0.2, 0.3], epoch_s=0.0)
    phi = np.eye(6)
    p_prop = propagate_covariance(p0, phi, epoch_tf_s=50.0)

    np.testing.assert_allclose(p_prop.matrix, p0.matrix)
    assert p_prop.epoch_s == 50.0


def test_covariance_propagation_orbit():
    """Propagate covariance in LEO orbit and verify variance growth / conservation properties."""
    mu = 3.986004418e14
    r0 = np.array([7000e3, 0.0, 0.0])
    v0_mag = math.sqrt(mu / 7000e3)
    v0 = np.array([0.0, v0_mag, 0.0])

    def acc_fn(t, r, v):
        return -mu / (np.linalg.norm(r)**3) * r

    cov0 = StateCovariance.from_diagonal([10.0, 10.0, 10.0], [0.01, 0.01, 0.01])
    prop = CovariancePropagator(acc_fn=acc_fn, mu=mu, dt=30.0)

    res = prop.propagate(r0, v0, cov0, (0.0, 600.0))
    p_final = res.propagated_covariance

    # Covariance must remain symmetric and positive semi-definite
    assert np.all(np.linalg.eigvalsh(p_final.matrix) >= 0.0)
    # Uncertainty in along-track direction naturally grows during orbit propagation
    assert p_final.sigma_pos_3d >= cov0.sigma_pos_3d


def test_process_noise():
    """Verify that process noise increases propagated covariance."""
    p0 = StateCovariance.from_isotropic(100.0, 0.1)
    phi = np.eye(6)
    pn = ProcessNoiseModel(enabled=True, q_matrix=np.diag([100.0]*3 + [0.01]*3))

    p_prop = propagate_covariance(p0, phi, epoch_tf_s=10.0, process_noise=pn)
    np.testing.assert_allclose(np.diag(p_prop.position_covariance), 100.0**2 + 100.0)


# ===========================================================================
# 4. Relative Covariance Tests
# ===========================================================================

def test_relative_covariance_independent():
    """For independent objects: P_rel = P_a + P_b."""
    cov_a = StateCovariance.from_diagonal([10.0, 20.0, 30.0], [0.1, 0.2, 0.3], epoch_s=10.0)
    cov_b = StateCovariance.from_diagonal([15.0, 25.0, 35.0], [0.15, 0.25, 0.35], epoch_s=10.0)

    rel_res = compute_relative_covariance(cov_a, cov_b)
    expected = cov_a.matrix + cov_b.matrix
    np.testing.assert_allclose(rel_res.relative_covariance.matrix, expected)
    assert rel_res.independent is True
    assert len(rel_res.assumptions) > 0


def test_relative_covariance_mismatch_errors():
    """Reject frame or epoch mismatch."""
    cov_a = StateCovariance.from_isotropic(10.0, 0.1, epoch_s=0.0, frame="ICRF")
    cov_b_wrong_frame = StateCovariance.from_isotropic(10.0, 0.1, epoch_s=0.0, frame="ECEF")
    cov_b_wrong_epoch = StateCovariance.from_isotropic(10.0, 0.1, epoch_s=100.0, frame="ICRF")

    with pytest.raises(CovarianceValidationError, match="FRAME MISMATCH"):
        compute_relative_covariance(cov_a, cov_b_wrong_frame)

    with pytest.raises(CovarianceValidationError, match="EPOCH MISMATCH"):
        compute_relative_covariance(cov_a, cov_b_wrong_epoch)


# ===========================================================================
# 5. B-Plane Uncertainty & Ellipse Tests
# ===========================================================================

def test_b_plane_uncertainty_projection():
    """Verify B-plane basis orthonormality and 2D covariance projection."""
    r_rel = np.array([100.0, 50.0, 0.0])  # miss vector
    v_rel = np.array([0.0, 0.0, 7500.0])  # high relative speed along Z

    p_rel_pos = np.diag([400.0**2, 100.0**2, 900.0**2])  # 3x3 position covariance
    b_unc = project_covariance_to_b_plane(p_rel_pos, r_rel, v_rel)

    assert b_unc.b_plane_covariance.shape == (2, 2)
    assert b_unc.sigma_major >= b_unc.sigma_minor
    assert -1.0 <= b_unc.correlation <= 1.0
    assert np.all(b_unc.eigenvalues >= 0.0)


# ===========================================================================
# 6. Hard-Body Radius Tests
# ===========================================================================

def test_hard_body_radius_presets():
    """Verify HBR calculations and presets."""
    iss = PRESET_ISS
    cube = PRESET_CUBESAT
    res = compute_hard_body_radius(iss, cube)

    expected_hbr = 54.0 + 0.3
    assert res.combined_hbr_m == pytest.approx(expected_hbr)
    assert res.object_a.name == "International Space Station"

    # Custom HBR
    custom_res = compute_hard_body_radius(custom_hbr_m=25.0)
    assert custom_res.combined_hbr_m == 25.0


def test_hard_body_radius_validation():
    """Negative radius must be rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        CollisionGeometry(physical_radius_m=-1.0)

    with pytest.raises(ValueError, match="cannot be smaller than physical"):
        CollisionGeometry(physical_radius_m=10.0, collision_radius_m=5.0)


# ===========================================================================
# 7. Collision Probability Tests & Physical Behaviors
# ===========================================================================

def test_collision_probability_zero_miss():
    """Zero miss distance with non-zero HBR must have Pc > 0."""
    r_rel = np.array([0.0, 0.0, 0.0])
    v_rel = np.array([0.0, 0.0, 7000.0])
    p_pos = np.diag([100.0**2, 100.0**2, 100.0**2])

    b_unc = project_covariance_to_b_plane(p_pos, r_rel, v_rel)
    res = compute_collision_probability(b_unc, hbr_m=20.0)

    assert res.probability > 0.0
    assert res.probability <= 1.0
    assert res.converged is True


def test_collision_probability_monotonicity_miss_distance():
    """Increasing miss distance with fixed covariance and HBR must decrease Pc."""
    v_rel = np.array([0.0, 0.0, 7000.0])
    p_pos = np.diag([200.0**2, 200.0**2, 200.0**2])

    pc_values = []
    miss_distances = [10.0, 50.0, 150.0, 300.0, 600.0]
    for d in miss_distances:
        r_rel = np.array([d, 0.0, 0.0])
        b_unc = project_covariance_to_b_plane(p_pos, r_rel, v_rel)
        res = compute_collision_probability(b_unc, hbr_m=15.0)
        pc_values.append(res.probability)

    # Strictly decreasing
    for i in range(len(pc_values) - 1):
        assert pc_values[i] > pc_values[i + 1]


def test_collision_probability_monotonicity_hbr():
    """Increasing HBR with fixed covariance and miss distance must increase Pc."""
    r_rel = np.array([100.0, 50.0, 0.0])
    v_rel = np.array([0.0, 0.0, 7000.0])
    p_pos = np.diag([200.0**2, 200.0**2, 200.0**2])
    b_unc = project_covariance_to_b_plane(p_pos, r_rel, v_rel)

    hbr_values = [2.0, 5.0, 15.0, 30.0, 60.0]
    pc_values = []
    for h in hbr_values:
        res = compute_collision_probability(b_unc, hbr_m=h)
        pc_values.append(res.probability)

    # Strictly increasing
    for i in range(len(pc_values) - 1):
        assert pc_values[i] < pc_values[i + 1]


def test_collision_probability_edge_cases():
    """Verify edge cases: HBR=0, huge separation, zero covariance."""
    r_rel = np.array([50.0, 0.0, 0.0])
    v_rel = np.array([0.0, 0.0, 7000.0])
    p_pos = np.diag([100.0**2, 100.0**2, 100.0**2])
    b_unc = project_covariance_to_b_plane(p_pos, r_rel, v_rel)

    # 1. HBR = 0 -> Pc = 0
    res_zero_hbr = compute_collision_probability(b_unc, hbr_m=0.0)
    assert res_zero_hbr.probability == 0.0

    # 2. Far separation -> Pc = 0
    r_far = np.array([500000.0, 0.0, 0.0])
    b_unc_far = project_covariance_to_b_plane(p_pos, r_far, v_rel)
    res_far = compute_collision_probability(b_unc_far, hbr_m=10.0)
    assert res_far.probability == 0.0


# ===========================================================================
# 8. Risk Classification Tests
# ===========================================================================

def test_risk_classification():
    """Verify threshold classification."""
    th = PROFILE_STANDARD  # low=1e-7, elevated=1e-5, high=1e-4

    r_low = classify_risk(1e-8, th)
    assert r_low.level == RiskLevel.LOW
    assert r_low.action_required is False

    r_elev = classify_risk(5e-6, th)
    assert r_elev.level == RiskLevel.ELEVATED
    assert r_elev.action_required is False

    r_high = classify_risk(5e-5, th)
    assert r_high.level == RiskLevel.HIGH
    assert r_high.action_required is True

    r_crit = classify_risk(5e-3, th)
    assert r_crit.level == RiskLevel.CRITICAL
    assert r_crit.action_required is True


# ===========================================================================
# 9. Full End-to-End Orchestration Test
# ===========================================================================

def test_full_uncertainty_conjunction_orchestration():
    """
    Run full end-to-end analysis and verify the 14-step calculation trace.

    Object B is inclined 5 deg relative to A, so the pair has a genuine node
    crossing roughly half an orbit into the window.  (The earlier version of
    this test used two co-planar orbits whose separation never reached a
    minimum; it produced no TCA at all and passed only because the
    orchestrator silently evaluated the window midpoint instead.)
    """
    r_a = 6778.137e3
    r_b = 6778.137e3 + 10.0  # 10m altitude separation
    mu = 3.986004418e14
    v_a = math.sqrt(mu / r_a)
    v_b = math.sqrt(mu / r_b)
    inc_b = math.radians(5.0)

    def pos_a(t):
        th = (v_a / r_a) * t
        return np.array([r_a * math.cos(th), r_a * math.sin(th), 0.0])

    def vel_a(t):
        th = (v_a / r_a) * t
        return np.array([-v_a * math.sin(th), v_a * math.cos(th), 0.0])

    def pos_b(t):
        th = (v_b / r_b) * t + 1e-4
        return np.array([
            r_b * math.cos(th),
            r_b * math.sin(th) * math.cos(inc_b),
            r_b * math.sin(th) * math.sin(inc_b),
        ])

    def vel_b(t):
        th = (v_b / r_b) * t + 1e-4
        return np.array([
            -v_b * math.sin(th),
            v_b * math.cos(th) * math.cos(inc_b),
            v_b * math.cos(th) * math.sin(inc_b),
        ])

    cov_a = StateCovariance.from_isotropic(100.0, 0.05, name="Sat A")
    cov_b = StateCovariance.from_isotropic(100.0, 0.05, name="Sat B")

    res = run_uncertainty_conjunction_analysis(
        pos_fn_a=pos_a,
        vel_fn_a=vel_a,
        pos_fn_b=pos_b,
        vel_fn_b=vel_b,
        initial_cov_a=cov_a,
        initial_cov_b=cov_b,
        t_start=0.0,
        t_end=3600.0,
        hbr_m=10.0,
    )

    # The pipeline must have actually run, not fallen back to a substitute point.
    assert res.conjunction_found is True
    assert res.analysis_status == "COMPLETE"

    assert res.collision_probability is not None
    assert 0.0 <= res.collision_probability.probability <= 1.0
    assert len(res.calculation_steps) == 14
    assert res.calculation_steps[0]["stepIndex"] == 1
    assert res.calculation_steps[13]["stepIndex"] == 14
    assert "RISK LEVEL" in res.calculation_steps[13]["result"]

    # The reported geometry must be the TCA geometry.
    assert res.tca_s is not None and not isinstance(res.tca_s, bool)
    sep_at_tca = float(np.linalg.norm(pos_a(res.tca_s) - pos_b(res.tca_s)))
    assert res.miss_distance_m == pytest.approx(sep_at_tca, rel=1e-9, abs=1e-6)

    # Verify JSON serialization
    d = res.to_dict()
    assert "conjunction_summary" in d
    assert "collision_probability" in d
    assert "risk_assessment" in d
    assert "calculation_steps" in d
    assert d["analysis_status"] == "COMPLETE"
    assert isinstance(d["conjunction_summary"]["tca_s"], float)
    assert not isinstance(d["conjunction_summary"]["tca_s"], bool)


def test_orchestration_refuses_to_classify_risk_without_a_tca():
    """
    Two co-planar orbits at slightly different radii drift past each other
    without ever reaching a closest approach inside the window.  The analysis
    must halt as indeterminate instead of manufacturing a Pc and a risk level.
    """
    r_a = 6778.137e3
    r_b = 6778.137e3 + 10.0
    mu = 3.986004418e14
    v_a = math.sqrt(mu / r_a)
    v_b = math.sqrt(mu / r_b)

    def pos_a(t):
        th = (v_a / r_a) * t
        return np.array([r_a * math.cos(th), r_a * math.sin(th), 0.0])

    def vel_a(t):
        th = (v_a / r_a) * t
        return np.array([-v_a * math.sin(th), v_a * math.cos(th), 0.0])

    def pos_b(t):
        th = (v_b / r_b) * t + 1e-4
        return np.array([r_b * math.cos(th), r_b * math.sin(th), 0.0])

    def vel_b(t):
        th = (v_b / r_b) * t + 1e-4
        return np.array([-v_b * math.sin(th), v_b * math.cos(th), 0.0])

    res = run_uncertainty_conjunction_analysis(
        pos_fn_a=pos_a,
        vel_fn_a=vel_a,
        pos_fn_b=pos_b,
        vel_fn_b=vel_b,
        initial_cov_a=StateCovariance.from_isotropic(100.0, 0.05, name="Sat A"),
        initial_cov_b=StateCovariance.from_isotropic(100.0, 0.05, name="Sat B"),
        t_start=0.0,
        t_end=300.0,
        hbr_m=10.0,
    )

    assert res.conjunction_found is False
    assert res.analysis_status == "INDETERMINATE_NO_CONJUNCTION"
    assert res.collision_probability is None
    assert res.tca_s is None
    assert res.miss_distance_m is None
    assert res.risk_assessment.level == RiskLevel.INDETERMINATE
    assert res.risk_assessment.action_required is False
    assert res.risk_assessment.probability is None


def test_classify_risk_refuses_a_missing_probability():
    """classify_risk must never invent a level from a Pc that was not computed."""
    with pytest.raises(ValueError):
        classify_risk(None)
    with pytest.raises(ValueError):
        classify_risk(float("nan"))
