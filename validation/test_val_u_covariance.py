"""
Phase 10 — Covariance Analysis and State Transition Matrix Analytical Validation.

Validates:
1. Static state test: Φ = I → P(t) = P₀ exactly within numerical tolerance.
2. Linear dynamics test: dx/dt = A x → Φ(t) = exp(A t) compared against matrix exponential.
3. Orbital STM conservation: Liouville's theorem det(Φ) = 1 for conservative Hamiltonian dynamics.
4. Finite-difference state sensitivity vs STM: δx(t) ≈ Φ(t, t0) δx(0).
5. Independent object covariance combination: P_rel = P₁ + P₂.
6. B-plane basis orthonormality and B · Ŝ = 0 orthogonality.
7. B-plane projected covariance positive semi-definiteness.
"""

import math
import pytest
import numpy as np
from scipy.linalg import expm

from theseus.uncertainty.covariance import StateCovariance
from theseus.uncertainty.state_transition import (
    gravity_jacobian,
    propagate_stm,
    build_dynamics_jacobian,
)
from theseus.uncertainty.propagation import propagate_covariance
from theseus.uncertainty.relative import compute_relative_covariance
from theseus.uncertainty.b_plane import project_covariance_to_b_plane
from theseus.conjunction.b_plane import compute_b_plane

MU_EARTH = 3.986004418e14  # m³/s²
R_LEO = 6778.137e3         # m


def test_val_static_state_covariance():
    """
    Validation Test 1: Static State & Zero-Acceleration Free Drift
    1. Zero time interval: Φ(t₀, t₀) = I₆, P(t₀) = P₀ exactly.
    2. Free rectilinear drift (a = 0):
       Exact Analytical STM:
           Φ(t, 0) = | I₃  t·I₃ |
                     | 0₃   I₃  |
       Exact Propagated Covariance:
           P_rr(t) = P_rr(0) + t (P_rv(0) + P_vr(0)) + t² P_vv(0)
           P_vv(t) = P_vv(0)
    """
    p0_mat = np.diag([100.0**2, 200.0**2, 300.0**2, 0.1**2, 0.2**2, 0.3**2])
    p0 = StateCovariance(matrix=p0_mat, epoch_s=0.0)

    # 1. Zero time interval
    def zero_acc(t, r, v):
        return np.zeros(3)

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.0])
    stm_res_0 = propagate_stm(zero_acc, r0, v0, (50.0, 50.0))
    np.testing.assert_allclose(stm_res_0.stm, np.eye(6), atol=1e-12)
    p_final_0 = propagate_covariance(p0, stm_res_0.stm, epoch_tf_s=50.0)
    np.testing.assert_allclose(p_final_0.matrix, p0_mat, atol=1e-12)

    # 2. Free drift over t = 1000 s
    t_drift = 1000.0
    stm_res_drift = propagate_stm(zero_acc, r0, v0, (0.0, t_drift), dt=100.0)

    # Expected analytical STM for a = 0
    phi_expected = np.eye(6)
    phi_expected[:3, 3:6] = t_drift * np.eye(3)
    np.testing.assert_allclose(stm_res_drift.stm, phi_expected, atol=1e-12)

    # Expected analytical covariance
    p_final_drift = propagate_covariance(p0, stm_res_drift.stm, epoch_tf_s=t_drift)
    p_expected = phi_expected @ p0_mat @ phi_expected.T
    np.testing.assert_allclose(p_final_drift.matrix, p_expected, atol=1e-12)


def test_val_linear_dynamics_stm():
    """
    Validation Test 2: Linear Dynamics Comparison
    Dynamics: dx/dt = A x (constant matrix A)
    Exact Analytical Solution: Φ(t, 0) = exp(A t).
    """
    # Define a 6x6 constant linear oscillator system
    omega = 0.001  # rad/s
    A = np.zeros((6, 6))
    A[:3, 3:6] = np.eye(3)
    A[3:6, :3] = -(omega**2) * np.eye(3)

    def linear_acc(t, r, v):
        return -(omega**2) * r

    r0 = np.array([100.0, 200.0, 300.0])
    v0 = np.array([0.1, 0.2, 0.3])
    t_span = (0.0, 500.0)

    # Numerical STM propagation using the general framework
    stm_res = propagate_stm(linear_acc, r0, v0, t_span, dt=10.0, integrator="rkf45", atol=1e-12, rtol=1e-12)

    # Analytical STM via matrix exponential
    stm_analytic = expm(A * 500.0)

    np.testing.assert_allclose(stm_res.stm, stm_analytic, rtol=1e-6, atol=1e-7)


def test_val_liouville_stm_determinant():
    """
    Validation Test 3: Liouville's Theorem / Hamiltonian Conservation
    For unperturbed gravitational orbital motion (conservative Hamiltonian system),
    the volume in phase space is conserved: det(Φ(t, t0)) = 1.0.
    """
    r0 = np.array([R_LEO, 0.0, 0.0])
    v_circ = math.sqrt(MU_EARTH / R_LEO)
    v0 = np.array([0.0, v_circ, 0.0])

    def acc_fn(t, r, v):
        return -MU_EARTH / (np.linalg.norm(r)**3) * r

    # Half orbit duration
    period = 2.0 * math.pi * math.sqrt(R_LEO**3 / MU_EARTH)
    stm_res = propagate_stm(acc_fn, r0, v0, (0.0, period / 2.0), mu=MU_EARTH, integrator="rkf45", atol=1e-11, rtol=1e-11)

    det_stm = float(np.linalg.det(stm_res.stm))
    # Expected: 1.0 within numerical integrator truncation tolerance
    assert det_stm == pytest.approx(1.0, abs=1e-4)


def test_val_stm_finite_difference_sensitivity():
    """
    Validation Test 4: Perturbation Sensitivity Mapping
    Verify δx(t) ≈ Φ(t, t0) δx(0) for nonlinear orbital propagation.
    """
    r0 = np.array([R_LEO, 0.0, 0.0])
    v_circ = math.sqrt(MU_EARTH / R_LEO)
    v0 = np.array([0.0, v_circ, 0.0])
    t_span = (0.0, 300.0)

    def acc_fn(t, r, v):
        return -MU_EARTH / (np.linalg.norm(r)**3) * r

    stm_res = propagate_stm(acc_fn, r0, v0, t_span, mu=MU_EARTH)
    phi = stm_res.stm
    nom_final = stm_res.nominal_state_tf

    # Apply small physical initial perturbation: 10 meters in X, 0.01 m/s in Y
    delta_x0 = np.array([10.0, 0.0, 0.0, 0.0, 0.01, 0.0])
    r0_pert = r0 + delta_x0[:3]
    v0_pert = v0 + delta_x0[3:6]

    # Full nonlinear trajectory propagation of perturbed state
    stm_res_pert = propagate_stm(acc_fn, r0_pert, v0_pert, t_span, mu=MU_EARTH)
    pert_final = stm_res_pert.nominal_state_tf

    # Actual nonlinear displacement δx_actual = x_pert(tf) - x_nom(tf)
    delta_x_actual = pert_final - nom_final

    # Linear STM predicted displacement δx_stm = Φ * δx₀
    delta_x_stm = phi @ delta_x0

    # Must agree closely for small perturbation
    np.testing.assert_allclose(delta_x_actual, delta_x_stm, rtol=1e-3, atol=1e-3)


def test_val_independent_covariance_combination():
    """
    Validation Test 5: Independent Object Covariance Combination
    P_rel = P₁ + P₂ analytically.
    """
    p1 = np.diag([100.0, 200.0, 300.0, 1.0, 2.0, 3.0])
    p2 = np.diag([50.0, 60.0, 70.0, 0.5, 0.6, 0.7])

    cov1 = StateCovariance(matrix=p1)
    cov2 = StateCovariance(matrix=p2)

    rel_res = compute_relative_covariance(cov1, cov2)
    expected_rel = p1 + p2

    np.testing.assert_allclose(rel_res.relative_covariance.matrix, expected_rel)
    assert rel_res.independent is True


def test_val_b_plane_orthonormality_and_orthogonality():
    """
    Validation Test 6: B-Plane Basis Orthonormality and Orthogonality
    Verify:
    Ŝ · T̂ = 0,  Ŝ · R̂ = 0,  T̂ · R̂ = 0
    |Ŝ| = 1,  |T̂| = 1,  |R̂| = 1
    B · Ŝ = 0 (B-vector is strictly in the B-plane)
    """
    r_rel = np.array([120.0, -85.0, 40.0])
    v_rel = np.array([1500.0, 6200.0, -3100.0])  # ~7 km/s relative speed

    b_res = compute_b_plane(r_rel, v_rel)
    assert b_res.applicable is True

    s_hat = b_res.s_hat
    t_hat = b_res.t_hat
    r_hat = b_res.r_hat
    b_vec = b_res.b_vector

    # Unit lengths
    assert np.linalg.norm(s_hat) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(t_hat) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(r_hat) == pytest.approx(1.0, abs=1e-12)

    # Mutual orthogonality
    assert float(np.dot(s_hat, t_hat)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.dot(s_hat, r_hat)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.dot(t_hat, r_hat)) == pytest.approx(0.0, abs=1e-12)

    # B-vector orthogonal to approach direction
    assert float(np.dot(b_vec, s_hat)) == pytest.approx(0.0, abs=1e-12)

    # Magnitude consistency: |B|² = (B·T)² + (B·R)²
    assert b_res.b_magnitude**2 == pytest.approx(b_res.b_dot_t**2 + b_res.b_dot_r**2, rel=1e-9)


def test_val_b_plane_covariance_positive_semidefinite():
    """
    Validation Test 7: Projected B-plane covariance P_B is positive semi-definite.
    """
    # General non-diagonal 3x3 position covariance
    A = np.array([
        [200.0, 30.0, -15.0],
        [30.0, 150.0, 40.0],
        [-15.0, 40.0, 300.0],
    ])
    P_rr = A @ A.T  # Guaranteed positive definite

    r_rel = np.array([50.0, 50.0, 0.0])
    v_rel = np.array([0.0, 7000.0, 1000.0])

    b_unc = project_covariance_to_b_plane(P_rr, r_rel, v_rel)
    assert np.all(b_unc.eigenvalues >= 0.0)
    assert b_unc.sigma_major >= b_unc.sigma_minor
    assert -1.0 <= b_unc.correlation <= 1.0
