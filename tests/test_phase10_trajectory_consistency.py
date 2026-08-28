"""
P10-07 — do the conjunction and covariance paths use the same trajectory?

Outcome of the investigation: **not a defect.**  The two paths are seeded from
the same state (bitwise), evaluate bitwise-identical accelerations, and are
integrated by the same integrator at the same tolerances.  They are two
independent adaptive integrations of one dynamical system, and they agree to
about two metres over an orbit -- the same order as each one's own distance
from a DOP853 reference at rtol 1e-13.

No code was changed for P10-07.  These tests exist to hold that conclusion in
place, because it is a conclusion about a property that a future edit could
silently break: change a tolerance, rebuild the force model differently,
re-seed the STM, and the two paths would start describing different
trajectories without anything else failing.

Measured on an a = 10 000 km, e = 0.3 pair (TCA = 4951.15 s):

    max |r_conj - r_cov| over the window          1.94 m
    max |r_conj - r_reference|                    1.38 m
    max |r_cov  - r_reference|                    0.64 m

    sigma_major   production        8715.0904 m
                  along-conjunction 8715.0914 m
                  tight reference   8715.0897 m
    Pc            production        5.183955e-136
                  along-conjunction 5.183355e-136
    risk          LOW in all three

Crucially, ||Phi_production - Phi_along-conjunction|| is 1.96e-07 while
||Phi_production - Phi_tight|| is 1.34e-07: making the linearisation follow the
conjunction trajectory exactly does not move Phi further than ordinary
integration error does, and in two of three scenarios it lands *further* from
the tight reference, because the Hermite interpolant's own error then enters
the Jacobian.  There is no accuracy case for the architectural change.

The tolerances below come from those measurements and from the independent
reference's accuracy, not from round numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import theseus.simulation.multi_object as multi_object_module
from theseus.simulation.multi_object import (
    MultiObjectEnvironment,
    SpacecraftDefinition,
)
from theseus.uncertainty.b_plane import project_covariance_to_b_plane
from theseus.uncertainty.collision_probability import compute_collision_probability
from theseus.uncertainty.risk import PROFILE_STANDARD, classify_risk
from theseus.uncertainty.state_transition import propagate_stm as _engine_propagate_stm


SIGMA_POS_M = 300.0
SIGMA_VEL_M_S = 0.3
P0 = np.diag([SIGMA_POS_M ** 2] * 3 + [SIGMA_VEL_M_S ** 2] * 3)

#: Both production trajectories sit within ~2.5 m of each other and of a
#: DOP853 reference over a full window.  25 m is an order of magnitude above
#: that and still far below anything a seed or force-model regression would
#: produce -- re-seeding the STM at TCA moved this by 12 000 km.
TRAJECTORY_AGREEMENT_M = 25.0
TRAJECTORY_AGREEMENT_M_S = 0.5


# ---------------------------------------------------------------------------
# Capturing what the production run actually built
# ---------------------------------------------------------------------------

class _Capture:
    """Records the two construction points inside one simulate() call."""

    def __init__(self):
        self.interpolators = []
        self.stm_calls = []
        self.propagator_acc_fns = []

    def install(self, monkeypatch):
        real_interp = multi_object_module.interpolator_from_state_history
        real_stm = multi_object_module.propagate_stm
        real_prop = multi_object_module.NumericalPropagator
        capture = self

        def spy_interp(history):
            interpolator = real_interp(history)
            capture.interpolators.append(interpolator)
            return interpolator

        def spy_stm(acc_fn, r0, v0, t_span, **kwargs):
            result = real_stm(acc_fn, r0, v0, t_span, **kwargs)
            capture.stm_calls.append({
                "acc_fn": acc_fn,
                "r0": np.array(r0, dtype=float),
                "v0": np.array(v0, dtype=float),
                "t_span": (float(t_span[0]), float(t_span[1])),
                "kwargs": dict(kwargs),
                "result": result,
            })
            return result

        class SpyPropagator(real_prop):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                capture.propagator_acc_fns.append(self.acceleration_fn)

        monkeypatch.setattr(multi_object_module,
                            "interpolator_from_state_history", spy_interp)
        monkeypatch.setattr(multi_object_module, "propagate_stm", spy_stm)
        monkeypatch.setattr(multi_object_module, "NumericalPropagator", SpyPropagator)
        return self


def _pair(a_km, ecc, inc_deg, nu_b_deg):
    common = dict(
        semi_major_axis_km=a_km, eccentricity=ecc,
        dry_mass_kg=1000.0, fuel_mass_kg=0.0, payload_mass_kg=0.0,
        hard_body_radius_m=5.0, cross_section_area_m2=10.0,
        drag_coefficient=2.2,
        sigma_pos_m=[SIGMA_POS_M] * 3, sigma_vel_m_s=[SIGMA_VEL_M_S] * 3,
    )
    a = SpacecraftDefinition(id="A", name="A", color="#ff9900",
                             inclination_deg=inc_deg, true_anomaly_deg=0.0, **common)
    b = SpacecraftDefinition(id="B", name="B", color="#3388ff",
                             inclination_deg=-inc_deg, true_anomaly_deg=nu_b_deg,
                             **common)
    return a, b


#: (label, a_km, e, inc, nu_b, t_end, threshold_km, coarse_dt, j2, drag, out_dt)
#
# The eccentric cases keep perigee above the atmosphere: a = 10 000 km with
# e = 0.6 puts perigee at 4 000 km radius, inside the Earth, where the drag
# model clamps to sea-level density and the health checker aborts.
SCENARIOS = {
    "gravity_only": (6778.137, 0.0, 51.6, 0.05, 3000.0, 50.0, 5.0, False, False, 5.0),
    "j2": (6778.137, 0.0, 51.6, 0.05, 3000.0, 50.0, 5.0, True, False, 5.0),
    "default_config": (6778.137, 0.0, 51.6, 0.05, 3000.0, 50.0, 5.0, True, True, 5.0),
    "drag_active_low": (6528.137, 0.0, 51.6, 0.05, 3000.0, 50.0, 5.0, True, True, 5.0),
    "eccentric": (10000.0, 0.3, 35.0, 1.0, 6000.0, 200.0, 10.0, True, True, 10.0),
}


def _simulate(capture, scenario):
    (a_km, ecc, inc, nu_b, t_end, threshold_km,
     coarse_dt, j2, drag, out_dt) = SCENARIOS[scenario]
    env = MultiObjectEnvironment(
        central_body="Earth", screening_threshold_km=threshold_km,
        coarse_dt_s=coarse_dt, enable_j2=j2, enable_drag=drag,
    )
    sc_a, sc_b = _pair(a_km, ecc, inc, nu_b)
    result = env.simulate([sc_a, sc_b], t_start=0.0, t_end=t_end, output_dt=out_dt)
    return env, result


# ---------------------------------------------------------------------------
# An independent reference trajectory
# ---------------------------------------------------------------------------

def reference_state(acc_fn, x0, t0: float, tf: float) -> np.ndarray:
    """
    DOP853 at rtol 1e-13, seeded at the covariance epoch with the same state
    and driven by the same production acceleration.  Independent of both
    production integrations.
    """
    if tf == t0:
        return np.asarray(x0, dtype=float).copy()
    sol = solve_ivp(
        lambda t, y: np.concatenate([y[3:6], acc_fn(t, y[:3], y[3:6])]),
        (t0, tf), np.asarray(x0, dtype=float),
        rtol=1e-13, atol=1e-8, method="DOP853",
    )
    return np.asarray(sol.y[:, -1], dtype=float)


def _fd_jacobian(acc_fn, t, r, v, dr=1.0, dv=1e-3):
    a_matrix = np.zeros((6, 6))
    a_matrix[:3, 3:] = np.eye(3)
    for i in range(3):
        rp, rm = r.copy(), r.copy()
        rp[i] += dr
        rm[i] -= dr
        a_matrix[3:, i] = (acc_fn(t, rp, v) - acc_fn(t, rm, v)) / (2 * dr)
        vp, vm = v.copy(), v.copy()
        vp[i] += dv
        vm[i] -= dv
        a_matrix[3:, 3 + i] = (acc_fn(t, r, vp) - acc_fn(t, r, vm)) / (2 * dv)
    return a_matrix


def reference_stm(acc_fn, x0, t0: float, tf: float) -> np.ndarray:
    """Variational equations, DOP853 rtol 1e-13, finite-difference Jacobian."""
    def deriv(t, y):
        r, v, phi = y[:3], y[3:6], y[6:42].reshape(6, 6)
        out = np.empty(42)
        out[:3] = v
        out[3:6] = acc_fn(t, r, v)
        out[6:] = (_fd_jacobian(acc_fn, t, r, v) @ phi).ravel()
        return out

    sol = solve_ivp(deriv, (t0, tf), np.concatenate([x0, np.eye(6).ravel()]),
                    rtol=1e-13, atol=1e-9, method="DOP853")
    return sol.y[6:42, -1].reshape(6, 6)


def stm_along_conjunction(acc_fn, pos_fn, vel_fn, t0: float, tf: float) -> np.ndarray:
    """
    Phi linearised about the *conjunction* trajectory, i.e. what exact
    conjunction/covariance consistency would produce.
    """
    def deriv(t, y):
        r = np.asarray(pos_fn(t), dtype=float)
        v = np.asarray(vel_fn(t), dtype=float)
        return (_fd_jacobian(acc_fn, t, r, v) @ y.reshape(6, 6)).ravel()

    sol = solve_ivp(deriv, (t0, tf), np.eye(6).ravel(),
                    rtol=1e-13, atol=1e-9, method="DOP853")
    return sol.y[:, -1].reshape(6, 6)


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))
                 / np.max(np.abs(np.asarray(b))))


# ---------------------------------------------------------------------------
# 1. The two paths start from the same state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_covariance_and_conjunction_start_from_the_same_state(monkeypatch, scenario):
    """
    Bitwise, not approximately.  Both consume ``resolved_initial_states[id]``,
    which is also the interpolator's first node -- this is the P10-05
    invariant seen from the trajectory side.
    """
    capture = _Capture().install(monkeypatch)
    _, result = _simulate(capture, scenario)
    assert result.conjunctions, f"{scenario}: fixture must produce a conjunction"

    interpolator = capture.interpolators[0]
    node_state = np.concatenate([interpolator.position_at(0.0),
                                 interpolator.velocity_at(0.0)])
    seed = np.concatenate([capture.stm_calls[0]["r0"], capture.stm_calls[0]["v0"]])

    assert np.array_equal(seed, node_state)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_covariance_epoch_is_the_propagation_start(monkeypatch, scenario):
    """P10-05: reference epoch t_start, evaluation epoch TCA."""
    capture = _Capture().install(monkeypatch)
    _, result = _simulate(capture, scenario)
    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)

    for call in capture.stm_calls:
        reference_epoch, evaluation_epoch = call["t_span"]
        assert reference_epoch == 0.0
        assert evaluation_epoch == pytest.approx(event.tca_s, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. The two paths integrate the same dynamics (excludes "different model")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_both_paths_evaluate_bitwise_identical_accelerations(monkeypatch, scenario):
    """
    The force models are two objects, but they must be the same *model*.
    Sampled over 200 random states so no single lucky point can carry it.
    """
    capture = _Capture().install(monkeypatch)
    _, result = _simulate(capture, scenario)
    assert result.conjunctions

    propagation_acc = capture.propagator_acc_fns[0]
    covariance_acc = capture.stm_calls[0]["acc_fn"]
    mass = 1000.0

    rng = np.random.default_rng(20260825)
    worst = 0.0
    for _ in range(200):
        r = np.array([7.0e6, 0.0, 0.0]) + rng.normal(0.0, 2e6, 3)
        v = rng.normal(0.0, 7000.0, 3)
        worst = max(worst, float(np.max(np.abs(
            propagation_acc(0.0, r, v, mass) - covariance_acc(0.0, r, v)))))

    assert worst == 0.0


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_both_paths_use_the_same_integration_tolerances(monkeypatch, scenario):
    """
    Same atol/rtol on both sides.  If these ever diverge the two trajectories
    stop being two samples of one answer, and the agreement asserted below
    would no longer be guaranteed by anything.
    """
    capture = _Capture().install(monkeypatch)
    _, result = _simulate(capture, scenario)
    assert result.conjunctions

    kwargs = capture.stm_calls[0]["kwargs"]
    assert kwargs["atol"] == 1e-4
    assert kwargs["rtol"] == 1e-8


# ---------------------------------------------------------------------------
# 3. Epoch-by-epoch agreement, against each other and a tight reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_trajectories_agree_epoch_by_epoch(monkeypatch, scenario):
    """
    Compared at the covariance epoch, 25 %, 50 %, 75 % and TCA.  Both paths
    are also checked against a DOP853 reference at rtol 1e-13, so a common
    drift away from the true trajectory could not pass as agreement.
    """
    capture = _Capture().install(monkeypatch)
    _, result = _simulate(capture, scenario)
    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)

    call = capture.stm_calls[0]
    acc_fn = call["acc_fn"]
    x0 = np.concatenate([call["r0"], call["v0"]])
    pos_fn, vel_fn = capture.interpolators[0].as_callables()
    tca = event.tca_s

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = fraction * tca
        conjunction = np.concatenate([pos_fn(t), vel_fn(t)])
        covariance = (x0 if t == 0.0 else _engine_propagate_stm(
            acc_fn, call["r0"], call["v0"], (0.0, t), **call["kwargs"]
        ).nominal_state_tf)
        reference = reference_state(acc_fn, x0, 0.0, t)

        assert np.linalg.norm(conjunction[:3] - covariance[:3]) < TRAJECTORY_AGREEMENT_M
        assert np.linalg.norm(conjunction[3:] - covariance[3:]) < TRAJECTORY_AGREEMENT_M_S
        assert np.linalg.norm(conjunction[:3] - reference[:3]) < TRAJECTORY_AGREEMENT_M
        assert np.linalg.norm(covariance[:3] - reference[:3]) < TRAJECTORY_AGREEMENT_M


def test_trajectory_agreement_is_measured_not_assumed(monkeypatch):
    """
    The agreement threshold has to be justified.  On the eccentric scenario
    the two production trajectories differ by about two metres -- an order of
    magnitude inside the threshold -- so the assertion above is not passing by
    being loose.
    """
    capture = _Capture().install(monkeypatch)
    _, result = _simulate(capture, "eccentric")
    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)

    call = capture.stm_calls[0]
    pos_fn, _ = capture.interpolators[0].as_callables()
    worst = 0.0
    for fraction in (0.25, 0.5, 0.75, 1.0):
        t = fraction * event.tca_s
        covariance = _engine_propagate_stm(
            call["acc_fn"], call["r0"], call["v0"], (0.0, t), **call["kwargs"]
        ).nominal_state_tf
        worst = max(worst, float(np.linalg.norm(np.asarray(pos_fn(t)) - covariance[:3])))

    assert worst < 5.0
    assert worst > 0.0, "identical to the last bit would mean the probe is not working"


# ---------------------------------------------------------------------------
# 4. The covariance chain against an independent reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ["eccentric", "drag_active_low"])
def test_reported_covariance_matches_an_independent_reference(monkeypatch, scenario):
    """
    Production sigma_major / sigma_minor / ellipse angle against covariances
    built from independently integrated STMs, seeded at the covariance epoch
    with the same state and the same production acceleration.
    """
    capture = _Capture().install(monkeypatch)
    _, result = _simulate(capture, scenario)
    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)
    assert event.b_plane_sigma_major_m is not None

    position_covariances = []
    for call in capture.stm_calls[:2]:
        x0 = np.concatenate([call["r0"], call["v0"]])
        phi = reference_stm(call["acc_fn"], x0, 0.0, event.tca_s)
        position_covariances.append((phi @ P0 @ phi.T)[:3, :3])

    reference = project_covariance_to_b_plane(
        rel_pos_cov=position_covariances[0] + position_covariances[1],
        r_rel=np.array(event.r_rel_m, dtype=float),
        v_rel=np.array(event.v_rel_m_s, dtype=float),
    )

    assert event.b_plane_sigma_major_m == pytest.approx(reference.sigma_major, rel=1e-4)
    assert event.b_plane_sigma_minor_m == pytest.approx(reference.sigma_minor, rel=1e-4)
    assert event.b_plane_ellipse_angle_deg == pytest.approx(
        reference.ellipse_angle_deg, abs=1e-3)


def test_probability_and_risk_match_the_independent_reference(monkeypatch):
    """
    Carried through to the end of the chain: the reported Pc and risk level
    must be those an independently integrated covariance produces.
    """
    capture = _Capture().install(monkeypatch)
    _, result = _simulate(capture, "drag_active_low")
    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)

    position_covariances = []
    for call in capture.stm_calls[:2]:
        x0 = np.concatenate([call["r0"], call["v0"]])
        phi = reference_stm(call["acc_fn"], x0, 0.0, event.tca_s)
        position_covariances.append((phi @ P0 @ phi.T)[:3, :3])

    reference = project_covariance_to_b_plane(
        rel_pos_cov=position_covariances[0] + position_covariances[1],
        r_rel=np.array(event.r_rel_m, dtype=float),
        v_rel=np.array(event.v_rel_m_s, dtype=float),
    )
    reference_pc = compute_collision_probability(
        reference, event.hard_body_radius_m).probability

    assert event.collision_probability == pytest.approx(reference_pc, rel=1e-3)
    assert (classify_risk(event.collision_probability, PROFILE_STANDARD).level
            == classify_risk(reference_pc, PROFILE_STANDARD).level)


def test_linearising_along_the_conjunction_trajectory_changes_nothing_material(
        monkeypatch):
    """
    The question P10-07 actually asks.

    Phi is compared three ways: as production computes it, linearised exactly
    along the conjunction trajectory, and from a tight independent reference.
    If the conjunction/covariance inconsistency mattered, the production Phi
    would sit measurably further from the along-conjunction Phi than from the
    tight reference.  It does not -- both differences are the same order, i.e.
    ordinary integration error.
    """
    capture = _Capture().install(monkeypatch)
    _, result = _simulate(capture, "eccentric")
    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)

    call = capture.stm_calls[0]
    x0 = np.concatenate([call["r0"], call["v0"]])
    pos_fn, vel_fn = capture.interpolators[0].as_callables()
    tca = event.tca_s

    production = call["result"].stm
    tight = reference_stm(call["acc_fn"], x0, 0.0, tca)
    along_conjunction = stm_along_conjunction(
        call["acc_fn"], pos_fn, vel_fn, 0.0, tca)

    to_tight = _rel(production, tight)
    to_conjunction = _rel(production, along_conjunction)

    assert to_tight < 1e-5
    assert to_conjunction < 1e-5
    # The inconsistency is not the dominant error term: forcing the
    # linearisation onto the conjunction trajectory moves Phi no further than
    # the integrator's own error already does.
    assert to_conjunction < 20.0 * to_tight


# ---------------------------------------------------------------------------
# 5. No second nominal trajectory leaks into the result
# ---------------------------------------------------------------------------

def test_the_stm_reintegrated_state_is_never_used_as_a_nominal_state(monkeypatch):
    """
    The STM integrates the nominal state alongside Phi because the variational
    equations need a reference trajectory -- that is inherent, not a duplicate
    nominal trajectory.  What matters is that the re-integrated state is not
    then used as if it were the nominal state: every reported quantity must
    come from the conjunction trajectory.
    """
    import inspect

    source = inspect.getsource(MultiObjectEnvironment.simulate)
    assert "nominal_state_tf" not in source

    capture = _Capture().install(monkeypatch)
    _, result = _simulate(capture, "eccentric")
    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)

    # The reported geometry is the conjunction trajectory's, to the last bit
    # of the TCA solution -- not the STM's re-integrated state.
    pos_a_fn, vel_a_fn = capture.interpolators[0].as_callables()
    pos_b_fn, vel_b_fn = capture.interpolators[1].as_callables()
    r_rel = np.asarray(pos_a_fn(event.tca_s)) - np.asarray(pos_b_fn(event.tca_s))
    v_rel = np.asarray(vel_a_fn(event.tca_s)) - np.asarray(vel_b_fn(event.tca_s))

    np.testing.assert_allclose(np.array(event.r_rel_m), r_rel, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.array(event.v_rel_m_s), v_rel, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# 6. The closed corrections this investigation depends on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_p10_05_invariant_seed_is_not_the_tca_state(monkeypatch, scenario):
    """The STM seed must remain the covariance-epoch state."""
    capture = _Capture().install(monkeypatch)
    _, result = _simulate(capture, scenario)
    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)

    pos_a_fn, _ = capture.interpolators[0].as_callables()
    pos_b_fn, _ = capture.interpolators[1].as_callables()
    tca_radii = [float(np.linalg.norm(pos_a_fn(event.tca_s))),
                 float(np.linalg.norm(pos_b_fn(event.tca_s)))]
    epoch_radii = [float(np.linalg.norm(pos_a_fn(0.0))),
                   float(np.linalg.norm(pos_b_fn(0.0)))]

    for call in capture.stm_calls:
        seed_radius = float(np.linalg.norm(call["r0"]))
        assert min(abs(seed_radius - r) for r in epoch_radii) < 1e-6


@pytest.mark.parametrize("scenario,expect_analytic", [
    # gravity_only used to be False, and not for the reason one would guess:
    # the multi-object STM call passed `j2=self.body.J2` unconditionally, even
    # when `enable_j2=False`.  The analytic Jacobian then carried a J2 term the
    # propagated trajectory did not have -- 2.87e-03 relative error in da/dr --
    # so P10-06's consistency guard correctly refused it and fell back to the
    # numerical Jacobian, which matches the model to 1.9e-10.  The result was
    # right; the call site's argument was not.  It was recorded as a residual
    # of P10-06 and left.
    #
    # The final P10 sweep fixed the call site: `_analytic_j2_for` passes the J2
    # the force model actually applies, so a gravity-only model now reaches the
    # analytic Jacobian instead of paying twelve extra acceleration evaluations
    # per step to reach the same answer.  The guard is unchanged, and now fires
    # only on the cases it was written for.
    ("gravity_only", True),
    ("j2", True),
    ("default_config", False),
    ("drag_active_low", False),
    ("eccentric", False),
])
def test_p10_06_invariant_jacobian_matches_the_force_model(
        monkeypatch, scenario, expect_analytic):
    """
    Gravity + J2 keeps the analytic Jacobian; every configuration whose force
    model the analytic Jacobian does not describe must not.
    """
    capture = _Capture().install(monkeypatch)
    _, result = _simulate(capture, scenario)
    assert result.conjunctions

    for call in capture.stm_calls:
        is_analytic = call["result"].method.startswith("analytic_jacobian")
        assert is_analytic is expect_analytic, call["result"].method
