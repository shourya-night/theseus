"""
P10-05 — the STM must be linearised about the trajectory its covariance
belongs to.

The defect
----------
``theseus/simulation/multi_object.py`` propagated each object's STM as::

    stm_a = propagate_stm(acc_fn_a,
                          tca.r_a, tca.v_a,        # the state AT TCA
                          (t_start, tca.tca),      # integrated FROM t_start
                          ...)

``propagate_stm`` establishes ``Phi = I`` at ``t_span[0]`` and integrates the
augmented state ``[r, v, vec(Phi)]`` forward, so ``(r0, v0)`` *is* the nominal
state at the reference epoch.  Passing the TCA state therefore linearises the
variational equations about a trajectory that begins at ``t_start`` holding
the state the object only reaches at TCA -- a trajectory that never existed.
The resulting matrix was then applied to ``P(t_start)``::

    cov_a_tca = propagate_covariance(cov_a_0, stm_a.stm, tca.tca)

The reference *epoch* label was right; the reference *trajectory* was not.

Measured on an a = 10 000 km, e = 0.6 orbit with TCA at 0.37 period:

    ||Phi_ascoded - Phi_reference|| / ||Phi_reference||   = 6.66e-01
    ||Phi_ascoded - Phi_fictitious|| / ||Phi_fictitious|| = 1.12e-09
    finite-difference error, seeded x(t0)                 = 1.3e-06
    finite-difference error, seeded x(TCA)                = 1.1e+00
    sigma_pos_3d at TCA                    14119 m -> 3767 m  (0.267x)

Understating the uncertainty is the dangerous direction: it shrinks the
covariance ellipse and moves Pc.

Near-circular orbits mask the *scalar* symptom -- at e = 0 the sigma ratio is
1.0013 -- but not the physics: the B-plane ellipse orientation is still wrong
by 29 degrees there, and Pc depends on the orientation.

What these tests pin
--------------------
1. The invariants that make an STM an STM, against an independent variational
   integration and against finite-difference propagation (groups A-D).
2. The reference-epoch contract at the multi-object call site: the state
   handed to ``propagate_stm`` must be the state at the covariance epoch
   (group E).
3. The end-to-end consequence: a real multi-object simulation's covariance at
   TCA, compared against an independently integrated reference (group F).

The reference lives in ``tests/_stm_reference.py`` and imports nothing from
``theseus``; its Jacobian is a central difference, so it cannot share an
algebraic mistake with the engine's analytic Jacobians either.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from theseus.uncertainty.covariance import StateCovariance
from theseus.uncertainty.propagation import propagate_covariance
from theseus.uncertainty.state_transition import propagate_stm

from tests._stm_reference import (
    J2_EARTH,
    MU_EARTH,
    RE_EARTH,
    acceleration,
    eccentric_state,
    finite_difference_column,
    period,
    propagate_state,
    reference_stm,
    sigma_pos_3d,
)


# An orbit eccentric enough that the defect cannot hide, with TCA well inside
# the window rather than at either end.
A_M = 10_000e3
ECC = 0.6
X0 = eccentric_state(A_M, ECC)
PERIOD = period(A_M)
T0 = 0.0
T_MID = 0.19 * PERIOD
T_TCA = 0.37 * PERIOD


def acc_fn(t, r, v):
    """Two-body + J2, matching the reference module's dynamics."""
    return acceleration(r, MU_EARTH, J2_EARTH, RE_EARTH)


def engine_stm(x0, t0, tf):
    return propagate_stm(acc_fn, x0[:3], x0[3:], (t0, tf),
                         mu=MU_EARTH, j2=J2_EARTH, radius=RE_EARTH)


def rel_matrix_error(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))
                 / np.max(np.abs(np.asarray(b))))


# ---------------------------------------------------------------------------
# A. Identity at the reference epoch
# ---------------------------------------------------------------------------

def test_zero_duration_gives_exact_identity():
    """Phi(t0, t0) = I, exactly -- not to a tolerance."""
    result = engine_stm(X0, T0, T0)
    np.testing.assert_array_equal(result.stm, np.eye(6))
    assert result.t0 == T0 and result.tf == T0


@pytest.mark.parametrize("epoch", (0.0, 1234.5, -900.0))
def test_identity_holds_at_any_reference_epoch(epoch):
    """The reference epoch is whatever the caller says it is."""
    result = engine_stm(X0, epoch, epoch)
    np.testing.assert_array_equal(result.stm, np.eye(6))


def test_stm_approaches_identity_as_the_interval_shrinks():
    """Continuity into the identity, so the zero case is not a special case."""
    previous = None
    for dt in (10.0, 1.0, 0.1, 1e-3):
        deviation = float(np.max(np.abs(engine_stm(X0, T0, T0 + dt).stm - np.eye(6))))
        if previous is not None:
            assert deviation < previous
        previous = deviation
    assert previous < 1e-2


def test_stm_records_the_epochs_it_was_built_between():
    """
    The reference and evaluation epochs must be recoverable from the result;
    an STM whose epochs are not stated cannot be checked by a caller.
    """
    result = engine_stm(X0, T0, T_TCA)
    assert result.t0 == T0
    assert result.tf == T_TCA
    assert result.to_dict()["t0_s"] == T0
    assert result.to_dict()["tf_s"] == T_TCA


# ---------------------------------------------------------------------------
# B. Composition
# ---------------------------------------------------------------------------

def test_composition_across_an_intermediate_epoch():
    """
    Phi(t2, t0) = Phi(t2, t1) Phi(t1, t0).

    Note the middle factor must be seeded with the state at t1 -- which is the
    very property P10-05 violates.  Composition is therefore a direct test of
    correct seeding, not only of integration accuracy.
    """
    x1 = propagate_state(X0, T0, T_MID, MU_EARTH, J2_EARTH, RE_EARTH)
    phi_10 = engine_stm(X0, T0, T_MID).stm
    phi_21 = engine_stm(x1, T_MID, T_TCA).stm
    phi_20 = engine_stm(X0, T0, T_TCA).stm

    assert rel_matrix_error(phi_21 @ phi_10, phi_20) < 1e-8


def test_composition_fails_when_the_middle_factor_is_misseeded():
    """
    The same composition, with the middle factor seeded from the *end* state
    instead of the intermediate one -- the exact shape of the P10-05 defect.
    It must not compose, or the invariant above would be vacuous.
    """
    x2 = propagate_state(X0, T0, T_TCA, MU_EARTH, J2_EARTH, RE_EARTH)
    phi_10 = engine_stm(X0, T0, T_MID).stm
    phi_21_bad = engine_stm(x2, T_MID, T_TCA).stm
    phi_20 = engine_stm(X0, T0, T_TCA).stm

    assert rel_matrix_error(phi_21_bad @ phi_10, phi_20) > 1e-2


# ---------------------------------------------------------------------------
# C. Against an independent variational integration and finite differences
# ---------------------------------------------------------------------------

def test_engine_stm_matches_an_independent_variational_integration():
    phi_ref, _ = reference_stm(X0, T0, T_TCA, MU_EARTH, J2_EARTH, RE_EARTH)
    assert rel_matrix_error(engine_stm(X0, T0, T_TCA).stm, phi_ref) < 1e-6


@pytest.mark.parametrize("column,delta", [(0, 50.0), (1, 50.0), (2, 50.0),
                                          (3, 0.05), (4, 0.05), (5, 0.05)])
def test_each_stm_column_matches_finite_difference_propagation(column, delta):
    """
    Column j of Phi is d x(tf) / d x0_j.  Recovered here by two full nonlinear
    propagations, which share no formulation with any STM.
    """
    phi = engine_stm(X0, T0, T_TCA).stm
    fd = finite_difference_column(X0, T0, T_TCA, column, delta,
                                  MU_EARTH, J2_EARTH, RE_EARTH)
    assert np.linalg.norm(phi[:, column] - fd) / np.linalg.norm(fd) < 1e-5


@pytest.mark.parametrize("scale", (0.1, 1.0, 10.0))
def test_stm_predicts_real_perturbations(scale):
    """dx(tf) ~= Phi dx0, against independently propagated perturbed states."""
    phi = engine_stm(X0, T0, T_TCA).stm
    x_end = propagate_state(X0, T0, T_TCA, MU_EARTH, J2_EARTH, RE_EARTH)
    rng = np.random.default_rng(20260825)

    for _ in range(4):
        dx0 = np.concatenate([rng.normal(0.0, scale, 3),
                              rng.normal(0.0, scale * 1e-3, 3)])
        dx_true = propagate_state(X0 + dx0, T0, T_TCA,
                                  MU_EARTH, J2_EARTH, RE_EARTH) - x_end
        assert (np.linalg.norm(phi @ dx0 - dx_true)
                / np.linalg.norm(dx_true)) < 1e-4


# ---------------------------------------------------------------------------
# D. Covariance consistency
# ---------------------------------------------------------------------------

def test_linear_covariance_matches_a_nonlinear_ensemble():
    """
    P(t) = Phi P0 Phi^T against a Monte-Carlo ensemble drawn from P0 and
    propagated nonlinearly.  Loose tolerance by design: the point is that the
    linear map reproduces the real spread, not that it is exact.
    """
    phi = engine_stm(X0, T0, T_TCA).stm
    p0 = np.diag([300.0 ** 2] * 3 + [0.3 ** 2] * 3)
    x_end = propagate_state(X0, T0, T_TCA, MU_EARTH, J2_EARTH, RE_EARTH)

    chol = np.linalg.cholesky(p0)
    rng = np.random.default_rng(3)
    deviations = np.array([
        propagate_state(X0 + chol @ rng.standard_normal(6), T0, T_TCA,
                        MU_EARTH, J2_EARTH, RE_EARTH) - x_end
        for _ in range(1500)
    ])
    p_mc = np.cov(deviations.T, bias=False)
    p_lin = phi @ p0 @ phi.T

    assert sigma_pos_3d(p_lin) == pytest.approx(sigma_pos_3d(p_mc), rel=0.05)


def test_covariance_propagation_uses_the_supplied_stm_unchanged():
    """
    P_tca = Phi P0 Phi^T, with no epoch fudge inside propagate_covariance.
    Pins that the epoch semantics live in the STM, where the fix belongs.
    """
    phi = engine_stm(X0, T0, T_TCA).stm
    p0 = StateCovariance.from_diagonal([250.0] * 3, [0.25] * 3, name="A")
    propagated = propagate_covariance(p0, phi, T_TCA)

    np.testing.assert_allclose(propagated.matrix, phi @ p0.matrix @ phi.T,
                               rtol=1e-12, atol=0.0)
    assert propagated.epoch_s == T_TCA


# ---------------------------------------------------------------------------
# E. The reference-epoch contract at the multi-object call site
# ---------------------------------------------------------------------------

def _eccentric_pair():
    """Two eccentric spacecraft whose paths bring them close inside the window."""
    from theseus.simulation.multi_object import SpacecraftDefinition

    common = dict(
        semi_major_axis_km=10_000.0,
        eccentricity=0.6,
        dry_mass_kg=1000.0,
        fuel_mass_kg=0.0,
        payload_mass_kg=0.0,
        hard_body_radius_m=10.0,
        sigma_pos_m=[300.0, 300.0, 300.0],
        sigma_vel_m_s=[0.3, 0.3, 0.3],
    )
    a = SpacecraftDefinition(id="ECC-A", name="Eccentric-A", color="#ff9900",
                             inclination_deg=35.0, true_anomaly_deg=0.0, **common)
    b = SpacecraftDefinition(id="ECC-B", name="Eccentric-B", color="#3388ff",
                             inclination_deg=-35.0, true_anomaly_deg=1.0, **common)
    return a, b


def _run_pair(monkeypatch=None):
    from theseus.simulation.multi_object import MultiObjectEnvironment

    env = MultiObjectEnvironment(
        central_body="Earth",
        screening_threshold_km=200.0,
        coarse_dt_s=10.0,
        enable_j2=False,
        enable_drag=False,
    )
    sc_a, sc_b = _eccentric_pair()
    # The pair meets near apoapsis, at t ~ 4973 s, where |r| is 16 000 km
    # against 4 000 km at the covariance epoch -- a 12 000 km separation
    # between the two candidate seed states, so no test below can pass by
    # accident.
    return env.simulate([sc_a, sc_b], t_start=0.0, t_end=6000.0, output_dt=10.0)


def test_multi_object_seeds_the_stm_at_the_covariance_epoch(monkeypatch):
    """
    The headline P10-05 assertion.

    Every state handed to ``propagate_stm`` must be the object's state at the
    covariance epoch (the propagation start), never its state at TCA.  Checked
    by recording the real call arguments during a real simulation.
    """
    from theseus.simulation import multi_object

    calls = []
    original = multi_object.propagate_stm

    def spy(acc_fn, r0, v0, t_span, **kwargs):
        calls.append({"r0": np.array(r0, dtype=float),
                      "v0": np.array(v0, dtype=float),
                      "t_span": (float(t_span[0]), float(t_span[1]))})
        return original(acc_fn, r0, v0, t_span, **kwargs)

    monkeypatch.setattr(multi_object, "propagate_stm", spy)
    result = _run_pair()

    assert result.conjunctions, "fixture must produce a conjunction to analyse"
    assert calls, "the covariance path must propagate an STM"

    # The state at the covariance epoch, taken from the simulation's own
    # recorded history rather than recomputed here.
    epoch_states = {}
    for obj in result.objects:
        first = obj.state_history[0]
        assert first["time_seconds"] == pytest.approx(0.0, abs=1e-9)
        epoch_states[obj.definition.id] = (
            np.array(first["position"], dtype=float),
            np.array(first["velocity"], dtype=float))

    for call in calls:
        reference_epoch, evaluation_epoch = call["t_span"]
        assert reference_epoch == pytest.approx(0.0, abs=1e-9)
        assert evaluation_epoch > reference_epoch

        matches = [
            oid for oid, (r, v) in epoch_states.items()
            if np.allclose(call["r0"], r, rtol=0, atol=1e-6)
            and np.allclose(call["v0"], v, rtol=0, atol=1e-9)
        ]
        assert matches, (
            "STM seeded with a state that is not any object's state at the "
            "covariance epoch t = 0"
        )


def test_multi_object_does_not_seed_the_stm_with_the_tca_state(monkeypatch):
    """
    The complement, stated separately so the failure mode is named.

    On an eccentric orbit the TCA state is far from the epoch state -- here the
    radius differs by more than 10 000 km -- so this is unambiguous.
    """
    from theseus.simulation import multi_object

    seeds = []
    original = multi_object.propagate_stm

    def spy(acc_fn, r0, v0, t_span, **kwargs):
        seeds.append(np.array(r0, dtype=float))
        return original(acc_fn, r0, v0, t_span, **kwargs)

    monkeypatch.setattr(multi_object, "propagate_stm", spy)
    result = _run_pair()

    assert result.conjunctions
    tca_radii = []
    for event in result.conjunctions:
        for obj in result.objects:
            states = obj.state_history
            times = np.array([s["time_seconds"] for s in states])
            idx = int(np.argmin(np.abs(times - event.tca_s)))
            tca_radii.append(float(np.linalg.norm(states[idx]["position"])))

    epoch_radii = [float(np.linalg.norm(o.state_history[0]["position"]))
                   for o in result.objects]

    # The fixture must actually separate the two epochs, or the test proves
    # nothing.
    assert max(tca_radii) - max(epoch_radii) > 1e6

    for seed in seeds:
        seed_radius = float(np.linalg.norm(seed))
        assert min(abs(seed_radius - r) for r in epoch_radii) < 1.0


# ---------------------------------------------------------------------------
# F. End-to-end covariance correctness through the multi-object pipeline
# ---------------------------------------------------------------------------

def test_multi_object_covariance_at_tca_matches_an_independent_reference():
    """
    The physical consequence, end to end.

    The simulation's reported B-plane uncertainty is compared against a
    covariance propagated with an independently integrated STM, seeded at the
    covariance epoch.  With the defect present the reported sigmas are far too
    small; the tolerance here is far wider than the integration error and far
    narrower than the defect.
    """
    result = _run_pair()
    assert result.conjunctions

    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)
    assert event.b_plane_sigma_major_m is not None, \
        "fixture must produce a B-plane-applicable encounter"

    epoch = {o.definition.id: (
        np.array(o.state_history[0]["position"], dtype=float),
        np.array(o.state_history[0]["velocity"], dtype=float))
        for o in result.objects}

    # Independent Phi(TCA, t0) for each object, from the reference module.
    covariances = []
    for object_id in (event.spacecraft_a_id, event.spacecraft_b_id):
        r_epoch, v_epoch = epoch[object_id]
        phi, _ = reference_stm(np.concatenate([r_epoch, v_epoch]),
                               0.0, event.tca_s, MU_EARTH, 0.0, RE_EARTH)
        p0 = np.diag([300.0 ** 2] * 3 + [0.3 ** 2] * 3)
        covariances.append(phi @ p0 @ phi.T)

    rel_pos_cov = covariances[0][:3, :3] + covariances[1][:3, :3]

    from theseus.uncertainty.b_plane import project_covariance_to_b_plane
    reference_projection = project_covariance_to_b_plane(
        rel_pos_cov=rel_pos_cov,
        r_rel=np.array(event.r_rel_m, dtype=float),
        v_rel=np.array(event.v_rel_m_s, dtype=float),
    )

    assert event.b_plane_sigma_major_m == pytest.approx(
        reference_projection.sigma_major, rel=0.05)
    assert event.b_plane_sigma_minor_m == pytest.approx(
        reference_projection.sigma_minor, rel=0.05)


def test_multi_object_reported_uncertainty_is_physically_large_enough():
    """
    A blunt scale check that does not depend on the projection at all.

    Over half an eccentric orbit, a 300 m per-axis initial position
    uncertainty grows substantially.  The defect collapsed it to a fraction of
    the correct value, so a floor well below the correct answer still
    separates the two.
    """
    result = _run_pair()
    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)

    combined = np.array(event.b_plane_covariance_m2, dtype=float)
    sigma = math.sqrt(max(np.trace(combined), 0.0))

    # Independently computed correct value for this fixture: 16 439 m.
    # The defect produced 7 435 m.  The floor sits between them, far from
    # both, so it separates the two without encoding either.
    assert sigma > 12_000.0


# ---------------------------------------------------------------------------
# G. The single-pair Phase 10 orchestration was already correct
# ---------------------------------------------------------------------------

def _single_pair_inputs():
    """An eccentric pair for the ``run_uncertainty_conjunction_analysis`` path."""
    from theseus.uncertainty.covariance import StateCovariance

    x_a = eccentric_state(A_M, ECC, inc_deg=35.0)
    x_b = eccentric_state(A_M, ECC, inc_deg=-35.0)

    def make(x0):
        cache = {}

        def state(t):
            t = float(t)
            if t not in cache:
                cache[t] = propagate_state(x0, 0.0, t, MU_EARTH, 0.0, RE_EARTH)
            return cache[t]

        return (lambda t: state(t)[:3]), (lambda t: state(t)[3:])

    pos_a, vel_a = make(x_a)
    pos_b, vel_b = make(x_b)
    cov = StateCovariance.from_diagonal([300.0] * 3, [0.3] * 3, name="P0")
    return (x_a, x_b), (pos_a, vel_a, pos_b, vel_b), cov


def test_single_pair_orchestration_seeds_the_stm_at_the_covariance_epoch(monkeypatch):
    """
    ``theseus/uncertainty/results.py`` already took ``pos_fn(t_start)`` as the
    STM seed, so it was never affected by P10-05.  Pinned so the two Phase 10
    entry points cannot drift apart again.
    """
    from theseus.uncertainty import results as results_module

    (x_a, x_b), fns, cov = _single_pair_inputs()
    pos_a, vel_a, pos_b, vel_b = fns

    seeds = []
    original = results_module.propagate_stm

    def spy(acc_fn, r0, v0, t_span, **kwargs):
        seeds.append((np.array(r0, dtype=float), np.array(v0, dtype=float),
                      float(t_span[0]), float(t_span[1])))
        return original(acc_fn, r0, v0, t_span, **kwargs)

    monkeypatch.setattr(results_module, "propagate_stm", spy)
    result = results_module.run_uncertainty_conjunction_analysis(
        pos_fn_a=pos_a, vel_fn_a=vel_a, pos_fn_b=pos_b, vel_fn_b=vel_b,
        initial_cov_a=cov, initial_cov_b=cov,
        t_start=0.0, t_end=6000.0,
        mu=MU_EARTH, j2=0.0, radius=RE_EARTH,
        screening_threshold_m=200e3, coarse_dt=30.0,
    )

    assert result.conjunction_found, "fixture must produce a conjunction"
    assert len(seeds) == 2

    for seed_r, seed_v, reference_epoch, evaluation_epoch in seeds:
        assert reference_epoch == 0.0
        assert evaluation_epoch == pytest.approx(result.tca_s)
        assert any(np.allclose(seed_r, x[:3], rtol=0, atol=1e-6)
                   and np.allclose(seed_v, x[3:], rtol=0, atol=1e-9)
                   for x in (x_a, x_b))


def test_single_pair_covariance_matches_an_independent_reference():
    """
    The complete Phase 10 path, checked end to end against independently
    integrated STMs.  Confirms the orchestration's covariance at TCA is the
    one the variational equations actually give.
    """
    from theseus.uncertainty import results as results_module

    (x_a, x_b), fns, cov = _single_pair_inputs()
    pos_a, vel_a, pos_b, vel_b = fns

    result = results_module.run_uncertainty_conjunction_analysis(
        pos_fn_a=pos_a, vel_fn_a=vel_a, pos_fn_b=pos_b, vel_fn_b=vel_b,
        initial_cov_a=cov, initial_cov_b=cov,
        t_start=0.0, t_end=6000.0,
        mu=MU_EARTH, j2=0.0, radius=RE_EARTH,
        screening_threshold_m=200e3, coarse_dt=30.0,
    )
    assert result.conjunction_found

    for x0, propagated in ((x_a, result.cov_a_tca), (x_b, result.cov_b_tca)):
        phi, _ = reference_stm(x0, 0.0, result.tca_s, MU_EARTH, 0.0, RE_EARTH)
        expected = phi @ cov.matrix @ phi.T
        assert sigma_pos_3d(propagated.matrix) == pytest.approx(
            sigma_pos_3d(expected), rel=1e-3)


# ---------------------------------------------------------------------------
# H. Known limitation, pinned rather than fixed
# ---------------------------------------------------------------------------

def test_backward_time_span_is_not_supported():
    """
    ``propagate_stm`` does not integrate backwards.  Every Phase 10 call site
    uses a forward span, so this remains unsupported.

    P10-05 recorded it as a known limitation and left it: given tf < t0 the
    integrator stepped forward anyway and the returned STM did not correspond
    to the requested span.  The final P10 sweep measured what that actually
    produced --

        ||Phi_back - I||_F           = 0.000000e+00
        ||Phi_back @ Phi_fwd - I||_F = 1.046464e+03

    -- the identity, exactly.  A covariance mapped through it came back
    unchanged, as though propagated with no dynamics at all: a wrong answer
    shaped exactly like a right one, which is the failure mode B-2 and P10-08
    both exist to remove.  It is still unsupported; it now says so.
    """
    forward = engine_stm(X0, T0, 600.0)
    x_600 = forward.nominal_state_tf

    with pytest.raises(ValueError) as excinfo:
        propagate_stm(acc_fn, x_600[:3], x_600[3:], (600.0, T0),
                      mu=MU_EARTH, j2=J2_EARTH, radius=RE_EARTH)
    assert "backward propagation" in str(excinfo.value)

    # The route the error message recommends still works.
    inverse = np.linalg.inv(forward.stm)
    assert np.allclose(inverse @ forward.stm, np.eye(6), atol=1e-6)
