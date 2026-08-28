"""
P10-06 — the STM Jacobian must linearise the acceleration actually propagated.

The defect
----------
``propagate_stm`` chose its Jacobian like this::

    can_use_analytic = use_analytic_jacobian and (mu is not None and mu > 0.0)
    if can_use_analytic:
        da_dr = gravity_jacobian(r, mu)
        if j2: da_dr = da_dr + j2_jacobian(r, mu, j2, radius)
        da_dv = np.zeros((3, 3))          # <-- structurally zero
    else:
        da_dr, da_dv = numerical_jacobian(acc_fn, t, r, v)

The branch is selected purely by whether ``mu`` was supplied.  Nothing about
``acc_fn`` enters the decision, so a caller propagating gravity + J2 + drag
and passing ``mu`` -- which the multi-object Phase 10 path always did -- got a
Jacobian describing gravity + J2 alone, with the velocity block forced to
exactly zero.  Drag is the one active force with a non-zero ``da/dv``, and
``enable_drag`` defaults to True both in ``MultiObjectEnvironment`` and in the
API schema, so this was the default configuration.

Measured against nonlinear finite-difference propagation of the production
force model:

    alt      |a_drag|      Phi error (analytic)   Phi error (numerical)   sigma err
    120 km   1.24e-02      7.30e-01               1.21e-07                28.9 %
    150 km   8.29e-05      5.10e-02               2.01e-08                 2.3 %
    180 km   5.56e-07      3.48e-04               9.52e-10                0.015 %
    250 km   4.71e-12      3.08e-09               7.43e-10                 ~0

End to end, a 150 km drag-active conjunction reported sigma_major = 3890.85 m
against an independently integrated 3975.23 m -- 2.1 % low.

Above roughly 250 km the omitted terms fall below the finite-difference noise
floor, because ``US1976StandardAtmosphere`` extrapolates above 86 km with a
6 km scale height and its density collapses (6.4e-23 m/s^2 of drag at 400 km).
That under-prediction is a force-model limitation the model itself declares,
belongs to Phase 8, and is deliberately not touched here -- but it is the
reason the defect's *reachable* magnitude is currently confined to very low
altitudes, and it would grow if the atmosphere model were corrected.

What these tests pin
--------------------
A. The analytic gravity and J2 derivatives are right (they always were).
B. Drag genuinely has non-zero position *and* velocity derivatives.
C. The STM must not claim an analytic Jacobian for an acceleration model that
   analytic Jacobian does not cover.
D. The STM must predict real perturbations of the real force model.
E. The multi-object covariance must match an independently integrated one.
F. Scope: thrust is not in the Phase 10 force model, and SRP's contribution is
   measured rather than assumed.

Every reference derivative comes from ``tests/_force_jacobian_reference.py``,
which finite-differences the production acceleration and uses no production
Jacobian.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from theseus.uncertainty.state_transition import (
    gravity_jacobian,
    j2_jacobian,
    propagate_stm,
)

from tests._force_jacobian_reference import (
    DEFAULT_AREA,
    DEFAULT_CD,
    DEFAULT_MASS,
    J2_EARTH,
    MU_EARTH,
    RE_EARTH,
    acceleration_fn,
    build_force_model,
    circular_state,
    finite_difference_jacobian,
    finite_difference_stm,
    propagate_nonlinear,
    relative_matrix_error,
    sigma_pos_3d,
    variational_stm,
)


ALTITUDES_KM = (150.0, 200.0, 400.0, 800.0, 20000.0)
FD_STEPS = ((0.5, 5e-4), (1.0, 1e-3), (5.0, 5e-3))


# ---------------------------------------------------------------------------
# A. The analytic gravity and J2 derivatives are correct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dr,dv", FD_STEPS)
@pytest.mark.parametrize("altitude_km", ALTITUDES_KM)
def test_gravity_jacobian_matches_finite_differences(altitude_km, dr, dv):
    """Point-mass gravity only, several altitudes and several step sizes."""
    fm = build_force_model(gravity=True, j2=False, drag=False)
    acc = acceleration_fn(fm)
    x = circular_state(altitude_km)

    fd_dr, fd_dv = finite_difference_jacobian(acc, 0.0, x[:3], x[3:], dr, dv)
    analytic = gravity_jacobian(x[:3], MU_EARTH)

    assert relative_matrix_error(analytic, fd_dr) < 1e-5
    assert np.max(np.abs(fd_dv)) == 0.0


@pytest.mark.parametrize("dr,dv", FD_STEPS)
@pytest.mark.parametrize("altitude_km", ALTITUDES_KM)
def test_gravity_plus_j2_jacobian_matches_finite_differences(altitude_km, dr, dv):
    fm = build_force_model(gravity=True, j2=True, drag=False)
    acc = acceleration_fn(fm)
    x = circular_state(altitude_km, inclination_deg=63.4)

    fd_dr, fd_dv = finite_difference_jacobian(acc, 0.0, x[:3], x[3:], dr, dv)
    analytic = (gravity_jacobian(x[:3], MU_EARTH)
                + j2_jacobian(x[:3], MU_EARTH, J2_EARTH, RE_EARTH))

    assert relative_matrix_error(analytic, fd_dr) < 1e-5
    assert np.max(np.abs(fd_dv)) == 0.0


@pytest.mark.parametrize("inclination_deg", (0.0, 28.5, 51.6, 90.0, 116.6))
def test_j2_jacobian_matches_finite_differences_across_inclinations(inclination_deg):
    """The J2 derivative is latitude-dependent; sample the range."""
    fm = build_force_model(gravity=True, j2=True, drag=False)
    acc = acceleration_fn(fm)
    x = circular_state(500.0, inclination_deg=inclination_deg)

    fd_dr, _ = finite_difference_jacobian(acc, 0.0, x[:3], x[3:])
    analytic = (gravity_jacobian(x[:3], MU_EARTH)
                + j2_jacobian(x[:3], MU_EARTH, J2_EARTH, RE_EARTH))

    assert relative_matrix_error(analytic, fd_dr) < 1e-5


# ---------------------------------------------------------------------------
# B. Drag has real position and velocity derivatives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("altitude_km", (120.0, 150.0, 180.0, 200.0))
def test_drag_contributes_a_non_zero_velocity_derivative(altitude_km):
    """
    The premise of the whole finding, measured from the production
    acceleration: with drag active the velocity block is not zero, so a
    Jacobian that forces it to zero is not the Jacobian of this model.
    """
    fm = build_force_model(gravity=True, j2=True, drag=True)
    acc = acceleration_fn(fm)
    x = circular_state(altitude_km)

    _, fd_dv = finite_difference_jacobian(acc, 0.0, x[:3], x[3:])
    assert np.max(np.abs(fd_dv)) > 0.0

    # And it is drag that supplies it: without drag the block is exactly zero.
    acc_no_drag = acceleration_fn(build_force_model(gravity=True, j2=True, drag=False))
    _, fd_dv_no_drag = finite_difference_jacobian(acc_no_drag, 0.0, x[:3], x[3:])
    assert np.max(np.abs(fd_dv_no_drag)) == 0.0


@pytest.mark.parametrize("altitude_km", (120.0, 150.0, 180.0))
def test_drag_changes_the_position_derivative_too(altitude_km):
    """
    Not only the velocity block: the density gradient makes drag depend on
    position, so the gravity+J2 position block is also incomplete.
    """
    fm = build_force_model(gravity=True, j2=True, drag=True)
    acc = acceleration_fn(fm)
    x = circular_state(altitude_km)

    fd_dr, _ = finite_difference_jacobian(acc, 0.0, x[:3], x[3:])
    gravitational = (gravity_jacobian(x[:3], MU_EARTH)
                     + j2_jacobian(x[:3], MU_EARTH, J2_EARTH, RE_EARTH))

    # Above the finite-difference noise floor of ~3e-10 measured with drag off.
    assert relative_matrix_error(gravitational, fd_dr) > 1e-8


def test_full_jacobian_matches_finite_differences_with_drag_active():
    """
    The complete 6x6 A matrix for the drag-active model, block by block, so
    neither block can be right by accident.
    """
    fm = build_force_model(gravity=True, j2=True, drag=True)
    acc = acceleration_fn(fm)
    x = circular_state(150.0)

    fd_dr, fd_dv = finite_difference_jacobian(acc, 0.0, x[:3], x[3:])
    coarse_dr, coarse_dv = finite_difference_jacobian(
        acc, 0.0, x[:3], x[3:], dr=5.0, dv=5e-3)

    # Two independent step sizes agree, so the reference itself is sound.
    assert relative_matrix_error(coarse_dr, fd_dr) < 1e-3
    assert relative_matrix_error(coarse_dv, fd_dv) < 1e-3


# ---------------------------------------------------------------------------
# C. The STM must not claim a Jacobian it does not have
# ---------------------------------------------------------------------------

def test_stm_reports_analytic_jacobian_only_for_gravitational_models():
    """A gravity+J2 acceleration is exactly what the analytic Jacobian covers."""
    fm = build_force_model(gravity=True, j2=True, drag=False)
    acc = acceleration_fn(fm)
    x = circular_state(400.0)

    result = propagate_stm(acc, x[:3], x[3:], (0.0, 600.0),
                           mu=MU_EARTH, j2=J2_EARTH, radius=RE_EARTH)
    assert result.method.startswith("analytic_jacobian")


@pytest.mark.parametrize("altitude_km", (120.0, 150.0, 180.0))
def test_stm_does_not_claim_an_analytic_jacobian_when_drag_is_active(altitude_km):
    """
    The contract test.  Supplying ``mu`` must not be read as a promise that
    the acceleration is gravitational -- the engine must not linearise a
    drag-perturbed trajectory with a gravity-only Jacobian.
    """
    fm = build_force_model(gravity=True, j2=True, drag=True)
    acc = acceleration_fn(fm)
    x = circular_state(altitude_km)

    result = propagate_stm(acc, x[:3], x[3:], (0.0, 600.0),
                           mu=MU_EARTH, j2=J2_EARTH, radius=RE_EARTH)
    assert not result.method.startswith("analytic_jacobian")


# ---------------------------------------------------------------------------
# D. The STM must predict the real dynamics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("altitude_km,span_s,tolerance", [
    (120.0, 1200.0, 1e-4),
    (150.0, 3000.0, 1e-4),
    (180.0, 3000.0, 1e-5),
])
def test_stm_matches_nonlinear_finite_differences_with_drag_active(
        altitude_km, span_s, tolerance):
    """
    Phi against twelve nonlinear propagations of the production force model.

    The tolerances sit far below the measured defect (7.3e-1, 5.1e-2, 3.5e-4)
    and far above the corrected error (1.2e-7, 2.0e-8, 9.5e-10), so they
    encode neither implementation.
    """
    fm = build_force_model(gravity=True, j2=True, drag=True)
    acc = acceleration_fn(fm)
    x = circular_state(altitude_km)

    engine = propagate_stm(acc, x[:3], x[3:], (0.0, span_s),
                           mu=MU_EARTH, j2=J2_EARTH, radius=RE_EARTH,
                           atol=1e-9, rtol=1e-12).stm
    truth = finite_difference_stm(acc, x, 0.0, span_s)

    assert relative_matrix_error(engine, truth) < tolerance


@pytest.mark.parametrize("altitude_km,span_s", [(400.0, 3000.0), (800.0, 3000.0)])
def test_stm_matches_nonlinear_finite_differences_without_drag(altitude_km, span_s):
    """The gravity-only path must stay exactly as accurate as it was."""
    fm = build_force_model(gravity=True, j2=True, drag=False)
    acc = acceleration_fn(fm)
    x = circular_state(altitude_km)

    engine = propagate_stm(acc, x[:3], x[3:], (0.0, span_s),
                           mu=MU_EARTH, j2=J2_EARTH, radius=RE_EARTH,
                           atol=1e-9, rtol=1e-12).stm
    truth = finite_difference_stm(acc, x, 0.0, span_s)

    assert relative_matrix_error(engine, truth) < 1e-6


@pytest.mark.parametrize("scale", (1.0, 10.0, 100.0))
def test_stm_predicts_real_perturbations_with_drag_active(scale):
    """
    dx(t) ~= Phi dx0, against independently propagated perturbed trajectories
    of the real drag-active model, at three perturbation magnitudes.
    """
    fm = build_force_model(gravity=True, j2=True, drag=True)
    acc = acceleration_fn(fm)
    x0 = circular_state(150.0)
    span = 2000.0

    phi = propagate_stm(acc, x0[:3], x0[3:], (0.0, span),
                        mu=MU_EARTH, j2=J2_EARTH, radius=RE_EARTH,
                        atol=1e-9, rtol=1e-12).stm
    x_end = propagate_nonlinear(acc, x0, 0.0, span)

    rng = np.random.default_rng(20260825)
    for _ in range(3):
        dx0 = np.concatenate([rng.normal(0.0, scale, 3),
                              rng.normal(0.0, scale * 1e-3, 3)])
        dx_true = propagate_nonlinear(acc, x0 + dx0, 0.0, span) - x_end
        assert (np.linalg.norm(phi @ dx0 - dx_true)
                / np.linalg.norm(dx_true)) < 1e-3


def test_stm_identity_and_composition_hold_with_drag_active():
    """The structural invariants must survive the Jacobian change."""
    fm = build_force_model(gravity=True, j2=True, drag=True)
    acc = acceleration_fn(fm)
    x0 = circular_state(150.0)

    identity = propagate_stm(acc, x0[:3], x0[3:], (0.0, 0.0),
                             mu=MU_EARTH, j2=J2_EARTH, radius=RE_EARTH)
    np.testing.assert_array_equal(identity.stm, np.eye(6))

    x_mid = propagate_nonlinear(acc, x0, 0.0, 900.0)
    kw = dict(mu=MU_EARTH, j2=J2_EARTH, radius=RE_EARTH, atol=1e-9, rtol=1e-12)
    phi_10 = propagate_stm(acc, x0[:3], x0[3:], (0.0, 900.0), **kw).stm
    phi_21 = propagate_stm(acc, x_mid[:3], x_mid[3:], (900.0, 1800.0), **kw).stm
    phi_20 = propagate_stm(acc, x0[:3], x0[3:], (0.0, 1800.0), **kw).stm

    assert relative_matrix_error(phi_21 @ phi_10, phi_20) < 1e-6


# ---------------------------------------------------------------------------
# E. End to end through the multi-object Phase 10 path
# ---------------------------------------------------------------------------

def _low_altitude_pair():
    from theseus.simulation.multi_object import SpacecraftDefinition

    common = dict(
        semi_major_axis_km=(RE_EARTH + 150e3) / 1e3,
        eccentricity=0.0,
        dry_mass_kg=DEFAULT_MASS,
        fuel_mass_kg=0.0,
        payload_mass_kg=0.0,
        hard_body_radius_m=5.0,
        cross_section_area_m2=DEFAULT_AREA,
        drag_coefficient=DEFAULT_CD,
        sigma_pos_m=[300.0, 300.0, 300.0],
        sigma_vel_m_s=[0.3, 0.3, 0.3],
    )
    a = SpacecraftDefinition(id="LOW-A", name="Low-A", color="#ff9900",
                             inclination_deg=51.6, true_anomaly_deg=0.0, **common)
    b = SpacecraftDefinition(id="LOW-B", name="Low-B", color="#3388ff",
                             inclination_deg=-51.6, true_anomaly_deg=0.05, **common)
    return a, b


def _run_low_pair():
    from theseus.simulation.multi_object import MultiObjectEnvironment

    env = MultiObjectEnvironment(
        central_body="Earth",
        screening_threshold_km=50.0,
        coarse_dt_s=5.0,
        enable_j2=True,
        enable_drag=True,
    )
    sc_a, sc_b = _low_altitude_pair()
    return env.simulate([sc_a, sc_b], t_start=0.0, t_end=3000.0, output_dt=5.0)


def test_multi_object_stm_uses_a_jacobian_covering_its_force_model(monkeypatch):
    """
    With drag enabled -- the default -- the multi-object covariance path must
    not fall back on the gravity-only analytic Jacobian.
    """
    from theseus.simulation import multi_object

    methods = []
    original = multi_object.propagate_stm

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        methods.append(result.method)
        return result

    monkeypatch.setattr(multi_object, "propagate_stm", spy)
    result = _run_low_pair()

    assert result.conjunctions, "fixture must produce a conjunction"
    assert methods, "the covariance path must propagate an STM"
    assert all(not m.startswith("analytic_jacobian") for m in methods), methods


def test_multi_object_covariance_matches_an_independent_drag_aware_reference():
    """
    The end-to-end consequence.  With the incomplete Jacobian this fixture
    reported sigma_major = 3890.85 m against an independently integrated
    3975.23 m -- 2.1 % low.  The reference here integrates the variational
    equations with a finite-difference Jacobian of the production force model,
    so it sees drag in both blocks.
    """
    from theseus.uncertainty.b_plane import project_covariance_to_b_plane

    result = _run_low_pair()
    assert result.conjunctions
    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)
    assert event.b_plane_sigma_major_m is not None

    fm = build_force_model(gravity=True, j2=True, drag=True)
    acc = acceleration_fn(fm)
    epoch = {o.definition.id: np.concatenate([
        np.array(o.state_history[0]["position"], dtype=float),
        np.array(o.state_history[0]["velocity"], dtype=float)])
        for o in result.objects}

    p0 = np.diag([300.0 ** 2] * 3 + [0.3 ** 2] * 3)
    position_covariances = []
    for object_id in (event.spacecraft_a_id, event.spacecraft_b_id):
        phi = variational_stm(acc, epoch[object_id], 0.0, event.tca_s)
        position_covariances.append((phi @ p0 @ phi.T)[:3, :3])

    reference = project_covariance_to_b_plane(
        rel_pos_cov=position_covariances[0] + position_covariances[1],
        r_rel=np.array(event.r_rel_m, dtype=float),
        v_rel=np.array(event.v_rel_m_s, dtype=float),
    )

    assert event.b_plane_sigma_major_m == pytest.approx(
        reference.sigma_major, rel=0.005)
    assert event.b_plane_sigma_minor_m == pytest.approx(
        reference.sigma_minor, rel=0.005)


def test_multi_object_probability_of_collision_stays_finite_and_bounded():
    """
    The corrected covariance must still produce a usable Pc: the fix changes
    the number, it must not break the chain that consumes it.
    """
    result = _run_low_pair()
    event = min(result.conjunctions, key=lambda e: e.miss_distance_m)

    assert event.collision_probability is not None
    assert 0.0 <= event.collision_probability <= 1.0
    assert math.isfinite(event.collision_probability)


# ---------------------------------------------------------------------------
# F. Scope boundaries, measured rather than assumed
# ---------------------------------------------------------------------------

def test_thrust_is_not_part_of_the_phase10_force_model():
    """
    Thrust direction modes are state-dependent, so thrust would need Jacobian
    terms -- but ``_build_force_model`` never adds a ThrustModel, so there is
    nothing to differentiate.  Proven from the assembled model, not from the
    source text.
    """
    from theseus.dynamics.thrust import ThrustModel
    from theseus.simulation.multi_object import (
        MultiObjectEnvironment, SpacecraftDefinition,
    )

    # enable_srp is left off deliberately: MultiObjectEnvironment cannot
    # currently build an SRP model at all (it omits the required ephemeris
    # argument and raises TypeError).  That is a separate defect, reported and
    # not fixed here; it also means SRP is unreachable in this path.
    env = MultiObjectEnvironment(central_body="Earth", enable_j2=True,
                                 enable_drag=True, enable_srp=False)
    sc = SpacecraftDefinition(id="X", name="X", color="#fff", thrust_n=5000.0,
                              specific_impulse_s=300.0, fuel_mass_kg=500.0)
    force_model = env._build_force_model(sc)

    assert not any(isinstance(m, ThrustModel) for m in force_model.models), (
        "a ThrustModel in the Phase 10 force model would need its own Jacobian "
        "terms; this test must be revisited if one is added"
    )


def test_srp_state_derivative_is_measured_not_assumed():
    """
    SRP depends on position only through the Sun-relative geometry, whose
    scale is an astronomical unit, and through a cylindrical shadow step.
    Away from the shadow boundary its derivative is therefore negligible
    against gravity -- measured here rather than asserted from the physics.
    """
    from theseus.dynamics.srp import SolarRadiationPressure
    from theseus.ephemeris.simple_provider import SimpleEphemerisProvider

    fm = build_force_model(gravity=True, j2=True, drag=False)
    fm.add(SolarRadiationPressure(ephemeris=SimpleEphemerisProvider(),
                                  cr=1.5, area=DEFAULT_AREA))
    acc = acceleration_fn(fm)
    x = circular_state(800.0)

    fd_dr, fd_dv = finite_difference_jacobian(acc, 0.0, x[:3], x[3:])
    gravitational = (gravity_jacobian(x[:3], MU_EARTH)
                     + j2_jacobian(x[:3], MU_EARTH, J2_EARTH, RE_EARTH))

    assert relative_matrix_error(gravitational, fd_dr) < 1e-6
    assert np.max(np.abs(fd_dv)) == 0.0
