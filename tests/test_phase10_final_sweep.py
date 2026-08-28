"""
FINAL P10 SWEEP -- the remaining known issues, each measured before being
classified and each fix pinned here.

P10-12 is the last numerical finding. This module covers the sweep that
followed it: every issue still on the P10 list was reproduced rather than
recalled, and the ones classified FIX NOW are regression-tested below.

    A  unconditional J2 at the multi-object STM call site   FIX NOW
    B  max_evals declared, documented, never read           FIX NOW
    C  non-finite request fields at the API                 FIX NOW
    D  custom hard-body radius precedence                   ALREADY CLOSED (P10-04)
    E  solar radiation pressure construction                FIX NOW
    F  backward STM integration returns the identity        FIX NOW
    G  covariance diagonal test uses a dimensional floor    FIX NOW
    H  B-1                                                  KNOWN LIMITATION

C is the most serious: it is B-2's defect at the boundary B-2 did not cover.
F is the most insidious: a wrong answer shaped exactly like a right one.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest
from fastapi.testclient import TestClient

from theseus.bodies.catalog import EARTH
from theseus.dynamics.force_model import CompositeForceModel
from theseus.dynamics.gravity import J2Perturbation, PointMassGravity
from theseus.server.app import app
from theseus.simulation.multi_object import (
    MultiObjectEnvironment,
    SolarRadiationPressureUnavailable,
    SpacecraftDefinition,
)
from theseus.uncertainty.covariance import (
    DIAGONAL_NOISE_RTOL,
    CovarianceValidationError,
    StateCovariance,
)
from theseus.uncertainty.state_transition import (
    analytic_jacobian_describes,
    propagate_stm,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app, raise_server_exceptions=False)


CONJUNCTION_REQUEST = {
    "object_a_alt_km": 400.0, "object_a_inc_deg": 51.6, "object_a_phase_deg": 0.0,
    "object_b_alt_km": 400.05, "object_b_inc_deg": 55.0, "object_b_phase_deg": 0.02,
    "analysis_duration_hours": 2.0, "screening_threshold_km": 100.0,
    "coarse_dt_s": 30.0,
}


def post_raw(client, path, payload, field, literal):
    """POST a body containing a non-standard JSON float literal."""
    body = json.dumps({**payload, field: 0.0}).replace(
        f'"{field}": 0.0', f'"{field}": {literal}')
    return client.post(path, content=body,
                       headers={"content-type": "application/json"})


# ---------------------------------------------------------------------------
# A. The STM call site asks for the J2 its force model actually applies
# ---------------------------------------------------------------------------

def _acceleration(force_model):
    return lambda t, r, v: force_model.compute_acceleration(t, r, v, 1000.0)


def test_analytic_jacobian_is_rejected_when_the_j2_argument_does_not_match():
    """
    The property the fix rests on, established directly on P10-06's guard: the
    analytic Jacobian is accepted only when the j2 it is built with matches the
    force model that produced the trajectory.
    """
    r = np.array([EARTH.radius + 700e3, 0.0, 0.0])
    v = np.array([0.0, 7500.0, 0.0])
    gravity_only = _acceleration(CompositeForceModel([PointMassGravity(EARTH)]))
    with_j2 = _acceleration(CompositeForceModel(
        [PointMassGravity(EARTH), J2Perturbation(EARTH)]))

    assert analytic_jacobian_describes(gravity_only, 0.0, r, v,
                                       mu=EARTH.mu, j2=0.0, radius=EARTH.radius)
    assert not analytic_jacobian_describes(gravity_only, 0.0, r, v,
                                           mu=EARTH.mu, j2=EARTH.J2,
                                           radius=EARTH.radius)
    assert analytic_jacobian_describes(with_j2, 0.0, r, v,
                                       mu=EARTH.mu, j2=EARTH.J2,
                                       radius=EARTH.radius)
    assert not analytic_jacobian_describes(with_j2, 0.0, r, v,
                                           mu=EARTH.mu, j2=0.0,
                                           radius=EARTH.radius)


def test_the_stm_call_site_uses_the_force_models_own_j2():
    """
    The call site passed ``j2=self.body.J2`` whatever the force model
    contained, so a two-body-only model -- which ``enable_j2=False`` produces
    and the environment API exposes -- was handed a J2 Jacobian, rejected by
    P10-06's guard, and silently downgraded to the numerical Jacobian.  Never
    wrong, always slower, and it made the guard fire on a case it was not
    written for.
    """
    env = MultiObjectEnvironment(central_body="Earth",
                                 enable_j2=False, enable_drag=False)
    sc = SpacecraftDefinition(id="A", name="A",
                              semi_major_axis_km=7078.137, inclination_deg=51.6)
    gravity_only = env._build_force_model(sc)
    assert env._analytic_jacobian_covers(gravity_only) is True
    assert env._analytic_j2_for(gravity_only, EARTH.J2) == 0.0

    env_j2 = MultiObjectEnvironment(central_body="Earth",
                                    enable_j2=True, enable_drag=False)
    with_j2 = env_j2._build_force_model(sc)
    assert env_j2._analytic_jacobian_covers(with_j2) is True
    assert env_j2._analytic_j2_for(with_j2, EARTH.J2) == pytest.approx(EARTH.J2)


def test_a_gravity_only_model_now_reaches_the_analytic_jacobian():
    """The consequence: the analytic path is actually taken."""
    env = MultiObjectEnvironment(central_body="Earth",
                                 enable_j2=False, enable_drag=False)
    sc = SpacecraftDefinition(id="A", name="A",
                              semi_major_axis_km=7078.137, inclination_deg=51.6)
    force_model = env._build_force_model(sc)
    j2 = env._analytic_j2_for(force_model, EARTH.J2)

    result = propagate_stm(
        _acceleration(force_model),
        np.array([EARTH.radius + 700e3, 0.0, 0.0]),
        np.array([0.0, 7500.0, 0.0]),
        (0.0, 300.0), mu=EARTH.mu, j2=j2, radius=EARTH.radius, dt=30.0)
    assert result.method.startswith("analytic_jacobian")

    # With the old unconditional argument it would have been the numerical one.
    downgraded = propagate_stm(
        _acceleration(force_model),
        np.array([EARTH.radius + 700e3, 0.0, 0.0]),
        np.array([0.0, 7500.0, 0.0]),
        (0.0, 300.0), mu=EARTH.mu, j2=EARTH.J2, radius=EARTH.radius, dt=30.0)
    assert downgraded.method.startswith("numerical_jacobian")

    # And the two agree, which is why this was never a correctness problem.
    assert np.allclose(result.stm, downgraded.stm, rtol=1e-7, atol=1e-9)


# ---------------------------------------------------------------------------
# B. max_evals says what it does
# ---------------------------------------------------------------------------

def test_max_evals_is_inert_and_now_says_so():
    """
    Documented as "maximum function evaluations for integration" and never
    read.  A caller who set it believed they had bounded the work.
    """
    from theseus.uncertainty.b_plane import project_covariance_to_b_plane
    from theseus.uncertainty.collision_probability import (
        REDUCTION_PANEL_CAP, compute_collision_probability)

    bpu = project_covariance_to_b_plane(
        rel_pos_cov=np.diag([500.0 ** 2, 50.0 ** 2, 1.0]),
        r_rel=np.array([0.0, 0.0, 0.0]), v_rel=np.array([0.0, 0.0, 1.0e4]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tight = compute_collision_probability(bpu, 10.0, max_evals=1)
        deprecations = [w for w in caught
                        if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert "never had any effect" in str(deprecations[0].message)
    assert str(REDUCTION_PANEL_CAP) in str(deprecations[0].message)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loose = compute_collision_probability(bpu, 10.0, max_evals=10 ** 9)
    # Still inert -- the warning is the fix, not a behaviour change.
    assert tight.probability == loose.probability
    assert tight.iterations == loose.iterations


def test_the_default_max_evals_does_not_warn():
    """No noise for callers who never touched it."""
    from theseus.uncertainty.b_plane import project_covariance_to_b_plane
    from theseus.uncertainty.collision_probability import compute_collision_probability

    bpu = project_covariance_to_b_plane(
        rel_pos_cov=np.diag([500.0 ** 2, 50.0 ** 2, 1.0]),
        r_rel=np.array([0.0, 0.0, 0.0]), v_rel=np.array([0.0, 0.0, 1.0e4]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_collision_probability(bpu, 10.0)
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# C. B-2's guarantee, extended to the API boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,literal", [
    ("object_a_alt_km", "NaN"),
    ("object_a_inc_deg", "Infinity"),
    ("object_b_phase_deg", "-Infinity"),
    ("screening_threshold_km", "NaN"),
    ("analysis_duration_hours", "-Infinity"),
    ("coarse_dt_s", "NaN"),
])
def test_a_non_finite_request_field_is_rejected_with_a_diagnostic(
        client, field, literal):
    """
    B-2: "a non-finite state must never produce zero conjunctions or another
    valid negative result".  JSON permits NaN and Infinity, Python's parser
    accepts them, and Pydantic passed them into float fields.  Measured before
    this guard:

        object_a_alt_km        = NaN       -> HTTP 500 Internal Server Error
        screening_threshold_km = NaN       -> HTTP 200, "events": []
        analysis_duration_hours= -Infinity -> HTTP 200, "events": []

    The 200s are the B-2 defect exactly: a non-finite input returning a
    successful analysis reporting no conjunctions.
    """
    response = post_raw(client, "/api/simulate/conjunction",
                        CONJUNCTION_REQUEST, field, literal)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "NON_FINITE_REQUEST_FIELD"
    assert field in body["message"]
    # A non-finite request has no analysis, not an empty one.
    assert "events" not in body
    assert "analysis_status" not in body


def test_a_non_finite_field_nested_in_a_covariance_is_rejected(client):
    """The guard must reach into sub-models and into lists, not just scalars."""
    payload = {**CONJUNCTION_REQUEST, "hard_body_radius_m": 15.0,
               "cov_a": {"sigma_pos_km": [0.1, 0.1, 0.0]}}
    body = json.dumps(payload).replace(
        '"sigma_pos_km": [0.1, 0.1, 0.0]', '"sigma_pos_km": [0.1, 0.1, NaN]')
    response = client.post("/api/simulate/conjunction/risk", content=body,
                           headers={"content-type": "application/json"})

    assert response.status_code == 422
    assert response.json()["error"] == "NON_FINITE_REQUEST_FIELD"
    assert "sigma_pos_km[2]" in response.json()["message"]


def test_a_finite_request_is_unaffected(client):
    """The control: the guard must not cost a valid request anything."""
    response = client.post("/api/simulate/conjunction", json=CONJUNCTION_REQUEST)
    assert response.status_code == 200
    assert len(response.json()["events"]) == 2


def test_an_ordinary_type_error_keeps_its_ordinary_shape(client):
    """Only the non-finite case is re-labelled; other 422s are untouched."""
    response = client.post("/api/simulate/conjunction",
                           json={**CONJUNCTION_REQUEST,
                                 "object_a_alt_km": "banana"})
    assert response.status_code == 422
    assert response.json().get("error") != "NON_FINITE_REQUEST_FIELD"
    assert response.json()["detail"][0]["type"] == "float_parsing"


def test_the_rejection_body_is_encodable(client):
    """
    The 422 became a 500 because FastAPI echoes the rejected input back and
    could not encode the NaN that caused the rejection.  The diagnostic is
    worthless if it cannot be delivered.
    """
    response = post_raw(client, "/api/simulate/conjunction",
                        CONJUNCTION_REQUEST, "object_a_alt_km", "NaN")
    assert response.status_code == 422
    text = response.text
    json.loads(text)                       # must round-trip
    assert "NaN" not in text.replace('"nan"', "")   # only as a quoted string


# ---------------------------------------------------------------------------
# D. Custom hard-body radius precedence -- already closed under P10-04
# ---------------------------------------------------------------------------

def test_custom_hbr_precedence_is_documented_and_unchanged():
    """
    A caller supplying both a geometry and a combined radius gets the combined
    radius.  That is a real precedence rule rather than a bug, it was
    documented under P10-04 with provenance recorded on both objects, and this
    sweep changes nothing about it -- pinned so it cannot drift silently.
    """
    from theseus.conjunction.geometry import CollisionGeometry
    from theseus.uncertainty.hard_body import compute_hard_body_radius

    a = CollisionGeometry.from_radius(7.0, name="A")
    b = CollisionGeometry.from_radius(1.5, name="B")

    assert compute_hard_body_radius(obj_a=a, obj_b=b).combined_hbr_m == 8.5

    override = compute_hard_body_radius(obj_a=a, obj_b=b, custom_hbr_m=3.0)
    assert override.combined_hbr_m == 3.0
    assert override.object_a.collision_radius_m == 1.5
    assert override.object_b.collision_radius_m == 1.5
    assert "caller-supplied" in override.object_a.source
    assert "split equally" in override.object_a.source


# ---------------------------------------------------------------------------
# E. Solar radiation pressure refuses rather than crashes
# ---------------------------------------------------------------------------

def test_enable_srp_raises_a_named_error_rather_than_a_type_error():
    """
    ``SolarRadiationPressure`` needs an ephemeris to locate the Sun and the
    call site never supplied one, so every ``enable_srp=True`` raised

        TypeError: SolarRadiationPressure.__init__() missing 1 required
        positional argument: 'ephemeris'

    Passing an ephemeris here would switch on a perturbation this project has
    never validated end to end.  A documented option that has never run is not
    made trustworthy by making it run.
    """
    env = MultiObjectEnvironment(central_body="Earth", enable_srp=True)
    sc = SpacecraftDefinition(id="A", name="A",
                              semi_major_axis_km=7078.137, inclination_deg=51.6)

    with pytest.raises(SolarRadiationPressureUnavailable) as excinfo:
        env._build_force_model(sc)
    assert "not available" in str(excinfo.value)
    assert "known limitation" in str(excinfo.value)
    assert isinstance(excinfo.value, NotImplementedError)


def test_enable_srp_is_a_501_not_a_500(client):
    """
    The API documents ``enable_srp`` and returned "Internal Server Error" for
    it.  501 says the request was fine and the feature is not implemented,
    which is the true statement.
    """
    payload = {
        "spacecraft": [
            {"id": "A", "name": "A", "semi_major_axis_km": 6778.137,
             "inclination_deg": 51.6},
            {"id": "B", "name": "B", "semi_major_axis_km": 6778.187,
             "inclination_deg": 55.0}],
        "duration_hours": 0.2, "dt_sec": 30.0, "enable_srp": True}

    response = client.post("/api/simulate/environment", json=payload)
    assert response.status_code == 501
    assert response.json()["error"] == "SOLAR_RADIATION_PRESSURE_UNAVAILABLE"

    payload["enable_srp"] = False
    assert client.post("/api/simulate/environment", json=payload).status_code == 200


# ---------------------------------------------------------------------------
# F. Backward STM propagation refuses rather than returning the identity
# ---------------------------------------------------------------------------

def test_backward_stm_propagation_is_refused():
    """
    The integrator steps forward only and terminated immediately when
    ``tf < t0``, returning Phi still at its initial value.  Measured over a
    600 s span:

        ||Phi_back - I||_F           = 0.000000e+00
        ||Phi_back @ Phi_fwd - I||_F = 1.046464e+03

    A covariance mapped through that identity came back unchanged, as though
    propagated with no dynamics at all -- a wrong answer shaped exactly like a
    right one.
    """
    acceleration = _acceleration(CompositeForceModel([PointMassGravity(EARTH)]))
    r0 = np.array([EARTH.radius + 700e3, 0.0, 0.0])
    v0 = np.array([0.0, 7500.0, 0.0])

    with pytest.raises(ValueError) as excinfo:
        propagate_stm(acceleration, r0, v0, (600.0, 0.0), mu=EARTH.mu, j2=0.0,
                      radius=EARTH.radius, dt=10.0)
    message = str(excinfo.value)
    assert "backward propagation" in message
    assert "identity" in message


def test_forward_propagation_and_the_documented_inverse_still_work():
    """The refusal must not cost the supported direction anything."""
    acceleration = _acceleration(CompositeForceModel([PointMassGravity(EARTH)]))
    r0 = np.array([EARTH.radius + 700e3, 0.0, 0.0])
    v0 = np.array([0.0, 7500.0, 0.0])

    forward = propagate_stm(acceleration, r0, v0, (0.0, 600.0), mu=EARTH.mu,
                            j2=0.0, radius=EARTH.radius, dt=10.0)
    stm = np.asarray(forward.stm)
    # A symplectic flow has unit determinant; this is the check the silent
    # identity would also have passed, which is why it went unnoticed.
    assert np.linalg.det(stm) == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.norm(stm - np.eye(6)) > 1.0

    # The route the error message recommends.
    inverse = np.linalg.inv(stm)
    assert np.linalg.norm(inverse @ stm - np.eye(6)) < 1e-6


def test_a_zero_length_span_is_still_the_identity():
    """t0 == tf is genuinely the identity and must stay allowed."""
    acceleration = _acceleration(CompositeForceModel([PointMassGravity(EARTH)]))
    result = propagate_stm(acceleration,
                           np.array([EARTH.radius + 700e3, 0.0, 0.0]),
                           np.array([0.0, 7500.0, 0.0]),
                           (300.0, 300.0), mu=EARTH.mu, j2=0.0,
                           radius=EARTH.radius, dt=10.0)
    assert result.method == "identity_t0"
    assert np.array_equal(np.asarray(result.stm), np.eye(6))


def test_the_phase_10_pipeline_never_propagates_backward():
    """
    Why the refusal costs no production behaviour: the covariance epoch is the
    window start and TCA is inside the window, so the span is always forward.
    """
    env = MultiObjectEnvironment(central_body="Earth", screening_threshold_km=100.0,
                                 coarse_dt_s=5.0)
    a = SpacecraftDefinition(id="A", name="A", semi_major_axis_km=6778.137,
                             inclination_deg=51.6, true_anomaly_deg=0.0)
    b = SpacecraftDefinition(id="B", name="B", semi_major_axis_km=6778.187,
                             inclination_deg=55.0, true_anomaly_deg=0.02)
    result = env.simulate([a, b], t_start=0.0, t_end=3000.0, output_dt=5.0)

    assert result.conjunctions
    for conjunction in result.conjunctions:
        assert conjunction.tca_s > 0.0


# ---------------------------------------------------------------------------
# G. The covariance diagonal test is dimensionless
# ---------------------------------------------------------------------------

def _diagonal_probe(relative_value, length_scale):
    """The same physical covariance, expressed in a different length unit."""
    matrix = np.diag([100.0, 100.0, 100.0, 0.01, 0.01, 0.01]) * length_scale ** 2
    matrix[0, 0] = relative_value * (100.0 * length_scale ** 2)
    return matrix


def _accepts(matrix):
    try:
        StateCovariance(matrix=matrix.copy(), name="probe")
        return True
    except CovarianceValidationError:
        return False


@pytest.mark.parametrize("relative_value,expected", [
    (-1e-16, True),     # roundoff: absorbed
    (-1e-9, False),     # genuine: rejected
    (-1e-6, False),     # genuine: rejected
])
def test_the_diagonal_verdict_does_not_depend_on_the_unit_system(
        relative_value, expected):
    """
    The threshold was the literal ``-1e-15`` -- an absolute value in whatever
    units the caller chose.  A variance is not dimensionless, so the same
    physical covariance changed verdict with the unit system:

        length unit  metres     P[0,0] = -1.0e-14  -> REJECTED
        length unit  kilometres P[0,0] = -1.0e-20  -> accepted
        length unit  megametres P[0,0] = -1.0e-26  -> accepted

    The same defect P10-10 removed from the branch criteria and P10-11 from
    the PSD test.
    """
    verdicts = [_accepts(_diagonal_probe(relative_value, scale))
                for scale in (1.0, 1e-3, 1e-6, 1e3)]
    assert len(set(verdicts)) == 1, "the verdict changed with the unit system"
    assert verdicts[0] is expected


def test_the_old_absolute_threshold_would_have_disagreed_with_itself():
    """
    Proof that the parametrised test above has teeth: the retired rule gives
    different answers for the same physical matrix.
    """
    old_rule = lambda m: not (m[0, 0] < -1e-15)
    verdicts = [old_rule(_diagonal_probe(-1e-9, scale))
                for scale in (1.0, 1e-3, 1e-6)]
    assert len(set(verdicts)) > 1, \
        "if the old rule were scale-free there was nothing to fix"


def test_the_velocity_block_is_judged_against_velocity_variances():
    """
    A single scale for the whole 6x6 would reintroduce the unit dependence:
    with sigma_r = 1 km and sigma_v = 1e-4 m/s the position block is twenty
    orders of magnitude larger, and any negative velocity variance would be
    absorbed as position-block roundoff.
    """
    matrix = np.diag([1e6, 1e6, 1e6, 1e-8, 1e-8, 1e-8])
    matrix[3, 3] = -1e-9 * 1e-8
    assert _accepts(matrix) is False

    # And genuine velocity-block roundoff is still absorbed.
    tiny = np.diag([1e6, 1e6, 1e6, 1e-8, 1e-8, 1e-8])
    tiny[3, 3] = -1e-16 * 1e-8
    assert _accepts(tiny) is True


def test_the_noise_floor_is_reported_in_the_rejection_message():
    """A rejection the caller cannot check is not much better than a crash."""
    with pytest.raises(CovarianceValidationError) as excinfo:
        StateCovariance(matrix=_diagonal_probe(-1e-6, 1.0), name="probe")
    message = str(excinfo.value)
    assert "rx" in message
    assert f"{DIAGONAL_NOISE_RTOL:.0e}" in message
    assert "largest variance in its block" in message


# ---------------------------------------------------------------------------
# H. Nothing in the sweep disturbed what came before
# ---------------------------------------------------------------------------

def test_the_recorded_production_probabilities_survive_the_sweep(client):
    """The values recorded under P10-04 and carried through every finding since."""
    for hbr, expected in ((15.0, 3.3437648624900262e-06),
                          (1.9, 5.365051327225372e-08),
                          (0.3, 1.3375480817909169e-09)):
        data = client.post("/api/simulate/conjunction/risk",
                           json={**CONJUNCTION_REQUEST,
                                 "hard_body_radius_m": hbr}).json()
        assert data["analysis_status"] == "COMPLETE"
        assert data["collision_probability"]["probability"] == pytest.approx(
            expected, rel=1e-12)
        assert data["collision_probability"]["converged"] is True


def test_b1_remains_deliberately_unfixed():
    """
    B-1 is unchanged by explicit instruction.  Pinned so that "unfixed" stays
    a decision rather than becoming an accident: if this test starts failing,
    something touched it.
    """
    from theseus.conjunction.tca import find_all_tca_with_diagnostics
    from theseus.orbital.circular import circular_orbit_from_altitude

    def orbit(altitude_km, inclination_deg, phase_deg):
        return circular_orbit_from_altitude(
            altitude_m=altitude_km * 1e3, body_radius_m=EARTH.radius,
            inclination_rad=math.radians(inclination_deg),
            phase_rad=math.radians(phase_deg), mu=EARTH.mu,
            raan_rad=0.0, epoch_s=0.0)

    a, b = orbit(400.0, 51.6, 0.0), orbit(400.05, 55.0, 0.02)
    pos_a, vel_a = a.as_callables()
    pos_b, vel_b = b.as_callables()

    r_rel = np.asarray(pos_a(0.0)) - np.asarray(pos_b(0.0))
    v_rel = np.asarray(vel_a(0.0)) - np.asarray(vel_b(0.0))
    # The range-rate at the window start is non-zero for this pair, so the
    # boundary behaviour B-1 describes is not exercised here; recorded rather
    # than asserted about, because B-1 must not be altered.
    assert math.isfinite(float(np.dot(r_rel, v_rel)))

    events, _ = find_all_tca_with_diagnostics(pos_a, vel_a, pos_b, vel_b,
                                              0.0, 7200.0, n_samples=200,
                                              tol=1e-6)
    assert len(events) == 2
