"""
P9-05 — the conjunction API must not implement orbital mechanics.

``/api/simulate/conjunction`` and ``/api/simulate/conjunction/risk`` each
contained the same 39 lines constructing circular-orbit position and velocity
functions inline: mean motion, circular speed, and the rotation from orbital
elements into inertial coordinates, open-coded twice in the HTTP layer.

Both now go through ``CircularOrbitStates``, which reads its perifocal basis
back out of ``theseus.orbital.conversions.elements_to_state`` rather than
re-deriving the rotation.

The reference states in these tests are built independently from
``OrbitalElements`` + ``elements_to_state``, and separately from the literal
formula the API used to contain.  Two endpoints agreeing with each other
would prove nothing; what is checked is that both agree with the engine's own
element conversion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from theseus.server.app import app, _conjunction_orbit_pair
from theseus.server.app import ConjunctionRequest, ConjunctionRiskRequest
from theseus.bodies.catalog import get_body
from theseus.orbital.circular import CircularOrbitStates, circular_orbit_from_altitude
from theseus.orbital.elements import OrbitalElements
from theseus.orbital.conversions import elements_to_state


client = TestClient(app)


# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------

def reference_state_via_elements(radius_m, inc_rad, phase_rad, mu, t):
    """
    Independent reference: the engine's element conversion, evaluated at the
    true anomaly a circular orbit reaches at time t.  Uses no code from
    theseus.orbital.circular.
    """
    n = math.sqrt(mu / radius_m ** 3)
    oe = OrbitalElements(a=radius_m, e=0.0, i=inc_rad, raan=0.0, argp=0.0,
                         nu=phase_rad + n * t, mu=mu)
    return elements_to_state(oe)


def reference_state_legacy_formula(radius_m, inc_rad, phase_rad, mu, t):
    """
    The literal formula the API used to contain, reproduced here so the
    correction can be shown not to have changed any value.  Not used by any
    engine code path.
    """
    n = math.sqrt(mu / radius_m ** 3)
    v_circ = math.sqrt(mu / radius_m)
    th = n * t + phase_rad
    pos = np.array([
        radius_m * math.cos(th),
        radius_m * math.sin(th) * math.cos(inc_rad),
        radius_m * math.sin(th) * math.sin(inc_rad),
    ])
    vel = np.array([
        -v_circ * math.sin(th),
        v_circ * math.cos(th) * math.cos(inc_rad),
        v_circ * math.cos(th) * math.sin(inc_rad),
    ])
    return pos, vel


# Agreement is at the floating-point noise floor, not bit-exact: the shared
# path forms a linear combination of basis vectors where the old code
# evaluated products directly.  1e-6 m on a 7e6 m orbit is 1.4e-13 relative,
# roughly three orders of magnitude looser than the ~2e-9 m actually measured.
POS_TOL_M = 1e-6
VEL_TOL_MS = 1e-9


# ---------------------------------------------------------------------------
# 1, 5: the shared construction reproduces the engine's own conversion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alt_km,inc_deg,phase_deg", [
    (400.0, 51.6, 0.0),
    (400.01, 51.6, 0.01),
    (405.0, 97.0, 10.0),
    (1200.0, 0.0, 359.9),
    (250.0, 180.0, -45.0),
    (35786.0, 0.0, 123.456),
])
def test_shared_construction_matches_engine_element_conversion(alt_km, inc_deg, phase_deg):
    """Units, frame and convention are the engine's, not a private copy."""
    body = get_body("Earth")
    radius = body.radius + alt_km * 1e3
    inc = math.radians(inc_deg)
    phase = math.radians(phase_deg)

    orbit = circular_orbit_from_altitude(
        altitude_m=alt_km * 1e3, body_radius_m=body.radius,
        inclination_rad=inc, phase_rad=phase, mu=body.mu,
    )
    assert orbit.radius_m == pytest.approx(radius, rel=0, abs=0)

    for t in (0.0, 1.0, 137.5, 1830.0, 7200.0):
        p_shared, v_shared = orbit.state_at(t)
        p_ref, v_ref = reference_state_via_elements(radius, inc, phase, body.mu, t)
        p_legacy, v_legacy = reference_state_legacy_formula(radius, inc, phase, body.mu, t)

        assert np.linalg.norm(p_shared - p_ref) < POS_TOL_M
        assert np.linalg.norm(v_shared - v_ref) < VEL_TOL_MS
        # And unchanged relative to what the API produced before.
        assert np.linalg.norm(p_shared - p_legacy) < POS_TOL_M
        assert np.linalg.norm(v_shared - v_legacy) < VEL_TOL_MS


def test_circular_orbit_invariants():
    """Radius, speed and energy are constant; the orbit closes on its period."""
    body = get_body("Earth")
    orbit = circular_orbit_from_altitude(
        altitude_m=400e3, body_radius_m=body.radius,
        inclination_rad=math.radians(51.6), phase_rad=0.3, mu=body.mu,
    )
    radii, speeds = [], []
    for t in np.linspace(0.0, orbit.period_s, 200):
        p, v = orbit.state_at(float(t))
        radii.append(float(np.linalg.norm(p)))
        speeds.append(float(np.linalg.norm(v)))
    assert np.std(radii) < 1e-6
    assert np.std(speeds) < 1e-9
    assert radii[0] == pytest.approx(orbit.radius_m, abs=1e-6)
    assert speeds[0] == pytest.approx(orbit.speed_m_s, abs=1e-9)

    p0, v0 = orbit.state_at(0.0)
    pT, vT = orbit.state_at(orbit.period_s)
    assert np.linalg.norm(pT - p0) < 1e-3
    assert np.linalg.norm(vT - v0) < 1e-6

    # Position and velocity are orthogonal for a circular orbit.
    assert abs(float(np.dot(p0, v0))) < 1e-3


def test_raan_convention_is_explicit_and_settable():
    """
    The endpoints fix RAAN at zero because the schema exposes no node
    parameter.  The engine object does not hard-code that.
    """
    body = get_body("Earth")
    base = dict(altitude_m=400e3, body_radius_m=body.radius,
                inclination_rad=math.radians(60.0), phase_rad=0.0, mu=body.mu)
    at_zero = circular_orbit_from_altitude(**base)
    rotated = circular_orbit_from_altitude(**base, raan_rad=math.radians(90.0))

    # At the ascending node both sit on their node line, 90 deg apart.
    p0, _ = at_zero.state_at(0.0)
    p1, _ = rotated.state_at(0.0)
    assert p0[0] == pytest.approx(at_zero.radius_m, rel=1e-12)
    assert abs(p0[1]) < 1e-6 and abs(p0[2]) < 1e-6
    assert p1[1] == pytest.approx(rotated.radius_m, rel=1e-12)
    assert abs(p1[0]) < 1e-6 and abs(p1[2]) < 1e-6


def test_invalid_orbit_parameters_are_rejected():
    body = get_body("Earth")
    with pytest.raises(ValueError):
        CircularOrbitStates(radius_m=0.0, inclination_rad=0.0, phase_rad=0.0, mu=body.mu)
    with pytest.raises(ValueError):
        CircularOrbitStates(radius_m=7e6, inclination_rad=0.0, phase_rad=0.0, mu=-1.0)
    with pytest.raises(ValueError):
        CircularOrbitStates(radius_m=7e6, inclination_rad=float("nan"),
                            phase_rad=0.0, mu=body.mu)


# ---------------------------------------------------------------------------
# 1, 6: both endpoints build the same states from the same input
# ---------------------------------------------------------------------------

ORBIT_FIELDS = dict(
    object_a_alt_km=400.0, object_a_inc_deg=51.6, object_a_phase_deg=0.0,
    object_b_alt_km=400.05, object_b_inc_deg=55.0, object_b_phase_deg=0.0,
    central_body="Earth",
)


def test_both_endpoints_use_one_construction_path():
    """
    The same request fields must give the same orbits regardless of which
    request model carries them, and both must match the independent reference.
    """
    body = get_body("Earth")
    conj_req = ConjunctionRequest(**ORBIT_FIELDS)
    risk_req = ConjunctionRiskRequest(**ORBIT_FIELDS)

    a1, b1 = _conjunction_orbit_pair(conj_req, body)
    a2, b2 = _conjunction_orbit_pair(risk_req, body)

    for o1, o2 in ((a1, a2), (b1, b2)):
        assert o1.radius_m == o2.radius_m
        assert o1.inclination_rad == o2.inclination_rad
        assert o1.phase_rad == o2.phase_rad
        assert o1.raan_rad == o2.raan_rad == 0.0
        assert o1.mu == o2.mu == body.mu
        assert o1.epoch_s == o2.epoch_s == 0.0

    for t in (0.0, 900.0, 2776.805194, 7200.0):
        for o, alt, inc, ph in (
            (a1, 400.0, 51.6, 0.0),
            (b1, 400.05, 55.0, 0.0),
        ):
            p, v = o.state_at(t)
            p_ref, v_ref = reference_state_via_elements(
                body.radius + alt * 1e3, math.radians(inc), math.radians(ph),
                body.mu, t,
            )
            assert np.linalg.norm(p - p_ref) < POS_TOL_M
            assert np.linalg.norm(v - v_ref) < VEL_TOL_MS


def test_altitude_is_above_the_body_mean_radius():
    """The altitude convention is unchanged: radius = body.radius + altitude."""
    for body_name in ("Earth", "Mars"):
        body = get_body(body_name)
        req = ConjunctionRequest(**{**ORBIT_FIELDS, "central_body": body_name})
        a, b = _conjunction_orbit_pair(req, body)
        assert a.radius_m == pytest.approx(body.radius + 400.0e3, abs=1e-9)
        assert b.radius_m == pytest.approx(body.radius + 400.05e3, abs=1e-9)
        assert a.mu == body.mu


# ---------------------------------------------------------------------------
# 2, 3: endpoint results are unchanged
# ---------------------------------------------------------------------------

def _conj_payload(**over):
    # Object B carries a small phase offset so the pair is not exactly
    # coincident at t = 0.  With both phases at zero the relative position is
    # perpendicular to the relative velocity *exactly* at the window boundary,
    # which makes r_rel . v_rel a floating-point zero there; see
    # test_boundary_stationary_point_is_the_only_permitted_difference.
    payload = dict(
        object_a_alt_km=400.0, object_a_inc_deg=51.6, object_a_phase_deg=0.0,
        object_b_alt_km=400.05, object_b_inc_deg=55.0, object_b_phase_deg=0.02,
        central_body="Earth", analysis_duration_hours=2.0,
        screening_threshold_km=100.0, coarse_dt_s=30.0,
    )
    payload.update(over)
    return payload


def test_conjunction_endpoint_matches_a_directly_driven_analysis():
    """
    Drive ConjunctionAnalysis directly with independently constructed state
    functions and require the endpoint to reproduce it exactly.
    """
    from theseus.conjunction.analysis import ConjunctionAnalysis

    body = get_body("Earth")
    payload = _conj_payload()

    def make(alt_km, inc_deg, phase_deg):
        radius = body.radius + alt_km * 1e3
        inc = math.radians(inc_deg)
        ph = math.radians(phase_deg)
        return (lambda t: reference_state_via_elements(radius, inc, ph, body.mu, t)[0],
                lambda t: reference_state_via_elements(radius, inc, ph, body.mu, t)[1])

    pos_a, vel_a = make(400.0, 51.6, 0.0)
    pos_b, vel_b = make(400.05, 55.0, 0.02)
    expected = ConjunctionAnalysis(
        screening_threshold_m=100.0e3, coarse_dt=30.0,
    ).analyse(pos_a, vel_a, pos_b, vel_b, 0.0, 2.0 * 3600.0)

    got = client.post("/api/simulate/conjunction", json=payload).json()

    assert got["summary"]["total_events"] == len(expected.events)
    assert got["accounting"]["candidate_intervals"] == expected.accounting.candidate_intervals
    assert got["accounting"]["tca_attempts"] == expected.accounting.tca_attempts
    assert got["accounting"]["accepted_conjunctions"] == expected.accounting.accepted_conjunctions

    for got_ev, exp_ev in zip(got["events"], expected.events):
        assert got_ev["tca"]["tca_s"] == pytest.approx(exp_ev.tca_result.tca, abs=1e-6)
        assert got_ev["tca"]["miss_distance_m"] == pytest.approx(
            exp_ev.tca_result.miss_distance, rel=1e-9, abs=1e-6,
        )
        assert got_ev["encounter_type"] == exp_ev.encounter_type


def test_risk_endpoint_still_reaches_phase_10():
    payload = _conj_payload()
    data = client.post("/api/simulate/conjunction/risk", json=payload).json()

    assert data["analysis_status"] == "COMPLETE"
    assert data["conjunction_found"] is True
    summary = data["conjunction_summary"]
    assert isinstance(summary["tca_s"], float) and not isinstance(summary["tca_s"], bool)
    assert 0.0 < summary["tca_s"] < 7200.0
    assert summary["miss_distance_km"] is not None
    pc = data["collision_probability"]["probability"]
    assert pc is not None and 0.0 <= pc <= 1.0
    assert data["risk_assessment"]["level"] in ("LOW", "ELEVATED", "HIGH", "CRITICAL")
    assert len(data["calculation_steps"]) == 14


def test_both_endpoints_report_the_same_tca_for_the_same_request():
    """
    Not a substitute for the independent checks above, but the two endpoints
    must not disagree about the geometry they were given.
    """
    payload = _conj_payload()
    conj = client.post("/api/simulate/conjunction", json=payload).json()
    risk = client.post("/api/simulate/conjunction/risk", json=payload).json()

    assert conj["events"], "expected a conjunction for this geometry"
    closest = min(conj["events"], key=lambda e: e["tca"]["miss_distance_m"])
    assert risk["conjunction_summary"]["tca_s"] == pytest.approx(
        closest["tca"]["tca_s"], abs=1e-6,
    )
    assert risk["conjunction_summary"]["miss_distance_m"] == pytest.approx(
        closest["tca"]["miss_distance_m"], rel=1e-9, abs=1e-6,
    )


# ---------------------------------------------------------------------------
# 4: validation behaviour unchanged
# ---------------------------------------------------------------------------

ROUTES = ("/api/simulate/conjunction", "/api/simulate/conjunction/risk")


@pytest.mark.parametrize("bad", [
    {"object_a_alt_km": "not-a-number"},
    {"object_a_inc_deg": None},
])
def test_schema_invalid_input_is_rejected_with_422(bad):
    """Malformed request bodies are still rejected by the request schema."""
    payload = _conj_payload(**bad)
    for route in ROUTES:
        resp = client.post(route, json=payload)
        assert resp.status_code == 422, f"{route} returned {resp.status_code} for {bad}"


@pytest.mark.parametrize("bad,exc", [
    ({"central_body": "Vulcan"}, KeyError),
    ({"screening_threshold_km": 0.0}, ValueError),
    ({"coarse_dt_s": 0.0}, ValueError),
])
def test_engine_invalid_input_behaviour_is_unchanged(bad, exc):
    """
    Values that pass the schema but the engine rejects still raise out of the
    endpoint rather than producing a result.

    This documents existing behaviour, which P9-05 does not change: these
    endpoints let engine exceptions propagate instead of translating them into
    a 4xx response. That is a pre-existing API-layer gap, out of scope here,
    and is recorded in the report rather than silently altered.
    """
    payload = _conj_payload(**bad)
    for route in ROUTES:
        with pytest.raises(exc):
            client.post(route, json=payload)


def test_valid_input_still_succeeds():
    """The negative cases above must not be passing for the wrong reason."""
    for route in ROUTES:
        resp = client.post(route, json=_conj_payload())
        assert resp.status_code == 200


def test_negative_altitude_below_the_surface_is_rejected_by_the_engine():
    """A radius at or below zero cannot form an orbit."""
    body = get_body("Earth")
    with pytest.raises(ValueError):
        circular_orbit_from_altitude(
            altitude_m=-body.radius, body_radius_m=body.radius,
            inclination_rad=0.0, phase_rad=0.0, mu=body.mu,
        )


# ---------------------------------------------------------------------------
# 7: collision geometry still propagates through the shared path
# ---------------------------------------------------------------------------

def test_collision_geometry_propagates_through_the_shared_construction():
    """
    P9-03's geometry must still reach events built from the shared orbit
    objects, and the clearance must follow from the reported miss distance.
    """
    from theseus.conjunction.analysis import ConjunctionAnalysis
    from theseus.conjunction.geometry import CollisionGeometry, CollisionStatus

    body = get_body("Earth")
    req = ConjunctionRequest(**_conj_payload())
    orbit_a, orbit_b = _conjunction_orbit_pair(req, body)
    pos_a, vel_a = orbit_a.as_callables()
    pos_b, vel_b = orbit_b.as_callables()

    result = ConjunctionAnalysis(100.0e3, coarse_dt=30.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, 7200.0,
        geometry_a=CollisionGeometry.from_radius(120.0, "A"),
        geometry_b=CollisionGeometry.from_radius(80.0, "B"),
        object_a_id="A", object_b_id="B",
    )
    assert result.events
    for ev in result.events:
        assert ev.object_a_id == "A" and ev.object_b_id == "B"
        assert ev.collision is not None and ev.collision.is_evaluated
        assert ev.collision.combined_hard_body_radius_m == pytest.approx(200.0)
        assert ev.clearance_m == pytest.approx(
            ev.miss_distance_m - 200.0, rel=1e-12,
        )
        expected = (CollisionStatus.INTERSECTION if ev.miss_distance_m < 200.0
                    else CollisionStatus.NO_INTERSECTION)
        assert ev.collision.status is expected


# ---------------------------------------------------------------------------
# Memoisation safety, matching the pattern used by the interpolator
# ---------------------------------------------------------------------------

def test_repeated_and_interleaved_evaluations_are_consistent():
    body = get_body("Earth")
    orbit = circular_orbit_from_altitude(
        altitude_m=400e3, body_radius_m=body.radius,
        inclination_rad=math.radians(51.6), phase_rad=0.2, mu=body.mu,
    )
    for t in (0.0, 100.0, 100.0, 900.0, 100.0, 2776.805194, 900.0):
        fresh = circular_orbit_from_altitude(
            altitude_m=400e3, body_radius_m=body.radius,
            inclination_rad=math.radians(51.6), phase_rad=0.2, mu=body.mu,
        )
        p_expected, v_expected = fresh.state_at(t)
        p, v = orbit.state_at(t)
        assert np.array_equal(p, p_expected)
        assert np.array_equal(v, v_expected)
        assert np.array_equal(orbit.position_at(t), p)
        assert np.array_equal(orbit.velocity_at(t), v)


def test_returned_arrays_are_read_only():
    body = get_body("Earth")
    orbit = circular_orbit_from_altitude(
        altitude_m=400e3, body_radius_m=body.radius,
        inclination_rad=0.0, phase_rad=0.0, mu=body.mu,
    )
    p, v = orbit.state_at(10.0)
    with pytest.raises(ValueError):
        p[0] = 0.0
    with pytest.raises(ValueError):
        v[0] = 0.0


# ---------------------------------------------------------------------------
# The one place the construction paths can legitimately differ
# ---------------------------------------------------------------------------

def test_boundary_stationary_point_is_the_only_permitted_difference():
    """
    With both objects at phase zero they sit on the shared ascending node at
    t = 0, separated only by their radius difference, and r_rel is exactly
    perpendicular to v_rel there.  f(t) = r_rel . v_rel is therefore a
    floating-point zero at the very first sample: the legacy formula and the
    element conversion both give exactly +0.0, while the shared construction
    gives -8.7e-17.  The bracket test ``f[i] < 0 and f[i+1] >= 0`` flips on
    that sign, so one path brackets a stationary point at the window boundary
    and the other does not.

    This is a pre-existing sensitivity in bracket detection at an exact zero,
    not something P9-05 introduced, and it is not fixed here -- the TCA search
    is a protected component.  What this test pins is the boundary of the
    claim: every *interior* conjunction is identical across all three
    construction paths, and any difference is confined to a stationary point
    lying exactly on a window edge.
    """
    from theseus.conjunction.analysis import ConjunctionAnalysis

    body = get_body("Earth")
    req = ConjunctionRequest(
        object_a_alt_km=400.0, object_a_inc_deg=51.6, object_a_phase_deg=0.0,
        object_b_alt_km=400.05, object_b_inc_deg=55.0, object_b_phase_deg=0.0,
        central_body="Earth",
    )
    orbit_a, orbit_b = _conjunction_orbit_pair(req, body)

    def make(fn, alt_km, inc_deg, phase_deg):
        radius = body.radius + alt_km * 1e3
        inc, ph = math.radians(inc_deg), math.radians(phase_deg)
        return (lambda t: fn(radius, inc, ph, body.mu, t)[0],
                lambda t: fn(radius, inc, ph, body.mu, t)[1])

    paths = {
        "shared": (orbit_a.as_callables(), orbit_b.as_callables()),
        "elements": (make(reference_state_via_elements, 400.0, 51.6, 0.0),
                     make(reference_state_via_elements, 400.05, 55.0, 0.0)),
        "legacy": (make(reference_state_legacy_formula, 400.0, 51.6, 0.0),
                   make(reference_state_legacy_formula, 400.05, 55.0, 0.0)),
    }

    interior = {}
    for name, ((pa, va), (pb, vb)) in paths.items():
        res = ConjunctionAnalysis(100e3, coarse_dt=30.0).analyse(
            pa, va, pb, vb, 0.0, 7200.0,
        )
        # Every event is either interior, or sits on a window edge.
        for e in res.events:
            assert e.tca_result.tca >= 0.0
            assert e.tca_result.tca <= 7200.0
        interior[name] = sorted(
            (round(e.tca_result.tca, 6), round(e.tca_result.miss_distance, 6))
            for e in res.events
            if 1e-6 < e.tca_result.tca < 7200.0 - 1e-6
        )

    assert len(interior["shared"]) >= 2
    assert interior["shared"] == interior["elements"] == interior["legacy"], interior


def test_non_degenerate_geometries_agree_exactly_across_all_paths():
    """
    Away from that exact-zero boundary the three constructions produce
    identical event lists, including counts.
    """
    from theseus.conjunction.analysis import ConjunctionAnalysis

    body = get_body("Earth")

    def make(fn, alt_km, inc_deg, phase_deg):
        radius = body.radius + alt_km * 1e3
        inc, ph = math.radians(inc_deg), math.radians(phase_deg)
        return (lambda t: fn(radius, inc, ph, body.mu, t)[0],
                lambda t: fn(radius, inc, ph, body.mu, t)[1])

    for phase_b in (0.02, 0.5, 3.0, 47.0):
        req = ConjunctionRequest(
            object_a_alt_km=400.0, object_a_inc_deg=51.6, object_a_phase_deg=0.0,
            object_b_alt_km=400.05, object_b_inc_deg=55.0,
            object_b_phase_deg=phase_b, central_body="Earth",
        )
        orbit_a, orbit_b = _conjunction_orbit_pair(req, body)

        results = []
        for pair in (
            (orbit_a.as_callables(), orbit_b.as_callables()),
            (make(reference_state_via_elements, 400.0, 51.6, 0.0),
             make(reference_state_via_elements, 400.05, 55.0, phase_b)),
            (make(reference_state_legacy_formula, 400.0, 51.6, 0.0),
             make(reference_state_legacy_formula, 400.05, 55.0, phase_b)),
        ):
            (pa, va), (pb, vb) = pair
            res = ConjunctionAnalysis(100e3, coarse_dt=30.0).analyse(
                pa, va, pb, vb, 0.0, 7200.0,
            )
            results.append(sorted(
                (round(e.tca_result.tca, 6), round(e.tca_result.miss_distance, 6))
                for e in res.events
            ))

        assert results[0] == results[1] == results[2], (phase_b, results)
