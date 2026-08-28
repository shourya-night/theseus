"""
P9-03 — deterministic collision geometry in the Phase 9 conjunction model.

Phase 9 answers a geometric question about the nominal trajectories:

    clearance = miss_distance - (R_A + R_B)

    clearance < 0  -> INTERSECTION      the bodies interpenetrate
    clearance = 0  -> GRAZING           surfaces touch exactly
    clearance > 0  -> NO_INTERSECTION   the bodies pass clear

Phase 10 answers a different question -- the probability of collision under
trajectory uncertainty -- and the two must not be conflated.  A close approach
is not a collision, and a non-zero Pc is not a collision.

Expected answers here are computed from the definitions above and from
closed-form rectilinear geometry, never by re-running the code under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from theseus.conjunction.analysis import ConjunctionAnalysis
from theseus.conjunction.geometry import (
    CollisionGeometry,
    CollisionAssessment,
    CollisionStatus,
    assess_collision_geometry,
    combined_hard_body_radius,
)

from tests._conjunction_reference import (
    R_LEO,
    circular_orbit,
    rectilinear,
    rectilinear_closest_approach,
    independent_closest_approach,
)


def geom(radius: float, name: str = "obj") -> CollisionGeometry:
    return CollisionGeometry.from_radius(radius, name=name)


# ---------------------------------------------------------------------------
# The three-way classification, from the definitions
# ---------------------------------------------------------------------------

def test_obvious_non_collision():
    """Case 1: a 10 km miss between two 5 m objects is not a collision."""
    a = assess_collision_geometry(10_000.0, geom(5.0), geom(5.0))
    assert a.status is CollisionStatus.NO_INTERSECTION
    assert a.combined_hard_body_radius_m == pytest.approx(10.0)
    assert a.clearance_m == pytest.approx(9990.0)
    assert a.is_physical_intersection is False
    assert a.is_evaluated is True


def test_exact_grazing_contact():
    """Case 2: miss distance exactly equal to the combined radius."""
    a = assess_collision_geometry(12.0, geom(5.0), geom(7.0))
    assert a.combined_hard_body_radius_m == pytest.approx(12.0)
    assert a.clearance_m == pytest.approx(0.0, abs=1e-12)
    assert a.status is CollisionStatus.GRAZING
    # Grazing is contact, not a clean pass.
    assert a.is_physical_intersection is True


def test_grazing_tolerance_band():
    """A caller may widen the grazing band; the default band is exactly zero."""
    near = assess_collision_geometry(12.5, geom(5.0), geom(7.0))
    assert near.status is CollisionStatus.NO_INTERSECTION

    with_tol = assess_collision_geometry(12.5, geom(5.0), geom(7.0),
                                         grazing_tolerance_m=1.0)
    assert with_tol.status is CollisionStatus.GRAZING
    assert with_tol.clearance_m == pytest.approx(0.5)

    with pytest.raises(ValueError):
        assess_collision_geometry(12.5, geom(5.0), geom(7.0), grazing_tolerance_m=-1.0)


def test_definite_collision():
    """Case 3: centres closer than the combined radius."""
    a = assess_collision_geometry(3.0, geom(5.0), geom(7.0))
    assert a.status is CollisionStatus.INTERSECTION
    assert a.clearance_m == pytest.approx(-9.0)
    assert a.is_physical_intersection is True


def test_unequal_body_radii():
    """
    Case 4: a 54 m ISS-scale sphere and a 0.3 m CubeSat.  The same 40 m miss
    is a strike for the pair and a clean pass for two CubeSats -- which is
    exactly why miss distance alone is not a collision criterion.
    """
    iss = CollisionGeometry(name="ISS", physical_radius_m=15.0,
                            collision_radius_m=54.0, shape="box_wing")
    cube = geom(0.3, "3U CubeSat")

    assert combined_hard_body_radius(iss, cube) == pytest.approx(54.3)

    strike = assess_collision_geometry(40.0, iss, cube)
    assert strike.status is CollisionStatus.INTERSECTION
    assert strike.clearance_m == pytest.approx(-14.3)

    pass_by = assess_collision_geometry(40.0, cube, cube)
    assert pass_by.status is CollisionStatus.NO_INTERSECTION
    assert pass_by.clearance_m == pytest.approx(39.4)


def test_zero_radius_bodies():
    """
    Case 5: point masses.  Only exact coincidence is contact; any positive
    separation is a clean pass.
    """
    point = geom(0.0, "point")
    assert assess_collision_geometry(1e-9, point, point).status is CollisionStatus.NO_INTERSECTION
    assert assess_collision_geometry(0.0, point, point).status is CollisionStatus.GRAZING
    assert assess_collision_geometry(0.0, point, point).clearance_m == pytest.approx(0.0)


def test_very_small_bodies():
    """Case 6: sub-metre debris, where the interesting scale is centimetres."""
    a = geom(0.05, "fragment A")
    b = geom(0.07, "fragment B")
    assert assess_collision_geometry(0.13, a, b).status is CollisionStatus.NO_INTERSECTION
    assert assess_collision_geometry(0.11, a, b).status is CollisionStatus.INTERSECTION
    assert assess_collision_geometry(0.12, a, b).clearance_m == pytest.approx(0.0, abs=1e-12)


def test_large_bodies():
    """Case 7: two large structures whose enclosing spheres dominate."""
    a = CollisionGeometry(name="Station", physical_radius_m=20.0, collision_radius_m=110.0)
    b = CollisionGeometry(name="Depot", physical_radius_m=10.0, collision_radius_m=65.0)
    assert combined_hard_body_radius(a, b) == pytest.approx(175.0)
    assert assess_collision_geometry(200.0, a, b).status is CollisionStatus.NO_INTERSECTION
    assert assess_collision_geometry(150.0, a, b).status is CollisionStatus.INTERSECTION


def test_coincident_positions():
    """Case 8: zero miss distance is unambiguously an intersection."""
    a = assess_collision_geometry(0.0, geom(2.0), geom(3.0))
    assert a.status is CollisionStatus.INTERSECTION
    assert a.clearance_m == pytest.approx(-5.0)


@pytest.mark.parametrize("miss,hbr_a,hbr_b,expect_intersection", [
    (9.99, 5.0, 5.0, True),
    (10.01, 5.0, 5.0, False),
    (99.9, 60.0, 40.0, True),
    (100.1, 60.0, 40.0, False),
    (0.29, 0.1, 0.2, True),
    (0.31, 0.1, 0.2, False),
])
def test_threshold_is_the_combined_radius_not_the_miss_distance(
    miss, hbr_a, hbr_b, expect_intersection,
):
    """
    The core requirement: miss < combined radius is an intersection, and
    miss > combined radius is not -- at every scale.
    """
    a = assess_collision_geometry(miss, geom(hbr_a), geom(hbr_b))
    assert a.is_physical_intersection is expect_intersection
    assert (a.clearance_m < 0.0) is expect_intersection


# ---------------------------------------------------------------------------
# Missing geometry must not read as "no collision"
# ---------------------------------------------------------------------------

def test_absent_geometry_is_unknown_not_safe():
    for ga, gb in ((None, geom(5.0)), (geom(5.0), None), (None, None)):
        a = assess_collision_geometry(1.0, ga, gb)
        assert a.status is CollisionStatus.UNKNOWN
        assert a.is_evaluated is False
        assert math.isnan(a.combined_hard_body_radius_m)
        assert math.isnan(a.clearance_m)
        assert a.to_dict()["evaluated"] is False


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        assess_collision_geometry(-1.0, geom(1.0), geom(1.0))
    with pytest.raises(ValueError):
        assess_collision_geometry(float("nan"), geom(1.0), geom(1.0))
    with pytest.raises(ValueError):
        CollisionGeometry(physical_radius_m=-1.0)
    with pytest.raises(ValueError):
        CollisionGeometry(physical_radius_m=5.0, collision_radius_m=2.0)


# ---------------------------------------------------------------------------
# Integration with the validated Phase 9 pipeline
# ---------------------------------------------------------------------------

def _head_on_pair(miss_m: float, closing_speed: float, t_ca: float):
    half = 0.5 * closing_speed
    pos_a, vel_a = rectilinear([-half * t_ca, 0.0, 0.0], [half, 0.0, 0.0])
    pos_b, vel_b = rectilinear([half * t_ca, miss_m, 0.0], [-half, 0.0, 0.0])
    return pos_a, vel_a, pos_b, vel_b


def test_close_high_speed_encounter_is_not_called_a_collision():
    """
    Case 9: a 30 m miss at 15 km/s between two 5 m objects.  Alarming, and
    still not a collision: clearance is +20 m.  The event must report the
    close approach and refuse the word collision.
    """
    pos_a, vel_a, pos_b, vel_b = _head_on_pair(30.0, 15_000.0, 1830.0)
    t_ca_true, miss_true = rectilinear_closest_approach(
        pos_a(0.0), vel_a(0.0), pos_b(0.0), vel_b(0.0),
    )

    result = ConjunctionAnalysis(screening_threshold_m=50e3, coarse_dt=60.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, 3600.0,
        geometry_a=geom(5.0, "A"), geometry_b=geom(5.0, "B"),
        object_a_id="A", object_b_id="B",
    )

    events = [e for e in result.events if e.tca_result.validated]
    assert events, "the encounter must still be detected"
    ev = min(events, key=lambda e: e.tca_result.miss_distance)

    assert ev.tca_result.tca == pytest.approx(t_ca_true, abs=1e-3)
    assert ev.miss_distance_m == pytest.approx(miss_true, rel=1e-9, abs=1e-6)
    assert ev.collision.status is CollisionStatus.NO_INTERSECTION
    assert ev.clearance_m == pytest.approx(20.0, abs=1e-6)
    assert ev.is_physical_intersection is False
    assert ev.object_a_id == "A" and ev.object_b_id == "B"


def test_pipeline_reports_intersection_when_bodies_are_large_enough():
    """Same encounter geometry, bigger bodies: now it is a strike."""
    pos_a, vel_a, pos_b, vel_b = _head_on_pair(30.0, 15_000.0, 1830.0)
    result = ConjunctionAnalysis(screening_threshold_m=50e3, coarse_dt=60.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, 3600.0,
        geometry_a=geom(25.0, "A"), geometry_b=geom(10.0, "B"),
    )
    ev = min((e for e in result.events if e.tca_result.validated),
             key=lambda e: e.tca_result.miss_distance)
    assert ev.collision.status is CollisionStatus.INTERSECTION
    assert ev.collision.combined_hard_body_radius_m == pytest.approx(35.0)
    assert ev.clearance_m == pytest.approx(-5.0, abs=1e-6)
    assert ev.is_physical_intersection is True


def test_geometry_is_evaluated_at_the_validated_tca():
    """
    The clearance must be derived from the TCA miss distance, not from a
    coarse screening sample.  Verified by recomputing the separation from the
    trajectories at the reported TCA.
    """
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=90.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO + 200.0, phase_deg=90.0009, inc_deg=170.0)
    t_ca_true, miss_true = independent_closest_approach(pos_a, pos_b, 0.0, 7200.0)

    result = ConjunctionAnalysis(screening_threshold_m=50e3, coarse_dt=60.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, 7200.0,
        geometry_a=geom(3.0), geometry_b=geom(4.0),
    )
    ev = min((e for e in result.events if e.tca_result.validated),
             key=lambda e: e.tca_result.miss_distance)

    sep_at_tca = float(np.linalg.norm(pos_a(ev.tca_result.tca) - pos_b(ev.tca_result.tca)))
    assert ev.collision.miss_distance_m == pytest.approx(sep_at_tca, rel=1e-12)
    assert ev.clearance_m == pytest.approx(sep_at_tca - 7.0, rel=1e-12)
    assert ev.tca_result.tca == pytest.approx(t_ca_true, abs=1e-2)

    # A screening sample would have given a wildly different separation.
    assert abs(sep_at_tca - miss_true) < 1.0


def test_analysis_without_geometry_reports_unknown():
    """Omitting geometry must not silently produce 'no intersection'."""
    pos_a, vel_a, pos_b, vel_b = _head_on_pair(30.0, 15_000.0, 1830.0)
    result = ConjunctionAnalysis(screening_threshold_m=50e3, coarse_dt=60.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, 3600.0,
    )
    ev = min((e for e in result.events if e.tca_result.validated),
             key=lambda e: e.tca_result.miss_distance)
    assert ev.collision.status is CollisionStatus.UNKNOWN
    assert ev.is_physical_intersection is None
    assert ev.clearance_m is None

    summary = result.to_dict()["summary"]
    assert summary["collision_geometry_evaluated"] is False
    assert summary["physical_intersections"] == 0


def test_serialisation_carries_the_three_distinct_quantities():
    pos_a, vel_a, pos_b, vel_b = _head_on_pair(30.0, 15_000.0, 1830.0)
    result = ConjunctionAnalysis(screening_threshold_m=50e3, coarse_dt=60.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, 3600.0,
        geometry_a=geom(5.0, "A"), geometry_b=geom(5.0, "B"),
    )
    payload = result.to_dict()
    ev = payload["events"][0]
    coll = ev["collision"]

    assert coll["miss_distance_m"] == pytest.approx(30.0, abs=1e-6)
    assert coll["combined_hard_body_radius_m"] == pytest.approx(10.0)
    assert coll["clearance_m"] == pytest.approx(20.0, abs=1e-6)
    assert coll["status"] == "NO_INTERSECTION"
    assert coll["object_a"]["name"] == "A"

    assert payload["summary"]["collision_geometry_evaluated"] is True
    assert payload["summary"]["smallest_clearance_m"] == pytest.approx(20.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Case 10: multiple pairs with different radii, through the simulator path
# ---------------------------------------------------------------------------

def test_multiple_spacecraft_pairs_with_different_radii():
    """
    Case 10.  Three objects with different hard-body radii.  Each pair's
    intersection verdict must follow its own combined radius, and must match
    the verdict recomputed from the reported miss distance independently.
    """
    from theseus.simulation.multi_object import MultiObjectEnvironment, SpacecraftDefinition

    radii = {"SC-1": 2.0, "SC-2": 40.0, "SC-3": 0.25}

    def mk(sc_id, ta, inc):
        return SpacecraftDefinition(
            id=sc_id, name=sc_id, semi_major_axis_km=6778.137, eccentricity=0.0,
            inclination_deg=inc, raan_deg=0.0, arg_periapsis_deg=0.0,
            true_anomaly_deg=ta, central_body="Earth",
            hard_body_radius_m=radii[sc_id],
            dry_mass_kg=500.0, fuel_mass_kg=0.0, payload_mass_kg=0.0,
            cross_section_area_m2=10.0,
        )

    env = MultiObjectEnvironment(central_body="Earth", enable_drag=True)
    res = env.simulate([mk("SC-1", 0.0, 0.0), mk("SC-2", 0.05, 170.0),
                        mk("SC-3", 0.1, 120.0)], 0.0, 7200.0, output_dt=30.0)

    assert res.conjunctions
    for c in res.conjunctions:
        expected_hbr = radii[c.spacecraft_a_id] + radii[c.spacecraft_b_id]
        assert c.hard_body_radius_m == pytest.approx(expected_hbr)

        expected_clearance = c.miss_distance_m - expected_hbr
        assert c.clearance_m == pytest.approx(expected_clearance, rel=1e-12)

        expected_intersection = c.miss_distance_m <= expected_hbr
        assert c.is_physical_collision is expected_intersection
        assert c.collision_status in (
            "NO_INTERSECTION", "GRAZING", "INTERSECTION",
        )
        if expected_intersection:
            assert c.collision_status in ("INTERSECTION", "GRAZING")
        else:
            assert c.collision_status == "NO_INTERSECTION"

    # Every reported physical collision must also appear in the collision list.
    struck = {c.event_id.replace("CONJ-", "") for c in res.conjunctions
              if c.is_physical_collision}
    listed = {c.collision_id.replace("COLL-", "") for c in res.collisions}
    assert struck == listed


def test_spacecraft_definition_geometry_adapter_uses_the_existing_radius():
    """No second dimension is introduced: the adapter reads hard_body_radius_m."""
    from theseus.simulation.multi_object import SpacecraftDefinition

    sc = SpacecraftDefinition(id="X", name="X", hard_body_radius_m=7.5)
    g = sc.get_collision_geometry()
    assert isinstance(g, CollisionGeometry)
    assert g.collision_radius_m == pytest.approx(7.5)
    assert g.physical_radius_m == pytest.approx(7.5)
    assert g.name == "X"
    assert "hard_body_radius_m" in g.source

    debris = SpacecraftDefinition(id="D", name="D", hard_body_radius_m=0.4, is_debris=True)
    assert debris.get_collision_geometry().object_type == "debris"


# ---------------------------------------------------------------------------
# Phase 9 geometry and Phase 10 probability stay separate
# ---------------------------------------------------------------------------

def test_deterministic_geometry_is_independent_of_collision_probability():
    """
    A non-zero probability of collision does not make a clean pass into an
    intersection, and a zero Pc does not clear a real one.  The Phase 9
    verdict depends only on miss distance and radii.
    """
    from theseus.uncertainty.b_plane import BPlaneUncertainty
    from theseus.uncertainty.collision_probability import compute_collision_probability
    from theseus.conjunction.b_plane import BPlaneResult

    miss = 30.0
    hbr_a, hbr_b = 5.0, 5.0
    determin = assess_collision_geometry(miss, geom(hbr_a), geom(hbr_b))
    assert determin.status is CollisionStatus.NO_INTERSECTION

    # Same encounter, a large covariance -> a clearly non-zero Pc.
    sigma = 200.0
    p_b = np.array([[sigma ** 2, 0.0], [0.0, sigma ** 2]])
    eigvals, eigvecs = np.linalg.eigh(p_b)
    b_unc = BPlaneUncertainty(
        b_plane_covariance=p_b, b_dot_t=miss, b_dot_r=0.0,
        sigma_t=sigma, sigma_r=sigma, cov_tr=0.0, correlation=0.0,
        sigma_major=sigma, sigma_minor=sigma,
        ellipse_angle_deg=0.0, ellipse_angle_rad=0.0,
        eigenvalues=eigvals, eigenvectors=eigvecs,
        b_plane_result=BPlaneResult(True, "fixture", b_dot_t=miss, b_dot_r=0.0,
                                    s_hat=np.array([0.0, 0.0, 1.0]),
                                    t_hat=np.array([1.0, 0.0, 0.0]),
                                    r_hat=np.array([0.0, 1.0, 0.0])),
    )
    pc = compute_collision_probability(b_unc, hbr_a + hbr_b).probability
    assert pc > 0.0

    # Pc is non-zero, yet the nominal bodies did not touch.
    assert determin.is_physical_intersection is False
    assert determin.clearance_m > 0.0
