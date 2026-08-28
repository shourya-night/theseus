"""
P10-04 — a combined hard-body radius below 2 m raised an exception.

The defect
----------
``compute_hard_body_radius(custom_hbr_m=h)`` built its two per-object
geometries as::

    CollisionGeometry(name="Object A", collision_radius_m=h / 2.0)

leaving ``physical_radius_m`` at the dataclass default of 1.0 m.  The
geometry's own consistency check -- a collision radius may not be smaller than
the structural radius it encloses -- then fired for every ``h < 2.0``::

    HBR = 2.000000  ->  ok
    HBR = 1.999999  ->  ValueError: Collision radius (0.9999995 m) cannot be
                        smaller than physical radius (1.0 m)

The API supplies ``hard_body_radius_m`` on every risk request (default 10 m),
so this branch is always taken, and every CubeSat, small-debris and fragment
conjunction returned HTTP 500.  Two 3U CubeSats have a combined HBR near
0.6 m; a trackable fragment pair is smaller still.  The excluded range is
exactly the range where a hard-body radius matters most, because Pc scales
with the disk area.

In the working range the fixed 1.0 m was reported to the caller as the
objects' ``physical_radius_m`` -- a structural dimension nobody had supplied.

What these tests pin
--------------------
1. Every non-negative combined HBR constructs, including the 1.999999 cliff,
   CubeSat scales, millimetres, and zero.
2. ``combined_hbr_m`` is the supplied value *exactly*, and the equal split
   sums back to it exactly.
3. The reported per-object radii are derived from the supplied number and
   carry their provenance, rather than a placeholder.
4. The newly-reachable probabilities are *correct*, not merely computable:
   each is checked against a quadrature reference derived from the definition
   of Pc, independent of the engine's integrator.
5. Everything outside the ``custom_hbr_m`` branch is untouched.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from theseus.uncertainty.b_plane import project_covariance_to_b_plane
from theseus.uncertainty.collision_probability import compute_collision_probability
from theseus.uncertainty.hard_body import (
    PRESET_CUBESAT,
    PRESET_DEBRIS_SMALL,
    PRESET_ISS,
    CollisionGeometry,
    compute_hard_body_radius,
)


# Values that used to raise.  1.999999 is the cliff edge; 0.6 is two 3U
# CubeSats; 0.2 is a pair of trackable fragments; 0.0 is the documented
# zero-cross-section case that compute_collision_probability handles.
FORMERLY_REJECTED = (1.999999, 1.9, 1.0, 0.6, 0.5, 0.3, 0.2, 0.12, 0.05,
                     0.01, 0.001, 0.0)

FORMERLY_ACCEPTED = (2.0, 5.0, 10.0, 20.0, 54.0, 108.0)


# ---------------------------------------------------------------------------
# An independent Pc reference
# ---------------------------------------------------------------------------

def encounter():
    """A realistic encounter geometry, projected into the B-plane."""
    r_rel = np.array([120.0, -45.0, 30.0])
    v_rel = np.array([-8200.0, 11400.0, 260.0])
    rel_pos_cov = np.array([
        [180.0 ** 2, 2100.0, -900.0],
        [2100.0, 95.0 ** 2, 450.0],
        [-900.0, 450.0, 140.0 ** 2],
    ])
    return project_covariance_to_b_plane(
        rel_pos_cov=rel_pos_cov, r_rel=r_rel, v_rel=v_rel)


def _gaussian(bpu):
    """The encounter-plane density N(b, P_B), built from first principles."""
    P = np.asarray(bpu.b_plane_covariance, dtype=float)
    b = np.array([bpu.b_dot_t, bpu.b_dot_r], dtype=float)
    P_inv = np.linalg.inv(P)
    scale = 1.0 / (2.0 * math.pi * math.sqrt(float(np.linalg.det(P))))

    def pdf(x: float, y: float) -> float:
        d = np.array([x, y]) - b
        return scale * math.exp(-0.5 * float(d @ P_inv @ d))

    return pdf


def reference_pc(bpu, radius_m: float) -> float:
    """
    Pc from its definition: the probability that the relative position in the
    encounter plane falls inside the collision disk,

        Pc = P(|x| <= R),   x ~ N(b, P_B)

    integrated in polar coordinates about the disk centre.  This shares no
    code with ``compute_collision_probability``.
    """
    if radius_m <= 0.0:
        return 0.0
    pdf = _gaussian(bpu)
    value, _ = integrate.dblquad(
        lambda th, r: pdf(r * math.cos(th), r * math.sin(th)) * r,
        0.0, radius_m, 0.0, 2.0 * math.pi,
        epsabs=1e-18, epsrel=1e-12,
    )
    return float(value)


# ---------------------------------------------------------------------------
# 1. The formerly-rejected range now constructs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hbr", FORMERLY_REJECTED)
def test_small_combined_hbr_no_longer_raises(hbr):
    result = compute_hard_body_radius(custom_hbr_m=hbr)
    assert result.combined_hbr_m == hbr


def test_the_two_metre_cliff_is_gone():
    """
    The defect had a sharp edge: 2.0 worked and anything below it did not.
    Walking across that edge must now be continuous.
    """
    below = compute_hard_body_radius(custom_hbr_m=1.999999).combined_hbr_m
    at = compute_hard_body_radius(custom_hbr_m=2.0).combined_hbr_m
    above = compute_hard_body_radius(custom_hbr_m=2.000001).combined_hbr_m
    assert below < at < above
    assert at - below == pytest.approx(1e-6, rel=1e-6)


@pytest.mark.parametrize("hbr", FORMERLY_REJECTED + FORMERLY_ACCEPTED)
def test_combined_hbr_is_the_supplied_value_exactly(hbr):
    """No rounding, no clamping, no minimum floor."""
    assert compute_hard_body_radius(custom_hbr_m=hbr).combined_hbr_m == hbr


@pytest.mark.parametrize("hbr", FORMERLY_REJECTED + FORMERLY_ACCEPTED)
def test_the_split_sums_back_to_the_combined_radius(hbr):
    """
    Only the sum is physically meaningful, so the sum is what must be exact.
    """
    r = compute_hard_body_radius(custom_hbr_m=hbr)
    assert (r.object_a.collision_radius_m
            + r.object_b.collision_radius_m) == r.combined_hbr_m
    assert r.object_a.collision_radius_m == r.object_b.collision_radius_m


# ---------------------------------------------------------------------------
# 2. The reported geometry is derived, not fabricated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hbr", FORMERLY_REJECTED + FORMERLY_ACCEPTED)
def test_physical_radius_is_derived_from_the_supplied_hbr(hbr):
    """
    The placeholder was the defect.  The structural radius must scale with the
    supplied number rather than sitting at a fixed 1.0 m -- which was both what
    broke small radii and, above 2 m, a dimension no caller had given.
    """
    r = compute_hard_body_radius(custom_hbr_m=hbr)
    for geom in (r.object_a, r.object_b):
        assert geom.physical_radius_m == hbr / 2.0
        assert geom.physical_radius_m <= geom.collision_radius_m


def test_reported_physical_radius_is_no_longer_a_fixed_placeholder():
    """Two different combined radii must not report the same structure."""
    small = compute_hard_body_radius(custom_hbr_m=4.0)
    large = compute_hard_body_radius(custom_hbr_m=40.0)
    assert small.object_a.physical_radius_m != large.object_a.physical_radius_m
    assert large.object_a.physical_radius_m == 20.0


@pytest.mark.parametrize("hbr", (0.3, 2.0, 54.0))
def test_derived_geometry_carries_its_provenance(hbr):
    """
    A reader must be able to tell a derived radius from a measured one, or the
    fix would replace one fabricated number with a better-looking one.
    """
    r = compute_hard_body_radius(custom_hbr_m=hbr)
    for geom in (r.object_a, r.object_b):
        assert geom.source
        assert "supplied" in geom.source.lower()
        assert str(hbr) in geom.source


def test_result_serialises_the_derived_radii():
    d = compute_hard_body_radius(custom_hbr_m=0.6).to_dict()
    assert d["combined_hbr_m"] == 0.6
    assert d["object_a"]["collision_radius_m"] == 0.3
    assert d["object_a"]["physical_radius_m"] == 0.3
    assert d["object_b"]["collision_radius_m"] == 0.3


# ---------------------------------------------------------------------------
# 3. Invalid input is still refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", (-1e-9, -0.5, -10.0))
def test_negative_combined_hbr_is_still_rejected(bad):
    with pytest.raises(ValueError, match="non-negative"):
        compute_hard_body_radius(custom_hbr_m=bad)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_non_finite_combined_hbr_is_still_rejected(bad):
    """
    Caught either by the negativity guard or by the geometry's own finiteness
    check.  Widening the accepted range must not have opened this door.
    """
    with pytest.raises(ValueError):
        compute_hard_body_radius(custom_hbr_m=bad)


# ---------------------------------------------------------------------------
# 4. The newly-reachable probabilities are correct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hbr", (0.001, 0.01, 0.05, 0.12, 0.3, 0.5, 0.6,
                                 1.0, 1.9, 1.999999))
def test_small_hbr_probability_matches_an_independent_integral(hbr):
    """
    The point of the fix is not that these calls stop throwing -- it is that
    the numbers they now return are right.  Checked against a polar quadrature
    of N(b, P_B) over the collision disk, derived from the definition of Pc.
    """
    bpu = encounter()
    combined = compute_hard_body_radius(custom_hbr_m=hbr).combined_hbr_m
    pc = compute_collision_probability(bpu, combined).probability
    assert pc == pytest.approx(reference_pc(bpu, hbr), rel=1e-9)


@pytest.mark.parametrize("hbr", (2.0, 10.0, 54.0))
def test_large_hbr_probability_still_matches_the_same_reference(hbr):
    """The range that already worked must not have moved."""
    bpu = encounter()
    combined = compute_hard_body_radius(custom_hbr_m=hbr).combined_hbr_m
    pc = compute_collision_probability(bpu, combined).probability
    assert pc == pytest.approx(reference_pc(bpu, hbr), rel=1e-9)


@pytest.mark.parametrize("hbr", (0.001, 0.01, 0.05))
def test_small_disk_limit_is_recovered(hbr):
    """
    A second, independent check on the small-HBR regime: for a disk much
    smaller than the uncertainty the density is effectively constant across
    it, so Pc -> pi R^2 * pdf(0).  Agreement here means the small values are
    right for the right reason, not by coincidence of two integrators.
    """
    bpu = encounter()
    pdf = _gaussian(bpu)
    combined = compute_hard_body_radius(custom_hbr_m=hbr).combined_hbr_m
    pc = compute_collision_probability(bpu, combined).probability
    assert pc == pytest.approx(math.pi * hbr * hbr * pdf(0.0, 0.0), rel=1e-4)


def test_zero_hbr_gives_zero_probability_through_the_documented_path():
    """
    ``compute_collision_probability`` documents an ``hbr_m <= 0`` case, but it
    was unreachable through ``compute_hard_body_radius`` because constructing
    the geometry raised first.
    """
    bpu = encounter()
    combined = compute_hard_body_radius(custom_hbr_m=0.0).combined_hbr_m
    result = compute_collision_probability(bpu, combined)
    assert result.probability == 0.0
    assert result.method == "analytic_zero_hbr"


def test_probability_is_monotonic_in_hard_body_radius():
    """
    A structural property no single-point comparison catches: a larger
    collision cross-section cannot be less likely to be hit.  Spans the
    formerly-excluded range and the formerly-working one in one sweep.
    """
    bpu = encounter()
    radii = np.geomspace(1e-3, 50.0, 40)
    pcs = [compute_collision_probability(
        bpu, compute_hard_body_radius(custom_hbr_m=float(h)).combined_hbr_m
    ).probability for h in radii]
    assert all(a <= b + 1e-18 for a, b in zip(pcs, pcs[1:]))
    assert pcs[0] < pcs[-1]


# ---------------------------------------------------------------------------
# 5. Nothing outside the custom-HBR branch moved
# ---------------------------------------------------------------------------

def test_preset_geometry_path_is_unchanged():
    r = compute_hard_body_radius(obj_a=PRESET_ISS, obj_b=PRESET_CUBESAT)
    assert r.combined_hbr_m == PRESET_ISS.collision_radius_m + PRESET_CUBESAT.collision_radius_m
    assert r.object_a is PRESET_ISS
    assert r.object_b is PRESET_CUBESAT
    assert r.object_a.physical_radius_m == 15.0


def test_explicit_radius_path_is_unchanged():
    r = compute_hard_body_radius(radius_a_m=0.3, radius_b_m=0.1)
    assert r.combined_hbr_m == pytest.approx(0.4)
    assert r.object_a.collision_radius_m == 0.3
    assert r.object_a.physical_radius_m == 0.3


def test_default_path_is_unchanged():
    r = compute_hard_body_radius()
    assert r.combined_hbr_m == 10.0
    assert r.object_a.collision_radius_m == 5.0
    assert r.object_a.physical_radius_m == 2.0


def test_small_presets_still_combine_without_a_custom_hbr():
    """
    Two small objects supplied as real geometries were never affected -- the
    defect was specific to the custom-HBR branch.  Pinned so the distinction
    stays visible.
    """
    r = compute_hard_body_radius(obj_a=PRESET_CUBESAT, obj_b=PRESET_DEBRIS_SMALL)
    assert r.combined_hbr_m == pytest.approx(0.4)


def test_custom_hbr_takes_precedence_over_supplied_geometry():
    """
    Pre-existing, undocumented-until-now behaviour, pinned rather than
    changed: a caller passing both real geometries and a combined HBR gets the
    combined HBR, and the geometries are discarded.  Recorded here so that if
    this precedence is ever revisited it is a deliberate decision against a
    stated baseline, not an accident.
    """
    r = compute_hard_body_radius(obj_a=PRESET_ISS, obj_b=PRESET_CUBESAT,
                                 custom_hbr_m=0.5)
    assert r.combined_hbr_m == 0.5
    assert r.object_a is not PRESET_ISS
    assert r.object_a.collision_radius_m == 0.25


def test_collision_geometry_constructor_check_is_intact():
    """
    The fix supplies a consistent physical radius; it does not weaken the
    check that caught the inconsistency.  That check belongs to the closed
    Phase 9 collision-geometry work and must still fire.
    """
    with pytest.raises(ValueError, match="cannot be smaller"):
        CollisionGeometry(physical_radius_m=1.0, collision_radius_m=0.5)


# ---------------------------------------------------------------------------
# 6. Through the risk API
# ---------------------------------------------------------------------------

def _risk_payload(hbr):
    return {
        "object_a_alt_km": 400.0, "object_a_inc_deg": 51.6, "object_a_phase_deg": 0.0,
        "object_b_alt_km": 400.05, "object_b_inc_deg": 55.0, "object_b_phase_deg": 0.02,
        "central_body": "Earth", "analysis_duration_hours": 2.0,
        "screening_threshold_km": 100.0, "coarse_dt_s": 30.0,
        "hard_body_radius_m": hbr,
    }


@pytest.mark.parametrize("hbr", (1.9, 0.6, 0.3, 0.12, 0.05))
def test_risk_api_accepts_cubesat_scale_hard_body_radii(hbr):
    """Each of these returned HTTP 500 before the fix."""
    from fastapi.testclient import TestClient
    from theseus.server.app import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/simulate/conjunction/risk", json=_risk_payload(hbr))

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_status"] == "COMPLETE"
    assert data["hard_body"]["combined_hbr_m"] == hbr
    assert (data["hard_body"]["object_a"]["collision_radius_m"]
            + data["hard_body"]["object_b"]["collision_radius_m"]) == pytest.approx(hbr)

    pc = data["collision_probability"]["probability"]
    assert pc is not None and 0.0 <= pc <= 1.0
    assert data["risk_assessment"]["level"] != "INDETERMINATE"


def test_risk_api_probability_scales_with_the_hard_body_radius():
    """
    A smaller spacecraft cannot be more likely to be hit.  Run through the
    API so the scaling is pinned end to end, not only at the function.
    """
    from fastapi.testclient import TestClient
    from theseus.server.app import app

    client = TestClient(app)
    pcs = []
    for hbr in (0.12, 0.6, 1.9, 10.0):
        data = client.post("/api/simulate/conjunction/risk",
                           json=_risk_payload(hbr)).json()
        pcs.append(data["collision_probability"]["probability"])

    assert all(a < b for a, b in zip(pcs, pcs[1:]))


def test_risk_api_large_hard_body_radius_is_unchanged():
    """The range that already worked must return what it always returned."""
    from fastapi.testclient import TestClient
    from theseus.server.app import app

    client = TestClient(app)
    data = client.post("/api/simulate/conjunction/risk",
                       json=_risk_payload(15.0)).json()

    assert data["analysis_status"] == "COMPLETE"
    assert data["hard_body"]["combined_hbr_m"] == 15.0
    assert data["collision_probability"]["probability"] > 0.0
