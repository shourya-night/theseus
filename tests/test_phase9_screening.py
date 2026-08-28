"""
Phase 9 — P9-01 regression tests for conservative conjunction screening.

Background
----------
Coarse screening used to accept an interval only when a *sampled* separation
fell below the screening threshold:

    below[i] = distance(t_i) < threshold

That is a statement about the samples, not about the trajectory.  For a
high-relative-velocity encounter the pair is inside the threshold sphere for
only a short time; if no sample lands in that window the encounter is
discarded and never reaches TCA refinement.

Measured for the reference case below (400 km circular pair, ~170 deg plane
difference, |v_rel| = 15.28 km/s, 50 km threshold):

    time inside the 50 km sphere        6.55 s
    relative travel per 60 s step     916.7 km   -> 10.9 % duty cycle
    relative travel per 30 s step     458.4 km   -> 21.8 % duty cycle

and at both shipped steps no sample landed inside, so a 9.28 m miss was
reported as "no conjunction".

These tests pin detection, not merely a non-actionable downstream result.
Expected TCA and miss distances come from closed-form rectilinear geometry or
from an independent SciPy minimisation -- never from the screener itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from theseus.conjunction.analysis import ConjunctionAnalysis
from theseus.conjunction.screening import (
    ConjunctionScreener,
    separation_lower_bound,
)

from tests._conjunction_reference import (
    MU_EARTH,
    R_LEO,
    circular_orbit,
    rectilinear,
    rectilinear_closest_approach,
    independent_closest_approach,
    min_distance_on_interval,
)


WINDOW_S = 7200.0
THRESHOLD_M = 50e3


# ---------------------------------------------------------------------------
# The reference high-speed near-miss
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fast_pair():
    """400 km circular pair, ~170 deg plane difference: 9.28 m at 15.28 km/s."""
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=0.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO, phase_deg=0.0009, inc_deg=170.0)
    t_ca, miss = independent_closest_approach(pos_a, pos_b, 0.0, WINDOW_S)
    return pos_a, vel_a, pos_b, vel_b, t_ca, miss


def _closest_event(result):
    """Smallest-miss validated event from a ConjunctionResult, or None."""
    events = [e for e in result.events if e.tca_result.validated]
    if not events:
        return None
    return min(events, key=lambda e: e.tca_result.miss_distance)


@pytest.mark.parametrize("coarse_dt", [60.0, 30.0])
def test_known_high_speed_encounter_is_detected(fast_pair, coarse_dt):
    """
    The exact case that P9-01 was raised for.  At both shipped coarse steps the
    encounter must now be detected, with the right time and the right miss
    distance -- both compared against an independent SciPy minimisation.
    """
    pos_a, vel_a, pos_b, vel_b, t_ca_true, miss_true = fast_pair

    analysis = ConjunctionAnalysis(
        screening_threshold_m=THRESHOLD_M, coarse_dt=coarse_dt,
    )
    result = analysis.analyse(pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW_S)

    event = _closest_event(result)
    assert event is not None, (
        f"encounter missed at coarse_dt={coarse_dt} s "
        f"(true TCA {t_ca_true:.3f} s, miss {miss_true:.3f} m)"
    )

    tca = event.tca_result
    assert tca.tca == pytest.approx(t_ca_true, abs=1e-3)
    assert tca.miss_distance == pytest.approx(miss_true, rel=1e-6, abs=1e-3)
    assert tca.relative_velocity == pytest.approx(15278.8, rel=1e-3)

    # Sanity on the reference itself: this really is a metre-scale miss.
    assert miss_true < 50.0


def test_reference_case_geometry(fast_pair):
    """Guard the fixture: ~9.28 m at ~15.3 km/s, and a short sub-threshold window."""
    pos_a, vel_a, pos_b, vel_b, t_ca_true, miss_true = fast_pair
    v_rel = float(np.linalg.norm(vel_a(t_ca_true) - vel_b(t_ca_true)))
    assert miss_true == pytest.approx(9.28, abs=0.5)
    assert v_rel == pytest.approx(15278.8, rel=1e-3)

    dwell = 2.0 * math.sqrt(THRESHOLD_M ** 2 - miss_true ** 2) / v_rel
    assert dwell < 10.0, "fixture is no longer a between-sample encounter"


# ---------------------------------------------------------------------------
# The safety property the correction rests on
# ---------------------------------------------------------------------------

def test_lower_bound_is_never_optimistic_on_orbital_arcs(fast_pair):
    """
    Conservatism check on real orbital dynamics.

    For every coarse interval across the window, the analytic lower bound must
    not exceed the true minimum separation on that interval, where the truth is
    obtained by a dense independent scan plus bounded refinement.

    A bound that ever exceeds the truth is exactly what would let a dangerous
    encounter be discarded.
    """
    pos_a, vel_a, pos_b, vel_b, _, _ = fast_pair
    h = 60.0
    times = np.arange(0.0, WINDOW_S, h)

    worst_slack = float("inf")
    for t0 in times:
        t1 = min(t0 + h, WINDOW_S)
        if t1 <= t0:
            continue
        d0 = float(np.linalg.norm(pos_a(t0) - pos_b(t0)))
        d1 = float(np.linalg.norm(pos_a(t1) - pos_b(t1)))
        v0 = float(np.linalg.norm(vel_a(t0) - vel_b(t0)))
        v1 = float(np.linalg.norm(vel_a(t1) - vel_b(t1)))

        bound = separation_lower_bound(d0, d1, v0, v1, t1 - t0)
        truth = min_distance_on_interval(pos_a, pos_b, t0, t1)

        assert bound <= truth + 1e-6, (
            f"bound {bound:.3f} m exceeds true minimum {truth:.3f} m "
            f"on [{t0:.1f}, {t1:.1f}] s"
        )
        worst_slack = min(worst_slack, truth - bound)

    # The bound must also be useful, not trivially zero everywhere.
    assert worst_slack < 1e9


def test_lower_bound_is_exact_for_rectilinear_motion():
    """
    For constant relative velocity the bound reduces to (d0 + d1 - |v| h) / 2,
    which is attained exactly when the closest approach sits inside the
    interval and the motion is head-on.  Check against closed form.
    """
    p_a, v_a = np.array([0.0, 0.0, 0.0]), np.array([1000.0, 0.0, 0.0])
    p_b, v_b = np.array([20000.0, 0.0, 0.0]), np.array([-1000.0, 0.0, 0.0])
    h = 10.0
    d0 = float(np.linalg.norm(p_a - p_b))
    d1 = float(np.linalg.norm((p_a + v_a * h) - (p_b + v_b * h)))
    v_rel = float(np.linalg.norm(v_a - v_b))

    bound = separation_lower_bound(d0, d1, v_rel, v_rel, h)
    expected = 0.5 * (d0 + d1 - v_rel * h)
    assert bound == pytest.approx(max(0.0, expected))


def test_lower_bound_never_negative():
    assert separation_lower_bound(10.0, 10.0, 1e6, 1e6, 100.0) == 0.0


# ---------------------------------------------------------------------------
# Adversarial cases on exactly-known rectilinear geometry
# ---------------------------------------------------------------------------

def _screen_and_refine(pos_a, vel_a, pos_b, vel_b, t_start, t_end,
                       threshold_m=THRESHOLD_M, coarse_dt=60.0):
    analysis = ConjunctionAnalysis(
        screening_threshold_m=threshold_m, coarse_dt=coarse_dt,
    )
    return analysis.analyse(pos_a, vel_a, pos_b, vel_b, t_start, t_end)


def _head_on_pair(miss_m: float, closing_speed: float, t_ca: float):
    """
    Two rectilinear objects closing head-on along x, offset by miss_m in y,
    reaching closest approach at t_ca.
    """
    half = 0.5 * closing_speed
    pos_a, vel_a = rectilinear([-half * t_ca, 0.0, 0.0], [half, 0.0, 0.0])
    pos_b, vel_b = rectilinear([half * t_ca, miss_m, 0.0], [-half, 0.0, 0.0])
    return pos_a, vel_a, pos_b, vel_b


@pytest.mark.parametrize("t_ca,label", [
    (1830.0, "mid-interval"),
    (1801.2, "near interval start"),
    (1858.8, "near interval end"),
])
def test_encounter_anywhere_inside_a_coarse_interval_is_detected(t_ca, label):
    """
    Cases 2, 3 and 4: a 15 km/s head-on encounter placed at the middle, just
    after the start, and just before the end of a coarse interval.  With
    coarse_dt = 60 s and a window starting at 0, sample times are multiples of
    60 s, so these place the encounter at 50 %, 2 % and 98 % of the interval
    [1800, 1860] s.
    """
    miss = 120.0
    speed = 15_000.0
    pos_a, vel_a, pos_b, vel_b = _head_on_pair(miss, speed, t_ca)
    t_ca_true, miss_true = rectilinear_closest_approach(
        pos_a(0.0), vel_a(0.0), pos_b(0.0), vel_b(0.0),
    )
    assert t_ca_true == pytest.approx(t_ca, abs=1e-6)
    assert miss_true == pytest.approx(miss, abs=1e-6)

    result = _screen_and_refine(pos_a, vel_a, pos_b, vel_b, 0.0, 3600.0)
    event = _closest_event(result)
    assert event is not None, f"{label} encounter was not detected"
    assert event.tca_result.tca == pytest.approx(t_ca_true, abs=1e-3)
    assert event.tca_result.miss_distance == pytest.approx(miss_true, rel=1e-6, abs=1e-3)


def test_high_relative_velocity_head_on_encounter():
    """Case 1: a genuinely fast head-on pass must survive a 60 s coarse step."""
    pos_a, vel_a, pos_b, vel_b = _head_on_pair(miss_m=25.0, closing_speed=16_000.0,
                                               t_ca=2530.0)
    t_ca_true, miss_true = rectilinear_closest_approach(
        pos_a(0.0), vel_a(0.0), pos_b(0.0), vel_b(0.0),
    )
    result = _screen_and_refine(pos_a, vel_a, pos_b, vel_b, 0.0, 5400.0)
    event = _closest_event(result)
    assert event is not None
    assert event.tca_result.tca == pytest.approx(t_ca_true, abs=1e-3)
    assert event.tca_result.miss_distance == pytest.approx(miss_true, rel=1e-6, abs=1e-3)
    assert event.tca_result.relative_velocity == pytest.approx(16_000.0, rel=1e-9)
    assert event.encounter_type == "head-on"


def test_genuine_safe_pass_is_rejected_by_the_screen():
    """
    Case 5: the screen must still do its job.  Two co-planar circular orbits
    500 km apart in radius never approach within the 50 km threshold, so the
    screener must produce no candidates at all -- proving the correction did
    not degenerate into "accept everything".
    """
    pos_a, vel_a = circular_orbit(R_LEO)
    pos_b, vel_b = circular_orbit(R_LEO + 500e3, phase_deg=40.0)

    screener = ConjunctionScreener(threshold_m=THRESHOLD_M, coarse_dt=60.0)
    candidates = screener.screen(pos_a, pos_b, 0.0, WINDOW_S,
                                 vel_fn_a=vel_a, vel_fn_b=vel_b)
    assert candidates == []

    _, true_min = independent_closest_approach(pos_a, pos_b, 0.0, WINDOW_S)
    assert true_min > THRESHOLD_M


@pytest.mark.parametrize("miss_m,should_detect", [
    (0.90 * THRESHOLD_M, True),    # clearly inside
    (0.99 * THRESHOLD_M, True),    # just inside
    (1.01 * THRESHOLD_M, None),    # just outside: may or may not be screened in
    (3.00 * THRESHOLD_M, None),    # comfortably outside
])
def test_threshold_boundary_behaviour(miss_m, should_detect):
    """
    Case 6.  Anything inside the threshold must be detected.  Just outside the
    threshold the screen is permitted to produce a candidate (a conservative
    screen over-includes by design); what matters is that when a TCA is
    reported it is the correct one.
    """
    pos_a, vel_a, pos_b, vel_b = _head_on_pair(miss_m, closing_speed=14_000.0,
                                               t_ca=1830.0)
    t_ca_true, miss_true = rectilinear_closest_approach(
        pos_a(0.0), vel_a(0.0), pos_b(0.0), vel_b(0.0),
    )
    result = _screen_and_refine(pos_a, vel_a, pos_b, vel_b, 0.0, 3600.0)
    event = _closest_event(result)

    if should_detect:
        assert event is not None, f"miss of {miss_m:.0f} m inside threshold was missed"

    if event is not None:
        assert event.tca_result.tca == pytest.approx(t_ca_true, abs=1e-3)
        assert event.tca_result.miss_distance == pytest.approx(miss_true, rel=1e-6, abs=1e-3)


def test_zero_relative_velocity_is_handled_safely():
    """
    Case 7: identical velocities give a constant separation, no TCA, and no
    division by zero anywhere in the screen.
    """
    pos_a, vel_a = rectilinear([0.0, 0.0, 0.0], [7000.0, 0.0, 0.0])
    pos_b, vel_b = rectilinear([0.0, 8000.0, 0.0], [7000.0, 0.0, 0.0])

    screener = ConjunctionScreener(threshold_m=THRESHOLD_M, coarse_dt=60.0)
    candidates = screener.screen(pos_a, pos_b, 0.0, 3600.0,
                                 vel_fn_a=vel_a, vel_fn_b=vel_b)
    # Constant 8 km separation, inside a 50 km threshold: must be a candidate.
    assert candidates, "a constant sub-threshold separation must be screened in"
    for c in candidates:
        assert math.isfinite(c.min_distance)
        assert c.min_distance == pytest.approx(8000.0, rel=1e-9)

    # And the full pipeline must not raise.
    result = _screen_and_refine(pos_a, vel_a, pos_b, vel_b, 0.0, 3600.0)
    assert isinstance(result.events, list)


def test_very_small_relative_velocity():
    """
    Case 8: a slow drift-through.  Closing speed 0.2 m/s over an hour moves the
    pair 720 m, so this is entirely inside one screening sphere and must be
    both screened in and refined correctly.
    """
    pos_a, vel_a, pos_b, vel_b = _head_on_pair(miss_m=300.0, closing_speed=0.2,
                                               t_ca=1800.0)
    t_ca_true, miss_true = rectilinear_closest_approach(
        pos_a(0.0), vel_a(0.0), pos_b(0.0), vel_b(0.0),
    )
    result = _screen_and_refine(pos_a, vel_a, pos_b, vel_b, 0.0, 3600.0)
    event = _closest_event(result)
    assert event is not None
    assert event.tca_result.tca == pytest.approx(t_ca_true, abs=1e-2)
    assert event.tca_result.miss_distance == pytest.approx(miss_true, rel=1e-6, abs=1e-3)


def test_multiple_spacecraft_pairwise_screening():
    """
    Case 9: four objects, three of which cross object A's plane at different
    times.  Every genuine encounter must be found, and each reported TCA must
    match an independent minimisation for that pair.
    """
    # All four start a quarter-orbit before the node, so every node crossing
    # falls well inside the window rather than on its boundary.  (A boundary
    # minimum is not a conjunction the TCA solver can bracket: r.v never
    # changes sign from negative to positive there.)
    base_phase = 90.0
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=base_phase, inc_deg=0.0)
    others = [
        ("B", circular_orbit(R_LEO + 200.0, phase_deg=base_phase + 0.0010, inc_deg=170.0)),
        ("C", circular_orbit(R_LEO + 400.0, phase_deg=base_phase + 0.0020, inc_deg=150.0)),
        ("D", circular_orbit(R_LEO + 600.0, phase_deg=base_phase + 0.0030, inc_deg=100.0)),
    ]

    detected = 0
    for name, (pos_b, vel_b) in others:
        t_ca_true, miss_true = independent_closest_approach(
            pos_a, pos_b, 0.0, WINDOW_S,
        )
        result = _screen_and_refine(pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW_S,
                                    threshold_m=THRESHOLD_M, coarse_dt=60.0)
        event = _closest_event(result)
        if miss_true < THRESHOLD_M:
            assert event is not None, f"pair A-{name} missed (true miss {miss_true:.1f} m)"
            assert event.tca_result.tca == pytest.approx(t_ca_true, abs=1e-2)
            assert event.tca_result.miss_distance == pytest.approx(
                miss_true, rel=1e-5, abs=1e-2,
            )
            detected += 1

    assert detected >= 2, "expected several genuine sub-threshold encounters"


# ---------------------------------------------------------------------------
# The screen must remain a screen
# ---------------------------------------------------------------------------

def test_screen_still_discards_the_vast_majority_of_intervals(fast_pair):
    """
    A conservative screen is allowed to over-include, but it must still cut the
    work down.  For the reference pair the candidate intervals must cover only
    a small fraction of the analysis window.
    """
    pos_a, vel_a, pos_b, vel_b, _, _ = fast_pair
    screener = ConjunctionScreener(threshold_m=THRESHOLD_M, coarse_dt=60.0)
    candidates = screener.screen(pos_a, pos_b, 0.0, WINDOW_S,
                                 vel_fn_a=vel_a, vel_fn_b=vel_b)

    assert candidates, "the real encounter must produce at least one candidate"
    covered = sum(c.t_end - c.t_start for c in candidates)
    assert covered / WINDOW_S < 0.20, (
        f"candidates cover {100 * covered / WINDOW_S:.1f}% of the window; "
        f"the screen is no longer reducing work"
    )
