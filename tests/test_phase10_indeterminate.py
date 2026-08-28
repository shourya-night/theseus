"""
Phase 10 — regression tests for fabricated risk assessments.

Background
----------
Phase 9 screening can fail to detect a genuine high-relative-velocity
conjunction when the coarse sampling step is too large for the encounter
geometry.  When that happened, ``run_uncertainty_conjunction_analysis``
silently evaluated the encounter at the *midpoint of the analysis window*
and ran the full Phase 10 pipeline on that arbitrary point.

For the geometry exercised below, that produced:

    events found      : 0
    reported tca_s    : True            <- boolean, not seconds
    reported miss (km): 0.0155          <- |b0| projection, not |r_rel|
    trace step 7      : "Miss Distance = 10836.3972 km"
    Pc                : 6.542452e-05
    RISK              : HIGH | action_required: True

i.e. a collision-avoidance recommendation for two objects 10 836 km apart,
while the *actual* closest approach in the window was ~9.3 m at
~15.3 km/s and was never found at all.

These tests pin the required behaviour:

1. Absence of a valid TCA must never be classified as HIGH risk or set
   ``action_required``.
2. ``tca_s`` must be a real time in seconds or ``None`` -- never a bool.
3. A reported miss distance must correspond to the actual TCA geometry.
4. A genuine conjunction must still produce a normal Phase 10 analysis.

The reference values used here are derived independently of the engine:
the closest-approach truth is obtained with ``scipy.optimize.minimize_scalar``
applied directly to |r_a(t) - r_b(t)| for analytic circular orbits.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from theseus.uncertainty.covariance import StateCovariance
from theseus.uncertainty.results import run_uncertainty_conjunction_analysis
from theseus.uncertainty.risk import RiskLevel


MU_EARTH = 3.986004418e14      # m^3/s^2
R_LEO = 6778.137e3             # 400 km altitude
WINDOW_S = 7200.0              # 2 hour analysis window


# ---------------------------------------------------------------------------
# Analytic circular-orbit fixtures (independent of the engine propagator)
# ---------------------------------------------------------------------------

def circular_orbit(radius: float, phase_deg: float = 0.0, inc_deg: float = 0.0):
    """Position/velocity callables for a circular orbit, RAAN = 0."""
    n = math.sqrt(MU_EARTH / radius ** 3)
    v_circ = math.sqrt(MU_EARTH / radius)
    phi0 = math.radians(phase_deg)
    inc = math.radians(inc_deg)

    def pos_fn(t: float) -> np.ndarray:
        th = n * t + phi0
        return np.array([
            radius * math.cos(th),
            radius * math.sin(th) * math.cos(inc),
            radius * math.sin(th) * math.sin(inc),
        ])

    def vel_fn(t: float) -> np.ndarray:
        th = n * t + phi0
        return np.array([
            -v_circ * math.sin(th),
            v_circ * math.cos(th) * math.cos(inc),
            v_circ * math.cos(th) * math.sin(inc),
        ])

    return pos_fn, vel_fn


def independent_closest_approach(pos_a, pos_b, t_start: float, t_end: float,
                                 n_scan: int = 20000, n_candidates: int = 60):
    """
    Independently determine (t_ca, miss_distance) by direct minimisation of
    |r_a(t) - r_b(t)|.  Deliberately does NOT use any engine TCA machinery:
    no conjunction module, no r.v root solve, no engine propagator.

    A single-pass grid scan is not sufficient here.  At 15 km/s relative speed
    a 0.36 s grid step spans 5.4 km, so a metre-scale minimum is invisible to
    the grid and two distinct minima separated by centimetres cannot be ranked.
    The search is therefore hierarchical: a coarse scan proposes candidate
    regions, each is re-scanned finely, and the best sub-interval of each is
    then closed with a bounded golden-section minimisation.
    """
    def dist(t: float) -> float:
        return float(np.linalg.norm(np.asarray(pos_a(t)) - np.asarray(pos_b(t))))

    ts = np.linspace(t_start, t_end, n_scan)
    ds = np.array([dist(t) for t in ts])
    step = ts[1] - ts[0]

    # Coarse candidates: the n_candidates smallest samples, plus the window
    # endpoints (a minimum can sit against a boundary).
    order = np.argsort(ds)[:n_candidates]
    candidate_idx = set(int(i) for i in order) | {0, len(ts) - 1}

    best_t, best_d = float(t_start), float("inf")
    for i in sorted(candidate_idx):
        lo = max(t_start, ts[i] - step)
        hi = min(t_end, ts[i] + step)
        if hi <= lo:
            continue
        # Fine re-scan inside the coarse cell, then close it properly.
        sub = np.linspace(lo, hi, 400)
        sub_d = np.array([dist(t) for t in sub])
        j = int(np.argmin(sub_d))
        sub_lo = sub[max(j - 1, 0)]
        sub_hi = sub[min(j + 1, len(sub) - 1)]
        if sub_hi > sub_lo:
            res = minimize_scalar(
                dist, bounds=(sub_lo, sub_hi), method="bounded",
                options={"xatol": 1e-10},
            )
            t_cand, d_cand = float(res.x), float(res.fun)
        else:
            t_cand, d_cand = float(sub[j]), float(sub_d[j])
        if d_cand < best_d:
            best_t, best_d = t_cand, d_cand

    return best_t, best_d


# The pathological pair: same radius, ~170 deg plane difference (so the
# relative speed at the node is ~15.3 km/s), tiny phase offset so that the
# node crossing is a near-miss rather than an exact intersection.
FAST_ENCOUNTER_A = dict(radius=R_LEO, phase_deg=0.0, inc_deg=0.0)
FAST_ENCOUNTER_B = dict(radius=R_LEO, phase_deg=0.0009, inc_deg=170.0)


@pytest.fixture(scope="module")
def fast_encounter():
    pos_a, vel_a = circular_orbit(**FAST_ENCOUNTER_A)
    pos_b, vel_b = circular_orbit(**FAST_ENCOUNTER_B)
    t_ca, miss = independent_closest_approach(pos_a, pos_b, 0.0, WINDOW_S)
    return pos_a, vel_a, pos_b, vel_b, t_ca, miss


def default_covariance(name: str) -> StateCovariance:
    return StateCovariance.from_diagonal(
        [300.0, 300.0, 300.0], [0.3, 0.3, 0.3], name=name,
    )


# ---------------------------------------------------------------------------
# Sanity: the scenario really is what the audit described
# ---------------------------------------------------------------------------

def test_scenario_is_a_genuine_high_speed_near_miss(fast_encounter):
    """The fixture must actually be a ~9 m miss at ~15.3 km/s."""
    pos_a, vel_a, pos_b, vel_b, t_ca, miss = fast_encounter

    assert miss < 100.0, f"expected a sub-100 m closest approach, got {miss:.3f} m"

    v_rel = float(np.linalg.norm(np.asarray(vel_a(t_ca)) - np.asarray(vel_b(t_ca))))
    assert 15.0e3 < v_rel < 15.6e3, f"expected ~15.3 km/s relative speed, got {v_rel:.1f} m/s"


def test_midpoint_separation_is_thousands_of_km(fast_encounter):
    """
    At the analysis-window midpoint -- the point the old fallback silently
    used -- the objects are thousands of km apart.  This is the separation
    that was previously reported as a 15.5 m miss.
    """
    pos_a, _, pos_b, _, _, _ = fast_encounter
    t_mid = 0.5 * WINDOW_S
    sep = float(np.linalg.norm(np.asarray(pos_a(t_mid)) - np.asarray(pos_b(t_mid))))
    assert sep > 5_000e3, f"expected a multi-thousand-km midpoint separation, got {sep/1e3:.1f} km"


# ---------------------------------------------------------------------------
# Core regression: a missed conjunction must not become a risk assessment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coarse_dt", [60.0, 30.0])
def test_missed_conjunction_produces_no_fabricated_risk(fast_encounter, coarse_dt):
    """
    With the shipped Phase 9 screening defaults (module default 60 s and API
    default 30 s) this encounter may not be detected.  Whatever the detection
    outcome, the analysis must never manufacture an actionable risk level from
    an undetected encounter.
    """
    pos_a, vel_a, pos_b, vel_b, t_ca_true, miss_true = fast_encounter

    result = run_uncertainty_conjunction_analysis(
        pos_fn_a=pos_a, vel_fn_a=vel_a,
        pos_fn_b=pos_b, vel_fn_b=vel_b,
        initial_cov_a=default_covariance("Object A"),
        initial_cov_b=default_covariance("Object B"),
        t_start=0.0, t_end=WINDOW_S,
        hbr_m=20.0,
        screening_threshold_m=50e3,
        coarse_dt=coarse_dt,
    )

    payload = result.to_dict()
    summary = payload["conjunction_summary"]
    risk = payload["risk_assessment"]

    found = bool(result.conjunction_result.events)

    if not found:
        # No valid TCA: the result must declare itself indeterminate and must
        # not be actionable.
        assert result.conjunction_found is False
        assert result.risk_assessment is None or \
            result.risk_assessment.level != RiskLevel.HIGH
        assert risk.get("action_required") is not True, (
            "an undetected conjunction produced action_required=True"
        )
        assert risk.get("level") not in ("HIGH", "CRITICAL"), (
            f"an undetected conjunction was classified as {risk.get('level')}"
        )
        assert summary["tca_s"] is None
        assert summary["miss_distance_km"] is None
        assert payload["collision_probability"]["probability"] is None
    else:
        # If it *was* detected, the geometry must be the real one.
        assert summary["tca_s"] == pytest.approx(t_ca_true, abs=1.0)
        assert summary["miss_distance_km"] * 1e3 == pytest.approx(miss_true, abs=1.0)


@pytest.mark.parametrize("coarse_dt", [60.0, 30.0])
def test_tca_s_is_never_a_boolean(fast_encounter, coarse_dt):
    """
    ``tca_s`` previously carried the B-plane *applicability flag*.  It must be
    a real time in seconds, or None -- never a bool.
    """
    pos_a, vel_a, pos_b, vel_b, _, _ = fast_encounter

    result = run_uncertainty_conjunction_analysis(
        pos_fn_a=pos_a, vel_fn_a=vel_a,
        pos_fn_b=pos_b, vel_fn_b=vel_b,
        initial_cov_a=default_covariance("Object A"),
        initial_cov_b=default_covariance("Object B"),
        t_start=0.0, t_end=WINDOW_S,
        hbr_m=20.0,
        screening_threshold_m=50e3,
        coarse_dt=coarse_dt,
    )

    tca_s = result.to_dict()["conjunction_summary"]["tca_s"]
    assert not isinstance(tca_s, bool), f"tca_s serialised as a bool: {tca_s!r}"
    assert tca_s is None or isinstance(tca_s, float)


@pytest.mark.parametrize("coarse_dt", [60.0, 30.0])
def test_no_high_risk_when_objects_are_thousands_of_km_apart(fast_encounter, coarse_dt):
    """
    The specific failure from the audit: HIGH risk / action_required reported
    for an evaluation point where the true separation was 10 836 km.

    Whatever point the analysis reports, the *actual* separation at that point
    must be consistent with the risk it assigns.  An actionable risk level
    requires a separation that is at least plausibly comparable to the
    uncertainty scale -- not thousands of kilometres.
    """
    pos_a, vel_a, pos_b, vel_b, _, _ = fast_encounter

    result = run_uncertainty_conjunction_analysis(
        pos_fn_a=pos_a, vel_fn_a=vel_a,
        pos_fn_b=pos_b, vel_fn_b=vel_b,
        initial_cov_a=default_covariance("Object A"),
        initial_cov_b=default_covariance("Object B"),
        t_start=0.0, t_end=WINDOW_S,
        hbr_m=20.0,
        screening_threshold_m=50e3,
        coarse_dt=coarse_dt,
    )

    payload = result.to_dict()
    risk = payload["risk_assessment"]
    if risk.get("level") not in ("HIGH", "CRITICAL") and risk.get("action_required") is not True:
        return  # not actionable, nothing further to check

    # Actionable: verify the reported geometry against ground truth at the
    # reported time, computed directly from the orbit fixtures.
    tca_s = payload["conjunction_summary"]["tca_s"]
    assert isinstance(tca_s, float)
    true_sep = float(np.linalg.norm(np.asarray(pos_a(tca_s)) - np.asarray(pos_b(tca_s))))
    assert true_sep < 100e3, (
        f"risk level {risk.get('level')} / action_required={risk.get('action_required')} "
        f"reported at a point where the objects are {true_sep/1e3:.1f} km apart"
    )


def test_reported_miss_distance_matches_true_tca_geometry(fast_encounter):
    """
    The reported miss distance must be |r_rel| at the reported TCA, not the
    B-plane projection |b0| and not a value taken from some other point.
    Uses a screening step fine enough for this geometry so a TCA is found.
    """
    pos_a, vel_a, pos_b, vel_b, t_ca_true, miss_true = fast_encounter

    result = run_uncertainty_conjunction_analysis(
        pos_fn_a=pos_a, vel_fn_a=vel_a,
        pos_fn_b=pos_b, vel_fn_b=vel_b,
        initial_cov_a=default_covariance("Object A"),
        initial_cov_b=default_covariance("Object B"),
        t_start=0.0, t_end=WINDOW_S,
        hbr_m=20.0,
        screening_threshold_m=50e3,
        coarse_dt=5.0,
    )

    assert result.conjunction_found is True
    summary = result.to_dict()["conjunction_summary"]

    assert summary["tca_s"] == pytest.approx(t_ca_true, abs=1e-3)
    reported_miss_m = summary["miss_distance_km"] * 1e3
    assert reported_miss_m == pytest.approx(miss_true, rel=1e-6, abs=1e-3)

    # And it must equal |r_rel| recomputed at the reported time.
    sep_at_tca = float(np.linalg.norm(
        np.asarray(pos_a(summary["tca_s"])) - np.asarray(pos_b(summary["tca_s"]))
    ))
    assert reported_miss_m == pytest.approx(sep_at_tca, rel=1e-6, abs=1e-3)


# ---------------------------------------------------------------------------
# The happy path must still work
# ---------------------------------------------------------------------------

def test_valid_conjunction_still_produces_full_analysis():
    """
    A well-separated-in-time, comfortably detectable conjunction must still
    run the complete Phase 10 pipeline: propagated covariances, B-plane
    projection, a finite Pc in [0, 1], a risk classification, and the
    14-step calculation trace.
    """
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=0.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO + 50.0, phase_deg=0.4, inc_deg=12.0)

    t_ca_true, miss_true = independent_closest_approach(pos_a, pos_b, 0.0, WINDOW_S)

    result = run_uncertainty_conjunction_analysis(
        pos_fn_a=pos_a, vel_fn_a=vel_a,
        pos_fn_b=pos_b, vel_fn_b=vel_b,
        initial_cov_a=default_covariance("Object A"),
        initial_cov_b=default_covariance("Object B"),
        t_start=0.0, t_end=WINDOW_S,
        hbr_m=20.0,
        screening_threshold_m=200e3,
        coarse_dt=5.0,
    )

    assert result.conjunction_found is True
    assert result.risk_assessment is not None
    assert result.collision_probability is not None

    payload = result.to_dict()
    summary = payload["conjunction_summary"]

    assert isinstance(summary["tca_s"], float)
    assert summary["tca_s"] == pytest.approx(t_ca_true, abs=1e-2)
    assert summary["miss_distance_km"] * 1e3 == pytest.approx(miss_true, rel=1e-6, abs=1e-3)
    assert summary["relative_velocity_km_s"] > 0.0

    pc = payload["collision_probability"]["probability"]
    assert pc is not None
    assert 0.0 <= pc <= 1.0

    assert payload["risk_assessment"]["level"] in ("LOW", "ELEVATED", "HIGH", "CRITICAL")
    assert len(result.calculation_steps) == 14
    assert result.calculation_steps[0]["stepIndex"] == 1
    assert result.calculation_steps[13]["stepIndex"] == 14


def test_indeterminate_result_trace_is_explicit():
    """
    When no conjunction exists at all (two widely separated coplanar orbits
    that never approach), the analysis must say so in the calculation trace
    rather than silently producing numbers.
    """
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=0.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO + 3000e3, phase_deg=180.0, inc_deg=0.0)

    result = run_uncertainty_conjunction_analysis(
        pos_fn_a=pos_a, vel_fn_a=vel_a,
        pos_fn_b=pos_b, vel_fn_b=vel_b,
        initial_cov_a=default_covariance("Object A"),
        initial_cov_b=default_covariance("Object B"),
        t_start=0.0, t_end=WINDOW_S,
        hbr_m=20.0,
        screening_threshold_m=50e3,
        coarse_dt=30.0,
    )

    assert result.conjunction_found is False
    payload = result.to_dict()

    assert payload["analysis_status"] == "INDETERMINATE_NO_CONJUNCTION"
    assert payload["conjunction_summary"]["tca_s"] is None
    assert payload["collision_probability"]["probability"] is None
    assert payload["risk_assessment"]["level"] == "INDETERMINATE"
    assert payload["risk_assessment"]["action_required"] is False

    titles = [s.get("title", "") for s in result.calculation_steps]
    assert any("No Conjunction" in t or "Indeterminate" in t for t in titles), titles
