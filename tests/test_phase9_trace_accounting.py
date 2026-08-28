"""
P9-04 — the Phase 9 calculation trace must be truthful.

The trace used to report ``f"{len(all_tcas)} TCA(s) found"``, where
``all_tcas`` was the list of *converged* refinements.  Solutions that failed
local-minimum validation were then dropped with a bare ``continue``, so the
trace could claim more TCAs than the result contained, and nothing recorded
why the difference existed.  Three of the five counts a reader needs were not
observable from ``analyse()`` at all.

These tests compare every trace count against the returned collections, and
pin the accounting invariants for this architecture.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from theseus.conjunction.analysis import ConjunctionAnalysis, ConjunctionAccounting
from theseus.conjunction.screening import ConjunctionScreener
from theseus.conjunction.tca import (
    find_tca,
    find_all_tca,
    find_all_tca_with_diagnostics,
)

from tests._conjunction_reference import (
    R_LEO,
    circular_orbit,
    rectilinear,
    rectilinear_closest_approach,
    independent_closest_approach,
)


WINDOW = 7200.0
THRESHOLD = 50e3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(result, title: str):
    """The single trace step with the given title, or None."""
    hits = [s for s in result.calculation_steps if s.get("title") == title]
    assert len(hits) <= 1, f"{title!r} appears {len(hits)} times"
    return hits[0] if hits else None


def _check_invariants(result) -> ConjunctionAccounting:
    """
    Assert the documented accounting invariants and cross-check every count
    against the returned collections.
    """
    a = result.accounting

    # Each stage narrows the previous one.
    assert a.intervals_accepted <= a.intervals_screened
    assert a.candidate_intervals <= a.intervals_accepted
    assert a.tca_converged <= a.tca_attempts
    assert a.tca_validated <= a.tca_converged
    assert a.accepted_conjunctions == a.tca_validated

    # Rejections must account for the whole difference at each stage.
    assert a.tca_rejected_not_converged == a.tca_attempts - a.tca_converged
    assert a.tca_rejected_not_validated == a.tca_converged - a.tca_validated

    # Counts must match the actual returned collections, not an estimate.
    assert a.candidate_intervals == len(result.candidate_intervals)
    assert a.accepted_conjunctions == len(result.events)

    # Non-negativity throughout.
    for name, value in a.to_dict().items():
        if isinstance(value, int):
            assert value >= 0, f"{name} is negative"

    return a


def _head_on_pair(miss_m: float, closing_speed: float, t_ca: float):
    half = 0.5 * closing_speed
    pos_a, vel_a = rectilinear([-half * t_ca, 0.0, 0.0], [half, 0.0, 0.0])
    pos_b, vel_b = rectilinear([half * t_ca, miss_m, 0.0], [-half, 0.0, 0.0])
    return pos_a, vel_a, pos_b, vel_b


# ---------------------------------------------------------------------------
# 1: zero candidate intervals
# ---------------------------------------------------------------------------

def test_zero_candidate_intervals():
    """Screening rejects everything: no attempts, and the trace says so."""
    pos_a, vel_a = circular_orbit(R_LEO)
    pos_b, vel_b = circular_orbit(R_LEO + 500e3, phase_deg=40.0)

    result = ConjunctionAnalysis(THRESHOLD, coarse_dt=60.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW,
    )
    a = _check_invariants(result)

    assert a.candidate_intervals == 0
    assert a.tca_attempts == 0
    assert a.tca_converged == 0
    assert a.accepted_conjunctions == 0
    assert result.events == []

    step = _step(result, "No Candidate Intervals — Analysis Complete")
    assert step is not None
    assert step["substitutions"]["tca_attempts"] == 0
    assert step["substitutions"]["intervals_screened"] == a.intervals_screened
    # The old "N TCA(s) found" wording must not appear anywhere.
    assert not any("TCA(s) found" in str(s.get("result", ""))
                   for s in result.calculation_steps)


# ---------------------------------------------------------------------------
# 2, 3: one and many candidates producing TCAs
# ---------------------------------------------------------------------------

def test_single_candidate_single_tca():
    pos_a, vel_a, pos_b, vel_b = _head_on_pair(120.0, 14_000.0, 1830.0)
    result = ConjunctionAnalysis(THRESHOLD, coarse_dt=60.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, 3600.0,
    )
    a = _check_invariants(result)

    assert a.candidate_intervals >= 1
    assert a.tca_attempts >= 1
    assert a.accepted_conjunctions == 1
    assert len(result.events) == 1

    step = _step(result, "TCA Refinement")
    assert step["substitutions"]["tca_attempts"] == a.tca_attempts
    assert step["substitutions"]["tca_converged"] == a.tca_converged
    assert step["substitutions"]["tca_validated"] == a.tca_validated
    assert len(step["iterations"]) == a.tca_converged


def test_multiple_candidates_multiple_tcas():
    """Two crossing LEO orbits give a node crossing roughly twice per orbit."""
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=90.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO + 200.0, phase_deg=90.0009, inc_deg=170.0)

    result = ConjunctionAnalysis(THRESHOLD, coarse_dt=60.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW,
    )
    a = _check_invariants(result)

    assert a.candidate_intervals >= 2
    assert a.accepted_conjunctions >= 2

    step = _step(result, "Conjunctions Reported")
    assert step is not None
    assert step["substitutions"]["accepted_conjunctions"] == len(result.events)


def test_tca_attempts_may_exceed_candidate_intervals():
    """
    The documented departure from a naive chain: one candidate span can hold
    several sign changes of r_rel . v_rel, so attempts are not bounded by
    candidate intervals.  Verified by counting brackets independently.
    """
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=90.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO, phase_deg=90.0, inc_deg=170.0)

    # One huge threshold makes the whole window a single candidate span.
    analysis = ConjunctionAnalysis(1e12, coarse_dt=60.0)
    result = analysis.analyse(pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW)
    a = _check_invariants(result)

    assert a.candidate_intervals == 1
    assert a.tca_attempts > a.candidate_intervals, (
        "this fixture is meant to contain several brackets in one span"
    )

    # Independent bracket count over the same span, from the definition.
    ci = result.candidate_intervals[0]
    ts = np.linspace(ci.t_start, ci.t_end, 200)

    def f(t):
        r = np.asarray(pos_a(t)) - np.asarray(pos_b(t))
        v = np.asarray(vel_a(t)) - np.asarray(vel_b(t))
        return float(np.dot(r, v))

    fv = [f(t) for t in ts]
    brackets = sum(1 for i in range(len(fv) - 1) if fv[i] < 0 <= fv[i + 1])
    assert a.tca_attempts == brackets


# ---------------------------------------------------------------------------
# 4: refinements that fail
# ---------------------------------------------------------------------------

def test_non_converged_refinements_are_counted_not_claimed():
    """
    A refinement that does not converge must be counted as an attempt and as a
    failure, never as a TCA.  Verified against the solver's own diagnostics.
    """
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=90.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO + 300.0, phase_deg=90.001, inc_deg=140.0)

    results, diag = find_all_tca_with_diagnostics(
        pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW, n_samples=200,
    )
    assert diag.attempts == diag.brackets_found
    assert diag.converged == len(results)
    assert diag.non_converged == diag.attempts - diag.converged
    assert diag.converged <= diag.attempts

    # The thin wrapper must return exactly the same list.
    assert [r.tca for r in find_all_tca(
        pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW, n_samples=200,
    )] == [r.tca for r in results]


def test_a_converged_solve_on_a_maximum_is_not_a_valid_tca():
    """
    The mechanism behind the original discrepancy: Brent converges on a
    positive-to-negative crossing of r_rel . v_rel, which is the moment of
    *greatest* separation.  The solve converges; the result is not a TCA.

    Reaching this through find_all_tca is rare because it only brackets
    negative-to-positive crossings, which is why the old trace inconsistency
    was latent rather than routine -- but the accounting must handle it.
    """
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=0.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO, phase_deg=0.0009, inc_deg=170.0)

    def f(t):
        r = np.asarray(pos_a(t)) - np.asarray(pos_b(t))
        v = np.asarray(vel_a(t)) - np.asarray(vel_b(t))
        return float(np.dot(r, v))

    ts = np.linspace(0.0, WINDOW, 4000)
    fv = [f(t) for t in ts]
    bracket = next((ts[i], ts[i + 1]) for i in range(len(fv) - 1)
                   if fv[i] > 0 >= fv[i + 1])

    res = find_tca(pos_a, vel_a, pos_b, vel_b, bracket[0], bracket[1])
    assert res.converged is True
    assert res.validated is False
    assert "local maximum" in res.validation_note


def test_validation_failures_are_counted_and_explained(monkeypatch):
    """
    Converged-but-unvalidated solutions must be counted as rejections, must
    not appear as conjunctions, and must be explained in the trace with their
    reasons.

    The solver is a validated component and is not modified; instead its
    search is stubbed for this one test so the accounting sees a controlled
    mix of validated and rejected solves.  What is under test here is the
    accounting, not the refinement.
    """
    import theseus.conjunction.analysis as analysis_mod
    from theseus.conjunction.tca import TCAResult, TCASearchDiagnostics

    def make(tca_s: float, validated: bool, note: str) -> TCAResult:
        return TCAResult(
            tca=tca_s, miss_distance=1234.0, relative_velocity=7000.0,
            r_rel=np.array([1234.0, 0.0, 0.0]), v_rel=np.array([0.0, 7000.0, 0.0]),
            r_a=np.array([R_LEO, 0.0, 0.0]), v_a=np.array([0.0, 7600.0, 0.0]),
            r_b=np.array([R_LEO - 1234.0, 0.0, 0.0]), v_b=np.array([0.0, 600.0, 0.0]),
            converged=True, iterations=7,
            validated=validated, validation_note=note,
        )

    stub_results = [
        make(100.0, True, "Validated: distance is locally minimised at TCA"),
        make(200.0, False, "TCA SOLUTION INVALID: local maximum (objects receding → approaching), not minimum"),
        make(300.0, False, "TCA SOLUTION INVALID: solution outside analysis window"),
    ]
    stub_diag = TCASearchDiagnostics(
        samples=200, brackets_found=5, attempts=5, converged=3, non_converged=2,
    )

    def fake_search(*args, **kwargs):
        return list(stub_results), stub_diag

    monkeypatch.setattr(analysis_mod, "find_all_tca_with_diagnostics", fake_search)

    pos_a, vel_a, pos_b, vel_b = _head_on_pair(120.0, 14_000.0, 1830.0)
    result = ConjunctionAnalysis(THRESHOLD, coarse_dt=60.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, 3600.0,
    )
    a = _check_invariants(result)

    n_spans = a.candidate_intervals
    assert a.tca_attempts == 5 * n_spans
    assert a.tca_converged == 3 * n_spans
    assert a.tca_validated == 1 * n_spans
    assert a.tca_rejected_not_converged == 2 * n_spans
    assert a.tca_rejected_not_validated == 2 * n_spans
    assert a.accepted_conjunctions == 1 * n_spans
    assert len(result.events) == 1 * n_spans

    step = _step(result, "Rejected TCA Solutions")
    assert step is not None
    assert step["substitutions"]["rejected_count"] == a.tca_rejected_not_validated
    assert len(step["substitutions"]["rejected"]) == a.tca_rejected_not_validated
    for entry in step["substitutions"]["rejected"]:
        assert entry["reason"]
    assert any("local maximum" in r for r in a.rejection_reasons)
    assert any("outside analysis window" in r for r in a.rejection_reasons)

    # The refinement step must report the attempt and the failure, and must
    # never present an attempt as a TCA.
    tca_step = _step(result, "TCA Refinement")
    assert tca_step["substitutions"]["tca_attempts"] == a.tca_attempts
    assert tca_step["substitutions"]["tca_validated"] == a.tca_validated
    assert "TCA(s) found" not in tca_step["result"]


def test_all_solves_rejected_yields_an_explicit_no_conjunction_step(monkeypatch):
    """
    The exact reported failure: candidates existed, refinements ran, nothing
    survived validation.  The old trace claimed TCAs had been found and then
    simply stopped.  It must now end with an explicit statement.
    """
    import theseus.conjunction.analysis as analysis_mod
    from theseus.conjunction.tca import TCAResult, TCASearchDiagnostics

    rejected = TCAResult(
        tca=500.0, miss_distance=99.0, relative_velocity=7000.0,
        r_rel=np.array([99.0, 0.0, 0.0]), v_rel=np.array([0.0, 7000.0, 0.0]),
        r_a=np.zeros(3), v_a=np.zeros(3), r_b=np.zeros(3), v_b=np.zeros(3),
        converged=True, iterations=4, validated=False,
        validation_note="TCA SOLUTION INVALID: local maximum (objects receding → approaching), not minimum",
    )
    monkeypatch.setattr(
        analysis_mod, "find_all_tca_with_diagnostics",
        lambda *a, **k: ([rejected], TCASearchDiagnostics(
            samples=200, brackets_found=1, attempts=1, converged=1, non_converged=0,
        )),
    )

    pos_a, vel_a, pos_b, vel_b = _head_on_pair(120.0, 14_000.0, 1830.0)
    result = ConjunctionAnalysis(THRESHOLD, coarse_dt=60.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, 3600.0,
    )
    a = _check_invariants(result)

    assert a.candidate_intervals >= 1
    assert a.tca_attempts >= 1
    assert a.tca_converged >= 1
    assert a.tca_validated == 0
    assert a.accepted_conjunctions == 0
    assert result.events == []

    terminal = _step(result, "No Conjunction Accepted")
    assert terminal is not None
    assert terminal is result.calculation_steps[-1]
    assert terminal["substitutions"]["accepted_conjunctions"] == 0
    assert terminal["substitutions"]["tca_attempts"] == a.tca_attempts
    assert _step(result, "Conjunctions Reported") is None
    assert _step(result, "Rejected TCA Solutions") is not None

    tca_step = _step(result, "TCA Refinement")
    assert tca_step["status"] == "no_solution"


# ---------------------------------------------------------------------------
# 5: validated TCAs that sit beyond the screening threshold
# ---------------------------------------------------------------------------

def test_events_beyond_the_screening_threshold_are_surfaced():
    """
    The conservative screen admits intervals it cannot prove clear, so an
    accepted event may have a miss distance above the threshold.  That is not
    a rejection, but the trace must say how many such events there are rather
    than leaving the reader to notice.
    """
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=90.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO + 200.0, phase_deg=90.0009, inc_deg=170.0)

    result = ConjunctionAnalysis(1e12, coarse_dt=60.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW,
    )
    a = _check_invariants(result)

    expected = sum(1 for e in result.events
                   if e.tca_result.miss_distance > result.screening_threshold_m)
    assert a.accepted_beyond_screening_threshold == expected

    step = _step(result, "Conjunctions Reported")
    assert step["substitutions"]["accepted_beyond_screening_threshold"] == expected


# ---------------------------------------------------------------------------
# 6: overlapping / adjacent candidate intervals
# ---------------------------------------------------------------------------

def test_merged_candidate_spans_are_counted_once():
    """
    Adjacent accepted intervals merge into one span, so candidate_intervals is
    at most intervals_accepted.  Cross-checked against the screener directly.
    """
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=90.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO + 200.0, phase_deg=90.0009, inc_deg=170.0)

    screener = ConjunctionScreener(threshold_m=THRESHOLD, coarse_dt=15.0)
    candidates, diag = screener.screen_with_diagnostics(
        pos_a, pos_b, 0.0, WINDOW, vel_fn_a=vel_a, vel_fn_b=vel_b,
    )
    assert diag.candidate_intervals == len(candidates)
    assert diag.intervals_accepted >= diag.candidate_intervals
    assert diag.intervals_rejected == diag.intervals_screened - diag.intervals_accepted

    # No span overlaps its neighbour, and all are ordered.
    for prev, nxt in zip(candidates[:-1], candidates[1:]):
        assert prev.t_end <= nxt.t_start

    result = ConjunctionAnalysis(THRESHOLD, coarse_dt=15.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW,
    )
    a = _check_invariants(result)
    assert a.candidate_intervals == len(candidates)
    assert a.intervals_screened == diag.intervals_screened


# ---------------------------------------------------------------------------
# 7, 8: the established regression cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coarse_dt", [60.0, 30.0])
def test_accounting_on_the_known_high_speed_encounter(coarse_dt):
    """The 9.28 m encounter: detection unchanged, accounting consistent."""
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=0.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO, phase_deg=0.0009, inc_deg=170.0)
    t_ca_true, miss_true = independent_closest_approach(pos_a, pos_b, 0.0, WINDOW)

    result = ConjunctionAnalysis(THRESHOLD, coarse_dt=coarse_dt).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW,
    )
    a = _check_invariants(result)

    assert a.accepted_conjunctions >= 1
    best = min(result.events, key=lambda e: e.tca_result.miss_distance)
    assert best.tca_result.tca == pytest.approx(t_ca_true, abs=1e-3)
    assert best.tca_result.miss_distance == pytest.approx(miss_true, rel=1e-6, abs=1e-3)

    assert _step(result, "Conjunctions Reported") is not None
    assert _step(result, "No Conjunction Accepted") is None


def test_no_conjunction_case_accounting():
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=0.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO + 3000e3, phase_deg=180.0, inc_deg=0.0)

    result = ConjunctionAnalysis(THRESHOLD, coarse_dt=30.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW,
    )
    a = _check_invariants(result)
    assert a.accepted_conjunctions == 0
    assert result.events == []


# ---------------------------------------------------------------------------
# 9: multi-spacecraft
# ---------------------------------------------------------------------------

def test_multi_spacecraft_accounting_is_per_pair_consistent():
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=90.0, inc_deg=0.0)
    others = [
        circular_orbit(R_LEO + 200.0, phase_deg=90.0010, inc_deg=170.0),
        circular_orbit(R_LEO + 400.0, phase_deg=90.0020, inc_deg=150.0),
        circular_orbit(R_LEO + 600.0, phase_deg=90.0030, inc_deg=100.0),
    ]
    for pos_b, vel_b in others:
        result = ConjunctionAnalysis(THRESHOLD, coarse_dt=60.0).analyse(
            pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW,
        )
        _check_invariants(result)


# ---------------------------------------------------------------------------
# Trace truthfulness as a whole
# ---------------------------------------------------------------------------

def test_every_trace_count_matches_the_returned_result():
    """
    Sweep a range of geometries and thresholds; for each, every count that
    appears anywhere in the trace must agree with the accounting, which in
    turn agrees with the returned collections.
    """
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=90.0, inc_deg=0.0)
    cases = [
        (circular_orbit(R_LEO + 200.0, phase_deg=90.0009, inc_deg=170.0), THRESHOLD, 60.0),
        (circular_orbit(R_LEO + 200.0, phase_deg=90.0009, inc_deg=170.0), 1e12, 60.0),
        (circular_orbit(R_LEO + 500e3, phase_deg=40.0, inc_deg=0.0), THRESHOLD, 60.0),
        (circular_orbit(R_LEO + 1000.0, phase_deg=91.0, inc_deg=30.0), 200e3, 30.0),
    ]
    for (pos_b, vel_b), threshold, dt in cases:
        result = ConjunctionAnalysis(threshold, coarse_dt=dt).analyse(
            pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW,
        )
        a = _check_invariants(result)

        for step in result.calculation_steps:
            subs = step.get("substitutions") or {}
            for key, value in subs.items():
                if not isinstance(value, int):
                    continue
                if hasattr(a, key):
                    assert value == getattr(a, key), (
                        f"trace step {step['title']!r} reports {key}={value} "
                        f"but the accounting says {getattr(a, key)}"
                    )

        # Step indices must be unique and ascending.
        indices = [s["stepIndex"] for s in result.calculation_steps]
        assert indices == sorted(indices)
        assert len(indices) == len(set(indices))

        # Exactly one terminal step, always present.
        terminals = [s for s in result.calculation_steps if s["title"] in (
            "Conjunctions Reported", "No Conjunction Accepted",
            "No Candidate Intervals — Analysis Complete",
        )]
        assert len(terminals) == 1, [s["title"] for s in result.calculation_steps]
        assert terminals[0] is result.calculation_steps[-1]


def test_accounting_is_serialised():
    pos_a, vel_a, pos_b, vel_b = _head_on_pair(120.0, 14_000.0, 1830.0)
    result = ConjunctionAnalysis(THRESHOLD, coarse_dt=60.0).analyse(
        pos_a, vel_a, pos_b, vel_b, 0.0, 3600.0,
    )
    payload = result.to_dict()
    assert "accounting" in payload
    acc = payload["accounting"]
    assert acc["accepted_conjunctions"] == len(payload["events"])
    assert acc["candidate_intervals"] == len(payload["candidate_intervals"])
    assert acc["tca_converged"] <= acc["tca_attempts"]
