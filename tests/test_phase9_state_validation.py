"""
B-2 — a non-finite state must never be reportable as "no conjunction".

The defect
----------
Every stage of Phase 9 is floating-point arithmetic that propagates NaN
silently.  Before this fix:

* ``ConjunctionScreener.screen`` computed a NaN separation bound, evaluated
  ``NaN < threshold`` as False, and *rejected the interval as provably clear*;
* the TCA search evaluated ``f(t) = r_rel . v_rel`` as NaN, found no
  ``f[i] < 0 and f[i+1] >= 0`` sign change, and reported no closest approach;
* ``ConjunctionAnalysis.analyse`` returned zero events with a complete,
  internally consistent trace.

So a corrupted or diverged trajectory was indistinguishable in the output from
a genuinely clear pass -- the same failure class as the fabricated Phase 10
risk output: an absence of valid information rendered as a valid negative
answer.

What these tests pin
--------------------
1. Every non-finite state that enters the pipeline raises
   :class:`NonFiniteStateError`, at screening, at TCA refinement, and through
   ``ConjunctionAnalysis``, for both objects, both quantities, every vector
   component, and NaN / +inf / -inf alike.
2. The failure is *not* a "nothing found" result.  Each corruption test is
   paired with the identical uncorrupted scenario, which is asserted to
   produce a real conjunction -- so an exception cannot be mistaken for "there
   was nothing there anyway".
3. The diagnostic names which object, which quantity, which component, and at
   what time.
4. Finite states behave exactly as before: identical candidates, identical
   TCAs, identical accounting, identical arrays -- and the same number of
   trajectory evaluations, so the guard adds checks and not work.

The expected TCA used to prove (2) comes from ``tests/_conjunction_reference``,
which imports nothing from ``theseus.conjunction``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from theseus.conjunction.analysis import ConjunctionAnalysis
from theseus.conjunction.screening import ConjunctionScreener
from theseus.conjunction.state_validation import (
    NonFiniteStateError,
    QUANTITY_POSITION,
    QUANTITY_VELOCITY,
    guard_state_function,
    validate_state_vector,
)
from theseus.conjunction.tca import (
    find_all_tca,
    find_all_tca_with_diagnostics,
    find_tca,
)

from tests._conjunction_reference import (
    R_LEO,
    circular_orbit,
    independent_closest_approach,
)


WINDOW = 7200.0
THRESHOLD = 50e3
COARSE_DT = 30.0

NON_FINITE_VALUES = (float("nan"), float("inf"), float("-inf"))
COMPONENTS = (0, 1, 2)


# ---------------------------------------------------------------------------
# Scenario: a genuine crossing conjunction, so "no events" is never the
# uncorrupted answer.  Every corruption test below is built on this pair.
# ---------------------------------------------------------------------------

def clean_pair():
    """Two crossing circular orbits with a real conjunction in the window."""
    pos_a, vel_a = circular_orbit(R_LEO, phase_deg=0.0, inc_deg=0.0)
    pos_b, vel_b = circular_orbit(R_LEO + 60.0, phase_deg=0.3, inc_deg=4.0)
    return pos_a, vel_a, pos_b, vel_b


@pytest.fixture(scope="module")
def reference_tca():
    """
    Independent closest approach for the clean scenario.

    Obtained by direct minimisation of |r_a(t) - r_b(t)| with SciPy, with no
    involvement from theseus.conjunction, so it can be used to assert that the
    uncorrupted scenario really does contain a conjunction.
    """
    pos_a, _, pos_b, _ = clean_pair()
    return independent_closest_approach(pos_a, pos_b, 0.0, WINDOW)


def corrupt(fn, *, component: int, value: float, when=None):
    """
    Wrap a trajectory function so it returns a non-finite component.

    Parameters
    ----------
    component : int
        Which component of the returned vector to poison.
    value : float
        The non-finite value to inject.
    when : callable, optional
        ``when(t) -> bool``.  When supplied, only times satisfying it are
        corrupted, which lets a test place the corruption *after* the first
        few pipeline stages have already succeeded.
    """
    def corrupted(t: float) -> np.ndarray:
        out = np.array(fn(t), dtype=np.float64)
        if when is None or when(float(t)):
            out[component] = value
        return out
    return corrupted


def counting(fn):
    """Wrap a trajectory function so it records how often it was evaluated."""
    calls = []

    def counted(t: float) -> np.ndarray:
        calls.append(float(t))
        return fn(t)

    counted.calls = calls
    return counted


# ---------------------------------------------------------------------------
# 0. The clean scenario really does produce a conjunction
# ---------------------------------------------------------------------------

def test_clean_scenario_produces_a_real_conjunction(reference_tca):
    """
    Anchor for every corruption test below.

    If this scenario produced no events, "raises instead of returning zero
    events" would be a vacuous claim -- zero events would have been correct.
    """
    t_ref, d_ref = reference_tca
    assert 0.0 < t_ref < WINDOW
    assert d_ref < THRESHOLD

    pos_a, vel_a, pos_b, vel_b = clean_pair()
    result = ConjunctionAnalysis(
        screening_threshold_m=THRESHOLD, coarse_dt=COARSE_DT
    ).analyse(pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW)

    assert len(result.events) >= 1
    best = min(result.events, key=lambda e: e.miss_distance_m)
    assert best.tca_result.tca == pytest.approx(t_ref, abs=1e-3)
    assert best.miss_distance_m == pytest.approx(d_ref, rel=1e-6)


# ---------------------------------------------------------------------------
# 1. Screening
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", NON_FINITE_VALUES)
@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("obj", ("A", "B"))
@pytest.mark.parametrize("quantity", (QUANTITY_POSITION, QUANTITY_VELOCITY))
def test_screening_raises_for_every_object_quantity_component_and_value(
    obj, quantity, component, value
):
    """
    36 cases: {A, B} x {position, velocity} x {x, y, z} x {nan, +inf, -inf}.

    Before the fix each of these returned a candidate list -- usually empty,
    because a NaN bound compares False against the threshold and the interval
    is discarded as clear.
    """
    pos_a, vel_a, pos_b, vel_b = clean_pair()
    fns = {"A": {"position": "pos_a", "velocity": "vel_a"},
           "B": {"position": "pos_b", "velocity": "vel_b"}}
    target = fns[obj][quantity]
    local = {"pos_a": pos_a, "vel_a": vel_a, "pos_b": pos_b, "vel_b": vel_b}
    local[target] = corrupt(local[target], component=component, value=value)

    screener = ConjunctionScreener(threshold_m=THRESHOLD, coarse_dt=COARSE_DT)
    with pytest.raises(NonFiniteStateError) as excinfo:
        screener.screen(
            local["pos_a"], local["pos_b"], 0.0, WINDOW,
            vel_fn_a=local["vel_a"], vel_fn_b=local["vel_b"],
            object_a_id="A", object_b_id="B",
        )

    err = excinfo.value
    assert err.object_id == obj
    assert err.quantity == quantity
    assert component in err.invalid_indices
    assert err.invalid_components == ("xyz"[component],)


def test_screening_does_not_silently_reject_a_corrupted_interval():
    """
    The specific pre-fix mechanism: NaN bound, ``NaN < threshold`` is False,
    interval rejected as provably clear.

    The corruption here is placed only in the second half of the window, so
    screening has already produced finite bounds for the first half before it
    meets the bad state.  A partially-corrupted screen must still fail rather
    than return the candidates it managed to find.
    """
    pos_a, vel_a, pos_b, vel_b = clean_pair()
    bad_a = corrupt(pos_a, component=1, value=float("nan"),
                    when=lambda t: t > WINDOW / 2)

    screener = ConjunctionScreener(threshold_m=THRESHOLD, coarse_dt=COARSE_DT)
    with pytest.raises(NonFiniteStateError) as excinfo:
        screener.screen(bad_a, pos_b, 0.0, WINDOW, vel_fn_a=vel_a, vel_fn_b=vel_b)

    assert excinfo.value.time_s > WINDOW / 2


def test_screening_reports_the_time_of_the_offending_state():
    pos_a, vel_a, pos_b, vel_b = clean_pair()
    bad_b = corrupt(vel_b, component=2, value=float("-inf"))

    screener = ConjunctionScreener(threshold_m=THRESHOLD, coarse_dt=COARSE_DT)
    with pytest.raises(NonFiniteStateError) as excinfo:
        screener.screen(pos_a, pos_b, 0.0, WINDOW, vel_fn_a=vel_a, vel_fn_b=bad_b)

    err = excinfo.value
    assert err.time_s is not None
    assert 0.0 <= err.time_s <= WINDOW
    assert "not finite" in str(err)
    assert "z = -inf" in str(err)


def test_screening_without_velocity_functions_still_validates_positions():
    """
    The central-difference fallback evaluates positions at t +/- delta.  Those
    evaluations feed the speed estimate and must be validated too.
    """
    pos_a, _, pos_b, _ = clean_pair()
    bad_a = corrupt(pos_a, component=0, value=float("inf"))

    screener = ConjunctionScreener(threshold_m=THRESHOLD, coarse_dt=COARSE_DT)
    with pytest.raises(NonFiniteStateError):
        screener.screen(bad_a, pos_b, 0.0, WINDOW)


# ---------------------------------------------------------------------------
# 2. Array-based screening entry point
# ---------------------------------------------------------------------------

def _sampled_pair(n: int = 241):
    pos_a, vel_a, pos_b, vel_b = clean_pair()
    times = np.linspace(0.0, WINDOW, n)
    return (
        times,
        np.array([pos_a(t) for t in times]),
        np.array([pos_b(t) for t in times]),
        np.array([vel_a(t) for t in times]),
        np.array([vel_b(t) for t in times]),
    )


@pytest.mark.parametrize("value", NON_FINITE_VALUES)
@pytest.mark.parametrize("which", ("positions_a", "positions_b",
                                   "velocities_a", "velocities_b"))
def test_screen_from_arrays_rejects_non_finite_samples(which, value):
    times, pa, pb, va, vb = _sampled_pair()
    arrays = {"positions_a": pa, "positions_b": pb,
              "velocities_a": va, "velocities_b": vb}
    arrays[which] = arrays[which].copy()
    arrays[which][117, 2] = value

    screener = ConjunctionScreener(threshold_m=THRESHOLD, coarse_dt=COARSE_DT)
    with pytest.raises(NonFiniteStateError) as excinfo:
        screener.screen_from_arrays(
            times, arrays["positions_a"], arrays["positions_b"],
            arrays["velocities_a"], arrays["velocities_b"],
        )

    err = excinfo.value
    assert "sample 117" in err.object_id
    assert err.invalid_components == ("z",)
    assert err.quantity == (QUANTITY_POSITION if which.startswith("positions")
                            else QUANTITY_VELOCITY)


def test_screen_from_arrays_rejects_non_finite_times():
    """
    A NaN time poisons every interval bound computed from it in exactly the
    same way a NaN state does, so it is refused the same way.
    """
    times, pa, pb, va, vb = _sampled_pair()
    times = times.copy()
    times[40] = float("nan")

    screener = ConjunctionScreener(threshold_m=THRESHOLD, coarse_dt=COARSE_DT)
    with pytest.raises(NonFiniteStateError) as excinfo:
        screener.screen_from_arrays(times, pa, pb, va, vb)

    assert "sample 40" in excinfo.value.object_id
    assert excinfo.value.quantity == "time"


def test_screen_from_arrays_unchanged_for_finite_samples():
    times, pa, pb, va, vb = _sampled_pair()
    screener = ConjunctionScreener(threshold_m=THRESHOLD, coarse_dt=COARSE_DT)
    candidates = screener.screen_from_arrays(times, pa, pb, va, vb)

    assert candidates, "the clean sampled scenario must still yield candidates"
    assert all(np.isfinite([c.t_start, c.t_end, c.min_distance, c.lower_bound]).all()
               for c in candidates)


# ---------------------------------------------------------------------------
# 3. TCA refinement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", NON_FINITE_VALUES)
def test_find_tca_raises_instead_of_returning_none(value, reference_tca):
    """
    ``find_tca`` returning None means "no bracket exists in this interval".
    With a NaN state no bracket can be found, so the corrupted call used to be
    indistinguishable from a genuinely quiet interval.
    """
    t_ref, _ = reference_tca
    pos_a, vel_a, pos_b, vel_b = clean_pair()
    lo, hi = t_ref - 120.0, t_ref + 120.0

    # The uncorrupted interval brackets a real TCA.
    clean = find_tca(pos_a, vel_a, pos_b, vel_b, lo, hi, tol=1e-6)
    assert clean is not None and clean.validated

    bad_a = corrupt(pos_a, component=0, value=value)
    with pytest.raises(NonFiniteStateError):
        find_tca(bad_a, vel_a, pos_b, vel_b, lo, hi, tol=1e-6)


@pytest.mark.parametrize("quantity,index", [
    (QUANTITY_POSITION, 0), (QUANTITY_POSITION, 1), (QUANTITY_POSITION, 2),
    (QUANTITY_VELOCITY, 0), (QUANTITY_VELOCITY, 1), (QUANTITY_VELOCITY, 2),
])
def test_find_all_tca_raises_instead_of_returning_empty(quantity, index):
    pos_a, vel_a, pos_b, vel_b = clean_pair()

    assert find_all_tca(pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW, n_samples=400), \
        "the clean scenario must contain at least one TCA"

    fns = [pos_a, vel_a, pos_b, vel_b]
    slot = 0 if quantity == QUANTITY_POSITION else 1
    fns[slot] = corrupt(fns[slot], component=index, value=float("nan"))

    with pytest.raises(NonFiniteStateError) as excinfo:
        find_all_tca(*fns, 0.0, WINDOW, n_samples=400)

    assert excinfo.value.quantity == quantity
    assert excinfo.value.invalid_indices == (index,)


def test_find_all_tca_with_diagnostics_raises_rather_than_reporting_zero():
    """
    The diagnostics variant must not report a truthful-looking
    "0 brackets, 0 attempts, 0 converged" summary for a corrupted trajectory.
    """
    pos_a, vel_a, pos_b, vel_b = clean_pair()
    bad_b = corrupt(pos_b, component=2, value=float("inf"))

    with pytest.raises(NonFiniteStateError):
        find_all_tca_with_diagnostics(
            pos_a, vel_a, bad_b, vel_b, 0.0, WINDOW, n_samples=200,
        )


def test_tca_diagnostic_uses_caller_supplied_object_ids():
    pos_a, vel_a, pos_b, vel_b = clean_pair()
    bad_b = corrupt(pos_b, component=1, value=float("nan"))

    with pytest.raises(NonFiniteStateError) as excinfo:
        find_all_tca(pos_a, vel_a, bad_b, vel_b, 0.0, WINDOW, n_samples=100,
                     object_a_id="STARLINK-1234", object_b_id="COSMOS-2251-DEB")

    assert excinfo.value.object_id == "COSMOS-2251-DEB"


# ---------------------------------------------------------------------------
# 4. Full pipeline through ConjunctionAnalysis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", NON_FINITE_VALUES)
@pytest.mark.parametrize("slot,obj,quantity", [
    (0, "SAT-A", QUANTITY_POSITION),
    (1, "SAT-A", QUANTITY_VELOCITY),
    (2, "SAT-B", QUANTITY_POSITION),
    (3, "SAT-B", QUANTITY_VELOCITY),
])
def test_analyse_raises_rather_than_returning_zero_events(slot, obj, quantity, value):
    """
    The headline B-2 assertion.

    The same scenario without corruption yields at least one conjunction
    (pinned by ``test_clean_scenario_produces_a_real_conjunction``), so an
    empty result here would have been a false negative, not an honest one.
    """
    fns = list(clean_pair())
    fns[slot] = corrupt(fns[slot], component=1, value=value)

    analysis = ConjunctionAnalysis(screening_threshold_m=THRESHOLD, coarse_dt=COARSE_DT)
    with pytest.raises(NonFiniteStateError) as excinfo:
        analysis.analyse(*fns, 0.0, WINDOW,
                         object_a_id="SAT-A", object_b_id="SAT-B")

    err = excinfo.value
    assert err.object_id == obj
    assert err.quantity == quantity
    assert err.invalid_components == ("y",)


def test_analyse_raises_when_corruption_appears_only_mid_window():
    """
    The dangerous shape: the first trace steps ("Acquire Object A State" etc.)
    succeed on finite states, and the corruption is only met during screening.

    A partially-completed analysis must still fail rather than return the
    portion of the window it managed to process.
    """
    pos_a, vel_a, pos_b, vel_b = clean_pair()
    bad_b = corrupt(pos_b, component=0, value=float("nan"),
                    when=lambda t: t > 1800.0)

    analysis = ConjunctionAnalysis(screening_threshold_m=THRESHOLD, coarse_dt=COARSE_DT)
    with pytest.raises(NonFiniteStateError) as excinfo:
        analysis.analyse(pos_a, vel_a, bad_b, vel_b, 0.0, WINDOW)

    assert excinfo.value.time_s > 1800.0


def test_analyse_raises_when_only_one_component_of_one_state_is_bad():
    """A single poisoned component is enough; the other five are irrelevant."""
    pos_a, vel_a, pos_b, vel_b = clean_pair()
    bad_a = corrupt(pos_a, component=2, value=float("nan"))

    with pytest.raises(NonFiniteStateError) as excinfo:
        ConjunctionAnalysis(
            screening_threshold_m=THRESHOLD, coarse_dt=COARSE_DT
        ).analyse(bad_a, vel_a, pos_b, vel_b, 0.0, WINDOW)

    err = excinfo.value
    assert err.invalid_indices == (2,)
    assert math.isfinite(err.values[0]) and math.isfinite(err.values[1])


def test_analyse_never_produces_a_result_object_for_a_bad_state():
    """
    Stated as a property rather than a single case: across every single-slot,
    single-component corruption, ``analyse`` returns no ConjunctionResult at
    all -- not an empty one, not a partial one.
    """
    for slot in range(4):
        for component in COMPONENTS:
            fns = list(clean_pair())
            fns[slot] = corrupt(fns[slot], component=component, value=float("nan"))
            analysis = ConjunctionAnalysis(
                screening_threshold_m=THRESHOLD, coarse_dt=COARSE_DT)
            with pytest.raises(NonFiniteStateError):
                analysis.analyse(*fns, 0.0, WINDOW)


# ---------------------------------------------------------------------------
# 5. The diagnostic itself
# ---------------------------------------------------------------------------

def test_validate_state_vector_reports_every_bad_component():
    with pytest.raises(NonFiniteStateError) as excinfo:
        validate_state_vector(
            [float("nan"), 2.0, float("inf")],
            object_id="DEBRIS-7", quantity=QUANTITY_POSITION, time_s=42.5,
        )

    err = excinfo.value
    assert err.invalid_indices == (0, 2)
    assert err.invalid_components == ("x", "z")
    assert err.object_id == "DEBRIS-7"
    assert err.time_s == 42.5
    assert "42.500000 s" in str(err)


def test_diagnostic_is_strict_json_serialisable():
    """
    ``nan`` and ``inf`` have no JSON literal.  The diagnostic must survive a
    strict round-trip, or the API response it feeds is unparseable by a
    conforming client.
    """
    with pytest.raises(NonFiniteStateError) as excinfo:
        validate_state_vector([1.0, float("nan"), float("-inf")],
                              object_id="A", quantity=QUANTITY_VELOCITY, time_s=0.0)

    encoded = json.dumps(excinfo.value.to_dict())

    def _reject(constant):
        raise AssertionError(f"non-JSON constant {constant!r} in diagnostic")

    decoded = json.loads(encoded, parse_constant=_reject)
    assert decoded["error"] == "NON_FINITE_STATE"
    assert decoded["quantity"] == "velocity"
    assert decoded["invalid_components"] == ["y", "z"]
    assert decoded["values"] == ["1.0", "nan", "-inf"]


def test_non_finite_state_error_is_a_value_error():
    """Callers already handling invalid numeric input keep working."""
    assert issubclass(NonFiniteStateError, ValueError)


# ---------------------------------------------------------------------------
# 6. Finite states are behaviourally identical
# ---------------------------------------------------------------------------

def test_guard_returns_the_identical_array_object():
    """
    Trajectory sources in this engine return read-only float64 arrays and
    memoise them.  The guard must hand back the same object, or the memo is
    defeated and a caller could mutate a cached state.
    """
    payload = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    payload.flags.writeable = False

    guarded = guard_state_function(lambda t: payload,
                                   object_id="A", quantity=QUANTITY_POSITION)
    out = guarded(0.0)
    assert out is payload
    assert out.flags.writeable is False


def test_guard_is_idempotent():
    """
    Screening, TCA and analysis all guard the same functions.  Re-wrapping
    must be a no-op, or the per-evaluation cost multiplies by the number of
    stages.
    """
    base = lambda t: np.zeros(3)
    once = guard_state_function(base, object_id="A", quantity=QUANTITY_POSITION)
    twice = guard_state_function(once, object_id="A", quantity=QUANTITY_POSITION)
    assert twice is once

    other = guard_state_function(once, object_id="B", quantity=QUANTITY_POSITION)
    assert other is not once


def test_guard_does_not_add_trajectory_evaluations():
    """
    Validation is a check on states already being fetched, not an extra fetch.
    Screening evaluates each position function once per coarse sample; that is
    what it must still do.
    """
    pos_a, vel_a, pos_b, vel_b = clean_pair()
    counted_a = counting(pos_a)

    screener = ConjunctionScreener(threshold_m=THRESHOLD, coarse_dt=COARSE_DT)
    _, diag = screener.screen_with_diagnostics(
        counted_a, pos_b, 0.0, WINDOW, vel_fn_a=vel_a, vel_fn_b=vel_b)

    assert len(counted_a.calls) == diag.samples


def test_finite_analysis_matches_the_independent_reference(reference_tca):
    """
    End-to-end: with validation in place the pipeline still reproduces the
    externally-computed closest approach, so nothing about the science moved.
    """
    t_ref, d_ref = reference_tca
    pos_a, vel_a, pos_b, vel_b = clean_pair()

    result = ConjunctionAnalysis(
        screening_threshold_m=THRESHOLD, coarse_dt=COARSE_DT
    ).analyse(pos_a, vel_a, pos_b, vel_b, 0.0, WINDOW,
              object_a_id="SAT-A", object_b_id="SAT-B")

    best = min(result.events, key=lambda e: e.miss_distance_m)
    assert best.tca_result.tca == pytest.approx(t_ref, abs=1e-3)
    assert best.miss_distance_m == pytest.approx(d_ref, rel=1e-6)
    assert best.tca_result.validated

    acc = result.accounting
    assert acc.accepted_conjunctions == len(result.events) == acc.tca_validated
    assert acc.candidate_intervals <= acc.intervals_accepted <= acc.intervals_screened


def test_guard_passes_lists_and_integer_arrays_through_unchanged():
    """Callers that return plain sequences keep working."""
    guarded = guard_state_function(lambda t: [1, 2, 3],
                                   object_id="A", quantity=QUANTITY_POSITION)
    out = guarded(0.0)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# 6b. The multi-object simulator boundary
# ---------------------------------------------------------------------------

def test_screening_diagnostic_uses_caller_supplied_object_ids():
    """
    The multi-spacecraft pipeline screens N(N-1)/2 pairs.  "Object A is not
    finite" is useless there; the diagnostic must name the spacecraft.
    """
    pos_a, vel_a, pos_b, vel_b = clean_pair()
    bad_a = corrupt(pos_a, component=0, value=float("nan"))

    screener = ConjunctionScreener(threshold_m=THRESHOLD, coarse_dt=COARSE_DT)
    with pytest.raises(NonFiniteStateError) as excinfo:
        screener.screen(bad_a, pos_b, 0.0, WINDOW,
                        vel_fn_a=vel_a, vel_fn_b=vel_b,
                        object_a_id="DEBRIS-04", object_b_id="TARGET-01")

    assert excinfo.value.object_id == "DEBRIS-04"


def test_multi_object_pipeline_passes_spacecraft_ids_to_the_guard():
    """
    Pin the wiring: the simulator's screening and TCA calls carry the real
    spacecraft identifiers, so a diverged propagation is attributable.
    """
    import inspect

    from theseus.simulation import multi_object

    source = inspect.getsource(multi_object.MultiObjectEnvironment.simulate)
    assert "object_a_id=id_a, object_b_id=id_b" in source
    assert source.count("object_a_id=id_a, object_b_id=id_b") == 2


def test_interpolator_rejects_non_finite_nodes_at_construction():
    """
    Second line of defence, pre-existing: the simulator's state source refuses
    to be built from a diverged propagation at all.  Pinned here because it is
    what keeps the guard from being the *only* thing between a NaN and a
    result.
    """
    from theseus.propagation.interpolation import TrajectoryInterpolator

    times = np.linspace(0.0, 600.0, 11)
    pos = np.tile(np.array([R_LEO, 0.0, 0.0]), (11, 1))
    vel = np.tile(np.array([0.0, 7660.0, 0.0]), (11, 1))

    bad_pos = pos.copy()
    bad_pos[4, 1] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        TrajectoryInterpolator(times, bad_pos, vel)

    bad_vel = vel.copy()
    bad_vel[7, 2] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        TrajectoryInterpolator(times, pos, bad_vel)


# ---------------------------------------------------------------------------
# 7. Failure propagation through Phase 10
# ---------------------------------------------------------------------------

def test_phase10_orchestration_raises_rather_than_reporting_indeterminate():
    """
    Phase 10 already has an INDETERMINATE path, for "the window contained no
    validated closest approach".  A non-finite state must not be routed into
    it: INDETERMINATE is a statement about a completed analysis of valid
    trajectories, and reusing it here would restate the B-2 defect one layer
    up -- an unusable input reported as an inconclusive but legitimate result.
    """
    from theseus.uncertainty.covariance import StateCovariance
    from theseus.uncertainty.results import run_uncertainty_conjunction_analysis

    pos_a, vel_a, pos_b, vel_b = clean_pair()
    bad_a = corrupt(pos_a, component=0, value=float("nan"))
    cov = StateCovariance.from_diagonal([500.0] * 3, [0.5] * 3, name="test")

    with pytest.raises(NonFiniteStateError):
        run_uncertainty_conjunction_analysis(
            pos_fn_a=bad_a, vel_fn_a=vel_a, pos_fn_b=pos_b, vel_fn_b=vel_b,
            initial_cov_a=cov, initial_cov_b=cov,
            t_start=0.0, t_end=WINDOW,
            screening_threshold_m=THRESHOLD, coarse_dt=COARSE_DT,
        )


# ---------------------------------------------------------------------------
# 8. The API boundary
# ---------------------------------------------------------------------------

class _StubOrbit:
    """Minimal stand-in for CircularOrbitStates carrying a poisoned state."""

    def __init__(self, pos_fn, vel_fn):
        self._pos, self._vel = pos_fn, vel_fn

    def as_callables(self):
        return self._pos, self._vel


def _patch_endpoint_orbits(monkeypatch, corrupt_slot: int, value=float("nan")):
    """Make both conjunction endpoints build a corrupted trajectory pair."""
    from theseus.server import app as app_module

    fns = list(clean_pair())
    fns[corrupt_slot] = corrupt(fns[corrupt_slot], component=1, value=value)

    def fake_pair(req, body):
        return _StubOrbit(fns[0], fns[1]), _StubOrbit(fns[2], fns[3])

    monkeypatch.setattr(app_module, "_conjunction_orbit_pair", fake_pair)


@pytest.mark.parametrize("endpoint", [
    "/api/simulate/conjunction",
    "/api/simulate/conjunction/risk",
])
@pytest.mark.parametrize("slot,expected_object,expected_quantity", [
    (0, "A", QUANTITY_POSITION),
    (3, "B", QUANTITY_VELOCITY),
])
def test_api_reports_non_finite_state_explicitly(
    monkeypatch, endpoint, slot, expected_object, expected_quantity
):
    """
    A non-finite state reaching either conjunction endpoint must produce an
    explicit, diagnosable failure -- never a 200 carrying zero conjunctions,
    and never an unqualified 500.
    """
    from fastapi.testclient import TestClient
    from theseus.server.app import app

    _patch_endpoint_orbits(monkeypatch, slot)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(endpoint, json={"analysis_duration_hours": WINDOW / 3600.0,
                                           "screening_threshold_km": THRESHOLD / 1e3,
                                           "coarse_dt_s": COARSE_DT})

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "NON_FINITE_STATE"
    assert body["object_id"] == expected_object
    assert body["quantity"] == expected_quantity
    assert body["invalid_components"] == ["y"]
    assert body["time_s"] is not None
    assert "not finite" in body["message"]

    # No analysis payload of any kind: a non-finite state has no result,
    # not an empty one.
    for forbidden in ("events", "summary", "accounting", "risk_assessment",
                      "collision_probability", "conjunction_found"):
        assert forbidden not in body


def test_api_non_finite_response_is_strict_json(monkeypatch):
    """The 422 body must be parseable by a conforming JSON client."""
    from fastapi.testclient import TestClient
    from theseus.server.app import app

    _patch_endpoint_orbits(monkeypatch, 0, value=float("-inf"))
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/simulate/conjunction", json={})

    def _reject(constant):
        raise AssertionError(f"non-JSON constant {constant!r} in response")

    decoded = json.loads(response.text, parse_constant=_reject)
    assert decoded["values"][1] == "-inf"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_api_non_finite_request_field_is_not_silently_analysed(literal):
    """
    A non-finite *request field* must never yield a successful analysis.

    B-2 pinned that property and recorded the shape of the failure as a coarse
    500 -- the API error-handling gap logged as N-5 in the Phase 9 gate report,
    deliberately left open at the time.

    The final P10 sweep closed it. The gap was worse than N-5 recorded: a
    non-finite ``screening_threshold_km`` or ``analysis_duration_hours``
    returned **HTTP 200 with an empty event list**, which is precisely the
    outcome B-2 exists to forbid. The boundary now rejects every non-finite
    numeric field, including ones nested in sub-models and lists, with a
    diagnostic naming the field.

    The B-2 assertion is unchanged and still first: the request never yields a
    successful analysis. What follows it is the improved shape.
    """
    from fastapi.testclient import TestClient
    from theseus.server.app import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/api/simulate/conjunction",
        content='{"object_a_alt_km": %s}' % literal,
        headers={"content-type": "application/json"},
    )

    assert response.status_code != 200
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "NON_FINITE_REQUEST_FIELD"
    assert "object_a_alt_km" in body["message"]
    # A non-finite request has no analysis, not an empty one.
    assert "events" not in body


def test_api_valid_request_is_unaffected():
    """The guarded pipeline still serves an ordinary request unchanged."""
    from fastapi.testclient import TestClient
    from theseus.server.app import app

    client = TestClient(app)
    response = client.post("/api/simulate/conjunction", json={
        "object_a_alt_km": 400.0, "object_a_inc_deg": 51.6, "object_a_phase_deg": 0.0,
        "object_b_alt_km": 400.05, "object_b_inc_deg": 55.0, "object_b_phase_deg": 0.02,
        "analysis_duration_hours": 2.0, "screening_threshold_km": 100.0,
        "coarse_dt_s": 30.0,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["total_events"] >= 1
    assert data["accounting"]["accepted_conjunctions"] == data["summary"]["total_events"]
    assert all(np.isfinite(e["tca"]["tca_s"]) for e in data["events"])
