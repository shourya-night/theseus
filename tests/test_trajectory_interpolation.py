"""
P9-02 — trajectory interpolation accuracy.

The multi-object conjunction path stores position *and* velocity at every
propagated node, then interpolated position linearly between them and threw
the velocities away.  Linear interpolation replaces a curved arc with its
chord, so the error is the arc's sagitta:

    e_linear  ~  r (n h)^2 / 8

which at 400 km altitude and ~69 s node spacing is kilometres -- far larger
than the propagator's own node accuracy, and larger than the miss distances a
conjunction analysis exists to resolve.

Cubic Hermite uses the stored velocities and is O(dt^4).

Truth in these tests never comes from another interpolation.  It is either the
closed-form two-body solution or a SciPy DOP853 integration at rtol 1e-13,
evaluated directly at the requested time.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from theseus.propagation.interpolation import (
    TrajectoryInterpolator,
    InterpolatedState,
    hermite_interpolate,
    interpolator_from_state_history,
    METHOD_HERMITE,
    METHOD_LINEAR,
)

from tests._trajectory_reference import (
    MU_EARTH,
    kepler_propagator,
    j2_reference,
    sample_reference,
    linear_position,
)


R_LEO = 6778.137e3
NODE_DT = 69.0          # the spacing actually observed in the multi-object path
WINDOW = 7200.0


def _grid(dt: float = NODE_DT, t_end: float = WINDOW) -> np.ndarray:
    return np.arange(0.0, t_end + 0.5 * dt, dt)


def _probes(times: np.ndarray, n: int = 1501) -> np.ndarray:
    """
    Evaluation times strictly inside the node range.

    Probing past the last node exercises clamping, not interpolation; a probe
    24 s beyond the final node of a LEO trajectory differs from the truth by
    184 km purely because the interpolant correctly refuses to extrapolate.
    """
    return np.linspace(float(times[0]), float(times[-1]), n)


def hermite_position_bound(dt: float, r_peri: float, mu: float = MU_EARTH,
                           safety: float = 8.0) -> float:
    """
    Analytic tolerance for cubic Hermite position error on a Keplerian arc.

    The interpolation error of a cubic Hermite polynomial on a smooth function
    is bounded by

        max |r - p3|  <=  (dt^4 / 384) * max |r''''|

    For two-body motion the fourth derivative scales as |r''''| ~ mu^2 / r^5,
    worst at perigee.  The *safety* factor covers the difference between that
    scaling and the true fourth derivative on an eccentric arc.

    This is a derived bound, not a value fitted to the implementation: it
    predicts 0.65 m for a 400 km circular orbit at 69 s spacing with
    safety = 1, against 0.6555 m measured.
    """
    return safety * (dt ** 4 / 384.0) * (mu ** 2 / r_peri ** 5)


def hermite_velocity_bound(dt: float, r_peri: float, mu: float = MU_EARTH,
                           safety: float = 8.0) -> float:
    """
    Companion bound for velocity error.

    Differentiating the error polynomial lowers the order by one, giving a
    velocity error of order (position error) x 1/dt up to a small constant.
    """
    return hermite_position_bound(dt, r_peri, mu, safety) * 5.0 / dt


def _errors(interp: TrajectoryInterpolator, state_at, sample_times):
    """Position and velocity error against the reference at arbitrary times."""
    pe, ve = [], []
    for t in sample_times:
        st = interp.state_at(float(t))
        r_true, v_true = state_at(float(t))
        pe.append(float(np.linalg.norm(st.position - r_true)))
        ve.append(float(np.linalg.norm(st.velocity - v_true)))
    return np.array(pe), np.array(ve)


def _linear_errors(times, positions, state_at, sample_times):
    """Position error of the pre-correction linear interpolation, for contrast."""
    return np.array([
        float(np.linalg.norm(linear_position(times, positions, float(t)) - state_at(float(t))[0]))
        for t in sample_times
    ])


# ---------------------------------------------------------------------------
# 1-4: accuracy against independent references
# ---------------------------------------------------------------------------

def test_circular_orbit_accuracy_against_analytic_two_body():
    """Case 3: circular orbit, exact closed-form truth."""
    state_at = kepler_propagator(a=R_LEO, e=0.0, inc_deg=51.6, nu0_deg=0.0)
    times = _grid()
    pos, vel = sample_reference(state_at, times)
    interp = TrajectoryInterpolator(times, pos, vel)

    probes = _probes(times)
    pe, ve = _errors(interp, state_at, probes)

    assert pe.max() < hermite_position_bound(NODE_DT, R_LEO), \
        f"max position error {pe.max():.3f} m"
    assert ve.max() < hermite_velocity_bound(NODE_DT, R_LEO), \
        f"max velocity error {ve.max():.6f} m/s"

    # And it must beat the chord by orders of magnitude on the same nodes.
    lin = _linear_errors(times, pos, state_at, probes)
    assert lin.max() > 1e3, "the reference case should show km-scale chord error"
    assert pe.max() < lin.max() / 100.0


def test_eccentric_orbit_accuracy_against_analytic_two_body():
    """Case 2: eccentric orbit, where the speed varies strongly along the arc."""
    a = 12000e3
    state_at = kepler_propagator(a=a, e=0.6, inc_deg=30.0, argp_deg=40.0, nu0_deg=0.0)
    period = 2.0 * math.pi * math.sqrt(a ** 3 / MU_EARTH)
    times = _grid(dt=NODE_DT, t_end=period)
    pos, vel = sample_reference(state_at, times)
    interp = TrajectoryInterpolator(times, pos, vel)

    probes = _probes(times, 2001)
    pe, ve = _errors(interp, state_at, probes)

    r_peri = a * (1.0 - 0.6)
    assert pe.max() < hermite_position_bound(NODE_DT, r_peri), \
        f"max position error {pe.max():.3f} m"
    assert ve.max() < hermite_velocity_bound(NODE_DT, r_peri), \
        f"max velocity error {ve.max():.6f} m/s"

    lin = _linear_errors(times, pos, state_at, probes)
    assert pe.max() < lin.max() / 100.0


def test_perturbed_orbit_accuracy_against_dense_output_reference():
    """
    Case 4: two-body + J2, the force model the multi-object path actually uses.
    Truth is SciPy DOP853 dense output at rtol 1e-13.
    """
    r0 = np.array([R_LEO, 0.0, 0.0])
    v0 = np.array([0.0, math.sqrt(MU_EARTH / R_LEO) * math.cos(math.radians(51.6)),
                   math.sqrt(MU_EARTH / R_LEO) * math.sin(math.radians(51.6))])
    state_at = j2_reference(r0, v0, (0.0, WINDOW + 100.0))

    times = _grid()
    pos, vel = sample_reference(state_at, times)
    interp = TrajectoryInterpolator(times, pos, vel)

    probes = _probes(times)
    pe, ve = _errors(interp, state_at, probes)

    assert pe.max() < hermite_position_bound(NODE_DT, R_LEO), \
        f"max position error {pe.max():.3f} m"
    assert ve.max() < hermite_velocity_bound(NODE_DT, R_LEO), \
        f"max velocity error {ve.max():.6f} m/s"

    lin = _linear_errors(times, pos, state_at, probes)
    assert pe.max() < lin.max() / 100.0


def test_error_scales_as_fourth_order():
    """
    Halving the node spacing must cut the error by roughly 2^4, confirming the
    interpolant is genuinely cubic-Hermite and not accidentally lower order.
    """
    state_at = kepler_propagator(a=R_LEO, e=0.0, inc_deg=0.0)
    maxima = []
    for dt in (120.0, 60.0, 30.0):
        times = _grid(dt=dt, t_end=3600.0)
        pos, vel = sample_reference(state_at, times)
        interp = TrajectoryInterpolator(times, pos, vel)
        probes = np.linspace(0.0, 3600.0, 3001)
        pe, _ = _errors(interp, state_at, probes)
        maxima.append(pe.max())

    for coarse, fine in zip(maxima[:-1], maxima[1:]):
        ratio = coarse / fine
        assert 8.0 < ratio < 32.0, f"error ratio {ratio:.1f} is not fourth order"


# ---------------------------------------------------------------------------
# 5-7: evaluation points
# ---------------------------------------------------------------------------

def test_nodes_are_reproduced_exactly():
    """Case 5: interpolation at a node must return the stored state exactly."""
    state_at = kepler_propagator(a=R_LEO, e=0.35, inc_deg=20.0, nu0_deg=10.0)
    times = _grid(dt=100.0, t_end=3000.0)
    pos, vel = sample_reference(state_at, times)
    interp = TrajectoryInterpolator(times, pos, vel)

    for i, t in enumerate(times):
        st = interp.state_at(float(t))
        assert np.array_equal(st.position, pos[i]), f"node {i} position not exact"
        assert np.array_equal(st.velocity, vel[i]), f"node {i} velocity not exact"
        assert st.clamped is False


def test_midpoint_and_near_endpoint_evaluation():
    """Cases 6 and 7: halfway, and arbitrarily close to either end."""
    state_at = kepler_propagator(a=R_LEO, e=0.0, inc_deg=45.0)
    times = _grid(dt=NODE_DT, t_end=1380.0)
    pos, vel = sample_reference(state_at, times)
    interp = TrajectoryInterpolator(times, pos, vel)

    for i in range(len(times) - 1):
        t0, t1 = float(times[i]), float(times[i + 1])
        dt = t1 - t0
        for frac in (1e-9, 1e-6, 1e-3, 0.5, 1.0 - 1e-3, 1.0 - 1e-6, 1.0 - 1e-9):
            t = t0 + frac * dt
            st = interp.state_at(t)
            r_true, v_true = state_at(t)
            assert np.linalg.norm(st.position - r_true) < 1.0
            assert np.linalg.norm(st.velocity - v_true) < 1e-3


def test_out_of_range_times_are_clamped_and_flagged():
    state_at = kepler_propagator(a=R_LEO, e=0.0)
    times = _grid(dt=100.0, t_end=1000.0)
    pos, vel = sample_reference(state_at, times)
    interp = TrajectoryInterpolator(times, pos, vel)

    before = interp.state_at(-500.0)
    after = interp.state_at(9999.0)
    assert before.clamped is True and after.clamped is True
    assert np.array_equal(before.position, pos[0])
    assert np.array_equal(after.position, pos[-1])


# ---------------------------------------------------------------------------
# 8-9: continuity and internal consistency
# ---------------------------------------------------------------------------

def test_position_and_velocity_are_continuous_across_nodes():
    """
    Cases 8 and 9: C1 continuity, proved exactly rather than statistically.

    At a shared node the polynomial of the left interval evaluated at s = 1 and
    the polynomial of the right interval evaluated at s = 0 must both equal the
    stored node state.  That is what makes position and velocity continuous.

    (Probing at t +/- eps and differencing does not test this: the difference
    it measures is 2*eps*|v| of genuine motion, which for a LEO orbit is
    already 0.02 m at eps = 1e-6 s.)
    """
    state_at = kepler_propagator(a=9000e3, e=0.4, inc_deg=63.4)
    times = _grid(dt=90.0, t_end=5400.0)
    pos, vel = sample_reference(state_at, times)
    interp = TrajectoryInterpolator(times, pos, vel)

    for i in range(len(times) - 2):
        dt_left = float(times[i + 1] - times[i])
        dt_right = float(times[i + 2] - times[i + 1])

        p_left, v_left = hermite_interpolate(
            pos[i], vel[i], pos[i + 1], vel[i + 1], dt_left, 1.0,
        )
        p_right, v_right = hermite_interpolate(
            pos[i + 1], vel[i + 1], pos[i + 2], vel[i + 2], dt_right, 0.0,
        )

        assert np.array_equal(p_left, p_right), f"position jump at node {i + 1}"
        assert np.array_equal(v_left, v_right), f"velocity jump at node {i + 1}"
        assert np.array_equal(p_left, pos[i + 1])
        assert np.array_equal(v_left, vel[i + 1])

    # Practical no-jump check: the change across a node over a short window
    # must be the motion itself, not motion plus a step.
    eps = 1e-6
    for t in times[1:-1]:
        left = interp.state_at(float(t) - eps)
        right = interp.state_at(float(t) + eps)
        expected = 2.0 * eps * float(np.linalg.norm(left.velocity))
        moved = float(np.linalg.norm(right.position - left.position))
        assert abs(moved - expected) < 1e-9 + 1e-3 * expected


def test_returned_velocity_is_the_derivative_of_returned_position():
    """
    The interpolated velocity must be the analytic derivative of the
    interpolated position -- not a separate estimate.  Checked by central
    differences of the interpolant itself.
    """
    state_at = kepler_propagator(a=R_LEO, e=0.2, inc_deg=28.5, nu0_deg=77.0)
    times = _grid(dt=NODE_DT, t_end=3450.0)
    pos, vel = sample_reference(state_at, times)
    interp = TrajectoryInterpolator(times, pos, vel)

    h = 1e-4
    worst = 0.0
    for i in range(len(times) - 1):
        t0, t1 = float(times[i]), float(times[i + 1])
        for frac in (0.13, 0.5, 0.87):
            t = t0 + frac * (t1 - t0)
            fd = (interp.position_at(t + h) - interp.position_at(t - h)) / (2.0 * h)
            worst = max(worst, float(np.linalg.norm(fd - interp.velocity_at(t))))
    assert worst < 1e-4, f"velocity is not the derivative of position ({worst:.3e} m/s)"


def test_no_overshoot_on_a_smooth_arc():
    """
    A cubic interpolant can overshoot on non-smooth data.  On a well-sampled
    orbit it must not: the interpolated radius has to stay inside the true
    radius range of the arc, with only round-off slack.
    """
    a, e = 10000e3, 0.5
    state_at = kepler_propagator(a=a, e=e, inc_deg=10.0)
    period = 2.0 * math.pi * math.sqrt(a ** 3 / MU_EARTH)
    times = _grid(dt=NODE_DT, t_end=period)
    pos, vel = sample_reference(state_at, times)
    interp = TrajectoryInterpolator(times, pos, vel)

    r_min_true = a * (1.0 - e)
    r_max_true = a * (1.0 + e)
    probes = np.linspace(0.0, period, 4001)
    radii = np.array([np.linalg.norm(interp.position_at(float(t))) for t in probes])

    assert radii.min() > r_min_true - 10.0
    assert radii.max() < r_max_true + 10.0

    speeds = np.array([np.linalg.norm(interp.velocity_at(float(t))) for t in probes])
    v_peri = math.sqrt(MU_EARTH * (1.0 + e) / (a * (1.0 - e)))
    v_apo = math.sqrt(MU_EARTH * (1.0 - e) / (a * (1.0 + e)))
    assert speeds.min() > v_apo - 1.0
    assert speeds.max() < v_peri + 1.0


# ---------------------------------------------------------------------------
# API semantics
# ---------------------------------------------------------------------------

def test_hermite_basis_endpoint_identities():
    r0 = np.array([1.0, 2.0, 3.0])
    v0 = np.array([0.1, 0.2, 0.3])
    r1 = np.array([4.0, 5.0, 6.0])
    v1 = np.array([0.4, 0.5, 0.6])
    p, v = hermite_interpolate(r0, v0, r1, v1, 10.0, 0.0)
    assert np.array_equal(p, r0) and np.array_equal(v, v0)
    p, v = hermite_interpolate(r0, v0, r1, v1, 10.0, 1.0)
    assert np.array_equal(p, r1) and np.array_equal(v, v1)


def test_missing_velocity_requires_explicit_opt_in():
    times = np.array([0.0, 10.0, 20.0])
    pos = np.zeros((3, 3))
    with pytest.raises(ValueError, match="velocities are required"):
        TrajectoryInterpolator(times, pos)

    interp = TrajectoryInterpolator(times, pos, allow_linear_fallback=True)
    assert interp.method == METHOD_LINEAR
    assert interp.state_at(5.0).method == METHOD_LINEAR


def test_method_is_reported_on_every_state():
    state_at = kepler_propagator(a=R_LEO, e=0.0)
    times = _grid(dt=100.0, t_end=1000.0)
    pos, vel = sample_reference(state_at, times)
    interp = TrajectoryInterpolator(times, pos, vel)
    assert interp.method == METHOD_HERMITE
    assert interp.state_at(350.0).method == METHOD_HERMITE
    assert isinstance(interp.state_at(350.0), InterpolatedState)


def test_time_ordering_is_validated():
    pos = np.zeros((3, 3))
    vel = np.zeros((3, 3))
    with pytest.raises(ValueError, match="strictly increasing"):
        TrajectoryInterpolator(np.array([0.0, 10.0, 5.0]), pos, vel)
    with pytest.raises(ValueError, match="strictly increasing"):
        TrajectoryInterpolator(np.array([0.0, 10.0, 10.0]), pos, vel)


def test_shape_and_finiteness_validation():
    times = np.array([0.0, 10.0])
    with pytest.raises(ValueError):
        TrajectoryInterpolator(times, np.zeros((3, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        TrajectoryInterpolator(times, np.zeros((2, 3)), np.zeros((2, 2)))
    bad = np.zeros((2, 3))
    bad[1, 0] = np.nan
    with pytest.raises(ValueError):
        TrajectoryInterpolator(times, bad, np.zeros((2, 3)))


def test_single_node_trajectory():
    interp = TrajectoryInterpolator(
        np.array([5.0]), np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]]),
    )
    st = interp.state_at(99.0)
    assert np.array_equal(st.position, np.array([1.0, 2.0, 3.0]))
    assert st.clamped is True


def test_interpolator_from_state_history_round_trip():
    from theseus.core.state import SimulationState, StateHistory

    state_at = kepler_propagator(a=R_LEO, e=0.1, inc_deg=15.0)
    times = _grid(dt=100.0, t_end=1000.0)
    pos, vel = sample_reference(state_at, times)

    hist = StateHistory()
    for i, t in enumerate(times):
        hist.append(SimulationState(time=float(t), position=pos[i].copy(),
                                    velocity=vel[i].copy(), mass=1000.0))

    interp = interpolator_from_state_history(hist)
    assert interp.method == METHOD_HERMITE
    assert interp.node_count == len(times)
    probes = _probes(times, 401)
    pe, ve = _errors(interp, state_at, probes)
    r_peri = R_LEO * (1.0 - 0.1)
    assert pe.max() < hermite_position_bound(100.0, r_peri)
    assert ve.max() < hermite_velocity_bound(100.0, r_peri)


# ---------------------------------------------------------------------------
# 10: the conjunction geometry this was fixed for
# ---------------------------------------------------------------------------

def test_multi_object_conjunction_geometry_accuracy():
    """
    Case 10.  Two crossing orbits sampled at 69 s, the spacing the multi-object
    path actually produces.  The miss distance recovered from the interpolated
    trajectories must match the miss distance computed from the exact analytic
    states, to metres rather than kilometres.
    """
    from scipy.optimize import minimize_scalar

    state_a = kepler_propagator(a=R_LEO, e=0.0, inc_deg=0.0, nu0_deg=90.0)
    state_b = kepler_propagator(a=R_LEO + 200.0, e=0.0, inc_deg=170.0, nu0_deg=90.0009)

    times = _grid(dt=NODE_DT, t_end=WINDOW)
    pa, va = sample_reference(state_a, times)
    pb, vb = sample_reference(state_b, times)

    ia = TrajectoryInterpolator(times, pa, va)
    ib = TrajectoryInterpolator(times, pb, vb)

    def d_true(t: float) -> float:
        return float(np.linalg.norm(state_a(t)[0] - state_b(t)[0]))

    def d_interp(t: float) -> float:
        return float(np.linalg.norm(ia.position_at(t) - ib.position_at(t)))

    def d_linear(t: float) -> float:
        return float(np.linalg.norm(
            linear_position(times, pa, t) - linear_position(times, pb, t)
        ))

    # Locate the true closest approach independently, then compare all three
    # distance functions in a tight neighbourhood of it.
    scan = np.linspace(0.0, WINDOW, 40000)
    k = int(np.argmin([d_true(float(t)) for t in scan]))
    lo, hi = scan[max(k - 1, 0)], scan[min(k + 1, len(scan) - 1)]
    res = minimize_scalar(d_true, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10})
    t_ca, miss_true = float(res.x), float(res.fun)

    probes = np.linspace(t_ca - 100.0, t_ca + 100.0, 401)
    err_hermite = max(abs(d_interp(float(t)) - d_true(float(t))) for t in probes)
    err_linear = max(abs(d_linear(float(t)) - d_true(float(t))) for t in probes)

    assert err_linear > 100.0, "expected large chord error near a fast encounter"
    assert err_hermite < 5.0, f"Hermite separation error {err_hermite:.3f} m"
    assert err_hermite < err_linear / 50.0

    # And the interpolated minimum must land on the true one.
    res_i = minimize_scalar(d_interp, bounds=(t_ca - 50.0, t_ca + 50.0),
                            method="bounded", options={"xatol": 1e-10})
    assert float(res_i.x) == pytest.approx(t_ca, abs=1e-2)
    assert float(res_i.fun) == pytest.approx(miss_true, abs=5.0)


# ---------------------------------------------------------------------------
# Memoisation must not be able to return a wrong state
# ---------------------------------------------------------------------------

def test_repeated_and_interleaved_evaluations_are_consistent():
    """
    The interpolator memoises its last evaluation because consumers ask for
    position and velocity at the same instant back to back.  Repeating a time,
    interleaving different times, and re-requesting an earlier time must all
    give identical results to a fresh interpolator.
    """
    state_at = kepler_propagator(a=R_LEO, e=0.25, inc_deg=33.0, nu0_deg=12.0)
    times = _grid(dt=NODE_DT, t_end=2070.0)
    pos, vel = sample_reference(state_at, times)
    interp = TrajectoryInterpolator(times, pos, vel)

    probe_times = [100.0, 100.0, 900.0, 100.0, 1500.0, 900.0, 100.0]
    seen: dict[float, tuple] = {}
    for t in probe_times:
        fresh = TrajectoryInterpolator(times, pos, vel)
        expected = fresh.state_at(t)
        got = interp.state_at(t)
        assert np.array_equal(got.position, expected.position)
        assert np.array_equal(got.velocity, expected.velocity)
        if t in seen:
            assert np.array_equal(got.position, seen[t][0])
            assert np.array_equal(got.velocity, seen[t][1])
        seen[t] = (got.position, got.velocity)

    # position_at and velocity_at at the same time must agree with state_at.
    for t in (250.0, 1234.5):
        st = interp.state_at(t)
        assert np.array_equal(interp.position_at(t), st.position)
        assert np.array_equal(interp.velocity_at(t), st.velocity)


def test_returned_arrays_are_read_only():
    """A caller must not be able to mutate a memoised state in place."""
    state_at = kepler_propagator(a=R_LEO, e=0.0)
    times = _grid(dt=100.0, t_end=1000.0)
    pos, vel = sample_reference(state_at, times)
    interp = TrajectoryInterpolator(times, pos, vel)

    st = interp.state_at(350.0)
    with pytest.raises(ValueError):
        st.position[0] = 0.0
    with pytest.raises(ValueError):
        interp.velocity_at(350.0)[0] = 0.0
