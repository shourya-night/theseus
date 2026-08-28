"""
Trajectory interpolation between propagated state nodes.

A numerical propagator produces states on a discrete time grid.  Downstream
consumers -- conjunction screening, TCA refinement, output resampling -- need
the state at arbitrary times between those nodes.  How that gap is filled sets
a hard floor on the accuracy of everything computed from it.

Why linear interpolation is not adequate
----------------------------------------
Linear interpolation of position uses only the two bracketing positions and
throws away the velocities that the propagator already stored.  It replaces a
curved arc by its chord, so the error is the sagitta of that arc.  For a
near-circular orbit of radius r sampled at spacing h with mean motion n:

    e_linear  ≈  r · (n h)² / 8        (chord–arc offset at mid-interval)

At 400 km altitude (r = 6.78e6 m, n = 1.13e-3 rad/s) a 69 s spacing gives
≈ 5.2 km.  That is three orders of magnitude larger than the propagator's own
node accuracy and larger than any miss distance a conjunction analysis is
trying to resolve.

Cubic Hermite interpolation
---------------------------
With position *and* velocity available at both ends of an interval, the unique
cubic matching all four conditions is the Hermite interpolant.  On
[t₀, t₁] with Δt = t₁ − t₀ and s = (t − t₀)/Δt ∈ [0, 1]:

    r(t) = h₀₀(s) r₀ + h₁₀(s) Δt v₀ + h₀₁(s) r₁ + h₁₁(s) Δt v₁

    h₀₀(s) =  2s³ − 3s² + 1
    h₁₀(s) =   s³ − 2s² + s
    h₀₁(s) = −2s³ + 3s²
    h₁₁(s) =   s³ −  s²

The velocity is the *analytic derivative of the same polynomial*, not a
separate estimate:

    v(t) = (1/Δt)[h₀₀'(s) r₀ + h₀₁'(s) r₁] + h₁₀'(s) v₀ + h₁₁'(s) v₁

    h₀₀'(s) =  6s² − 6s
    h₁₀'(s) =  3s² − 4s + 1
    h₀₁'(s) = −6s² + 6s
    h₁₁'(s) =  3s² − 2s

so the returned position and velocity are always mutually consistent:
v(t) ≡ dr/dt exactly, everywhere in the interval.

Properties
----------
- **Exact at nodes.**  h₀₀(0) = 1 and h₀₁(1) = 1 with every other basis
  function vanishing, so r(t₀) = r₀ and r(t₁) = r₁ to the last bit; likewise
  v(t₀) = v₀ and v(t₁) = v₁.
- **C¹ continuous.**  Adjacent intervals share both the position and the
  velocity at the node they meet at, so neither jumps.
- **Order.**  The interpolation error is O(Δt⁴) against a smooth trajectory,
  versus O(Δt²) for linear interpolation.
- **Time ordering.**  The node times must be strictly increasing; this is
  validated at construction rather than assumed.

Limitations
-----------
Cubic Hermite is an interpolant, not a propagator.  It reproduces the sampled
arc; it does not know the equations of motion, and it cannot recover detail
finer than the node spacing.  Like any polynomial interpolant it can overshoot
if the underlying data is not smooth on the scale of the interval -- across a
discontinuity such as an impulsive burn, an interval that straddles the event
is interpolated as if the state were smooth.  Sample such events at their own
node so the discontinuity lands on an interval boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


METHOD_HERMITE = "hermite_cubic"
METHOD_LINEAR = "linear_position_only"


@dataclass(frozen=True)
class InterpolatedState:
    """
    A trajectory state produced by interpolation.

    Attributes
    ----------
    time : float
        Evaluation time (s).
    position : np.ndarray
        Position (m).
    velocity : np.ndarray
        Velocity (m/s).  For the Hermite method this is the exact derivative
        of the interpolated position; for the linear fallback it is the
        interpolated node velocity, which is *not* the derivative of the
        interpolated position.
    method : str
        Which interpolation produced this state -- ``hermite_cubic`` or
        ``linear_position_only``.
    clamped : bool
        True when the requested time fell outside the node range and the state
        was taken from the nearest endpoint rather than interpolated.
    """
    time: float
    position: np.ndarray
    velocity: np.ndarray
    method: str
    clamped: bool = False


def hermite_basis(s: float) -> tuple[float, float, float, float]:
    """Hermite basis functions h₀₀, h₁₀, h₀₁, h₁₁ at s ∈ [0, 1]."""
    s2 = s * s
    s3 = s2 * s
    return (
        2.0 * s3 - 3.0 * s2 + 1.0,   # h00
        s3 - 2.0 * s2 + s,           # h10
        -2.0 * s3 + 3.0 * s2,        # h01
        s3 - s2,                     # h11
    )


def hermite_basis_derivative(s: float) -> tuple[float, float, float, float]:
    """Derivatives h₀₀', h₁₀', h₀₁', h₁₁' with respect to s."""
    s2 = s * s
    return (
        6.0 * s2 - 6.0 * s,          # h00'
        3.0 * s2 - 4.0 * s + 1.0,    # h10'
        -6.0 * s2 + 6.0 * s,         # h01'
        3.0 * s2 - 2.0 * s,          # h11'
    )


def hermite_interpolate(
    r0: np.ndarray, v0: np.ndarray,
    r1: np.ndarray, v1: np.ndarray,
    dt: float, s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cubic Hermite position and velocity on one interval.

    Parameters
    ----------
    r0, v0 : (3,) arrays
        State at the start of the interval (m, m/s).
    r1, v1 : (3,) arrays
        State at the end of the interval (m, m/s).
    dt : float
        Interval duration (s).  Must be > 0.
    s : float
        Normalised position in the interval, 0 at the start, 1 at the end.

    Returns
    -------
    (position, velocity)
        Velocity is the exact time derivative of the returned position curve.
    """
    h00, h10, h01, h11 = hermite_basis(s)
    position = h00 * r0 + (h10 * dt) * v0 + h01 * r1 + (h11 * dt) * v1

    d00, d10, d01, d11 = hermite_basis_derivative(s)
    velocity = (d00 * r0 + d01 * r1) / dt + d10 * v0 + d11 * v1

    return position, velocity


class TrajectoryInterpolator:
    """
    Cubic Hermite interpolation over a sampled trajectory.

    Parameters
    ----------
    times : (N,) array
        Node times (s), strictly increasing.
    positions : (N, 3) array
        Node positions (m).
    velocities : (N, 3) array, optional
        Node velocities (m/s).  Required for Hermite interpolation.  When
        omitted, ``allow_linear_fallback`` must be set explicitly -- the
        fallback is never silent.
    allow_linear_fallback : bool
        Permit construction without velocities, using linear interpolation of
        position.  The resulting states are labelled ``linear_position_only``
        so a consumer can tell that the O(Δt²) chord error applies.

    Raises
    ------
    ValueError
        On mismatched shapes, non-finite values, non-increasing times, or
        missing velocities without an explicit fallback opt-in.
    """

    def __init__(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        *,
        allow_linear_fallback: bool = False,
    ) -> None:
        t = np.asarray(times, dtype=np.float64)
        p = np.asarray(positions, dtype=np.float64)

        if t.ndim != 1:
            raise ValueError(f"times must be 1-D, got shape {t.shape}")
        if p.shape != (len(t), 3):
            raise ValueError(
                f"positions must have shape ({len(t)}, 3), got {p.shape}"
            )
        if len(t) == 0:
            raise ValueError("trajectory must contain at least one node")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)):
            raise ValueError("trajectory contains non-finite times or positions")
        if len(t) > 1 and not np.all(np.diff(t) > 0.0):
            bad = int(np.argmin(np.diff(t)))
            raise ValueError(
                "node times must be strictly increasing; "
                f"t[{bad}] = {t[bad]} is not before t[{bad + 1}] = {t[bad + 1]}"
            )

        if velocities is None:
            if not allow_linear_fallback:
                raise ValueError(
                    "velocities are required for Hermite interpolation. "
                    "Pass velocities, or set allow_linear_fallback=True to accept "
                    "linear position interpolation and its O(dt^2) chord error."
                )
            self._velocities = None
            self.method = METHOD_LINEAR
        else:
            v = np.asarray(velocities, dtype=np.float64)
            if v.shape != p.shape:
                raise ValueError(
                    f"velocities must have shape {p.shape}, got {v.shape}"
                )
            if not np.all(np.isfinite(v)):
                raise ValueError("trajectory contains non-finite velocities")
            self._velocities = v
            self.method = METHOD_HERMITE

        self._times = t
        self._positions = p

        # Single-entry memo.  Consumers such as the TCA solver ask for the
        # position and the velocity at the same instant, one after the other;
        # without this the interval search and the cubic are evaluated twice
        # for every time.  Keyed on the exact float, so it can only ever return
        # the state it would have recomputed.
        self._cache_t: Optional[float] = None
        self._cache_state: Optional[InterpolatedState] = None

    # -- properties ----------------------------------------------------------

    @property
    def t_start(self) -> float:
        return float(self._times[0])

    @property
    def t_end(self) -> float:
        return float(self._times[-1])

    @property
    def node_count(self) -> int:
        return len(self._times)

    # -- evaluation ----------------------------------------------------------

    def _bracket(self, t: float) -> tuple[int, float, bool]:
        """
        Locate the interval containing *t*.

        Returns (index of the left node, normalised position s, clamped flag).
        A time exactly on a node yields s = 0 (or s = 1 at the final node), so
        the interpolant reproduces the stored state bit-for-bit.
        """
        n = len(self._times)
        clamped = False
        if t < self._times[0]:
            t = float(self._times[0])
            clamped = True
        elif t > self._times[-1]:
            t = float(self._times[-1])
            clamped = True

        if n == 1:
            return 0, 0.0, clamped

        idx = int(np.searchsorted(self._times, t, side="right")) - 1
        idx = max(0, min(idx, n - 2))
        dt = self._times[idx + 1] - self._times[idx]
        s = (t - self._times[idx]) / dt
        return idx, float(s), clamped

    def state_at(self, t: float) -> InterpolatedState:
        """Interpolated state at time *t* (s), clamped to the node range."""
        t = float(t)
        if self._cache_t is not None and t == self._cache_t:
            return self._cache_state

        state = self._evaluate(t)
        self._cache_t = t
        self._cache_state = state
        return state

    def _evaluate(self, t: float) -> InterpolatedState:
        idx, s, clamped = self._bracket(t)
        n = len(self._times)

        if n == 1:
            vel = (self._velocities[0] if self._velocities is not None
                   else np.zeros(3))
            return self._freeze(t, self._positions[0], vel, clamped, self.method)

        dt = float(self._times[idx + 1] - self._times[idx])
        r0 = self._positions[idx]
        r1 = self._positions[idx + 1]

        if self._velocities is not None:
            v0 = self._velocities[idx]
            v1 = self._velocities[idx + 1]
            pos, vel = hermite_interpolate(r0, v0, r1, v1, dt, s)
        else:
            pos = (1.0 - s) * r0 + s * r1
            vel = (r1 - r0) / dt          # secant, the only estimate available

        return self._freeze(t, pos, vel, clamped, self.method)

    @staticmethod
    def _freeze(t: float, pos: np.ndarray, vel: np.ndarray,
                clamped: bool, method: str = METHOD_HERMITE) -> InterpolatedState:
        """
        Build the state with read-only arrays.

        States are memoised, so handing out a writeable view would let a
        caller mutate a cached result and silently corrupt every later
        evaluation at the same time.
        """
        pos = np.array(pos, dtype=np.float64)
        vel = np.array(vel, dtype=np.float64)
        pos.flags.writeable = False
        vel.flags.writeable = False
        return InterpolatedState(time=t, position=pos, velocity=vel,
                                 method=method, clamped=clamped)

    def position_at(self, t: float) -> np.ndarray:
        """Interpolated position (m) at time *t*."""
        return self.state_at(t).position

    def velocity_at(self, t: float) -> np.ndarray:
        """Interpolated velocity (m/s) at time *t*."""
        return self.state_at(t).velocity

    def as_callables(self):
        """
        Return ``(position_fn, velocity_fn)`` for APIs that take separate
        position and velocity callables, such as the conjunction pipeline.
        """
        return self.position_at, self.velocity_at

    def resample(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the interpolant on a new time grid.

        Returns
        -------
        (positions, velocities) : (M, 3) arrays
        """
        grid = np.asarray(times, dtype=np.float64)
        pos = np.empty((len(grid), 3), dtype=np.float64)
        vel = np.empty((len(grid), 3), dtype=np.float64)
        for i, t in enumerate(grid):
            st = self.state_at(float(t))
            pos[i] = st.position
            vel[i] = st.velocity
        return pos, vel


def interpolator_from_state_history(history, *, allow_linear_fallback: bool = False
                                    ) -> TrajectoryInterpolator:
    """
    Build a :class:`TrajectoryInterpolator` from a :class:`StateHistory`.

    ``StateHistory`` nodes always carry both position and velocity, so the
    Hermite path is always available for propagated trajectories.
    """
    return TrajectoryInterpolator(
        times=np.asarray(history.times, dtype=np.float64),
        positions=np.asarray(history.positions, dtype=np.float64),
        velocities=np.asarray(history.velocities, dtype=np.float64),
        allow_linear_fallback=allow_linear_fallback,
    )
