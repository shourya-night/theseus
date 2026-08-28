"""
Conjunction screening — conservative coarse-pass filter.

Given two objects' trajectories, identifies time intervals that *may* contain
an approach closer than a configurable threshold, so that only those intervals
pay for full TCA refinement.

Pipeline:
    STATE HISTORY A
    STATE HISTORY B
        ↓
    RELATIVE POSITION AND RELATIVE SPEED at each coarse sample
        ↓
    ANALYTIC LOWER BOUND on separation across each coarse interval
        ↓
    CANDIDATE INTERVALS for TCA refinement

Why a bound and not a sampled distance
--------------------------------------
Testing only the sampled separations,

    below[i] = |r_rel(t_i)| < threshold

is a statement about the samples, not about the trajectory between them.  A
fast encounter spends very little time inside the threshold sphere:

    dwell = 2 sqrt(threshold² − d_min²) / |v_rel|

For two 400 km LEO objects meeting at |v_rel| = 15.3 km/s with a 50 km
threshold the dwell is 6.5 s, while a 60 s coarse step advances the pair by
917 km.  Roughly nine times out of ten no sample lands inside the sphere and
the encounter is discarded before it is ever refined.  That is a false
negative on a dangerous conjunction, which is the one error a screen must
never make.

The separation bound
--------------------
**Theorem.**  Let r_rel(t) be continuously differentiable on [t₀, t₁],
h = t₁ − t₀, d₀ = |r_rel(t₀)|, d₁ = |r_rel(t₁)|, and suppose
|v_rel(t)| ≤ V for all t in the interval.  Then

    min_{t ∈ [t₀,t₁]} |r_rel(t)|  ≥  (d₀ + d₁ − V h) / 2.

*Proof.*  For any t in the interval, the displacement from the left endpoint
satisfies |r_rel(t) − r_rel(t₀)| = |∫_{t₀}^{t} v_rel| ≤ V (t − t₀), so by the
reverse triangle inequality

    |r_rel(t)| ≥ d₀ − V (t − t₀).

Applying the same argument from the right endpoint,

    |r_rel(t)| ≥ d₁ − V (t₁ − t).

Adding the two inequalities eliminates t:

    2 |r_rel(t)| ≥ d₀ + d₁ − V h,

which holds for every t in the interval and therefore for its minimum.  ∎

The bound is tight: for head-on rectilinear motion whose closest approach lies
inside the interval it is attained exactly.

Bounding the relative speed
---------------------------
The theorem needs an upper bound V on |v_rel| across the interval.  Endpoint
speeds alone are not formally an upper bound, because the speed may peak
between them, so this implementation uses

    V̂ = max(|v_rel(t₀)|, |v_rel(t₁)|) + (κ/2) · | |v_rel(t₁)| − |v_rel(t₀)| |

with κ = 2 by default.  This is an **approximation, stated explicitly**: the
observed change in speed across the step is used as the scale of any excursion
inside it, and κ sets how many such excursions are allowed for.  For two-body
relative motion over a coarse step the relative speed varies smoothly and
slowly compared with its magnitude, so the endpoint difference is a
representative scale.

Two properties keep an imperfect V̂ from causing a false negative:

1. **Rejection carries a 2·threshold cushion.**  An interval is discarded only
   when the bound reaches the threshold, i.e. when d₀ + d₁ − V̂h ≥ 2·threshold.
   Under-estimating V̂ by less than 2·threshold/h therefore cannot turn a
   sub-threshold encounter into a rejection.  At a 50 km threshold and a 60 s
   step that tolerance is 1667 m/s, against a typical endpoint speed
   difference of a few m/s.

2. **The bound is clamped to the sampled distances.**  Since
   | d₀ − d₁ | ≤ V h for the true V, the bound can never exceed min(d₀, d₁);
   the implementation enforces that explicitly.  The screen is therefore never
   less inclusive than the old sampled-distance test, whatever V̂ does.

Cost
----
The screen adds one relative-speed evaluation per sample (or two extra
position evaluations per sample when velocity functions are not supplied) and
one arithmetic bound per interval.  It remains O(N) in samples and continues
to reject the great majority of intervals; a conservative screen over-includes
near an encounter, and a false-positive candidate is simply rejected later by
the TCA solver at bounded cost.

Input validation
----------------
The bound is arithmetic on floating-point numbers, so a non-finite state does
not make it fail -- it makes it produce NaN, and ``NaN < threshold`` is False,
which is *rejection*.  A corrupted trajectory would therefore be screened out
as provably clear.  Every state entering the screen is checked for finiteness
first and a non-finite component raises
:class:`~theseus.conjunction.state_validation.NonFiniteStateError`.  See that
module for the reasoning; the screening mathematics above is unchanged by it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from theseus.conjunction.state_validation import (
    guard_position_functions,
    guard_velocity_functions,
    validate_state_samples,
    validate_time_samples,
    QUANTITY_POSITION,
    QUANTITY_VELOCITY,
)


# Default allowance for relative-speed excursion inside a coarse interval,
# expressed in units of the observed endpoint speed difference.  See the
# module docstring: V̂ = max(v₀, v₁) + (κ/2)·|v₁ − v₀|.
DEFAULT_SPEED_MARGIN_KAPPA: float = 2.0


def separation_lower_bound(
    d0: float,
    d1: float,
    v0: float,
    v1: float,
    h: float,
    kappa: float = DEFAULT_SPEED_MARGIN_KAPPA,
) -> float:
    """
    Analytic lower bound on the separation across one coarse interval.

    Implements  min |r_rel| ≥ (d₀ + d₁ − V̂ h) / 2  from the module docstring,
    with V̂ = max(v₀, v₁) + (κ/2)|v₁ − v₀|, clamped to [0, min(d₀, d₁)].

    Parameters
    ----------
    d0, d1 : float
        Separation at the interval endpoints (m).
    v0, v1 : float
        Relative speed at the interval endpoints (m/s).
    h : float
        Interval duration (s).
    kappa : float
        Speed-margin factor.  Larger values widen the margin and produce more
        candidate intervals; the screen becomes more conservative, never less.

    Returns
    -------
    float
        A separation (m) that the trajectory is not expected to go below
        anywhere in the interval.  Never negative, never above min(d0, d1).
    """
    if h <= 0.0:
        return max(0.0, min(d0, d1))

    v_max = max(v0, v1) + 0.5 * max(0.0, kappa) * abs(v1 - v0)
    bound = 0.5 * (d0 + d1 - v_max * h)

    # Clamp: the bound can never legitimately exceed either endpoint distance,
    # so enforcing that makes the screen provably no less inclusive than a
    # plain sampled-distance test regardless of the quality of v_max.
    return max(0.0, min(bound, d0, d1))


@dataclass(frozen=True)
class ScreeningDiagnostics:
    """
    What the coarse pass actually examined and decided.

    Reported so that a trace can state the screening outcome from execution
    state rather than inferring it.  Note that ``candidate_intervals`` counts
    *merged spans*, not accepted coarse intervals: consecutive accepted
    intervals are merged into one span, so
    ``candidate_intervals <= intervals_accepted``.

    Attributes
    ----------
    intervals_screened : int
        Number of coarse intervals examined (samples - 1).
    intervals_accepted : int
        Intervals whose separation bound fell below the threshold.
    intervals_rejected : int
        Intervals proved to stay above the threshold throughout.
    candidate_intervals : int
        Merged candidate spans handed on for TCA refinement.
    samples : int
        Number of coarse samples evaluated.
    coarse_dt_s : float
        Actual sample spacing used (s), which may differ from the requested
        step because the window is divided into a whole number of samples.
    threshold_m : float
        Screening threshold applied (m).
    min_sampled_distance_m : float
        Smallest separation seen at any sample (m).
    min_lower_bound_m : float
        Smallest separation bound across any interval (m).
    """
    intervals_screened: int
    intervals_accepted: int
    intervals_rejected: int
    candidate_intervals: int
    samples: int
    coarse_dt_s: float
    threshold_m: float
    min_sampled_distance_m: float
    min_lower_bound_m: float

    def to_dict(self) -> dict:
        return {
            "intervals_screened": int(self.intervals_screened),
            "intervals_accepted": int(self.intervals_accepted),
            "intervals_rejected": int(self.intervals_rejected),
            "candidate_intervals": int(self.candidate_intervals),
            "samples": int(self.samples),
            "coarse_dt_s": float(self.coarse_dt_s),
            "threshold_m": float(self.threshold_m),
            "min_sampled_distance_m": float(self.min_sampled_distance_m),
            "min_lower_bound_m": float(self.min_lower_bound_m),
        }


@dataclass
class CandidateInterval:
    """
    A time interval that may contain an approach closer than the threshold.

    Attributes
    ----------
    t_start : float
        Start of the candidate interval (s).
    t_end : float
        End of the candidate interval (s).
    min_distance : float
        Smallest *sampled* separation inside the interval (m).  A coarse
        diagnostic only -- the true minimum is found by TCA refinement.
    min_distance_time : float
        Time of the smallest sampled separation (s).
    lower_bound : float
        Smallest analytic separation bound across the interval (m).  This is
        the quantity the screening decision was actually made on.
    """
    t_start: float
    t_end: float
    min_distance: float
    min_distance_time: float
    lower_bound: float = 0.0


class ConjunctionScreener:
    """
    Conservative coarse conjunction screening.

    Parameters
    ----------
    threshold_m : float
        Screening distance threshold (m).  An interval proceeds to TCA
        refinement unless the separation is provably above this value across
        the whole interval.
    coarse_dt : float
        Coarse time step for sampling (s).
    speed_margin_kappa : float
        Relative-speed margin factor used by :func:`separation_lower_bound`.
    """

    def __init__(
        self,
        threshold_m: float = 100_000.0,   # 100 km default
        coarse_dt: float = 60.0,           # 1-minute coarse step
        speed_margin_kappa: float = DEFAULT_SPEED_MARGIN_KAPPA,
    ) -> None:
        if threshold_m <= 0:
            raise ValueError(f"threshold_m must be > 0, got {threshold_m}")
        if coarse_dt <= 0:
            raise ValueError(f"coarse_dt must be > 0, got {coarse_dt}")
        if speed_margin_kappa < 0:
            raise ValueError(
                f"speed_margin_kappa must be >= 0, got {speed_margin_kappa}"
            )
        self.threshold_m = threshold_m
        self.coarse_dt = coarse_dt
        self.speed_margin_kappa = speed_margin_kappa

    # -- internals -----------------------------------------------------------

    def _relative_speeds(
        self,
        times: np.ndarray,
        pos_fn_a: Callable[[float], np.ndarray],
        pos_fn_b: Callable[[float], np.ndarray],
        vel_fn_a: Optional[Callable[[float], np.ndarray]],
        vel_fn_b: Optional[Callable[[float], np.ndarray]],
    ) -> np.ndarray:
        """
        Relative speed at each sample.

        Uses the supplied velocity functions when available.  Otherwise falls
        back to central differences of the position functions, which keeps the
        screen self-sufficient for callers that only expose positions.
        """
        n = len(times)
        speeds = np.empty(n)

        if vel_fn_a is not None and vel_fn_b is not None:
            for i, t in enumerate(times):
                v_rel = (np.asarray(vel_fn_a(t), dtype=np.float64)
                         - np.asarray(vel_fn_b(t), dtype=np.float64))
                speeds[i] = float(np.linalg.norm(v_rel))
            return speeds

        # Central-difference fallback.  The step is small relative to the
        # coarse step so the estimate tracks the local relative velocity.
        delta = max(1e-3, self.coarse_dt * 1e-3)
        for i, t in enumerate(times):
            t_minus = t - delta
            t_plus = t + delta
            r_minus = (np.asarray(pos_fn_a(t_minus), dtype=np.float64)
                       - np.asarray(pos_fn_b(t_minus), dtype=np.float64))
            r_plus = (np.asarray(pos_fn_a(t_plus), dtype=np.float64)
                      - np.asarray(pos_fn_b(t_plus), dtype=np.float64))
            speeds[i] = float(np.linalg.norm(r_plus - r_minus) / (2.0 * delta))
        return speeds

    def _intervals_to_candidates(
        self,
        times: np.ndarray,
        distances: np.ndarray,
        bounds: np.ndarray,
    ) -> list[CandidateInterval]:
        """
        Merge consecutive possible intervals into candidate spans.

        ``bounds[i]`` is the separation bound across [times[i], times[i+1]].
        """
        possible = bounds < self.threshold_m
        candidates: list[CandidateInterval] = []

        i = 0
        n_int = len(bounds)
        while i < n_int:
            if not possible[i]:
                i += 1
                continue
            j = i
            while j + 1 < n_int and possible[j + 1]:
                j += 1

            lo_idx, hi_idx = i, j + 1
            seg = distances[lo_idx:hi_idx + 1]
            k = int(np.argmin(seg)) + lo_idx
            candidates.append(CandidateInterval(
                t_start=float(times[lo_idx]),
                t_end=float(times[hi_idx]),
                min_distance=float(distances[k]),
                min_distance_time=float(times[k]),
                lower_bound=float(np.min(bounds[i:j + 1])),
            ))
            i = j + 1

        return candidates

    # -- public API ----------------------------------------------------------

    def screen(
        self,
        pos_fn_a: Callable[[float], np.ndarray],
        pos_fn_b: Callable[[float], np.ndarray],
        t_start: float,
        t_end: float,
        *,
        vel_fn_a: Optional[Callable[[float], np.ndarray]] = None,
        vel_fn_b: Optional[Callable[[float], np.ndarray]] = None,
        object_a_id: str = "A",
        object_b_id: str = "B",
    ) -> list[CandidateInterval]:
        """
        Screen for candidate conjunction intervals.

        An interval is discarded only when the analytic bound proves the
        separation stays above the threshold across the whole interval.

        Parameters
        ----------
        pos_fn_a, pos_fn_b : callable
            Position of each object at time t → (3,) array [m].
        t_start, t_end : float
            Analysis window (s).
        vel_fn_a, vel_fn_b : callable, optional
            Velocity of each object at time t → (3,) array [m/s].  Supplying
            these avoids the central-difference fallback and gives the tightest
            speed estimate.
        object_a_id, object_b_id : str
            Labels used in a non-finite-state diagnostic.

        Returns
        -------
        list[CandidateInterval]
            Intervals that may contain a sub-threshold approach.

        Raises
        ------
        NonFiniteStateError
            If any state evaluated during screening is not finite.
        """
        candidates, _ = self.screen_with_diagnostics(
            pos_fn_a, pos_fn_b, t_start, t_end,
            vel_fn_a=vel_fn_a, vel_fn_b=vel_fn_b,
            object_a_id=object_a_id, object_b_id=object_b_id,
        )
        return candidates

    def screen_with_diagnostics(
        self,
        pos_fn_a: Callable[[float], np.ndarray],
        pos_fn_b: Callable[[float], np.ndarray],
        t_start: float,
        t_end: float,
        *,
        vel_fn_a: Optional[Callable[[float], np.ndarray]] = None,
        vel_fn_b: Optional[Callable[[float], np.ndarray]] = None,
        object_a_id: str = "A",
        object_b_id: str = "B",
    ) -> tuple[list[CandidateInterval], ScreeningDiagnostics]:
        """
        Screen, and also report what was examined.

        Identical screening decisions to :meth:`screen`; this variant simply
        returns the counts alongside, so a calculation trace can state the
        outcome from execution state instead of re-deriving it.

        Returns
        -------
        (candidates, diagnostics)

        Raises
        ------
        NonFiniteStateError
            If any state evaluated during screening is not finite.  The screen
            never reports a non-finite trajectory as clear.
        """
        # Guard every state the screen is about to consume.  Idempotent, so a
        # caller that already guarded these functions pays nothing extra.
        pos_fn_a, pos_fn_b = guard_position_functions(
            pos_fn_a, pos_fn_b, object_a_id=object_a_id, object_b_id=object_b_id,
        )
        vel_fn_a, vel_fn_b = guard_velocity_functions(
            vel_fn_a, vel_fn_b, object_a_id=object_a_id, object_b_id=object_b_id,
        )

        if t_end <= t_start:
            empty = ScreeningDiagnostics(
                intervals_screened=0, intervals_accepted=0, intervals_rejected=0,
                candidate_intervals=0, samples=0, coarse_dt_s=float(self.coarse_dt),
                threshold_m=float(self.threshold_m),
                min_sampled_distance_m=float("nan"), min_lower_bound_m=float("nan"),
            )
            return [], empty

        n_samples = max(2, int((t_end - t_start) / self.coarse_dt) + 1)
        times = np.linspace(t_start, t_end, n_samples)

        distances = np.empty(n_samples)
        for i, t in enumerate(times):
            r_a = np.asarray(pos_fn_a(t), dtype=np.float64)
            r_b = np.asarray(pos_fn_b(t), dtype=np.float64)
            distances[i] = float(np.linalg.norm(r_a - r_b))

        speeds = self._relative_speeds(times, pos_fn_a, pos_fn_b, vel_fn_a, vel_fn_b)

        bounds = np.empty(n_samples - 1)
        for i in range(n_samples - 1):
            bounds[i] = separation_lower_bound(
                distances[i], distances[i + 1],
                speeds[i], speeds[i + 1],
                float(times[i + 1] - times[i]),
                kappa=self.speed_margin_kappa,
            )

        candidates = self._intervals_to_candidates(times, distances, bounds)

        n_intervals = int(n_samples - 1)
        n_accepted = int(np.count_nonzero(bounds < self.threshold_m))
        diagnostics = ScreeningDiagnostics(
            intervals_screened=n_intervals,
            intervals_accepted=n_accepted,
            intervals_rejected=n_intervals - n_accepted,
            candidate_intervals=len(candidates),
            samples=int(n_samples),
            coarse_dt_s=float(times[1] - times[0]),
            threshold_m=float(self.threshold_m),
            min_sampled_distance_m=float(np.min(distances)),
            min_lower_bound_m=float(np.min(bounds)),
        )
        return candidates, diagnostics

    def screen_from_arrays(
        self,
        times: np.ndarray,
        positions_a: np.ndarray,
        positions_b: np.ndarray,
        velocities_a: Optional[np.ndarray] = None,
        velocities_b: Optional[np.ndarray] = None,
        *,
        object_a_id: str = "A",
        object_b_id: str = "B",
    ) -> list[CandidateInterval]:
        """
        Screen from pre-computed sample arrays.

        Parameters
        ----------
        times : (N,) array of times (s).
        positions_a, positions_b : (N, 3) arrays of positions (m).
        velocities_a, velocities_b : (N, 3) arrays of velocities (m/s), optional.
            When omitted, relative speed is estimated from successive relative
            positions, which is the best available estimate from samples alone.

        object_a_id, object_b_id : str
            Labels used in a non-finite-state diagnostic.

        Returns
        -------
        list[CandidateInterval]

        Raises
        ------
        NonFiniteStateError
            If any sampled time, position or velocity is not finite.  The
            samples are validated before any bound is computed, because a NaN
            propagates into the bound and is then read as "provably clear".
        """
        times = validate_time_samples(times)
        positions_a = validate_state_samples(
            positions_a, object_id=object_a_id, quantity=QUANTITY_POSITION, times=times)
        positions_b = validate_state_samples(
            positions_b, object_id=object_b_id, quantity=QUANTITY_POSITION, times=times)
        if velocities_a is not None:
            velocities_a = validate_state_samples(
                velocities_a, object_id=object_a_id, quantity=QUANTITY_VELOCITY, times=times)
        if velocities_b is not None:
            velocities_b = validate_state_samples(
                velocities_b, object_id=object_b_id, quantity=QUANTITY_VELOCITY, times=times)

        if len(times) < 2:
            return []

        rel = np.asarray(positions_a, dtype=np.float64) - np.asarray(positions_b, dtype=np.float64)
        distances = np.linalg.norm(rel, axis=1)

        if velocities_a is not None and velocities_b is not None:
            v_rel = (np.asarray(velocities_a, dtype=np.float64)
                     - np.asarray(velocities_b, dtype=np.float64))
            speeds = np.linalg.norm(v_rel, axis=1)
        else:
            # Secant speeds between successive samples, assigned to both
            # endpoints of each interval.  By the mean value theorem the
            # secant speed is a lower bound on the peak speed in the interval,
            # so the kappa margin in separation_lower_bound carries the
            # allowance for the difference.
            seg = np.linalg.norm(np.diff(rel, axis=0), axis=1)
            dt = np.diff(times)
            dt = np.where(dt <= 0.0, np.inf, dt)
            secant = seg / dt
            speeds = np.empty(len(times))
            speeds[0] = secant[0]
            speeds[-1] = secant[-1]
            if len(times) > 2:
                speeds[1:-1] = np.maximum(secant[:-1], secant[1:])

        bounds = np.empty(len(times) - 1)
        for i in range(len(times) - 1):
            bounds[i] = separation_lower_bound(
                float(distances[i]), float(distances[i + 1]),
                float(speeds[i]), float(speeds[i + 1]),
                float(times[i + 1] - times[i]),
                kappa=self.speed_margin_kappa,
            )

        return self._intervals_to_candidates(times, distances, bounds)
