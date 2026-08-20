"""
Conjunction screening — coarse-pass filter.

Given two objects' state histories, identifies time intervals where
the separation distance is below a configurable threshold.

Pipeline:
    STATE HISTORY A
    STATE HISTORY B
        ↓
    RELATIVE POSITION at each sample time
        ↓
    DISTANCE BELOW THRESHOLD?
        ↓
    CANDIDATE INTERVALS for TCA refinement

The screening operates on actual propagated/interpolated positions.
It does NOT declare a conjunction merely because orbital radii overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class CandidateInterval:
    """
    A time interval where two objects are within the screening threshold.

    Attributes
    ----------
    t_start : float
        Start of the candidate interval (s).
    t_end : float
        End of the candidate interval (s).
    min_distance : float
        Minimum separation found during coarse pass (m).
    min_distance_time : float
        Time of minimum separation in coarse pass (s).
    """
    t_start: float
    t_end: float
    min_distance: float
    min_distance_time: float


class ConjunctionScreener:
    """
    Coarse conjunction screening.

    Parameters
    ----------
    threshold_m : float
        Screening distance threshold (m).
        Only intervals where |r₁ − r₂| < threshold proceed to TCA refinement.
    coarse_dt : float
        Coarse time step for sampling (s).
    """

    def __init__(
        self,
        threshold_m: float = 100_000.0,   # 100 km default
        coarse_dt: float = 60.0,           # 1-minute coarse step
    ) -> None:
        if threshold_m <= 0:
            raise ValueError(f"threshold_m must be > 0, got {threshold_m}")
        if coarse_dt <= 0:
            raise ValueError(f"coarse_dt must be > 0, got {coarse_dt}")
        self.threshold_m = threshold_m
        self.coarse_dt = coarse_dt

    def screen(
        self,
        pos_fn_a: Callable[[float], np.ndarray],
        pos_fn_b: Callable[[float], np.ndarray],
        t_start: float,
        t_end: float,
    ) -> list[CandidateInterval]:
        """
        Screen for candidate conjunction intervals.

        Parameters
        ----------
        pos_fn_a : callable
            Function returning position of object A at time t → (3,) array [m].
        pos_fn_b : callable
            Function returning position of object B at time t → (3,) array [m].
        t_start : float
            Start of analysis window (s).
        t_end : float
            End of analysis window (s).

        Returns
        -------
        list[CandidateInterval]
            Intervals where separation < threshold.
        """
        if t_end <= t_start:
            return []

        # Sample at coarse time steps
        n_samples = max(2, int((t_end - t_start) / self.coarse_dt) + 1)
        times = np.linspace(t_start, t_end, n_samples)

        distances = np.empty(n_samples)
        for i, t in enumerate(times):
            r_a = np.asarray(pos_fn_a(t), dtype=np.float64)
            r_b = np.asarray(pos_fn_b(t), dtype=np.float64)
            distances[i] = float(np.linalg.norm(r_a - r_b))

        # Find intervals where distance < threshold
        below = distances < self.threshold_m
        candidates: list[CandidateInterval] = []

        in_interval = False
        interval_start = t_start
        min_dist = float("inf")
        min_dist_time = t_start

        for i in range(n_samples):
            if below[i]:
                if not in_interval:
                    # Start of new candidate interval
                    # Extend one step before if possible
                    interval_start = times[max(0, i - 1)]
                    in_interval = True
                    min_dist = distances[i]
                    min_dist_time = times[i]
                else:
                    if distances[i] < min_dist:
                        min_dist = distances[i]
                        min_dist_time = times[i]
            else:
                if in_interval:
                    # End of candidate interval
                    # Extend one step after
                    interval_end = times[min(n_samples - 1, i)]
                    candidates.append(CandidateInterval(
                        t_start=interval_start,
                        t_end=interval_end,
                        min_distance=min_dist,
                        min_distance_time=min_dist_time,
                    ))
                    in_interval = False
                    min_dist = float("inf")

        # Close any open interval
        if in_interval:
            candidates.append(CandidateInterval(
                t_start=interval_start,
                t_end=times[-1],
                min_distance=min_dist,
                min_distance_time=min_dist_time,
            ))

        return candidates

    def screen_from_arrays(
        self,
        times: np.ndarray,
        positions_a: np.ndarray,
        positions_b: np.ndarray,
    ) -> list[CandidateInterval]:
        """
        Screen from pre-computed position arrays.

        Parameters
        ----------
        times : (N,) array of times (s).
        positions_a : (N, 3) array of object A positions (m).
        positions_b : (N, 3) array of object B positions (m).

        Returns
        -------
        list[CandidateInterval]
        """
        if len(times) < 2:
            return []

        distances = np.linalg.norm(positions_a - positions_b, axis=1)
        below = distances < self.threshold_m
        candidates: list[CandidateInterval] = []

        in_interval = False
        interval_start = times[0]
        min_dist = float("inf")
        min_dist_time = times[0]

        for i in range(len(times)):
            if below[i]:
                if not in_interval:
                    interval_start = times[max(0, i - 1)]
                    in_interval = True
                    min_dist = float(distances[i])
                    min_dist_time = float(times[i])
                else:
                    if distances[i] < min_dist:
                        min_dist = float(distances[i])
                        min_dist_time = float(times[i])
            else:
                if in_interval:
                    interval_end = float(times[min(len(times) - 1, i)])
                    candidates.append(CandidateInterval(
                        t_start=float(interval_start),
                        t_end=interval_end,
                        min_distance=min_dist,
                        min_distance_time=min_dist_time,
                    ))
                    in_interval = False
                    min_dist = float("inf")

        if in_interval:
            candidates.append(CandidateInterval(
                t_start=float(interval_start),
                t_end=float(times[-1]),
                min_distance=min_dist,
                min_distance_time=min_dist_time,
            ))

        return candidates
